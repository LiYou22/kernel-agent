"""Second stage: make a verified kernel cheaper without ever making it wrong.

The harness already keeps an efficiency ledger (FLOPs, instructions, HBM bytes, each against the
op's minimum). This loop turns the worst excess into ONE optimisation directive, asks the model for
a rewrite, and accepts it only if it still passes dev AND holdout with no rule violation AND is
measurably cheaper. Correctness never regresses: a rejected rewrite leaves the best kernel in place.

Lives outside kagent/ so the correctness benchmark's code version is unchanged.

    python3 scripts/optimize.py --from runs/ours-L1234-seat-130.jsonl --rounds 4 --out runs/opt.jsonl
    python3 scripts/optimize.py --kernel 5=kernels/hand/l5_softmax.py --rounds 4
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kagent.agent import LiveModel, ScriptedModel, account, render  # noqa: E402
from kagent.extract import extract_code  # noqa: E402
from kagent.harness import verify  # noqa: E402
from kagent.levels import LEVELS  # noqa: E402
from kagent.rules import TILE_C, TILE_R  # noqa: E402
from roofline import measure  # noqa: E402  (scripts/ is on sys.path when run as a script)

TILE = TILE_R * TILE_C
# accept a rewrite only if roofline time (scripts/roofline.py) drops by IMPROVE; these ops are
# memory bound, so trading FLOPs for fewer passes (online softmax) is a win
IMPROVE = 0.05      # a rewrite must cut roofline time by at least 5%
MAX_OPS_GROWTH = 2.0  # instruction count is mostly an artefact of the column-loop rules: reported, capped

# where the excess usually comes from, per op; generic advice otherwise
READ_HINT = {
    "softmax": "Use an online softmax: in ONE pass over the column tiles keep a running max m and a "
               "running sum s per row (when m grows to m_new, rescale s by exp(m - m_new)), then a "
               "second pass writes exp(x - m) / s. That reads x twice instead of three times.",
    "rmsnorm": "Accumulate the sum of squares in the first pass and write the output in the second; "
               "do not read x a third time.",
    "layernorm": "Accumulate the sum and the sum of squared deviations from a running mean "
                 "(Welford) in one pass, then write the output in a second pass.",
    "matmul": "Reuse each loaded tile as much as possible: load an a tile once and use it against "
              "the whole N extent of the b tile before loading the next.",
}


def better(new, old):
    """Roofline time down by IMPROVE (instructions may grow, within reason), or time unchanged and
    instructions down by IMPROVE (per-element loops -> column vectors)."""
    faster = new["time_x"] <= old["time_x"] * (1 - IMPROVE) and new["ops"] <= old["ops"] * MAX_OPS_GROWTH
    leaner = new["time_x"] <= old["time_x"] * 1.01 and new["ops"] <= old["ops"] * (1 - IMPROVE)
    return faster or leaner


def directives(level, m):
    """Optimisation directives, worst excess first."""
    out = []
    if m["reads_x"] > 1 + IMPROVE:
        out.append(("reads", m["reads_x"],
                    f"It loads each input element {m['reads_x']:.2f} times from HBM; the minimum is "
                    f"1.00. Cut the number of passes over the input. "
                    + READ_HINT.get(level.name, "Fuse passes that read the same tiles, and keep "
                                                "values you need instead of reloading them.")))
    if m["flops_x"] is not None and m["flops_x"] > 1 + IMPROVE:
        out.append(("flops", m["flops_x"],
                    f"It performs {m['flops_x']:.2f}x the minimal FLOPs: some values are computed "
                    f"more than once (for example exp in two passes, or a product recomputed per "
                    f"column). Compute each value once and reuse it."))
    if m["elems_per_op"] < TILE / 32:
        out.append(("ops", TILE / max(m["elems_per_op"], 1),
                    f"It issues {m['ops']} instructions averaging {m['elems_per_op']:.0f} elements, "
                    f"while one tile holds {TILE}. Operate on whole column slices such as "
                    f"x[i0:i1, j] instead of single elements, and on whole tiles x[i0:i1, j0:j1] "
                    f"where the work is elementwise on same-shape arrays. Keep reductions as "
                    f"explicit loops over columns."))
    order = {"reads": 0, "flops": 1, "ops": 2}   # memory bound: traffic first
    return sorted(out, key=lambda d: order[d[0]])


def opt_prompt(level, code, m, text):
    return [("task", f"This numpy function `{level.signature}` is correct and must stay correct: "
                     f"{level.task} Tiles are at most {TILE_R} rows x {TILE_C} columns."),
            ("code", f"```python\n{code}```"),
            ("feedback", f"Efficiency on the largest test ({m['case']}): {text}"),
            ("task", "Rewrite it to be cheaper. Keep the same rules (explicit tile loops, no "
                     "whole-array numpy reductions, no fancy indexing, operands of the same shape). "
                     "Reply with the complete function in one ```python code block.")]


@dataclass
class Step:
    level: int
    source: str
    round: int
    target: str
    outcome: str        # accepted | not-cheaper | wrong | holdout-fail | violation | no-code
    before: dict
    after: dict | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    code: str | None = None


def optimise(level, code, model, rounds, source, log, say=print):
    if not verify(level, code).passed:
        say(f"L{level.num} {source}: input kernel does not pass; skipped")
        return None
    best, m = code, measure(level, code)
    first = dict(m)
    say(f"L{level.num} {source}: start  time {m['time_x']:.2f}x min  reads {m['reads_x']:.2f}x  "
        f"flops {(m['flops_x'] or 0):.2f}x  intensity {m['intensity']:.2f} FLOP/B  ops {m['ops']}")
    tried = set()
    for rnd in range(1, rounds + 1):
        todo = [d for d in directives(level, m) if d[0] not in tried]
        if not todo:
            say("   nothing left to optimise" if not tried else "   every direction tried")
            break
        target, _, text = todo[0]
        parts = opt_prompt(level, best, m, text)
        account(model, parts)
        r = model.complete(render(parts), f"opt:{source}:{rnd}")
        new = extract_code(r.content) if r.finish_reason == "stop" else None
        after, outcome = None, "no-code"
        if new:
            nrep = verify(level, new)
            if nrep.violations:
                outcome = "violation"
            elif not nrep.passed:
                outcome = "wrong"
            elif not verify(level, new, holdout=True).passed:
                outcome = "holdout-fail"
            else:
                after = measure(level, new)
                outcome = "accepted" if better(after, m) else "not-cheaper"
        log.write(json.dumps(asdict(Step(level.num, source, rnd, target, outcome, m, after,
                                         r.prompt_tokens, r.completion_tokens, r.latency_s,
                                         new))) + "\n")
        log.flush()
        say(f"   #{rnd} target={target:5s} -> {outcome}"
            + (f"  time {after['time_x']:.2f}x reads {after['reads_x']:.2f}x flops "
               f"{(after['flops_x'] or 0):.2f}x ops {after['ops']}" if after else ""))
        if outcome == "accepted":
            best, m = new, after
        else:
            tried.add(target)
    say(f"   final: time {first['time_x']:.2f}x -> {m['time_x']:.2f}x of the roofline minimum "
        f"(reads {first['reads_x']:.2f}x -> {m['reads_x']:.2f}x, flops {(first['flops_x'] or 0):.2f}x "
        f"-> {(m['flops_x'] or 0):.2f}x, ops {first['ops']} -> {m['ops']})")
    return first, m, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", help="agent JSONL: optimise every verified kernel in it")
    ap.add_argument("--kernel", action="append", default=[], help="LEVEL=path to a kernel file")
    ap.add_argument("--levels", type=int, nargs="*", help="only these levels")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--scripted", help="JSON list of replies, for testing without a model")
    ap.add_argument("--out", default=f"runs/opt-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    a = ap.parse_args()

    jobs = []
    if a.src:
        for line in Path(a.src).read_text().splitlines():
            r = json.loads(line)
            if r.get("type") == "result" and r["status"] == "verified":
                jobs.append((r["level"], r["code"], f"run{r['run']}"))
    for spec in a.kernel:
        n, path = spec.split("=", 1)
        jobs.append((int(n), Path(path).read_text(), Path(path).stem))
    if a.levels:
        jobs = [j for j in jobs if j[0] in a.levels]

    model = ScriptedModel(json.loads(Path(a.scripted).read_text())) if a.scripted else LiveModel()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as log:
        for n, code, source in jobs:
            optimise(LEVELS[n], code, model, a.rounds, source, log)
    print(f"log: {a.out}")


if __name__ == "__main__":
    main()
