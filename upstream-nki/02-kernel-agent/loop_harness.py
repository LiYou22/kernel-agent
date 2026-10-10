#!/usr/bin/env python3
"""
loop_harness.py — the agent's feedback loop, run against a scripted model.

agent.py's loop is: prompt -> reply -> checker -> one instruction -> next prompt. test_verifier.py
checks the checker. This checks the LOOP: what each prompt carried, what the loop did with each
kind of reply, and when it stopped. The model is replaced by a list of canned replies, so a run is
exact, repeatable and takes seconds, and a claim like "a truncated reply is never graded" is a
test rather than something read off one live log.

    python loop_harness.py                       # every scenario, pass or fail
    python loop_harness.py --list
    python loop_harness.py --show repair         # one scenario's prompts and replies, in full
    python loop_harness.py --script s.json --level 2 [--plan-tokens 250 ...]

A script file is a JSON list of replies in the order the model would give them. Each is a string,
or [finish_reason, text] to script a truncated answer: ["length", "```python\\nimport nki"].
`@reference` stands for the level's shipped reference kernel. Anything after the known flags is
passed to agent.py's own command line, so the loop runs with its real options and defaults.

Kernels are really simulated, so this needs the Neuron SDK. No endpoint, no tokens: token counts
here are chars/4 estimates.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field

import agent
from test_verifier import BUGS, REF, broken


def block(src):
    return f"```python\n{src}\n```"


@dataclass
class Run:
    """Everything one scripted run of a level did."""
    result: dict                                    # what solve() returned
    calls: list = field(default_factory=list)       # every model call: prompt, max_tokens, reply
    records: list = field(default_factory=list)     # the JSONL log, parsed
    output: str = ""                                # what the loop printed
    bank: dict = field(default_factory=dict)        # the memory bank afterwards, if one was used

    @property
    def prompts(self):
        """The prompts of the calls that asked for code (not the planning calls)."""
        return [c["prompt"] for c in self.calls if not c["max_tokens"]]

    @property
    def attempts(self):
        return [r for r in self.records if r["type"] == "attempt"]

    @property
    def keys(self):
        """The failure class of each round, in order."""
        return self.result["keys"]


def run(level, replies, bank=None, **opts):
    """Run agent.solve() on one level with `replies` in place of the model.

    `opts` are agent.py's command-line options by their Python names (rounds=6, plan_tokens=250).
    `bank` is a memory-bank path; without one the run neither reads nor writes a bank.
    """
    argv = ["--level", str(level), "--base", "http://scripted/v1", "--samples", "1",
            "--memory", bank or "none"]
    for k, v in opts.items():
        flag = "--" + k.replace("_", "-")
        argv += [flag] if v is True else [] if v is False else [flag, str(v)]
    a = agent.options(argv)
    a.tokens = agent.Tokens(a)
    a.tokens.exact, a.tokens.url = False, None      # no server: chars/4
    a.profile = agent.ContextProfile(a.context)

    script, calls, lock = list(replies), [], threading.Lock()

    def scripted(a_, prompt, max_tokens=None):
        with lock:
            r = script.pop(0) if script else ""
            finish, text = ("stop", r) if isinstance(r, str) else r
            calls.append(dict(prompt=prompt, max_tokens=max_tokens, finish=finish, reply=text))
        return agent.Reply(text, finish, len(prompt) // 4, len(text) // 4)

    log, out, real = io.StringIO(), io.StringIO(), agent.ask
    agent.ask = scripted
    try:
        with contextlib.redirect_stdout(out):
            result = agent.solve(a, level, log)
    finally:
        agent.ask = real
    return Run(result, calls, [json.loads(line) for line in log.getvalue().splitlines()],
               out.getvalue(), agent.load_bank(bank) if bank else {})


def transcript(r):
    """A run as a readable exchange: each prompt, the reply to it, then what the loop concluded."""
    out = []
    for i, c in enumerate(r.calls):
        kind = f"PLANNING CALL (max_tokens={c['max_tokens']})" if c["max_tokens"] else "CODE CALL"
        out += [f"\n>>>>>>>> call {i}: {kind}, prompt of {len(c['prompt'])} chars >>>>>>>>",
                c["prompt"], f"<<<<<<<< reply (finish={c['finish']}) <<<<<<<<", c["reply"]]
    out += ["\n======== what the loop printed ========", r.output,
            "======== result ========", json.dumps(r.result, indent=2)]
    return "\n".join(out)


# ---------------------------------------------------------------- scenarios
#
# Each one scripts the model and asserts what the loop must do with it. They return the Run, so
# --show can print the same exchange.

NO_JIT = broken(2, "@nki.jit\n", "")
FIXED_128 = BUGS[4][2]              # level 1: tile allocated at the limit -> crash:copy-shape
SUM_NOT_MEAN = BUGS[3][2]           # level 1: wrong numbers -> numeric:sum-not-mean
DEV_ONLY = broken(4, '  assert K == K_, "lhsT', '  assert K != 384\n  assert K == K_, "lhsT')


def scenario_repair():
    """One named repair: the verdict becomes one instruction, and the next reply is verified."""
    r = run(2, [block(NO_JIT), block(REF[2])])
    assert r.keys == ["rule:no-jit", "pass"], r.keys
    assert r.result["status"] == "verified" and r.result["confidence"] == 0.95, r.result
    first, repair = r.prompts
    assert "def ref_transpose2d" in first               # the reference goes in the first prompt
    assert "Put @nki.jit on the line directly above" in repair
    assert NO_JIT in repair                             # the code being repaired
    assert "def ref_transpose2d" not in repair          # the reference is not re-sent
    assert "earlier attempts" not in repair             # nothing before this one to recall
    return r


def scenario_first_prompt_has_no_rules_list():
    """Rules live in the checker: the first prompt never enumerates prohibitions."""
    r = run(2, [block(REF[2])])
    assert r.result["status"] == "verified"
    low = r.prompts[0].lower()
    assert not any(w in low for w in ("do not use", "must not", "forbidden", "banned")), low
    return r


def scenario_unusable_replies_are_not_graded():
    """Truncated, empty and code-less replies never reach the checker, and each changes the
    prompt -- the same prompt would get the same non-answer back."""
    r = run(2, [("length", block(REF[2])[:400]), block(NO_JIT), "", "The fix is easy.",
                block(REF[2])], rounds=8)
    assert r.keys == ["reply:truncated", "rule:no-jit", "reply:empty", "reply:no-code", "pass"]
    assert [a["code"] for a in r.attempts if a["key"].startswith("reply:")] == ["", "", ""]
    assert all(a["reward"] == 0.0 for a in r.attempts if a["key"].startswith("reply:"))
    assert len(set(r.prompts)) == len(r.prompts), "a prompt was re-sent unchanged"
    assert len(r.prompts[1]) < len(r.prompts[0])        # after truncation: a shorter first prompt
    assert "code block only" in r.prompts[3]            # after an empty answer to a repair
    assert "earlier attempts" not in r.prompts[4]       # twice running: drop the error memory
    return r


def scenario_same_class_stops_the_level():
    """The same failure class three rounds running, with no progress, is 'stuck'."""
    r = run(1, [block(FIXED_128)] * 8, rounds=8)
    assert r.result["status"] == "stuck" and r.result["rounds"] == 3, r.result
    assert r.keys == ["crash:copy-shape"] * 3
    assert len(r.calls) == 3                            # the remaining rounds were not spent
    assert len(set(r.prompts)) == 3, "a prompt was re-sent unchanged"
    return r


def scenario_progress_is_not_a_repeat():
    """The same class on a LATER line is the kernel getting further, so it does not count
    towards stopping."""
    later = broken(1, "out_tile = nl.ndarray(sum_tile.shape,",
                   "out_tile = nl.ndarray((128, sz_hout, sz_wout),")     # line 52 fixed, 66 not
    r = run(1, [block(FIXED_128), block(FIXED_128), block(later), block(REF[1])], rounds=8)
    assert r.keys == ["crash:copy-shape"] * 3 + ["pass"], r.keys
    assert r.result["status"] == "verified", r.result
    return r


def scenario_error_memory_is_carried():
    """Earlier attempts reach later prompts as one line each, and a repeated error is listed
    once with a count."""
    r = run(1, [block(SUM_NOT_MEAN), block(FIXED_128), block(FIXED_128 + "\n# again"),
                block(REF[1])], rounds=8)
    assert r.keys == ["numeric:sum-not-mean", "crash:copy-shape", "crash:copy-shape", "pass"]
    assert "earlier attempts" not in r.prompts[1]
    third, fourth = r.prompts[2], r.prompts[3]
    assert "first version -> numeric:sum-not-mean" in third     # round 0, recalled by its class
    assert "the checker's report above" in third                # round 1 is the report itself
    assert "MAXIMUM" in third                                   # ...with its instruction in full
    assert fourth.count("- ") >= 2 and "numeric:sum-not-mean" in fourth
    assert "dma_copy requires src and dst" in fourth.split("earlier attempts")[1]
    return r


def scenario_dev_only_is_not_verified():
    """A kernel that passes the shapes it was repaired against, and not the holdout, is reported
    as dev-only -- and the holdout report never reaches the model."""
    r = run(4, [block(DEV_ONLY)])
    assert r.result["status"] == "dev-only" and r.result["confidence"] == 0.3, r.result
    assert r.result["reward"] == 1.0 and r.result["holdout"] == "2/5", r.result
    assert len(r.calls) == 1 and "K=384" not in r.prompts[0]
    return r


def scenario_memory_bank_feeds_the_next_run():
    """What one run learned is in the next run's first prompt, each error once."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bank.json")
        one = run(1, [block(FIXED_128)] * 3, bank=path, rounds=3)
        assert "Notes from your earlier runs" not in one.prompts[0]
        assert len(one.bank["1"]["failed_patterns"]) == 1, one.bank     # 3 rounds, 1 error
        two = run(1, [block(FIXED_128), block(REF[1])], bank=path)
        assert "Notes from your earlier runs" in two.prompts[0]
        assert "dma_copy requires src and dst" in two.prompts[0]
        assert "Notes from your earlier runs" not in two.prompts[1]     # first prompt only
        entry = two.bank["1"]
        assert entry["best_known"]["confidence"] == 0.95, entry["best_known"]
        assert any(w.startswith("solved with") for w in entry["successful_patterns"]), entry
        # the fix superseded the failed pattern: the error is in the bank once, with its cure
        assert agent.memory_text(entry).count("dma_copy requires src and dst") == 1
        return two


def scenario_plan_then_code():
    """--plan-tokens: a capped planning call, then a coding call that carries the plan."""
    r = run(2, [("length", "- read shapes from in_tensor.shape\n- loop f1 and f2\n- then cop"),
                block(NO_JIT), "- add the decorator", block(REF[2])], plan_tokens=250)
    assert [c["max_tokens"] for c in r.calls] == [250, None, 250, None]
    assert r.keys == ["rule:no-jit", "pass"] and r.result["status"] == "verified"
    plan_call, code_call = r.calls[0]["prompt"], r.calls[1]["prompt"]
    assert "Do not write the code yet" in plan_call and "Do not write the code yet" not in code_call
    assert "- loop f1 and f2" in code_call and "then cop" not in code_call   # cut line dropped
    assert r.attempts[0]["plan_finish"] == "length" and r.attempts[0]["sections"]["plan"] > 0
    assert r.result["context"]["calls"] == 4            # the profiler counts both kinds of call
    assert "- loop f1 and f2" not in r.calls[2]["prompt"]   # a plan is not carried to next round
    return r


def scenario_token_split_is_logged():
    """Every attempt logs where its input tokens went and how full the context was."""
    r = run(2, [block(NO_JIT), block(REF[2])])
    first, repair = r.attempts
    assert set(first["sections"]) == {"task"} and first["sections"]["task"] > 0
    assert {"code", "feedback", "task"} <= set(repair["sections"])
    for a in r.attempts:
        c = a["context"]
        assert c["used"] == c["prompt"] + c["answer"] and c["limit"] == 4096
        assert abs(c["share"] - c["used"] / 4096) < 1e-3
    return r


def scenario_tiling_guide_goes_to_tiling_levels_only():
    """Levels with a shape larger than one tile get the tiling guide in the first prompt; the
    others, and an ordinary repair prompt, do not."""
    marker = "GPU Tiling Optimization Guide"
    assert [n for n in sorted(agent.nkibench.LEVELS) if agent.needs_tiling(n)] == [4, 5, 6, 7]
    tiled = run(4, [block(broken(4, "@nki.jit\n", "")), block(REF[4])])
    assert marker in tiled.prompts[0] and marker not in tiled.prompts[1]
    assert tiled.attempts[0]["sections"]["guide"] > 0 and "guide" not in tiled.attempts[1]["sections"]
    assert marker not in run(3, [block(REF[3])]).prompts[0]         # single tile: no guide
    assert marker not in run(4, [block(REF[4])], tiling_guide="none").prompts[0]
    return tiled


def scenario_untiled_dimension_gets_a_restructure_round():
    """A failure that means a dimension is not tiled is followed by a restructure round: the
    tiling guide is in the repair prompt, and it asks for loops instead of a one-line edit."""
    marker, keep = "GPU Tiling Optimization Guide", "keep everything else identical"
    no_k_loop = broken(4, """        nisa.dma_copy(dst=rhs_tile, 
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])""",
                       "        nisa.dma_copy(dst=rhs_tile, src=rhs[:, n * TILE_N:(n + 1) * TILE_N])")
    r = run(4, [block(no_k_loop), block(broken(4, "@nki.jit\n", "")), block(REF[4])])
    assert r.keys == ["crash:copy-shape", "rule:no-jit", "pass"], r.keys
    first, restructure, ordinary = r.prompts
    assert marker in restructure and "Restructure the kernel" in restructure
    assert keep not in restructure
    assert restructure.index("A checker reports") < restructure.index(marker)
    assert r.attempts[0]["restructure"] and r.attempts[1]["sections"]["guide"] > 0
    # the round after it is an ordinary repair again: no guide, one named change
    assert marker not in ordinary and keep in ordinary and "guide" not in r.attempts[2]["sections"]
    # the same class on a level that fits in one tile is not a tiling problem
    small = run(1, [block(FIXED_128), block(REF[1])])
    assert small.keys[0] == "crash:copy-shape" and keep in small.prompts[1]
    assert marker not in small.prompts[1]
    # with the guide off, the round is still a restructure round, without the guide
    off = run(4, [block(no_k_loop), block(REF[4])], tiling_guide="none")
    assert "Restructure the kernel" in off.prompts[1] and marker not in off.prompts[1]
    return r


def scenario_rounds_run_out():
    """Different failures every round is not 'stuck': the level ends as 'failed'."""
    r = run(1, [block(SUM_NOT_MEAN), block(FIXED_128), block(SUM_NOT_MEAN)], rounds=3)
    assert r.result["status"] == "failed" and r.result["confidence"] == 0.0, r.result
    assert r.result["reward"] == 0.625                  # the best round, 1 of 4 shapes
    return r


SCENARIOS = {name[len("scenario_"):]: fn for name, fn in list(globals().items())
             if name.startswith("scenario_")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="name the scenarios and stop")
    ap.add_argument("--show", metavar="SCENARIO", help="print one scenario's full exchange")
    ap.add_argument("--script", help="a JSON list of replies to run instead of the scenarios")
    ap.add_argument("--level", type=int, default=1, help="the level --script runs on")
    a, rest = ap.parse_known_args()

    if a.list:
        for name, fn in SCENARIOS.items():
            print(f"{name:36s} {fn.__doc__.split(chr(10))[0]}")
        return 0
    if a.script:
        replies = [tuple(r) if isinstance(r, list) else r for r in json.load(open(a.script))]
        ref = lambda t: block(REF[a.level]) if t.strip() == "@reference" else t
        replies = [ref(r) if isinstance(r, str) else (r[0], ref(r[1])) for r in replies]
        opts = {}
        while rest:                                      # the rest is agent.py's own options
            key = rest.pop(0).lstrip("-").replace("-", "_")
            opts[key] = rest.pop(0) if rest and not rest[0].startswith("--") else True
        print(transcript(run(a.level, replies, **opts)))
        return 0
    if a.show:
        if a.show not in SCENARIOS:
            sys.exit(f"no scenario named {a.show!r}; --list names them")
        print(transcript(SCENARIOS[a.show]()))
        return 0

    failed = 0
    for name, fn in SCENARIOS.items():
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {str(e)[:300] or 'assertion failed'}  "
                  f"(python loop_harness.py --show {name})")
    print(f"{len(SCENARIOS) - failed} of {len(SCENARIOS)} scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
