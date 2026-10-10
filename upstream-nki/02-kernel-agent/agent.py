#!/usr/bin/env python3
"""
agent.py — the loop: a model writes an NKI kernel, the harness grades it, the reason goes back,
it tries again.

    controller  picks a level                          nkibench.py
    generator   the model writes a kernel             the shared gpt-oss endpoint
    checker     rules, then simulate, then roofline   nkibench.py
    verifier    error bound, holdout, diagnosis       verifier.py
    loop        the failure becomes the next prompt   this file

What the loop adds to "run it and report": a reply that was truncated or empty never reaches the
checker; a verdict becomes ONE instruction with a failure class, and repeats are counted by class;
a kernel that passes is re-checked on holdout cases, which decides the confidence reported; and
every attempt logs its input tokens by section, counted by the server's own tokenizer.

Everything here runs on the CPU, so it does NOT need the Trainium device -- which is the point,
because the device is what Project 2 needs later for real timings, and a Neuron device cannot be
shared by two processes. The model therefore comes from the shared endpoint, not from a server on
your own chip.

    export GPTOSS_BASE_URL="https://..."
    python agent.py --level 1
    python agent.py --level 4 --rounds 6 --samples 4
    python agent.py --all
    python agent.py --offline --level 4        # no model; replays the reference kernel

Every attempt is appended to a JSONL file with its reward, failure class and token split, and
every level ends with a result record (status, confidence), so the log is the deliverable.
"""

import argparse
import ast
import difflib
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass

import numpy as np

import nkibench
import verifier

MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")

# The model writes to a hidden reasoning channel before it writes any answer. Measured on this
# endpoint: a coding task burned 900 tokens thinking and returned EMPTY content. See gptoss/README.
MIN_ANSWER_TOKENS = 2500

REASONING_KEYS = ("reasoning", "reasoning_content")


# ---------------------------------------------------------------- reward
#
# Graded, not pass/fail, so a near miss is distinguishable from nonsense and the loop has
# something to climb. Matches Project 1's shape: correctness dominates, and nothing else counts
# until the kernel is right.

WEIGHTS = dict(parses=0.1, rules=0.2, runs=0.2, correct=0.5)


@dataclass
class Verdict:
    """What the checker concluded about one kernel.

    `feedback` is an INSTRUCTION, never just a verdict. `brief` is the verdict alone, on one line,
    for the history of earlier attempts. `key` is the failure CLASS: the numbers in a message
    change every round and the class does not, so the loop counts repeats by it.
    """
    reward: float
    parts: dict
    feedback: str
    brief: str
    key: str
    passed: int = 0
    total: int = 0
    bound_use: float = None     # share of the float32 error bound the worst case spends
    cost: str = ""              # efficiency on the largest case: reported, not scored
    restructure: bool = False   # the fix is a missing tile loop, not an edit to one line

    @property
    def solved(self):
        return self.parts["correct"]


# What a crash IS, whatever names and numbers it mentions. `nisa.multiply` one round and
# `nisa.scalar_mul` the next are the same mistake -- guessing at an API -- and exact-text matching
# counted them as two.
CRASH_CLASSES = (
    (r"has no attribute", "invented-name"),
    (r"unexpected keyword argument", "invented-argument"),
    (r"got multiple values for argument", "argument-twice"),
    (r"'MemoryRegion' object is not callable", "region-called"),
    (r"same number of elements|could not be broadcast", "copy-shape"),
    (r"partition dimension \d+ exceeds", "partition-limit"),
    (r"contraction dimension \d+ exceeds", "k-limit"),
    (r"(stationary|moving) free dimension \d+ exceeds", "free-limit"),
    (r"Out-of-bound access", "out-of-bounds"),
    (r"must be in \[", "buffer-placement"),
    (r"both must be\. Got", "dtype-mismatch"),
    (r"at least 2 dimensions", "tile-rank"),
    (r"cannot reshape", "reshape"),
    (r"unsupported operand type", "python-operator"),
    (r"is not defined", "undefined-name"),
)


# Failure classes that mean a dimension is not tiled. No one-line edit fixes these -- measured on
# level 4, the model answered each of them by changing a tile's shape, three rounds running, and
# never added a loop -- so the repair round for them is a different kind of round.
UNTILED = ("crash:partition-limit", "crash:k-limit", "crash:free-limit", "numeric:k-overwrite",
           "numeric:tag:multi-")


def crash_class(exc):
    text = str(exc)
    for pattern, name in CRASH_CLASSES:
        if re.search(pattern, text):
            return f"crash:{name}"
    return f"crash:{type(exc).__name__}:{pattern_class(text.split(' (')[0])[:50]}"


def tiling_probe(kernel, level):
    """Which matmul dimensions does this kernel tile? {"K": True, "M": False, "N": True}.

    The level's own shapes cannot say: every one of them that is big in M is big in K too. So run
    the kernel on three shapes that each exceed the tile limit in ONE dimension. A crash or a
    wrong answer there means that dimension is not tiled. Returns {} for any other operation.

    Measured on level 4 without this: the checker reported the first dimension to crash, the
    model fixed K, the next round crashed on M, and the fix for M disturbed K -- one dimension
    per round, never all three right at once.
    """
    if nkibench.LEVELS[level]["ref"] is not nkibench.ref_matmul:
        return {}
    one = dict(K=nkibench.PMAX, M=nkibench.GEMM_STATIONARY_FMAX, N=nkibench.GEMM_MOVING_FMAX)
    tiled = {}
    for dim in one:
        sp = dict(one, **{dim: 2 * one[dim]})
        case = verifier.Case(sp, nkibench.label(sp, level), ())
        args = verifier.make_args(case, level)
        try:
            got, _ = nkibench.simulate_and_count(kernel, args)
            tiled[dim] = verifier.compare(level, case, args, got,
                                          nkibench.ref_matmul(*args)).status == "pass"
        except Exception:
            tiled[dim] = False
    return tiled


TILE_HOW = {
    "K": "K, in chunks of {k}: rows k0:k0 + {k} of BOTH lhsT and rhs. The K chunks of one output "
         "tile must add up, so allocate ONE psum tile before the K loop, call nisa.nc_matmul into "
         "it once per K chunk, and copy it out only after the K loop.",
    "M": "M, in chunks of {m}: columns m0:m0 + {m} of lhsT go into the stationary tile, and the "
         "result goes to rows m0:m0 + {m} of the output.",
    "N": "N, in chunks of {n}: columns n0:n0 + {n} of rhs go into the moving tile, and the result "
         "goes to columns n0:n0 + {n} of the output.",
}


def tiling_report(tiled):
    """The probe as an instruction: every dimension that still needs a loop, named at once, and
    the loops that already work named as things to keep."""
    if not tiled:
        return ""
    lim = dict(k=nkibench.PMAX, m=nkibench.GEMM_STATIONARY_FMAX, n=nkibench.GEMM_MOVING_FMAX)
    missing = [d for d, ok in tiled.items() if not ok]
    kept = [d for d, ok in tiled.items() if ok]
    if not missing:
        return ("Tested one dimension at a time, K, M and N are each tiled correctly; the kernel "
                "fails only when several are large together. The three loops must be NESTED: for "
                "each M chunk, for each N chunk, loop over all K chunks.")
    return ("Tested one dimension at a time: " + " and ".join(missing)
            + (" are NOT tiled" if len(missing) > 1 else " is NOT tiled")
            + (f"; {' and '.join(kept)} {'are' if len(kept) > 1 else 'is'} tiled correctly, so "
               f"keep {'those loops' if len(kept) > 1 else 'that loop'}" if kept else "")
            + f". Fix {'all of them' if len(missing) > 1 else 'it'} in this one reply. "
            "The kernel needs a loop over each of:\n"
            + "\n".join("  - " + TILE_HOW[d].format(**lim) for d in missing)
            + "\nNest them: for each M chunk, for each N chunk, loop over all K chunks. Every "
              "tile has the shape of one chunk, never of the whole tensor.")


def cost_line(level, case, counted, args, want):
    """Efficiency on one case, in the units the device charges: HBM bytes against the floor of
    reading each input once and writing the output once. Reported, not scored -- except on levels
    5 to 7, where nkibench's traffic bar already makes it the level."""
    floor = nkibench.minimum_hbm_bytes(args, want)
    if not counted.get("bytes") or not floor:
        return ""
    line = (f"Cost on {case.label}: {counted['bytes']:,} HBM bytes in {counted['transfers']} "
            f"transfers, {counted['bytes'] / floor:.2f}x the floor.")
    if nkibench.LEVELS[level]["ref"] is nkibench.ref_matmul:
        sp = case.spec
        line += " " + nkibench.explain_roofline(nkibench.roofline(
            nkibench.matmul_flops(sp["M"], sp["K"], sp["N"]), counted["bytes"]))
    return line


def grade(source, level, holdout=False):
    """Check one kernel and return a Verdict.

    The order is the order a repair has to happen in: it must parse, obey the rules, load, run,
    return the right shape, and then be right. Only the first thing wrong is reported, as one
    change to make.

    `holdout` runs the cases the model never sees instead of the level's own shapes. It decides
    how far a dev pass is trusted, and its report never goes back to the model.
    """
    parts = dict(parses=False, rules=False, runs=False, correct=False)
    base = lambda: sum(WEIGHTS[k] for k, v in parts.items() if v)

    if not source.strip():
        return Verdict(0.0, parts, "No code came back. Reply with one python code block "
                       "containing the kernel and nothing else.", "no code came back",
                       "reply:no-code")
    try:
        compile(source, "<candidate>", "exec")
        parts["parses"] = True
    except SyntaxError as e:
        return Verdict(0.0, parts,
                       f"The code does not parse: {e.msg} on line {e.lineno}. Send one complete "
                       f"python code block.", f"does not parse: {e.msg} on line {e.lineno}",
                       "syntax")

    violations = nkibench.check_rules(source, level)
    if violations:
        # Measured: "there is no function named X" repeated 15 rounds running, because it never
        # said what the function should look like. Every violation kind carries its fix, and
        # several kinds are handed over one at a time, structural ones first.
        d = verifier.rule_directive(level, violations, source)
        return Verdict(base(), parts, "A rule is broken, which scores zero however fast the "
                       "kernel is. " + d.text(), d.brief, d.key)
    parts["rules"] = True

    spec = nkibench.LEVELS[level]
    # One file per process: two runs on the same level used to share this path, so each could
    # load and grade the other's kernel.
    path = f"/tmp/_agent_level{level}_{os.getpid()}.py"
    with open(path, "w") as f:
        f.write(source)
    try:
        kernel = nkibench.load_kernel(path, spec["entry"])
    except ModuleNotFoundError as e:
        # Two very different causes, and conflating them aborted a whole run: either the SDK is
        # absent (an environment problem the model cannot fix), or the model imported a module it
        # invented -- observed: `import nki.nl`. Tell them apart by asking whether nki itself is
        # importable.
        try:
            import nki  # noqa: F401
            sdk_present = True
        except ImportError:
            sdk_present = False
        if not sdk_present:
            raise SystemExit(
                f"cannot import {e.name!r}, so no NKI kernel can be loaded here. Run this where "
                f"the Neuron SDK exists -- k8s/kernel-agent-job.yaml does that -- or use "
                f"--offline to exercise the loop without a kernel ever running.") from e
        return Verdict(base(), parts,
                       f"There is no module named {e.name!r}. The only imports that exist are: "
                       f"`import nki`, `import nki.language as nl`, and `import nki.isa as nisa`. "
                       f"Use exactly those three.", f"no module named {e.name!r}", "load:module")
    except Exception as e:
        return Verdict(base(), parts,
                       f"The file imports but {spec['entry']} could not be loaded: "
                       f"{type(e).__name__}: {e}", f"could not be loaded: {type(e).__name__}",
                       "load:crash")

    results, cost, biggest = [], "", -1
    for case in verifier.cases(level, holdout):
        args = verifier.make_args(case, level)
        before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
        want = spec["ref"](*args)
        try:
            got, counted = nkibench.simulate_and_count(kernel, args)
        except nkibench.NkiMissing as e:
            return Verdict(base(), parts, f"CANNOT SIMULATE: {e}", "cannot simulate", "env")
        except Exception as e:
            where, facts = failing_call(e, path, source)
            results.append(verifier.CaseResult(
                case, "error", enrich(f"raised {type(e).__name__}: {e}", where, facts), np.inf,
                brief=crash_brief(e, facts), key=crash_class(e)))
            # a copy whose one side has more rows than a tile holds is an untiled dimension too
            results[-1].oversize = any(sh and sh[0] > nkibench.PMAX
                                       for sh in (facts["shapes"].get(k) for k in ("src", "dst")))
            continue
        parts["runs"] = True
        touched = nkibench.check_inputs_untouched(before, args)
        if touched:
            results.append(verifier.CaseResult(case, "input", touched, np.inf,
                                               brief=touched.split(". ")[0][:140]))
            continue
        r = verifier.compare(level, case, args, got, want)
        if r.status == "pass":
            # A simulator warning about a hardware-correctness hazard counts as a failure even
            # when the numbers match on CPU: the kernel would be wrong on the device.
            hazards = [w for w in counted.get("warnings", [])
                       if "incorrect results on hardware" in w]
            over = None if holdout else nkibench.check_traffic_bar(level, counted, args, want)
            if hazards:
                r = verifier.CaseResult(
                    case, "hazard", "CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0]
                    + ". Fix that before anything else -- the simulator agrees with the reference "
                      "here and the device would not.", brief="wrong on hardware: " + hazards[0][:110])
            elif over:
                r = verifier.CaseResult(case, "traffic", over, brief=over.split(". ")[0][:140])
            elif nkibench.minimum_hbm_bytes(args, want) > biggest:
                biggest = nkibench.minimum_hbm_bytes(args, want)
                cost = cost_line(level, case, counted, args, want)
        results.append(r)

    total = len(results)
    passed = sum(r.status == "pass" for r in results)
    uses = [r.bound_use for r in results if r.status == "pass" and r.bound_use is not None]
    if passed < total:
        crashes = [r for r in results if r.status == "error"]
        if crashes:
            # It has to run before it can be right, so a crash outranks a wrong number. Crashes
            # of one class are one crash, whatever the shapes.
            first = crashes[0]
            same = [r for r in crashes if r.key == first.key]
            ok = [r.case.label for r in results if r.status == "pass"]
            pattern = (f" Passed: {'; '.join(ok)}. Failed: "
                       f"{'; '.join(r.case.label for r in results if r.status != 'pass')}."
                       if ok else "")
            d = verifier.Directive(
                first.key, f"On {first.case.label}: {first.msg}",
                f"{passed} of {total} shapes passed; {len(same)} raised this error."
                + pattern,
                (f"{passed} of {total} shapes passed; on {first.case.label}: " if passed else "")
                + first.brief)
        else:
            d = verifier.diagnose(level, results)
        untiled = needs_tiling(level) and (d.key.startswith(UNTILED) or bool(
            crashes and d.key == "crash:copy-shape" and getattr(crashes[0], "oversize", False)))
        feedback = d.text()
        if untiled and not holdout:
            # The crash above is only the FIRST untiled dimension the kernel ran into.
            report = tiling_report(tiling_probe(kernel, level))
            feedback += ("\n" + report) if report else ""
        return Verdict(base() + WEIGHTS["correct"] * passed / total, parts, feedback, d.brief,
                       d.key, passed, total, max(uses) if uses else None, restructure=untiled)

    parts["correct"] = True
    note = f"Correct on every shape ({total} of {total})."
    if uses:
        note += (f" The worst case uses {max(uses):.0%} of the proven float32 error bound.")
    else:
        note += " (No proven error bound for this operation: empirical tolerance.)"
    if cost:
        note += " " + cost
    return Verdict(sum(WEIGHTS.values()), parts, note, "correct on every shape", "pass",
                   passed, total, max(uses) if uses else None, cost)


# ---------------------------------------------------------------- prompting

# Every name here appears in the three shipped tutorial kernels, so none of it is invented. The
# model does not know this API and guesses plausible names -- nl.scalar, nl.value, nl.dot,
# tile.mean -- none of which exist. A short card of what IS real costs ~200 tokens and is
# documentation rather than the answer.
API_CARD = """Available NKI functions:

  @nki.jit                                  decorate the entry point
  nl.ndarray(shape, dtype=..., buffer=b)    allocate; b is nl.sbuf, nl.psum or nl.shared_hbm
  nl.affine_range(n)                        the loop
  nl.sum(view, axis=[i, j])                 reduce; axis is a list
  nl.float32, nl.bfloat16                   dtypes
  nisa.dma_copy(dst=, src=)                 move data between HBM and SBUF
  nisa.nc_matmul(dst=, stationary=, moving=)   matmul into a PSUM tile
  nisa.tensor_copy(dst=, src=)              copy, e.g. PSUM to SBUF
  nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=0.5)   scale by a constant
  tile.ap([[stride, count], ...])           a strided view, for reductions

nisa.nc_matmul has strict memory rules: dst must live in nl.psum, while stationary and moving must
both live in nl.sbuf. So the pattern is: dma_copy both operands from HBM into sbuf tiles, allocate a
psum tile for the result, call nc_matmul(dst=psum_tile, stationary=..., moving=...), then
tensor_copy from psum into an sbuf tile, then dma_copy that out to the shared_hbm output.
The left operand arrives already transposed, with K on the partition axis.

Slice tiles with ranges, e.g. a[0:128, 0:64]. A complete kernel looks like this:

import nki
import nki.isa as nisa
import nki.language as nl

@nki.jit
def copy_kernel(a):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    tile = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=a)
    nisa.dma_copy(dst=out, src=tile)
    return out

"""


def available_names(dotted):
    """Turn 'no attribute X' into 'here are the real ones'.

    Measured: told only `module 'nki.language' has no attribute 'value'`, the model guessed
    another invented name every round -- value, scalar, sbuf_scalar, dot. A verdict names the
    mistake and never the fix, so list what actually exists and let it choose.
    """
    import difflib
    import importlib
    mod_name, _, attr = dotted.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return ""
    names = [n for n in dir(mod) if not n.startswith("_")]
    close = difflib.get_close_matches(attr, names, n=6, cutoff=0.4)
    if close:
        return (f" `{mod_name}` has no `{attr}`. The closest real names are: "
                f"{', '.join(close)}. Pick one of those or use a different approach.")
    return (f" `{mod_name}` has no `{attr}`, and nothing similar exists. Its real names include: "
            f"{', '.join(sorted(names)[:25])}.")


def real_signature(func_name):
    """The actual signature of an NKI function, for when the model invents arguments."""
    import inspect
    for mod_name in ("nki.language", "nki.isa", "nki"):
        try:
            mod = __import__(mod_name, fromlist=["x"])
        except Exception:
            continue
        fn = getattr(mod, func_name, None)
        if fn is None:
            continue
        try:
            return f"{mod_name.split('.')[-1]}.{func_name}{inspect.signature(fn)}"
        except (TypeError, ValueError):
            return f"{mod_name.split('.')[-1]}.{func_name}"
    return ""


PMAX_NOTE = (f"{nkibench.PMAX} is the MAXIMUM a tile's first dimension may be, not the size to "
             f"allocate.")


def failing_call(exc, path, source):
    """The candidate's own line that raised, and the facts about the call it made.

    The simulator's messages carry element counts or a bare limit. The traceback has the rest:
    the last frame inside the candidate file is the offending line, and the frames below it still
    hold the tensors that were passed. Returns (where, facts); facts has `shapes` (by argument
    name), `expr` (the text the kernel wrote for each argument) and `kind` (what each plain name
    on that line is: an input, or a tile in sbuf, psum or shared_hbm).
    """
    lineno, shapes = None, {}
    tb = exc.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_filename == path:
            lineno, shapes = tb.tb_lineno, {}
        elif lineno:
            loc = tb.tb_frame.f_locals
            for k in ("dst", "src", "value", "stationary", "moving", "self", "shape"):
                if k in shapes or k not in loc:
                    continue
                try:
                    v = loc[k]
                    shapes[k] = tuple(int(n) for n in (v if isinstance(v, (tuple, list))
                                                       else v.shape))
                except Exception:
                    pass
        tb = tb.tb_next
    if "src" not in shapes and "value" in shapes:
        shapes["src"] = shapes["value"]            # `a[...] = b` is a copy too
    lines = source.splitlines()
    line = lines[lineno - 1].strip() if lineno and lineno <= len(lines) else ""
    facts = dict(lineno=lineno, line=line, shapes=shapes, expr={}, kind={}, source=source)
    try:
        stmt = ast.parse(line).body[0]
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in ("dst", "src", "stationary", "moving"):
                        facts["expr"].setdefault(kw.arg, ast.unparse(kw.value))
        if not facts["expr"] and isinstance(stmt, ast.Assign):
            facts["expr"] = dict(dst=ast.unparse(stmt.targets[0]), src=ast.unparse(stmt.value))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    facts["kind"].setdefault(arg.arg, "input")
            if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func).endswith("ndarray")):
                for kw in node.value.keywords:
                    if kw.arg == "buffer":
                        facts["kind"][node.targets[0].id] = ast.unparse(kw.value).split(".")[-1]
    except Exception:
        pass                                       # a multi-line statement; the shapes still stand
    return (f"line {lineno}, `{line}`" if line else ""), facts


def crash_brief(exc, facts):
    """One line for the attempt history: where it raised, what was raised, and the shapes."""
    shapes = facts.get("shapes", {})
    names = dict(dst="dst", src="src", stationary="stationary", moving="moving",
                 shape="shape asked for", self="tensor indexed")
    seen = ", ".join(f"{label} {shapes[k]}" for k, label in names.items()
                     if k in shapes and (k != "self" or "dst" not in shapes))
    at = f"line {facts['lineno']}: " if facts.get("lineno") else ""
    said = str(exc).split(", got ")[0].split(" (")[0][:90]
    return f"{at}{said}" + (f" [{seen}]" if seen else "")


def _tile(name, facts):
    """Is this plain name a tile the kernel allocated on chip (so its shape is the kernel's choice)?"""
    return facts["kind"].get(name) in ("sbuf", "psum")


def _copy_fix(facts, dst_n, src_n):
    """ONE literal edit for a copy whose two sides differ, in the kernel's own names.

    Measured on level 2: told "use a.shape", the model fixed one dimension of two; offered two
    options, it took the wrong one and invented a reshape. So name the line to change and what to
    write there, and offer nothing else.
    """
    shapes, expr = facts.get("shapes", {}), facts.get("expr", {})
    d, s = expr.get("dst", ""), expr.get("src", "")
    for tile, other, other_shape in ((d, s, shapes.get("src")), (s, d, shapes.get("dst"))):
        if not (tile.isidentifier() and _tile(tile, facts) and other):
            continue
        if other.isidentifier() and not _tile(other, facts):
            if other_shape and other_shape[0] > nkibench.PMAX:
                return (f"`{other}` has {other_shape[0]} rows, more than one tile may hold. Loop "
                        f"over its first dimension in chunks of at most {nkibench.PMAX} rows with "
                        f"nl.affine_range, and allocate `{tile}` inside the loop with the shape of "
                        f"the chunk being copied.")
            return (f"Change the one line that allocates `{tile}` so it reads "
                    f"`{tile} = nl.ndarray({other}.shape, dtype={other}.dtype, "
                    f"buffer=nl.{facts['kind'][tile]})`. Do not write that shape as numbers and "
                    f"do not reshape or slice `{other}`.")
        if not other.isidentifier():
            if other_shape and other_shape[0] > nkibench.PMAX:
                # Measured on level 4: the kernel tiled N and took every K row with `:`, and the
                # advice below told it to grow the tile to 256 rows, which no tile may have.
                return (f"`{tile}` is right as it is: the slice `{other}` is too big. It takes "
                        f"{other_shape[0]} rows along its first dimension, and one tile holds at "
                        f"most {nkibench.PMAX}. Tile that dimension too: loop over it in chunks "
                        f"of {nkibench.PMAX} with nl.affine_range, and inside the loop copy only "
                        f"rows k0:k0 + {nkibench.PMAX} of it, never `:`."
                        + (" For a matmul that dimension is K: call nisa.nc_matmul once per "
                           "chunk into ONE psum tile allocated outside that loop, so the chunks "
                           "add up." if "matmul" in facts.get("source", "") else ""))
            return (f"Change the one line that allocates `{tile}` so that its shape is exactly the "
                    f"shape of `{other}`"
                    + (f", which is {other_shape} here" if other_shape else "")
                    + f". Write that shape with the same variables the slice uses, not numbers.")
    big = next((k for k in ("src", "dst") if shapes.get(k) and shapes[k][0] > nkibench.PMAX), "")
    if big:
        # The same mistake when the call is spread over several lines and its text is not known.
        return (f"The {big} side takes {shapes[big][0]} rows along its first dimension, and one "
                f"tile holds at most {nkibench.PMAX}, so do not grow the tile to match. Tile that "
                f"dimension too: loop over it in chunks of {nkibench.PMAX} with nl.affine_range "
                f"and copy only rows k0:k0 + {nkibench.PMAX} per step, never `:`."
                + (" For a matmul that dimension is K: call nisa.nc_matmul once per chunk into "
                   "ONE psum tile allocated outside that loop, so the chunks add up."
                   if "matmul" in facts.get("source", "") else ""))
    return ("Give the on-chip tile exactly the shape of the data on the other side of the copy, "
            "read from that data's .shape rather than written as numbers."
            if dst_n > src_n else
            "Give the destination exactly the shape of the source, read from the source's .shape "
            "rather than written as numbers.")


def enrich(error_text, where="", facts=None):
    """Turn a simulator error into an instruction: where it happened, then what to change."""
    facts = facts or {}
    facts = dict(shapes=facts.get("shapes", {}), expr=facts.get("expr", {}),
                 kind=facts.get("kind", {}), line=facts.get("line", ""),
                 source=facts.get("source", ""))
    hint = _hint(error_text, facts)[len(error_text):]
    return error_text + (f" This is {where}." if where else "") + hint


def _hint(error_text, facts):
    """error_text plus the fix. Every branch returns error_text with a suffix."""
    shapes, expr = facts["shapes"], facts["expr"]
    if "'MemoryRegion' object is not callable" in error_text:
        return (error_text + " nl.sbuf, nl.psum and nl.shared_hbm are memory regions, not "
                "functions. Do not call them. Allocate with "
                "nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf) and pass the region as the "
                "buffer= argument.")
    m = re.search(r"(\w+)\(\) got an unexpected keyword argument '(\w+)'", error_text)
    if m:
        sig = real_signature(m.group(1))
        return (error_text + f" Remove the `{m.group(2)}=` argument."
                + (f" The real signature is {sig}." if sig else ""))
    if "unsupported operand type(s) for" in error_text and "NkiTensor" in error_text:
        return (error_text + " A tile is not a number, so Python operators like += do not work on "
                "one. Accumulate by allocating a PSUM tile with "
                "nl.ndarray(shape, nl.float32, buffer=nl.psum) and letting nisa.nc_matmul add into "
                "it across the loop, or combine two tiles with a nisa op rather than a Python "
                "operator.")
    m = re.search(r"dma_copy requires src and dst to have the same number of elements, "
                  r"got src=(\d+), dst=(\d+)", error_text)
    if m:
        # Measured: the old hint showed a hard-coded (128, 512) tile, and on inputs SMALLER than a
        # tile -- (32, 12) into (128, 128) -- the model kept its fixed 128s and returned the same
        # kernel four rounds running. State both shapes, and never show a number to copy.
        src, dst = int(m.group(1)), int(m.group(2))
        sizes = (f"dst has shape {shapes['dst']} and src has shape {shapes['src']}"
                 if "dst" in shapes and "src" in shapes
                 else f"dst holds {dst} elements and src holds {src}")
        return (error_text + f" Here {sizes}. nisa.dma_copy does not slice, pad or broadcast, so "
                f"the two shapes must be identical. {PMAX_NOTE} {_copy_fix(facts, dst, src)}")
    m = re.search(r"dma_copy (\w+) partition dimension (\d+) exceeds maximum (\d+)", error_text)
    if m:
        which, got, mx = m.group(1), int(m.group(2)), int(m.group(3))
        what = f"`{expr[which]}`" if expr.get(which) else f"the {which}"
        shape = f", which has shape {shapes[which]}," if which in shapes else ""
        return (error_text + f" One copy may move at most {mx} rows, and {what}{shape} has {got}. "
                f"Loop over that first dimension in chunks with nl.affine_range: inside the loop, "
                f"copy one chunk of at most {mx} rows and allocate its tile with that chunk's own "
                f"shape. Write the result back one chunk at a time in the same way. A dimension "
                f"that is already {mx} or smaller is used whole -- never pad it up to {mx}.")
    m = re.search(r"value array of shape \((\d+),?\) could not be broadcast to "
                  r"indexing result of shape \((\d+),?\)", error_text)
    if m:
        val, dst = int(m.group(1)), int(m.group(2))
        sizes = (f"the destination has shape {shapes['dst']} and the value has shape "
                 f"{shapes['src']}" if "dst" in shapes and "src" in shapes
                 else f"the destination holds {dst} elements and the value holds {val}")
        return (error_text + f" Here {sizes}. A copy or an assignment never reshapes or "
                f"broadcasts, so both sides must have the same shape. "
                f"{_copy_fix(facts, dst, val)}")
    m = re.search(r"Out-of-bound access for tensor .*? on dimension (\d+): "
                  r"index (?:range \[\d+, )?(\d+)\]? exceed dimension size of (\d+)",
                  error_text)
    if m:
        dim, hi, size = m.group(1), int(m.group(2)), int(m.group(3))
        whole = f" of a tensor whose shape is {shapes['self']}" if "self" in shapes else ""
        return (error_text + f" You indexed position {hi} on dimension {dim}{whole}, and that "
                f"dimension is only {size} long. Tile limits are a MAXIMUM, not a target. Whatever "
                f"produces that index -- a slice bound or a loop range -- must be read from the "
                f"tensor's own .shape[{dim}] instead of being a fixed number; in a chunked loop, "
                f"end each chunk at min(start + chunk, that size) so the last one is partial.")
    m = re.search(r"Matmul contraction dimension (\d+) exceeds pmax=(\d+)", error_text)
    if m:
        k, mx = int(m.group(1)), int(m.group(2))
        return (error_text + f" The contraction dimension K is {k} and one nc_matmul can only "
                f"contract {mx}. Split K into chunks of {mx} and accumulate: allocate ONE psum tile "
                f"OUTSIDE the K loop, call nisa.nc_matmul into that same psum tile once per chunk so "
                f"the partial products add up there, and only after the loop copy it out with "
                f"nisa.tensor_copy. Do not allocate a new psum tile per chunk and do not write partial "
                f"results to HBM.")
    m = re.search(r"Matmul (stationary|moving) free dimension (\d+) exceeds \w+=(\d+)", error_text)
    if m:
        which, got, mx = m.group(1), int(m.group(2)), int(m.group(3))
        dim, src = ("M", "lhsT") if which == "stationary" else ("N", "rhs")
        return (error_text + f" The {which} operand of one nc_matmul may be at most {mx} wide, "
                f"and this one is {got}: {dim} is not tiled. Do not change a tile's shape to fit. "
                f"Add a loop over {dim} in chunks of {mx} with nl.affine_range, load only columns "
                f"{dim.lower()}0:{dim.lower()}0 + {mx} of {src} into the {which} tile, and write "
                f"each result to its own {dim.lower()}0:{dim.lower()}0 + {mx} slice of the output.")
    m = re.search(r"(\w+) (?:dst|src)? ?must be in \['sbuf', 'psum'\], got shared_hbm", error_text)
    if m:
        return (error_text + f" `nisa.{m.group(1)}` only moves data between on-chip buffers, sbuf and "
                f"psum. To reach HBM -- the tensor you allocated with buffer=nl.shared_hbm and will "
                f"return -- use nisa.dma_copy instead. The usual sequence is nc_matmul into psum, "
                f"tensor_copy psum to sbuf, then dma_copy sbuf to the shared_hbm output.")
    m = re.search(r"(\w+) must be in \['(\w+)'\], got (\w+)", error_text)
    if m:
        which, needed, got = m.groups()
        place = {"psum": "nl.psum", "sbuf": "nl.sbuf"}.get(needed, needed)
        return (error_text + f" Allocate the `{which}` tile with buffer={place} instead of "
                f"nl.{got}. For nisa.nc_matmul: dst must be in nl.psum, and stationary and moving "
                f"must both be in nl.sbuf. Copy between them with nisa.tensor_copy.")
    if "both must be. Got" in error_text:
        return (error_text + " The two operands have different dtypes. Allocate each SBUF tile "
                "with the dtype of the input it holds (dtype=lhsT.dtype, dtype=rhs.dtype) instead "
                "of naming a dtype.")
    if "got multiple values for argument" in error_text:
        return (error_text + " Pass every argument by keyword, e.g. "
                "nisa.nc_matmul(dst=..., stationary=..., moving=...), so none is bound twice.")
    if "must have at least 2 dimensions" in error_text:
        # Measured on level 3: the old hint suggested the shape (1, N). The model wrote (1, 64)
        # for a matmul result that is (64, 512), and the next round died in a reshape.
        asked = shapes.get("shape")
        got = (f" The shape passed on that line is {asked}, which has {len(asked)} dimension"
               f"{'' if len(asked) == 1 else 's'}." if asked else "")
        sliced = (" Slicing a 2-D shape, as in t.shape[1:], leaves a single number."
                  if ".shape[" in facts["line"] and ":" in facts["line"] else "")
        return (error_text + f"{got} Every SBUF and PSUM tile needs two: a partition dimension "
                f"first, then a free dimension.{sliced} Write the shape as a tuple of two sizes, "
                f"(rows, cols), each read from the .shape of the data this tile will hold.")
    if re.search(r"cannot reshape (?:array|tensor) of size", error_text):
        if all(k in shapes for k in ("dst", "stationary", "moving")) \
                and shapes["dst"] != (shapes["stationary"][1], shapes["moving"][1]):
            st, mv = expr.get("stationary", "stationary"), expr.get("moving", "moving")
            return (error_text + f" Here stationary has shape {shapes['stationary']}, moving has "
                    f"shape {shapes['moving']} and dst has shape {shapes['dst']}. nisa.nc_matmul "
                    f"writes a result whose shape is (stationary.shape[1], moving.shape[1]). "
                    f"Change the one line that allocates `{expr.get('dst', 'dst')}` so its shape "
                    f"is ({st}.shape[1], {mv}.shape[1]).")
        drop = (" Remove the .reshape(...) call on that line."
                if ".reshape(" in facts["line"] else "")
        return (error_text + f" Do not reshape: a tensor keeps the shape it was allocated with."
                f"{drop} To move part of a tensor, slice it with bounds read from .shape, and give "
                f"the other side of the copy exactly the shape of that slice.")
    m = re.search(r"module '([\w.]+)' has no attribute '(\w+)'", error_text)
    if m:
        return error_text + available_names(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_text)
    if m:
        return (error_text + f" A {m.group(1)} is not a numpy array, so it has no "
                f"`{m.group(2)}`. Use the nl/nisa functions instead.")
    return error_text

# Compressed from nki_kernel_essentials.md, one short block per prompt rather than the document:
# the model reads 8192 tokens, and a checklist of rules makes it audit itself instead of answering.
# Three things in that document are deliberately NOT carried over, each checked on nki 0.6.0:
#   nl.load / nl.store  work, but bypass nisa.dma_copy, which is what nkibench counts HBM bytes
#                       through -- a kernel using them reports 0 bytes and no roofline.
#   nl.arange           does not exist in this SDK.
#   nl.matmul           does the whole matmul, which is what levels 3 to 7 are about writing.
NKI_MODEL = """How the chip is laid out:
  HBM   large off-chip memory. The kernel's arguments and the tensor it returns live here.
  SBUF  on-chip memory for input and intermediate tiles. The compute engines work on SBUF tiles.
  PSUM  on-chip memory where matmul results accumulate.
Data flows HBM -> SBUF -> compute -> SBUF -> HBM, and every HBM transfer is slow, so load a tile
once and reuse it. A tile's first axis is the partition axis (at most {pmax}); the rest are free
axes. The two are not interchangeable: which axis is which decides what an instruction can do."""

NKI_MATMUL = ("nisa.nc_matmul(dst=C, stationary=A, moving=B) computes C = A.T @ B, with A: [K, M], "
              "B: [K, N] and C: [M, N] accumulated in PSUM. One call takes K <= {pmax}, "
              "M <= {smax}, N <= {mmax}.")

NKI_LESSON = {
    1: "build the pooling windows as an access pattern over the tile and reduce them with nl.sum "
       "or nl.mean (same arguments); if you sum, divide by the window size.",
    2: "a transpose moves data across the partition/free layout. Get it correct first; after "
       "that the cost to reduce is the NUMBER of transfers, since many tiny ones are slow even "
       "when few bytes move.",
    3: "respect the operand layout and the tile limits, and move the result from PSUM to SBUF "
       "before it goes out to HBM.",
    4: "loop over tiles in M, N and K, and accumulate the K tiles of one output tile in the same "
       "PSUM tile.",
    5: "the same operand tiles are re-read on every pass of the inner loop. Move those loads out "
       "of the inner loop and reuse the tiles in SBUF.",
    6: "keep a block of several M and N tiles resident in SBUF, so each loaded tile serves "
       "several output tiles instead of one.",
    7: "block M, N and K together, so tiles stay resident and partial sums stay in PSUM across "
       "the whole contraction.",
    8: "compute scores = Q @ K.T / sqrt(d), a stable softmax over each row, then multiply by V. "
       "Subtract each row's maximum before nl.exp (nl.max finds it) or large scores overflow, and "
       "keep the seq x seq scores on chip -- never write them to HBM.",
}


def nki_notes(level, short=False):
    """The NKI background for one level. `short` is the reminder a repair prompt carries."""
    limits = dict(pmax=nkibench.PMAX, smax=nkibench.GEMM_STATIONARY_FMAX,
                  mmax=nkibench.GEMM_MOVING_FMAX)
    parts = [] if short else [NKI_MODEL.format(**limits)]
    if nkibench.LEVELS[level]["ref"] in (nkibench.ref_matmul, nkibench.ref_attention):
        parts.append(NKI_MATMUL.format(**limits))
    if level in NKI_LESSON:
        parts.append(f"What this kernel has to get right: {NKI_LESSON[level]}")
    return "\n".join(parts)


def needs_tiling(level):
    """Does this level have a shape that does not fit in one tile? Read from the level's own
    shapes, so a level added to nkibench.py is classified without being listed here."""
    spec = nkibench.LEVELS[level]
    return any(tag.startswith("multi-") for sp in spec["shapes"]
               for tag in verifier.shape_tags(spec["ref"], sp))


def tiling_guide(a, level):
    """The tiling guide, for the levels that need tiling; "" for the others, or with it off.

    The file is read on every level, so editing it changes the next run. It goes into the full
    first prompt and into restructure rounds -- repair rounds whose failure means a dimension is
    not tiled -- and nowhere else: at about 790 tokens it is not worth paying for on a round
    that is fixing a decorator or a buffer.
    """
    if a.tiling_guide.lower() in ("", "none") or not needs_tiling(level):
        return ""
    path = a.tiling_guide if os.path.isabs(a.tiling_guide) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), a.tiling_guide)
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError as e:
        print(f"  (no tiling guide: {e}. Running this level without it.)")
        return ""
    return ("Background on tiling, for reference. Its code is conceptual pseudocode, not NKI: "
            "write the kernel with the NKI functions listed further down.\n\n" + text)


def first_prompt(level, terse=0, notes=True, memory="", guide=""):
    """Deliberately short, and it does NOT list the rules.

    Measured twice in this repo: hand a model an enumerated list of prohibitions and it audits
    itself against each one and returns nothing, while a bigger budget only buys more thinking.
    So the rules live in the checker. Generate freely, let the checker object, then send back one
    named change.
    """
    s = nkibench.LEVELS[level]
    import inspect
    if terse >= 2:
        # Last resort. Measured on this endpoint: one-sentence prompts answered in 300-700
        # tokens while every structured, rule-carrying prompt spiralled.
        return (f"Write a Python function `{s['entry']}` decorated with @nki.jit that computes "
                f"the same thing as this, using nki.language as nl and nki.isa as nisa:\n\n"
                f"{inspect.getsource(s['ref'])}\n"
                f"Reply with one python code block.")
    if terse >= 1:
        # The matmul memory rules are the substance of levels 3 and 4, and the short prompt has to
        # carry them: measured, the agent cycled between "dst must be in ['psum']" and "moving must
        # be in ['sbuf']" because nothing told it where the operands live.
        mm = ("nisa.nc_matmul(dst=, stationary=, moving=) needs dst in nl.psum and both operands "
              "in nl.sbuf. So: dma_copy the operands HBM->sbuf, allocate a psum tile, nc_matmul "
              "into it, tensor_copy psum->sbuf, then dma_copy sbuf->the shared_hbm output you "
              "return. The left operand is already transposed, with K on the partition axis.\n"
              if level >= 3 else "")
        return (f"Write an AWS Neuron NKI kernel: a function `{s['entry']}` decorated with "
                f"@nki.jit that computes what this reference computes.\n\n"
                f"{inspect.getsource(s['ref'])}\n"
                f"Allocate with nl.ndarray(shape, dtype=..., buffer=nl.sbuf), move data with "
                f"nisa.dma_copy(dst=, src=), loop with nl.affine_range(n). A tile's partition "
                f"dimension is at most {nkibench.PMAX}.\n{mm}\n"
                f"Reply with one python code block.")
    return (
        f"Write an AWS Neuron NKI kernel.\n\n"
        f"Operation: {s['op']}\n"
        f"Entry point: a function named `{s['entry']}`, decorated with `@nki.jit`.\n"
        f"It must compute exactly what this NumPy reference computes:\n\n"
        f"{inspect.getsource(s['ref'])}\n"
        f"Hardware limits: a tile's partition dimension is at most {nkibench.PMAX}. For matmul, "
        f"the stationary free dimension is at most {nkibench.GEMM_STATIONARY_FMAX} and the "
        f"moving free dimension at most {nkibench.GEMM_MOVING_FMAX}.\n\n"
        + (f"{nki_notes(level)}\n\n" if notes else "")
        + (f"{guide}\n\n" if guide else "")
        + (f"{memory}\n\n" if memory else "") +
        f"Import nki, nki.language as nl, and nki.isa as nisa.\n\n{API_CARD}\n\n"
        f"Reply with ONE python code block containing the imports and the function. No prose.")


def describe_change(parent, code):
    """What one attempt changed, in one line, so the history can pair it with what happened."""
    if not (parent or "").strip():
        return "first version"
    if parent == code:
        return "returned the kernel unchanged"
    keep = lambda src: [ln.strip() for ln in src.splitlines()
                        if ln.strip() and not ln.strip().startswith("#")]
    a, b = keep(parent), keep(code)
    ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    if not ops:
        return "changed only comments"
    tag, i1, i2, j1, j2 = ops[0]
    old, new = (a[i1][:80] if i2 > i1 else ""), (b[j1][:80] if j2 > j1 else "")
    what = (f"changed `{old}` to `{new}`" if old and new
            else f"added `{new}`" if new else f"removed `{old}`")
    more = sum(max(o[2] - o[1], o[4] - o[3]) for o in ops) - 1
    return what + (f" (and {more} more line{'' if more == 1 else 's'})" if more else "")


def history_text(history):
    """The error memory as the repair prompt carries it: one line per earlier attempt.

    An error that came back several times is listed once, with its latest change and a count --
    three lines saying the same thing cost three times the tokens and say it once.
    """
    if not history:
        return ""
    last, count = {}, {}
    for i, (change, result) in enumerate(history):
        cls = error_class(result)
        last[cls], count[cls] = i, count.get(cls, 0) + 1
    lines = [f"- {change} -> {result}" + (f" ({count[error_class(result)]} times)"
                                         if count[error_class(result)] > 1 else "")
             for i, (change, result) in enumerate(history) if last[error_class(result)] == i]
    return ("For background only, these are the earlier attempts and what each one led "
            "to. Do not undo a change that removed an error, and do not repeat one that "
            "did not:\n" + "\n".join(lines))


def repair_prompt(level, source, feedback, history=(), notes=False, note="", restructure=False,
                  guide=""):
    """One named change, and the previous code. No rules list, no reference re-sent.

    The lesson this whole repo keeps re-learning: feeding a verifier's report back verbatim
    reproduces the same mistake, because a report says what is wrong and never what to do.

    `history` is the earlier attempts, one line each: what was changed and what the checker then
    said. It comes AFTER the instruction and is marked as background, so the report above stays
    the only thing being asked for -- otherwise the model goes back to fix errors already gone.

    `notes` adds the level's two-line NKI reminder. Measured: after round 0 the prompt carried
    no description of the task at all, and no repair round ever improved a score.

    `note` is one extra sentence for when the last reply was unusable, so that the prompt differs
    from the one that produced it.

    `restructure` makes this a restructure round, for a failure that means a dimension is not
    tiled. "Keep everything else identical" is the wrong request then: the fix is a new loop
    around existing lines, and under that sentence the model edited a tile's shape instead. So
    the round asks for the loops, says what may be rewritten, and carries `guide`, the tiling
    guide, right before the request.
    """
    remind = f"{nki_notes(level, short=True)}\n\n" if notes else ""
    past = history_text(history)
    ask_for = ("Change exactly what the checker names and keep everything else identical."
               if not restructure else
               "This cannot be fixed by editing one line or by changing a tile's shape: a "
               "dimension is not tiled. Restructure the kernel so that every dimension larger "
               "than its tile limit has its own loop over chunks, and each tile is loaded, "
               "computed and written per chunk. You may rewrite the loops and move allocations "
               "to do it; keep the imports, the function signature and the calls that already "
               "work.")
    return (
        f"This NKI kernel for {nkibench.LEVELS[level]['op']} is not right yet.\n\n{remind}"
        f"```python\n{source}\n```\n\n"
        f"A checker reports:\n{feedback}\n\n"
        + (f"{guide}\n\n" if restructure and guide else "") +
        f"{ask_for} Reply with "
        f"ONE python code block.{' ' + note if note else ''}{chr(10) * 2 + past if past else ''}")


# ---------------------------------------------------------------- the memory bank
#
# What earlier RUNS learned, kept on disk and read before a level starts. The attempt history above
# lasts one level of one run; this outlives the process. Per kernel it holds the patterns that
# worked, the patterns that failed and why, and the best result so far. Each list is capped, and a
# new pattern pushes out the oldest, so the block it adds to the prompt cannot grow without bound.

def pattern_class(text):
    """Two patterns are the same pattern if they differ only in their numbers."""
    return re.sub(r"\d+", "N", re.sub(r"^line \d+: ", "", text or "")).strip()


def error_class(text):
    """The error a verdict names, without where it happened or on which shapes: the key that
    decides whether the bank, or the error memory, already holds it."""
    text = re.sub(r"^\d+ of \d+ shapes passed; on .*?: ", "", text or "")
    named = re.match(r"(numeric:[\w:-]+): ", text)     # a diagnosed failure carries its class
    return named[1] if named else pattern_class(text.split(" [")[0])[:60].rstrip()


def cleared_errors(entry):
    """The errors some stored successful pattern already records as fixed."""
    return {error_class(w.split("-> cleared: ", 1)[1])
            for w in entry["successful_patterns"] if "-> cleared: " in w}


def kernel_signature(code):
    """A solved kernel as one line: the tiles it allocates, the calls it makes, its loop depth."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    tiles, calls = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.endswith("ndarray"):
            shape = node.args[0] if node.args else next(
                (kw.value for kw in node.keywords if kw.arg == "shape"), None)
            where = next((ast.unparse(kw.value).split(".")[-1] for kw in node.keywords
                          if kw.arg == "buffer"), "?")
            if shape is not None and where != "shared_hbm":
                tiles.append(f"{ast.unparse(shape)} in {where}")
        elif name.split(".")[0] in ("nisa", "nl") and not name.endswith("_range") \
                and name not in calls:
            calls.append(name)

    def depth(node):
        inner = max((depth(c) for c in ast.iter_child_nodes(node)), default=0)
        return inner + isinstance(node, ast.For)

    d = depth(tree)
    return "; ".join(x for x in (
        "tiles " + ", ".join(dict.fromkeys(tiles)) if tiles else "",
        "calls " + ", ".join(calls) if calls else "",
        f"{d} nested loop{'' if d == 1 else 's'}" if d else "no loops") if x)[:220]


def bank_entry(bank, level):
    spec = nkibench.LEVELS[level]
    return bank.setdefault(str(level), dict(
        kernel=spec["op"], entry=spec["entry"],
        shapes=[nkibench.label(c, level) for c in spec["shapes"]],
        successful_patterns=[], failed_patterns=[],
        best_known=dict(correctness="not attempted", reward=0.0)))


def load_bank(path):
    try:
        with open(path) as f:
            bank = json.load(f)
        return bank if isinstance(bank, dict) else {}
    except (OSError, ValueError):
        return {}


def remember(path, level, size, worked=None, failed=None, best=None):
    """Add one pattern to the bank on disk. Returns what was added, or "" if nothing was new.

    The bank is prompt text, so nothing goes in twice. A failure is skipped when its error is
    already there -- as a failed pattern, or as the error a successful pattern cleared, which says
    the same thing plus the fix -- and storing such a fix drops the failed pattern it supersedes.
    A new pattern goes on the end, and once a list holds `size` entries the oldest is dropped to
    make room. The file is re-read first and replaced in one step, so two runs sharing a bank do
    not truncate each other's file.
    """
    if not path:
        return ""
    bank = load_bank(path)
    entry = bank_entry(bank, level)
    added = ""
    if worked:
        # Compare what would be STORED: the stored text is cut to 220 characters, so comparing
        # the uncut text against it never matched, and one fix went in on every run.
        worked = worked[:220]
        have = {pattern_class(w) for w in entry["successful_patterns"]}
        if pattern_class(worked) not in have:
            entry["successful_patterns"] = (entry["successful_patterns"] + [worked[:220]])[-size:]
            fixed = cleared_errors(entry)
            entry["failed_patterns"] = [f for f in entry["failed_patterns"]
                                        if error_class(f["reason"]) not in fixed]
            added = "worked"
    if failed:
        change, reason = failed
        have = {error_class(f["reason"]) for f in entry["failed_patterns"]} | cleared_errors(entry)
        if error_class(reason) not in have:
            entry["failed_patterns"] = (entry["failed_patterns"]
                                        + [dict(change=change[:160], reason=reason[:160])])[-size:]
            added = "failed"
    rank = lambda b: (round(b.get("reward", 0.0), 6), b.get("confidence", 0.0))
    if best and rank(best) > rank(entry["best_known"]):
        entry["best_known"] = best
        added = added or "best"
    if added:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(bank, f, indent=2)
        os.replace(tmp, path)
    return added


def memory_text(entry):
    """The bank for one kernel, as the block the first prompt carries. Empty if nothing is known.

    An error is printed once: a failed pattern whose error a successful pattern cleared is left
    out, since that line already names the error and what fixed it.
    """
    fixed = cleared_errors(entry)
    worked = entry["successful_patterns"]
    failed = [f for f in entry["failed_patterns"] if error_class(f["reason"]) not in fixed]
    if not worked and not failed:
        return ""
    out = ["Notes from your earlier runs on this kernel:"]
    if worked:
        out.append("successful_patterns:")
        out += [f"  - {w}" for w in worked]
    if failed:
        out.append("failed_patterns (do not write these again):")
        for f in failed:
            out += [f"  - change: {f['change']}", f"    reason: {f['reason']}"]
    out.append(f"best_known: {entry['best_known']['correctness']}")
    return "\n".join(out)


def got_further(before, after):
    """Did a repair make progress? Each argument is (reward, one-line verdict)."""
    if after[0] > before[0] + 1e-9:
        return True
    was, now = (re.search(r"line (\d+): ", v) for v in (before[1], after[1]))
    return bool(after[0] >= before[0] and was and now and int(now[1]) > int(was[1]))


def failed_pattern(code, change, brief):
    """(change, reason) for the bank: the line that failed if there is one, else the edit made."""
    m = re.match(r"line (\d+): ", brief)
    lines = code.splitlines()
    if m and int(m.group(1)) <= len(lines):
        n, text = int(m.group(1)), lines[int(m.group(1)) - 1].strip()
        try:        # the whole statement, when the failing call is spread over several lines
            inside = [x for x in ast.walk(ast.parse(code)) if isinstance(x, ast.stmt)
                      and x.lineno <= n <= x.end_lineno and not hasattr(x, "body")]
            text = ast.unparse(min(inside, key=lambda x: x.end_lineno - x.lineno))
        except (SyntaxError, ValueError):
            pass
        return f"`{text}`", brief[m.end():]
    if change == "first version":
        change = "first version, " + (kernel_signature(code) or "which did not parse")
    return change, brief


CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text):
    """Pull out the code, and return NOTHING rather than prose.

    The old fallback returned the whole reply whenever it contained "def ", so a numbered list
    or a sentence reached the compiler and produced "invalid decimal literal on line 2" -- a
    parse error that blamed the model for the extractor's mistake.
    """
    text = text or ""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    # No fence: start at the first line that can legally begin a module and keep the rest.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |@nki|def )", line):
            return "\n".join(lines[i:]).strip()
    return ""


# ---------------------------------------------------------------- the model

@dataclass
class Reply:
    content: str
    finish: str = "stop"
    prompt_tokens: int = 0          # as the server counted them, chat template included
    completion_tokens: int = 0
    reasoning_chars: int = 0
    plan: "Reply" = None            # the planning call this answer followed, with --plan-tokens
    plan_text: str = ""             # that plan as it was handed to the coding call
    sent: str = ""                  # the prompt of the coding call, when a plan was added to it

    def unusable(self):
        """Why this reply must not reach the checker, or "" if it may.

        A truncated answer with half a kernel in it used to reach the compiler and come back as
        "does not parse", which blames the model for a budget problem and sends it to fix syntax
        that was never wrong.
        """
        if self.finish == "length":
            return "truncated"
        if not self.content.strip():
            return "empty"
        return "" if extract_code(self.content).strip() else "no-code"


class Tokens:
    """Token counts from the server's own tokenizer, so the budget is measured, not guessed.

    vLLM serves POST /tokenize beside /v1. Where that is missing (another server, or --offline)
    the count falls back to chars/4 for the rest of the run, and every log record says which it
    was in `exact_tokens`.
    """

    def __init__(self, a):
        self.model, self.cache = a.model, {}
        self.url = None if a.offline else re.sub(r"/v1/?$", "", a.base.rstrip("/")) + "/tokenize"
        self.exact = self.url is not None

    def count(self, text):
        if not text:
            return 0
        if text not in self.cache:
            n = None
            if self.exact:
                import httpx
                try:
                    r = httpx.post(self.url, json=dict(model=self.model, prompt=text),
                                   timeout=20, verify=False)
                    n = int(r.json()["count"])
                except Exception as e:
                    self.exact = False
                    print(f"    (no tokenizer at {self.url}: {type(e).__name__}. Token counts "
                          f"are chars/4 estimates from here on.)")
            self.cache[text] = n if n is not None else max(1, len(text) // 4)
        return self.cache[text]

    def sections(self, prompt, **named):
        """Where one prompt's input tokens went. `task` is whatever the named parts leave."""
        out = {k: self.count(v) for k, v in named.items() if v}
        out["task"] = max(0, self.count(prompt) - sum(out.values()))
        return out


class ContextProfile:
    """How full the context window is on every model call: tokens used / max context length.

    Used is prompt plus answer, since max-model-len covers both. The prompt count is the server's
    own (chat template included); the answer is what it actually generated. One row per call, so
    the run can say where it came closest to the ceiling and which part of the prompt put it there.
    """

    def __init__(self, limit):
        self.limit, self.rows = limit, []

    def record(self, level, rnd, reply, sections, call="code"):
        prompt = reply.prompt_tokens or sum(sections.values())
        used = prompt + reply.completion_tokens
        row = dict(level=level, round=rnd, call=call, prompt=prompt,
                   answer=reply.completion_tokens,
                   used=used, limit=self.limit, share=round(used / self.limit, 4),
                   largest=max(sections, key=sections.get) if sections else "")
        self.rows.append(row)
        return row

    @staticmethod
    def line(row):
        filled = min(20, round(20 * row["share"]))
        return (f"context {row['used']}/{row['limit']} ({row['share']:.0%}) "
                f"[{'#' * filled}{'.' * (20 - filled)}] "
                f"prompt {row['prompt']} + answer {row['answer']}"
                + ("  NEAR THE LIMIT" if row["share"] >= 0.9 else ""))

    def summary(self, level=None, start=0):
        """Peak and mean over the calls since `start`, for one level or all of them."""
        rows = [r for r in self.rows[start:] if level in (None, r["level"])]
        if not rows:
            return None
        peak = max(rows, key=lambda r: r["used"])
        return dict(calls=len(rows), peak=peak["used"], peak_share=peak["share"],
                    peak_round=peak["round"], peak_largest=peak["largest"],
                    mean_share=round(sum(r["share"] for r in rows) / len(rows), 4),
                    limit=self.limit)

    def report(self, level=None, start=0):
        m = self.summary(level, start)
        if not m:
            return "context: no model calls"
        return (f"context peak {m['peak']}/{m['limit']} ({m['peak_share']:.0%}) on round "
                f"{m['peak_round']}, largest part of that prompt: {m['peak_largest']}; "
                f"mean {m['mean_share']:.0%} over {m['calls']} call(s)")


def ask(a, prompt, max_tokens=None):
    """One model call. `max_tokens` caps the answer below the usual budget; the planning call
    uses it, and being cut off at that cap is the point there, not a failure."""
    import httpx
    # enable_thinking=False matters. Qwen3 reasons before answering, and with thinking on it
    # spent the whole budget there: the first cluster run returned "No code came back" at 54.7s
    # over and over, plus truncated fragments (invalid decimal literal, unterminated string).
    # max-model-len is prompt PLUS answer, so the answer budget is whatever the prompt leaves.
    # No `seed` is ever sent: on Neuron vLLM a per-request seed kills the engine for everyone.
    in_prompt = a.tokens.count(prompt)
    cap = max_tokens or a.max_tokens
    budget = min(cap, max(min(256, cap), a.context - in_prompt - 64))
    if budget < cap:
        print(f"    (prompt is {'' if a.tokens.exact else '~'}{in_prompt} tokens, so the answer "
              f"budget is capped at {budget} to stay inside the {a.context}-token context)")
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body,
                   timeout=900, verify=False)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    msg = ch.get("message", {})
    reasoning = next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")
    content = msg.get("content") or ""
    finish = ch.get("finish_reason") or "stop"
    usage = payload.get("usage") or {}
    if finish == "length" and not max_tokens:
        # Do not let a budget problem look like a model failure.
        print(f"    (TRUNCATED: finish_reason=length after {len(content)} chars. The answer was "
              f"cut off by the budget, so it is not graded. Prompt is {in_prompt} tokens; server "
              f"context is the ceiling.)")
    if not content.strip() and reasoning:
        # The single most common surprise on this endpoint, so name it rather than reporting
        # an empty answer as a model failure.
        print(f"    (empty answer, {len(reasoning)} chars of hidden reasoning, "
              f"finish={finish} — shorten the prompt rather than raising the budget)")
    return Reply(content, finish, usage.get("prompt_tokens") or in_prompt,
                 usage.get("completion_tokens") or len(content) // 4, len(reasoning))


# Thinking inside a budget. Qwen3's own thinking mode has no cap: measured, it spent the whole
# answer budget reasoning and returned nothing. So the reasoning is asked for as an ordinary
# answer with thinking off, in a call of its own whose max_tokens IS the cap, and the code is
# written by a second call that reads it. Each call has its own context window, so the plan
# never takes output room from the kernel; what it costs is its own tokens in the second prompt.
PLAN_ASK = ("\n\nDo not write the code yet. First think it through: in at most five short bullet "
            "points, say exactly what you will write or change, and why that fixes it. Name the "
            "functions, shapes and loop bounds you will use. Bullets only, no code block.")
PLAN_FOLLOW = ("\n\nYou already thought this through. Your plan:\n{plan}\n\nNow carry out that "
               "plan. Reply with ONE python code block and nothing else.")


def ask_planned(a, prompt):
    """Plan in at most --plan-tokens tokens, then write the code with the plan in the prompt."""
    plan = ask(a, prompt + PLAN_ASK, max_tokens=a.plan_tokens)
    text = plan.content.strip()
    if plan.finish == "length" and "\n" in text:
        text = text.rsplit("\n", 1)[0].rstrip()      # the cap cut the last line off mid-thought
    sent = prompt + PLAN_FOLLOW.format(plan=text) if text else prompt
    reply = ask(a, sent)
    reply.plan, reply.plan_text, reply.sent = plan, text, sent
    return reply


def ask_parallel(a, prompt, n):
    import concurrent.futures as cf
    one = ask_planned if a.plan_tokens else ask
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        return [f.result() for f in [ex.submit(one, a, prompt) for _ in range(n)]]


def offline_answers(level, n, rnd):
    """No model. Replays the shipped reference, preceded by a deliberately broken version, so the
    loop and the feedback path can be exercised with no endpoint. Never report a number."""
    ref = open(f"reference_level{level}.py").read()
    if rnd == 0:
        broken = ref.replace("@nki.jit", "", 1)
        return [Reply(f"```python\n{broken}\n```")] * n
    return [Reply(f"```python\n{ref}\n```")] * n


# ---------------------------------------------------------------- the loop

def solve(a, level, log):
    """One level: generate, verify, turn the verdict into one change, repeat.

    Returns a dict with the best reward, the rounds used, a status and a confidence:
        verified   passed the level's shapes AND the holdout cases the model never saw
        dev-only   passed the shapes it was repaired against, and not the holdout
        stuck      stopped early on one failure class coming back without progress
        failed     ran out of rounds
    """
    print(f"\n=========== level {level}: {nkibench.LEVELS[level]['op']} ===========")
    terse = a.terse
    # Read the bank before anything is asked: what earlier runs learned goes into the first prompt.
    # Offline runs replay the reference, so they neither read nor write it.
    bank_path = "" if (a.offline or a.memory.lower() in ("", "none")) else a.memory
    memory = ""
    if bank_path:
        known = bank_entry(load_bank(bank_path), level)
        memory = memory_text(known)
        print(f"  memory bank {bank_path}: {len(known['successful_patterns'])} successful and "
              f"{len(known['failed_patterns'])} failed patterns, best known "
              f"{known['best_known']['correctness']}")
    guide = tiling_guide(a, level)
    if guide:
        print(f"  this level needs tiling, so the first prompt carries {a.tiling_guide} "
              f"({'' if a.tokens.exact else '~'}{a.tokens.count(guide)} tokens)")
    prompt = first_prompt(level, terse, a.notes >= 1, memory, guide)
    # the named parts of `prompt`, for the token split; only the full prompt carries them
    pieces = dict(memory=memory, guide=guide) if terse == 0 else {}
    best = 0.0
    latest = None       # (code, verdict) of the attempt being repaired
    tried, keys, seen, streak, gated = [], [], {}, 0, 0
    history = []        # (what the attempt changed, what the checker said), oldest first
    shown = []          # the part of it the current repair prompt carries
    prior = None        # (reward, verdict) of the attempt the current one was repaired from
    tin = tout = 0
    first_call = len(a.profile.rows)    # this level's rows in the profile start here

    def done(status, rounds, confidence=verifier.FAILED, holdout=""):
        print("  " + a.profile.report(level, first_call))
        rec = dict(type="result", level=level, status=status, confidence=confidence,
                   reward=best, rounds=rounds, keys=keys, holdout=holdout,
                   tokens_in=tin, tokens_out=tout, exact_tokens=a.tokens.exact,
                   context=a.profile.summary(level, first_call))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        return rec

    def repeats(key, progressed=False):
        """Count this failure class, and say why to stop if it has stopped being informative.

        A class that comes back while the kernel still gets further -- a higher score, or a
        crash on a later line -- is progress, not a repeat.
        """
        nonlocal streak
        streak = streak + 1 if keys and keys[-1] == key and not progressed else 1
        keys.append(key)
        seen[key] = seen.get(key, 0) + (not progressed)
        if streak >= a.patience:
            return f"'{key}' {streak} rounds in a row with no progress"
        if seen[key] >= a.give_up_after:
            return (f"'{key}' for the {seen[key]}th time, alternating with "
                    f"{len(seen) - 1} other failure class(es)")
        return ""

    def stuck(why, rnd):
        print(f"  STOPPING this level: {why}. The agent is cycling between a fixed set of "
              f"mistakes rather than converging, so more rounds will not help. Classes seen:")
        for k, n in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"    {n}x  {k}")
        return done("stuck", rnd + 1)

    for rnd in range(a.rounds):
        t0 = time.perf_counter()
        replies = (offline_answers(level, a.samples, rnd) if a.offline
                   else ask_parallel(a, prompt, a.samples))
        # With a plan, each sample's coding prompt is the round's prompt plus its own plan.
        split = lambda r: a.tokens.sections(r.sent or prompt, **pieces, plan=r.plan_text)
        sections = split(replies[0])
        plans = [r.plan for r in replies if r.plan]
        tin += sum(r.prompt_tokens for r in replies + plans)
        tout += sum(r.completion_tokens for r in replies + plans)
        spent = (f"in={replies[0].prompt_tokens or sum(sections.values())} ["
                 + " ".join(f"{k}={v}" for k, v in sections.items() if v)
                 + f"] out={replies[0].completion_tokens}")
        graded = []
        fullest = None      # the sample of this round that used the most context
        if a.trace:
            print(f"\n>>>>>>>> level {level} round {rnd}: PROMPT ({len(prompt)} chars) >>>>>>>>\n"
                  f"{prompt}\n>>>>>>>> end of prompt >>>>>>>>")
        for i, reply in enumerate(replies):
            sections = split(reply)
            if reply.plan:
                a.profile.record(level, rnd, reply.plan,
                                 a.tokens.sections(prompt + PLAN_ASK, **pieces), call="plan")
            if a.trace and reply.plan:
                print(f"-------- level {level} round {rnd}: PLAN {i + 1} of {len(replies)} "
                      f"({reply.plan.completion_tokens} of {a.plan_tokens} tokens, "
                      f"finish={reply.plan.finish}) --------\n{reply.plan.content}\n"
                      f"-------- end of plan; it was added to the prompt above --------")
            if a.trace:
                print(f"<<<<<<<< level {level} round {rnd}: REPLY {i + 1} of {len(replies)} "
                      f"(finish={reply.finish}, {len(reply.content)} chars) <<<<<<<<\n"
                      f"{reply.content}\n<<<<<<<< end of reply <<<<<<<<")
            used = a.profile.record(level, rnd, reply, sections)
            if fullest is None or used["used"] > fullest["used"]:
                fullest = used
            # The reply gate: a truncated, empty or code-less reply is a fact about the prompt
            # and the budget, so it never reaches the checker.
            why = reply.unusable()
            src = "" if why else extract_code(reply.content)
            v = None if why else grade(src, level)
            if v:
                graded.append((v, src))
            log.write(json.dumps(dict(
                type="attempt", level=level, round=rnd, reward=v.reward if v else 0.0,
                parts=v.parts if v else dict.fromkeys(WEIGHTS, False),
                key=v.key if v else f"reply:{why}", sections=sections,
                prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
                exact_tokens=a.tokens.exact, finish=reply.finish, context=used,
                prompt_chars=len(prompt),
                reply_chars=len(reply.content), code=src, feedback=v.feedback if v else "",
                brief=v.brief if v else f"no usable reply: {why}",
                bound_use=v.bound_use if v else None, cost=v.cost if v else "",
                restructure=bool(v and v.restructure),
                **(dict(plan=reply.plan_text, plan_tokens=reply.plan.completion_tokens,
                        plan_finish=reply.plan.finish) if reply.plan else {}),
                **(dict(prompt=reply.sent or prompt, reply=reply.content) if a.trace else {})))
                + "\n")
        log.flush()
        spent += "\n  " + ContextProfile.line(fullest)
        if plans:
            spent += (f"\n  plan: {max(p.completion_tokens for p in plans)} of {a.plan_tokens} "
                      f"tokens" + (" (cut off at the cap)" if any(p.finish == "length"
                                                                 for p in plans) else ""))

        if not graded:
            why = replies[0].unusable()
            gated += 1
            print(f"round {rnd}: no usable reply ({why}), so nothing was graded  {spent}")
            stop = repeats(f"reply:{why}")
            if stop:
                return stuck(stop, rnd)
            # Sampling here is close to greedy: the same prompt gets the same non-answer, so the
            # prompt has to change, and it changes by getting shorter.
            if latest is None:
                terse = min(terse + 1, 2)
                keep = memory if gated < 2 else ""
                prompt = first_prompt(level, terse, a.notes >= 1, keep, guide)
                pieces = dict(memory=keep, guide=guide) if terse == 0 else {}
                print(f"  re-asking with a shorter prompt (terseness {terse})")
            else:
                past = shown if gated < 2 else []
                again = latest[1].restructure and gated < 2    # the long form once, then shorter
                prompt = repair_prompt(level, latest[0], latest[1].feedback, past,
                                       a.notes >= 2 and gated < 2,
                                       "Send the code block only: no explanation, no comments.",
                                       again, guide)
                pieces = dict(code=latest[0], feedback=latest[1].feedback,
                              history=history_text(past), guide=guide if again else "")
                print("  re-asking for the code block only"
                      + ("" if gated < 2 else ", without the earlier attempts"))
            continue
        gated = 0

        graded.sort(key=lambda g: g[0].reward, reverse=True)
        top, code = graded[0]
        best = max(best, top.reward)
        # A dev pass is re-checked on cases the model never saw. That decides how far the pass
        # is trusted; the holdout report is never fed back, or it would stop being a holdout.
        hold = grade(code, level, holdout=True) if top.solved else None
        verified = bool(hold and hold.solved)
        # Repair the LATEST attempt, not the best one. Rebuilding from the best attempt with the
        # best attempt's feedback is a fixed point: once a round scores worse, the prompt stops
        # changing, and a greedy model then returns the same answer forever. Measured: level 2
        # stuck at 0.10 for four rounds while the prompt still carried the 0.50 code.
        # The prompt that follows carries the attempts BEFORE this one; this one's own result is
        # the checker's report, so it joins the history only after that prompt is built.
        shown = history[-a.history:] if a.history else []
        change = describe_change(latest[0] if latest else "", code)
        if shown:
            shown = shown + [(change, "the checker's report above")]
        # In the bank and the history a diagnosed failure goes by its class, so that the same bug
        # with different numbers is recognised as the same entry.
        verdict = f"{top.key}: {top.brief}" if top.key.startswith("numeric:") else top.brief
        history.append((change, verdict))
        progressed = bool(prior) and got_further(prior, (top.reward, verdict))
        new = [
            # A SMALL repair that got further: the score rose, or the kernel now runs past
            # the line it used to stop on. Swapping one error for another on the same line is
            # not progress, and a rewrite of the whole kernel teaches nothing about what
            # fixed it.
            remember(bank_path, level, a.memory_size,
                     worked=f"{change} -> cleared: {pattern_class(prior[1])[:90]}")
            if progressed and change.startswith(("changed `", "added `", "removed `"))
            and int((re.search(r"\(and (\d+) more", change) or [0, 0])[1]) <= 2 else "",
            # Only a kernel that also passed the holdout is a pattern worth repeating.
            remember(bank_path, level, a.memory_size,
                     worked="solved with " + kernel_signature(code)) if verified else "",
            remember(bank_path, level, a.memory_size,
                     failed=failed_pattern(code, change, verdict)) if not top.solved else "",
            remember(bank_path, level, a.memory_size, best=dict(
                correctness="passed, and verified on holdout cases" if verified
                else f"passed its own shapes but not the holdout cases ({hold.brief[:120]})"
                if top.solved else f"{top.passed} of {top.total} shapes passed"
                if top.parts["runs"] else "does not run yet", reward=round(top.reward, 2),
                confidence=verifier.VERIFIED if verified else verifier.DEV_ONLY
                if top.solved else verifier.FAILED,
                detail=top.feedback[:300] if top.solved else verdict[:200])),
        ]
        if any(new):
            print(f"  memory bank: stored {', '.join(n for n in new if n)}")
        prior = (top.reward, verdict)
        latest = (code, top)

        same = bool(keys) and keys[-1] == top.key and not progressed
        if same and top.feedback == (tried[-1] if tried else None):
            # Collapse. Fifteen identical multi-line blocks is noise, not information.
            print(f"round {rnd}: same failure again ({top.reward:.2f}, best so far {best:.2f})  "
                  f"[{top.key}]  {spent}")
        else:
            print(f"round {rnd}: this round {top.reward:.2f}  best so far {best:.2f}  "
                  f"[{top.key}]  {spent}  ({time.perf_counter() - t0:.1f}s)")
            print(textwrap.indent(top.feedback[:1500], "  "))
        if top.solved:
            status, confidence = (("verified", verifier.VERIFIED) if verified
                                  else ("dev-only", verifier.DEV_ONLY))
            print(f"  SOLVED on round {rnd}: {status}, confidence {confidence:.2f}. Holdout "
                  f"{hold.passed} of {hold.total} cases passed.")
            if not verified:
                print("  It passes the shapes it was repaired against and not the ones it never "
                      "saw, so it fits those shapes rather than computing the operation:")
                print(textwrap.indent(hold.feedback[:500], "    "))
            print("  ---------------- the kernel ----------------")
            print(textwrap.indent(code, "  "))
            print("  -------------------------------------------")
            keys.append("pass")
            return done(status, rnd + 1, confidence, f"{hold.passed}/{hold.total}")
        stop = repeats(top.key, progressed)
        if stop:
            return stuck(stop, rnd)
        tried.append(top.feedback)
        if streak >= 2 and not a.history:
            # Sampling on this endpoint is greedy, so an unchanged prompt returns an unchanged
            # answer. Measured: the same TypeError 19 rounds running. Changing the prompt is the
            # only thing that can change the answer, so say what has already been tried.
            ledger = ("These approaches have already failed, so do something different:\n"
                      + "\n".join(f"- {t[:160]}" for t in dict.fromkeys(tried)))
            prompt = repair_prompt(level, code, top.feedback, notes=a.notes >= 2,
                                   restructure=top.restructure, guide=guide) + "\n\n" + ledger
            pieces = dict(code=code, feedback=top.feedback, history=ledger,
                          guide=guide if top.restructure else "")
            print(f"  same failure class {streak}x — adding a ledger of {len(set(tried))} failed "
                  f"attempts to break the repeat")
            continue
        if top.restructure:
            print("  a dimension is not tiled, so the next round is a restructure round"
                  + (f" and carries {a.tiling_guide}" if guide else ""))
        prompt = repair_prompt(level, code, top.feedback, shown, a.notes >= 2,
                               restructure=top.restructure, guide=guide)
        pieces = dict(code=code, feedback=top.feedback, history=history_text(shown),
                      guide=guide if top.restructure else "")
    print(f"  not solved in {a.rounds} rounds; best reward {best:.2f}")
    return done("failed", a.rounds)


def options(argv=None):
    """The command line, parsed. loop_harness.py calls this too, so it runs with the real defaults."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    ap.add_argument("--all", action="store_true", help="levels 1 to 4")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=MIN_ANSWER_TOKENS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL")
                    or os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--path", default="", help="path to append, e.g. /agg/v1 for gpt-oss")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the whole thing N times and report a solve RATE. One run is not a "
                         "result: measured, the same config scored 1.00, 1.00 and 0.50 on level 2 "
                         "across three runs with no code change.")
    ap.add_argument("--log", default="attempts.jsonl")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop a level once the same failure CLASS has come back this many "
                         "rounds in a row without the kernel getting any further")
    ap.add_argument("--give-up-after", type=int, default=4,
                    help="stop a level once one failure class has been seen this many times in "
                         "all, which catches a cycle between two or three mistakes. Measured: 15 "
                         "was pure waste, because the prompt had stopped changing.")
    ap.add_argument("--memory", default="memory_bank.json",
                    help="the memory bank: successful and failed patterns per kernel, read before "
                         "a level starts and updated as it runs. Pass `none` to run without it. "
                         "With it on, the runs of --repeat are NOT independent: each one starts "
                         "from what the ones before it stored.")
    ap.add_argument("--memory-size", type=int, default=10,
                    help="most patterns kept per kernel in each list; a new one replaces the "
                         "oldest")
    ap.add_argument("--tiling-guide", default="gpu_tiling_agent_guide.md",
                    help="a guide added to the first prompt of the levels that need tiling -- "
                         "those with a shape larger than one tile, which is levels 4 to 7. A "
                         "relative path is looked up next to agent.py. Pass `none` to run "
                         "without it, which is what to compare against")
    ap.add_argument("--notes", type=int, default=2, choices=(0, 1, 2),
                    help="NKI background compressed from nki_kernel_essentials.md. 1 puts the "
                         "memory model and this level's lesson in the first prompt; 2 also puts a "
                         "two-line reminder in every repair prompt; 0 is the prompts as they "
                         "were, which is what to compare against.")
    ap.add_argument("--history", type=int, default=3,
                    help="how many earlier attempts the repair prompt lists, one line each: what "
                         "was changed and what the checker then said. 0 restores the old ledger "
                         "of failure messages, which is what to compare against.")
    ap.add_argument("--terse", type=int, default=0, choices=(0, 1, 2),
                    help="starting prompt length. Measured on gpt-oss-20b: 0 produced 13,245 "
                         "chars of hidden reasoning and no answer, while 1 answered with code. "
                         "Qwen3-8B is fine at 0.")
    ap.add_argument("--context", type=int, default=4096,
                    help="the server's max-model-len; prompt + answer must fit inside it. The "
                         "context profiler reports every call's usage against this number, so "
                         "set it to what the server really has")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason first; costs budget, and it ran out")
    ap.add_argument("--plan-tokens", type=int, default=0,
                    help="let the model think first, inside a budget: before each kernel it "
                         "writes a short plan in a call of its own, capped at this many tokens, "
                         "and the plan goes into the prompt of the call that writes the code. "
                         "250 is a reasonable value. 0 (the default) is no planning call. It "
                         "doubles the number of model calls")
    ap.add_argument("--trace", action="store_true",
                    help="show the full input and output: print every prompt exactly as sent and "
                         "every raw reply, and store both in the log as `prompt` and `reply`")
    ap.add_argument("--offline", action="store_true")
    return ap.parse_args(argv)


def main():
    a = options()

    if not a.offline:
        # Validate before the first request. An empty or scheme-less value produces a hostname
        # starting with "." and 30 lines of httpx/idna traceback that say nothing about the cause.
        raw = (a.base or "").strip()
        if not raw:
            sys.exit("KERNEL_AGENT_BASE_URL is empty or unset. Point it at a model:\n"
                     "  export KERNEL_AGENT_BASE_URL=http://localhost:8000/v1\n"
                     "  export KERNEL_AGENT_MODEL=Qwen/Qwen3-8B\n"
                     "That is the server ./serve.sh started. The seat pods set these for you; if they are "
                     "missing, you are not in a seat pod. Or pass --offline to run with no model.\n"
                     "(Separately: the shared gpt-oss endpoint is reached by setting "
                     "GPTOSS_BASE_URL and passing --path /agg/v1 -- not needed for the local run.)")
        from urllib.parse import urlparse
        u = urlparse(raw)
        if u.scheme not in ("http", "https") or not u.netloc:
            sys.exit(f"base URL {raw!r} is not usable. It needs a scheme and a host, e.g. "
                     f"http://qwen3-8b:8000/v1.")
        a.base = raw.rstrip("/") + a.path
        print(f"endpoint {a.base}  model {a.model}")
    else:
        print("*** OFFLINE: replaying the reference kernel. Numbers are meaningless. ***")

    a.tokens = Tokens(a)
    a.profile = ContextProfile(a.context)
    levels = sorted(nkibench.LEVELS)[:4] if a.all else [a.level or 1]
    history = {lv: [] for lv in levels}

    with open(a.log, "a") as log:
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n################ run {rep + 1} of {a.repeat} ################")
            results = []
            for level in levels:
                results.append(solve(a, level, log))
                history[level].append(results[-1])

            print("\n=========== summary ===========")
            for r in results:
                print(f"  level {r['level']}  reward {r['reward']:.2f} after {r['rounds']} "
                      f"round(s)  {r['status']}, confidence {r['confidence']:.2f}  "
                      f"({r['tokens_in']} tokens in, {r['tokens_out']} out)")
            print(f"  verified {sum(r['status'] == 'verified' for r in results)}/{len(results)}")

    print("\n=========== context usage ===========")
    for lv in levels:
        print(f"  level {lv}: {a.profile.report(lv)}")
    print(f"  overall: {a.profile.report()}"
          + ("" if a.tokens.exact else "  (token counts are chars/4 estimates)"))

    if a.repeat > 1:
        # The number that actually means something. A solve rate over N runs survives the variance
        # that makes any single run uninterpretable. Only VERIFIED counts as solved: a kernel that
        # passes just the shapes it was repaired against has not been shown to be right.
        print(f"\n=========== over {a.repeat} runs ===========")
        for lv in levels:
            got = [r["reward"] for r in history[lv]]
            said = [r["status"] for r in history[lv]]
            conf = [r["confidence"] for r in history[lv]]
            print(f"  level {lv}: verified {said.count('verified')}/{len(got)}  "
                  f"dev-only {said.count('dev-only')}  stuck {said.count('stuck')}  "
                  f"mean confidence {sum(conf) / len(conf):.2f}  "
                  f"best {max(got):.2f}  worst {min(got):.2f}  "
                  f"mean {sum(got) / len(got):.2f}  all={[round(r, 2) for r in got]}")
        print("\n  Report the rate, not your best run. A level that solves 1 in 3 times is not solved.")
        if not a.offline and a.memory.lower() not in ("", "none"):
            print("  These runs shared a memory bank, so they are not independent samples.")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
