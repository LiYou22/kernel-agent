#!/usr/bin/env python3
"""
verifier.py — what the checker says after "it ran": is it right, how sure are we, and what ONE
change would fix it.

    tolerance    a proven float32 error bound per operation, not a chosen constant
    cases        the level's own shapes to repair against, and holdout cases the model never sees
    diagnosis    the pattern across cases, facts true of every wrong case, two diverse worst cases
    directive    one instruction with a failure CLASS, so a repeat is recognised when numbers move

agent.py runs the kernel; everything here is NumPy on what came back, so it needs no SDK.

TOLERANCE. A case passes iff, for every output element,
    |got - ref| <= gamma_k * scale,     gamma_k = k*u / (1 - k*u),   u = 2**-24 (float32)
where ref is float64 on the exact float32 inputs, `scale` is the magnitude from the backward error
analysis and k counts the roundings on the path to one output (Higham, Accuracy and Stability of
Numerical Algorithms, ch. 3-4):
    average pooling, window p x p   k = p*p + 2   scale = mean of |x| over the window
    transpose                       k = 0         it only moves data, so it must be bit-exact
    matmul, contraction K           k = K + 1     scale = |A|.T @ |B|
So any kernel that is correct in float32 passes whatever its summation order, and the tolerance
encodes what float32 can guarantee rather than a preference. Measured on the shipped reference
kernels: pooling spends at most 36% of its bound, matmul 4%, transpose 0%.
An error over the bound but under 2**-6 * scale is what a bfloat16 tile produces (u = 2**-8, a few
roundings), so it is labelled precision-sized and the agent is not sent hunting for a logic bug.
Attention has no proven bound here, so it keeps nkibench's empirical tolerance and says so.
"""

from dataclasses import dataclass, field

import numpy as np

import nkibench

U32 = 2.0 ** -24
# Errors below this (x scale) are precision, not logic: see TOLERANCE.
ROUNDING_CEILING = 2.0 ** -6
NEAR = 1e-3             # "these wrong outputs ARE that other quantity", x scale
TINY = float(np.finfo(np.float32).tiny)
ONE_TILE = 10 ** 9      # "this axis is never tiled", as a tile size

VERIFIED, DEV_ONLY, FAILED = 0.95, 0.3, 0.0     # priors; repeated runs calibrate them


def gamma(k):
    return k * U32 / (1 - k * U32)


def _pool(z, p):
    C, H, W = z.shape
    return z[:, :H // p * p, :W // p * p].reshape(C, H // p, p, W // p, p).mean(axis=(2, 4))


def error_bound(level, args):
    """(ref64, scale, k) for one case, or None when the operation has no proven bound."""
    ref = nkibench.LEVELS[level]["ref"]
    a = [x.astype(np.float64) if isinstance(x, np.ndarray) else x for x in args]
    if ref is nkibench.ref_avgpool2d:
        # p*p - 1 additions, the rounded reciprocal, the multiply by it, and one spare.
        x, p = a
        return _pool(x, p), _pool(np.abs(x), p), p * p + 2
    if ref is nkibench.ref_transpose2d:
        want = np.asarray(ref(*a), np.float64)
        return want, np.abs(want), 0
    if ref is nkibench.ref_matmul:
        A, B = a
        return A.T @ B, np.abs(A).T @ np.abs(B), A.shape[0] + 1
    return None


# ---------------------------------------------------------------- cases

@dataclass(eq=False)
class Case:
    spec: dict
    label: str
    tags: tuple
    seed: int = 0
    values: str = "normal"


# Shapes the model never sees. Each stays inside what the level promises: the shipped level-4
# reference asserts multiples of the tile sizes, so the matmul holdout has no partial tiles.
HOLDOUT = {
    nkibench.ref_avgpool2d: [dict(shape=(16, 12, 12), pool_size=3),
                             dict(shape=(128, 8, 8), pool_size=4),
                             dict(shape=(5, 6, 6), pool_size=2)],
    nkibench.ref_transpose2d: [dict(shape=(16, 6), shape2D=(2, 3)),
                               dict(shape=(128, 15), shape2D=(3, 5)),
                               dict(shape=(7, 16), shape2D=(4, 4))],
    nkibench.ref_matmul: [dict(K=384, M=128, N=512), dict(K=128, M=256, N=512),
                          dict(K=256, M=128, N=1024)],
    nkibench.ref_attention: [dict(seq=48, dim=64), dict(seq=128, dim=32)],
}

# Values a kernel that is only right on friendly data gets wrong. Holdout only.
HOSTILE = {
    nkibench.ref_avgpool2d: ("large-mean", "all-negative"),
    nkibench.ref_transpose2d: ("index-coded",),
    nkibench.ref_matmul: ("cancelling", "large-magnitude"),
    nkibench.ref_attention: ("large-scores",),
}


def shape_tags(ref, sp):
    if ref is nkibench.ref_matmul:
        t = [name for name, size, lim in (("multi-k-tile", sp["K"], nkibench.PMAX),
                                          ("multi-m-tile", sp["M"], nkibench.GEMM_STATIONARY_FMAX),
                                          ("multi-n-tile", sp["N"], nkibench.GEMM_MOVING_FMAX))
             if size > lim]
        return tuple(t or ["single-tile"])
    if ref is nkibench.ref_avgpool2d:
        C, H, W = sp["shape"]
        return (("partition-below-max" if C < nkibench.PMAX else "partition-at-max"),
                ("odd-pool" if sp["pool_size"] % 2 else "even-pool"))
    if ref is nkibench.ref_transpose2d:
        return (("partition-below-max" if sp["shape"][0] < nkibench.PMAX else "partition-at-max"),
                ("square" if sp["shape2D"][0] == sp["shape2D"][1] else "non-square"))
    if ref is nkibench.ref_attention:
        return (("partition-below-max" if sp["seq"] < nkibench.PMAX else "partition-at-max"),
                ("seq-gt-dim" if sp["seq"] > sp["dim"] else "seq-le-dim"))
    return ()


def cases(level, holdout=False):
    """Dev cases are the level's own shapes, exactly as nkibench grades them. Holdout cases are
    other shapes and another seed, plus each hostile value kind on the first holdout shape."""
    spec = nkibench.LEVELS[level]
    ref = spec["ref"]
    mk = lambda sp, seed, values="normal": Case(
        sp, nkibench.label(sp, level) + ("" if values == "normal" else f" [{values}]"),
        shape_tags(ref, sp) + (() if values == "normal" else (values,)), seed, values)
    if not holdout:
        return [mk(sp, 0) for sp in spec["shapes"]]
    shapes = HOLDOUT.get(ref) or spec["shapes"]
    if ref is nkibench.ref_matmul and all(shape_tags(ref, sp) == ("single-tile",)
                                          for sp in spec["shapes"]):
        shapes = spec["shapes"]     # a single-tile level is not asked to tile
    return ([mk(sp, 1) for sp in shapes]
            + [mk(shapes[0], 2, kind) for kind in HOSTILE.get(ref, ())])


def _hostile(x, kind):
    if kind == "large-mean":
        x = x + 1e4
    elif kind == "all-negative":
        x = -np.abs(x) - 0.5
    elif kind == "index-coded":
        x = np.arange(1, x.size + 1).reshape(x.shape)
    elif kind == "cancelling":
        x = np.where(x > 0, 1e3, -1e3)
    elif kind == "large-magnitude":
        x = x * 1e4
    elif kind == "large-scores":
        x = x * 20
    return x.astype(np.float32)


def make_args(case, level):
    args, _ = nkibench.make_inputs(case.spec, level, case.seed)
    if case.values == "normal":
        return args
    return tuple(_hostile(x, case.values) if isinstance(x, np.ndarray) else x for x in args)


# ---------------------------------------------------------------- one case

@dataclass
class CaseResult:
    case: Case
    status: str                 # pass | wrong | nonfinite | shape | error | input | hazard | traffic
    msg: str = ""
    worst: float = 0.0          # max |got - ref| / scale
    bound_use: float = None     # max |got - ref| / bound, where there is a proven bound
    facts: tuple = ()           # compared across cases to find what the wrong ones share
    brief: str = ""
    key: str = ""               # crash class, for status == "error"


def _ranges(coords, limit=3):
    coords = [int(c) for c in coords]
    runs, start = [], coords[0]
    for a, b in zip(coords, coords[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}..{a}")
            if b is not None:
                start = b
    more = f" +{len(runs) - limit} more" if len(runs) > limit else ""
    return ", ".join(runs[:limit]) + more


def _layout(ref):
    """(tile size, name) for each output axis."""
    if ref is nkibench.ref_matmul:
        return (nkibench.GEMM_STATIONARY_FMAX, nkibench.GEMM_MOVING_FMAX), ("row", "column")
    if ref is nkibench.ref_avgpool2d:
        return (nkibench.PMAX, ONE_TILE, ONE_TILE), ("channel", "row", "column")
    return (nkibench.PMAX, ONE_TILE), ("row", "column")


def _locate(bad, tiles, names):
    """(text, facts): where the wrong outputs are. A fact prefixed '~' is merely consistent -- a
    one-row output cannot SHOW 'only the last row' -- so it supports a cross-case fact without
    being evidence for it."""
    parts, facts = [], []
    for ax, T in enumerate(tiles):
        D = bad.shape[ax]
        others = tuple(i for i in range(bad.ndim) if i != ax)
        coords = np.nonzero(bad.any(axis=others) if others else bad)[0]
        nm, ntiles = names[ax], -(-D // T)
        if len(coords) == D:
            parts.append(f"every {nm}")
            if D == 1:
                facts.append(f"~only the last {nm} is wrong")
            continue
        tiles_bad = sorted({int(c) // T for c in coords})
        if list(coords) == [D - 1]:
            parts.append(f"only the last {nm} ({nm} {D - 1} of {D})")
            facts.append(f"only the last {nm} is wrong")
        elif ntiles == 1:
            parts.append(f"{len(coords)}/{D} {nm}s ({_ranges(coords)})")
        elif len(tiles_bad) < ntiles:
            parts.append(f"{nm}s {_ranges(coords)}, in {nm} tile(s) {tiles_bad} of {ntiles} "
                         f"(a tile is {T} {nm}s)")
            if tiles_bad == [0]:
                facts.append(f"wrong outputs lie only in the first {nm} tile")
            elif tiles_bad == list(range(1, ntiles)):
                facts.append(f"wrong outputs lie in every {nm} tile but the first")
        else:
            parts.append(f"{len(coords)}/{D} {nm}s ({_ranges(coords)})")
    return "; ".join(parts), facts


def _value_facts(g, ref, bad):
    gb, rb = g[bad], ref[bad]
    weak = "~" if gb.size == 1 else ""      # one wrong output is trivially "all equal"
    facts = []
    if gb.size and np.isfinite(gb[0]) and (gb == gb[0]).all() and not (rb == rb[0]).all():
        facts.append(f"{weak}every wrong output is exactly {gb[0]:g}")
    return facts


def _explain(ref, args, g, ref64, scale, bad):
    """What the wrong outputs ARE, per operation: the bug each one is the fingerprint of."""
    near = lambda other: bool(np.all(np.abs(g - other)[bad] <= NEAR * scale[bad]))
    a = [x.astype(np.float64) if isinstance(x, np.ndarray) else x for x in args]
    facts = []
    if ref is nkibench.ref_matmul:
        A, B = a
        K, T = A.shape[0], nkibench.PMAX
        if K > T:
            last = (K - 1) // T * T
            for name, lo, hi in (("last", last, K), ("first", 0, T)):
                if near(A[lo:hi].T @ B[lo:hi]):
                    facts.append(f"each wrong output equals the product over the {name} K tile "
                                 f"only")
                    break
    elif ref is nkibench.ref_avgpool2d:
        p = a[1]
        if p > 1 and near(ref64 * p * p):
            facts.append("each wrong output is the SUM of its window, not the mean")
    elif ref is nkibench.ref_transpose2d:
        x, (F1, F2) = a
        if near(x):
            facts.append("the output is the input unchanged, copied without transposing")
        elif F1 != F2 and near(x.reshape(-1, F2, F1).transpose(0, 2, 1).reshape(x.shape)):
            facts.append("the rows were transposed as F2 x F1 matrices instead of F1 x F2")
    return facts


def compare(level, case, args, got, want):
    """One case against its error bound. Returns a CaseResult; status "pass" means inside it."""
    ref = nkibench.LEVELS[level]["ref"]
    got = np.asarray(got)
    if got.shape != np.shape(want):
        msg = (f"returned shape {got.shape}, expected {np.shape(want)}. Check the output-size "
               f"arithmetic, not the values")
        return CaseResult(case, "shape", msg, np.inf, brief=msg.split(". ")[0])
    g = got.astype(np.float64)
    bound = error_bound(level, args)
    if bound is None:
        m = nkibench.describe_mismatch(got, want)
        if not m:
            return CaseResult(case, "pass")
        rms = float(np.sqrt((np.asarray(want, np.float64) ** 2).mean())) or 1.0
        bad = ~np.isfinite(g) | (np.abs(g - want) / rms > 2e-2)
        return CaseResult(case, "nonfinite" if not np.all(np.isfinite(g)) else "wrong", m,
                          np.inf, None, tuple(_value_facts(g, np.asarray(want, np.float64), bad)),
                          m.splitlines()[0][:140])
    ref64, scale, k = bound
    scale = np.maximum(scale, TINY)
    limit = gamma(k) * scale
    with np.errstate(all="ignore"):
        abserr = np.abs(g - ref64)
        abserr[np.isnan(abserr)] = np.inf
        err = abserr / scale
        use = np.where(abserr == 0, 0.0, abserr / np.maximum(limit, TINY))
    bad = abserr > limit
    worst, bound_use = float(err.max()), float(use.max())
    if not bad.any():
        return CaseResult(case, "pass", "", worst, bound_use)
    nonfinite = ~np.isfinite(g)
    idx = np.unravel_index(int(np.argmax(err)), err.shape)
    d = (g - ref64)[bad & ~nonfinite]
    direction = ("non-finite" if not d.size else "all too large" if (d > 0).all()
                 else "all too small" if (d < 0).all() else "mixed signs")
    at = ", ".join(str(int(i)) for i in idx)
    head = f"out[{at}] expected {ref64[idx]:.6g}, got {g[idx]:.6g}"
    if np.isfinite(g[idx]):
        head += (f" (error {err[idx]:.2g} x scale, "
                 + (f"{use[idx]:.3g}x the float32 error bound)" if k else
                    "and this operation only moves data, so the answer must match bit for bit)"))
    msg = f"{head}; {bad.mean():.0%} of outputs wrong, {direction}"
    if nonfinite.any():
        msg += f", {int(nonfinite.sum())} of them nan or inf"
    tiles, names = _layout(ref)
    where, facts = _locate(bad, tiles, names)
    facts += _value_facts(g, ref64, bad) + _explain(ref, args, g, ref64, scale, bad)
    msg += ("; wrong everywhere" if all(w.startswith("every ") for w in where.split("; "))
            else f"; wrong at: {where}")
    if nonfinite.any() and not np.isinf(g).any():
        facts.append("every non-finite output is nan")
    if not nonfinite.any() and worst < ROUNDING_CEILING:
        msg += ". The error is precision-sized, not a logic bug"
    return CaseResult(case, "nonfinite" if nonfinite.any() else "wrong", msg, worst, bound_use,
                      tuple(facts), f"{bad.mean():.0%} of outputs wrong: {head}"[:160])


# ---------------------------------------------------------------- across cases

def pattern_tag(results):
    """The case tags that best separate failing from passing cases: (tags, exclusive) or None.

    A tag shared with many passing cases is a coincidence, not a pattern, and saying nothing beats
    pointing the model at the wrong cause: it must cover >= 90% of the failures at >= 75%
    precision. Several tags can separate the cases equally well -- every shape with several M
    tiles here also has several N tiles -- and then all of them are returned, since the shapes
    cannot tell which one is the cause.
    """
    fails = [set(r.case.tags) for r in results if r.status != "pass"]
    passes = [set(r.case.tags) for r in results if r.status == "pass"]
    if not fails or not passes:
        return None
    scored = []
    for t in sorted(set().union(*fails)):
        nf, np_ = sum(t in f for f in fails), sum(t in p for p in passes)
        scored.append((nf / len(fails), nf / (nf + np_), t))
    exclusive = tuple(t for cover, precision, t in scored if cover == 1 and precision == 1)
    if exclusive:
        return exclusive, True
    cover, precision, t = max(scored, key=lambda x: x[0] * x[1])
    return ((t,), False) if cover >= 0.9 and precision >= 0.75 else None


def common_facts(fails, internal=True):
    """Facts true of at least 90% of the numerically wrong cases, and shown by at least one."""
    measured = [set(r.facts) for r in fails if r.status in ("wrong", "nonfinite")]
    strong, support = {}, {}
    for fs in measured:
        for f in fs:
            base = f.lstrip("~")
            support[base] = support.get(base, 0) + 1
            if not f.startswith("~"):
                strong[base] = strong.get(base, 0) + 1
    return [f for f in sorted(strong) if support[f] >= 0.9 * len(measured)
            and (internal or not f.startswith("every non-finite"))]


def _diverse(results, k):
    """Worst first, but two examples that differ in tags, so they are not the same bug twice."""
    picked, seen = [], set()
    for r in sorted(results, key=lambda r: -r.worst):
        if frozenset(r.case.tags) not in seen:
            picked.append(r)
            seen.add(frozenset(r.case.tags))
        if len(picked) == k:
            break
    return picked


def evidence(results, max_cases=2):
    """The distilled failure: counts, the cross-case pattern, the shared facts, the worst cases."""
    fails = [r for r in results if r.status != "pass"]
    lines = [f"{len(results) - len(fails)} of {len(results)} shapes passed."]
    pt = pattern_tag(results)
    if pt:
        tags, exclusive = pt
        nf = sum(tags[0] in r.case.tags for r in fails)
        named = " and ".join(f"'{t}'" for t in tags)
        lines.append(f"Pattern: every failing shape is {named} and no passing shape is."
                     if exclusive else
                     f"Pattern: {nf}/{len(fails)} failing shapes are {named}.")
    elif len(fails) < len(results):
        # Which shapes pass is a diagnosis in itself, so list them when no tag names the split.
        lines.append("Passed: " + "; ".join(r.case.label for r in results if r.status == "pass")
                     + ". Failed: " + "; ".join(r.case.label for r in fails) + ".")
    facts = common_facts(fails, internal=False)
    if facts:
        lines.append("In every wrong shape: " + "; ".join(facts) + ".")
    lines += [f"- {r.case.label}: {r.msg}" for r in _diverse(fails, max_cases)]
    return "\n".join(lines)


# ---------------------------------------------------------------- verdict -> instruction

@dataclass
class Directive:
    key: str            # failure class, e.g. "rule:framework-call", "numeric:k-overwrite"
    instruction: str    # the one change to make
    evidence: str = ""  # the distilled evidence shown with it
    brief: str = field(default="")

    def text(self):
        return (self.evidence + "\n" if self.evidence else "") + self.instruction


# A kernel can break several rules; fix the ones that change its structure first.
RULE_ORDER = ["no-kernel", "no-jit", "framework-call", "matmul-op", "host-transpose",
              "pmax-literal"]

_ON_CHIP = {
    nkibench.ref_avgpool2d: "Reduce on chip instead: build the pooling windows as an access "
                            "pattern over the SBUF tile with tile.ap([[stride, count], ...]), "
                            "reduce the two window axes with nl.sum(view, axis=[3, 4]), and scale "
                            "by 1.0 / (pool_size * pool_size) with nisa.tensor_scalar.",
    nkibench.ref_transpose2d: "Move the data yourself: loop over f1 and f2 with nl.affine_range "
                              "and copy one column of the SBUF tile at a time with "
                              "nisa.tensor_copy, from column f1*F2 + f2 of the input tile to "
                              "column f2*F1 + f1 of the output tile.",
    nkibench.ref_matmul: "Multiply on chip instead: nisa.nc_matmul(dst=psum_tile, "
                         "stationary=lhsT_tile, moving=rhs_tile), with both operands in SBUF "
                         "tiles and the result in a PSUM tile.",
    nkibench.ref_attention: "Compute it on chip instead: nisa.nc_matmul for the two products, and "
                            "nl.max, nl.exp and nl.sum over tiles for the softmax.",
}


def rule_kind(violation):
    for needle, kind in (("no function named", "no-kernel"), ("is not decorated", "no-jit"),
                         ("which hands the whole", "framework-call"),
                         ("`@` matmul operator", "matmul-op"),
                         ("transposes the input on the host", "host-transpose"),
                         ("exceeds the maximum of", "pmax-literal")):
        if needle in violation:
            return kind
    return "other"


def rule_directive(level, violations, source):
    """Order the violations and name ONE change: all the lines of the first kind."""
    import inspect
    import re
    spec = nkibench.LEVELS[level]
    by_kind = {}
    for v in violations:
        by_kind.setdefault(rule_kind(v), []).append(v)
    kind = min(by_kind, key=lambda k: RULE_ORDER.index(k) if k in RULE_ORDER else 99)
    vs = by_kind[kind]
    lines = source.splitlines()
    nums = [int(m[1]) for m in (re.match(r"line (\d+):", v) for v in vs) if m]
    where = ", ".join(f"line {n} `{lines[n - 1].strip()}`" for n in nums if 0 < n <= len(lines))
    called = ", ".join(dict.fromkeys(m[1] for m in (re.search(r"calls `([^`]+)`", v) for v in vs)
                                     if m))
    args = ", ".join(inspect.signature(spec["ref"]).parameters)
    fix = {
        "no-kernel": f"Start the function with exactly this line:  def {spec['entry']}({args}):  "
                     f"and put @nki.jit on the line above it.",
        "no-jit": f"Put @nki.jit on the line directly above `def {spec['entry']}`.",
        "framework-call": f"Remove the call to `{called}`: it hands the whole operation to a "
                          f"framework, and this level is about computing it in the kernel. "
                          f"{_ON_CHIP.get(spec['ref'], '')}",
        "matmul-op": f"Remove the `@` operator: it does the whole job on the host. "
                     f"{_ON_CHIP[nkibench.ref_matmul]}",
        "host-transpose": "Remove the `.T`: it transposes an input on the host. Use the input as "
                          "it arrives; for a matmul the left operand is already transposed, with "
                          "K on the partition axis, so it is the stationary operand as it is.",
        "pmax-literal": f"A tile's first dimension may be at most {nkibench.PMAX}. Do not "
                        f"allocate one tile for the whole tensor: loop over that dimension in "
                        f"chunks of at most {nkibench.PMAX} with nl.affine_range and allocate the "
                        f"tile inside the loop with the chunk's own shape.",
    }.get(kind, " ".join(vs))
    rest = len(violations) - len(vs)
    later = f" ({rest} other rule problem(s) will be handled after this one.)" if rest else ""
    return Directive(f"rule:{kind}", fix.strip() + later,
                     f"Rule broken at {where}." if where else "",
                     "rule violation: " + vs[0][:120])


TAG_MEANING = {
    "multi-k-tile": f"K is larger than {nkibench.PMAX}, so the contraction spans several tiles",
    "multi-m-tile": f"M is larger than {nkibench.GEMM_STATIONARY_FMAX}, so the output has several "
                    f"row tiles",
    "multi-n-tile": f"N is larger than {nkibench.GEMM_MOVING_FMAX}, so the output has several "
                    f"column tiles",
    "single-tile": "everything fits in one tile",
    "partition-below-max": f"the partition dimension is smaller than {nkibench.PMAX}",
    "partition-at-max": f"the partition dimension is exactly {nkibench.PMAX}",
    "non-square": "F1 and F2 differ",
    "square": "F1 equals F2",
    "odd-pool": "the pool size is odd",
    "even-pool": "the pool size is even",
    "seq-gt-dim": "seq is larger than dim",
    "seq-le-dim": "seq is at most dim",
}

# When the failures are exactly the cases with one tag, some tags have a known fix.
TAG_FIX = {
    "multi-k-tile": f"Results are wrong exactly when K is larger than {nkibench.PMAX}. Split K "
                    f"into chunks of {nkibench.PMAX} and let them add up: allocate ONE psum tile "
                    f"outside the K loop, call nisa.nc_matmul into that same tile once per chunk, "
                    f"and copy it out only after the loop.",
    "multi-m-tile": f"Results are wrong exactly when M is larger than "
                    f"{nkibench.GEMM_STATIONARY_FMAX}. Loop over M in chunks of that size, and "
                    f"give every chunk its own origin: read lhsT[:, m0:m1] and write "
                    f"result[m0:m1, ...], never the same rows twice.",
    "multi-n-tile": f"Results are wrong exactly when N is larger than "
                    f"{nkibench.GEMM_MOVING_FMAX}. Loop over N in chunks of that size, and give "
                    f"every chunk its own origin: read rhs[:, n0:n1] and write "
                    f"result[..., n0:n1], never the same columns twice.",
    "partition-below-max": f"Results are wrong exactly when the first dimension is smaller than "
                           f"{nkibench.PMAX}. {nkibench.PMAX} is a maximum, not a size: read "
                           f"every tile shape and loop bound from the input's .shape.",
    "non-square": "Results are wrong exactly when F1 and F2 differ, so the two are swapped "
                  "somewhere. The element at input column f1*F2 + f2 goes to output column "
                  "f2*F1 + f1.",
}


def diagnose(level, results):
    """The numerical verdict as ONE directive. Ordered by how much each finding narrows the
    search: what the wrong values are, then where they are, then which shapes they are on."""
    spec = nkibench.LEVELS[level]
    fails = [r for r in results if r.status != "pass"]
    ev = evidence(results)
    brief = (f"{len(results) - len(fails)} of {len(results)} shapes passed; on "
             f"{fails[0].case.label}: " if len(fails) < len(results) else "") + fails[0].brief
    D = lambda key, instruction: Directive(key, instruction, ev, brief)

    for status, key in (("input", "input-modified"), ("shape", "shape"),
                        ("hazard", "hardware-hazard")):
        hit = next((r for r in fails if r.status == status), None)
        if hit:
            return Directive(key, hit.msg if status != "shape" else
                             f"Wrong output shape on {hit.case.label}: {hit.msg}. Return exactly "
                             f"the reference's shape.", "", brief)
    if all(r.status == "traffic" for r in fails):
        return Directive("traffic", fails[0].msg,
                         f"{len(results) - len(fails)} of {len(results)} shapes are inside this "
                         f"level's traffic bar.", brief)
    fails = [r for r in fails if r.status in ("wrong", "nonfinite")] or fails
    facts = common_facts(fails)
    fact = lambda s: next((f for f in facts if s in f), None)
    pt = pattern_tag(results)
    tag = pt[0][0] if pt and len(pt[0]) == 1 else None      # one tag, or the cause is ambiguous
    where = next((f for f in facts if f.startswith("wrong outputs lie")), "")
    unwritten = ("The kernel ran, so the arithmetic is probably fine. Make every tile of the loop "
                 "reach the returned tensor -- PSUM to SBUF with nisa.tensor_copy, then SBUF to "
                 "the output with nisa.dma_copy -- and give each tile its own destination slice, "
                 "since writing every tile to the same place leaves the rest unwritten.")
    if any(r.status == "nonfinite" for r in fails):
        if fact("every non-finite output is nan") and spec["ref"] is not nkibench.ref_attention:
            # The simulator fills memory nothing wrote to with nan, so this is a location bug.
            return D("numeric:unwritten",
                     "Some outputs are nan: that part of the returned tensor was never written"
                     + (f" ({where})" if where else "") + ". " + unwritten)
        return D("numeric:nonfinite",
                 "Some outputs are nan or inf where the true answer is finite. Either a PSUM or "
                 "SBUF tile is read before anything wrote to it, or a step overflows: subtract "
                 "the row maximum before nl.exp. Make sure every tile is fully written before it "
                 "is read.")
    if all(r.bound_use is not None and r.worst < ROUNDING_CEILING for r in fails):
        return D("numeric:precision",
                 "The logic is right, but the rounding error is larger than float32 arithmetic "
                 "allows. Some tile holds the data at lower precision: allocate every SBUF tile "
                 "and the output with the input's own dtype (dtype=x.dtype), and PSUM tiles as "
                 "nl.float32. Do not use nl.bfloat16 or float16 anywhere.")
    f = fact("K tile only")
    if f:
        which = "last" if "last" in f else "first"
        return D("numeric:k-overwrite",
                 f"Each wrong output is the product over the {which} K tile only, so the K tiles "
                 f"are not being added together. Allocate ONE psum tile OUTSIDE the K loop, call "
                 f"nisa.nc_matmul into that same tile for every K chunk so the partial products "
                 f"accumulate there, and copy it out with nisa.tensor_copy only after the loop.")
    if fact("SUM of its window"):
        return D("numeric:sum-not-mean",
                 "Each output is the sum of its pooling window, not the mean. Scale the summed "
                 "tile by 1.0 / (pool_size * pool_size) with nisa.tensor_scalar(dst=, data=, "
                 "op0=nl.multiply, operand0=...) before copying it out.")
    if fact("copied without transposing"):
        return D("numeric:not-transposed",
                 "The output is the input unchanged: the data was copied, not transposed. The "
                 "element at input column f1*F2 + f2 must land at output column f2*F1 + f1.")
    if fact("F2 x F1 matrices"):
        return D("numeric:f1f2-swapped", TAG_FIX["non-square"])
    const = fact("every wrong output is exactly")
    if const and const.endswith(" 0"):
        return D("numeric:unwritten",
                 "Wrong outputs are all exactly 0: they were never written"
                 + (f" ({where})" if where else "") + ". " + unwritten)
    if tag in TAG_FIX:
        return D(f"numeric:tag:{tag}", TAG_FIX[tag])
    for axis in ("row", "column", "channel"):
        if fact(f"only the last {axis} is wrong"):
            return D(f"numeric:last-{axis}",
                     f"Only the last {axis} is wrong: an off-by-one in the {axis} bounds. A loop "
                     f"or slice must run up to the size itself, not size - 1.")
    if pt:
        return D("numeric:tag:" + "+".join(pt[0]),
                 "The kernel is right except when "
                 + ", or ".join(TAG_MEANING.get(t, t) for t in pt[0])
                 + ". Find the assumption in the code that these shapes break and remove it.")
    return D("numeric:other",
             f"The results are wrong on every kind of shape, so this is the core arithmetic or "
             f"the operand layout, not an edge case. Check each step against the operation: "
             f"{spec['op']}.")
