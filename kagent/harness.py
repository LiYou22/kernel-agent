"""Verification harness: rule check + numerical check + a failure description the agent can act on.

TOLERANCE. A case passes iff, for every output element,
    |got - ref| <= RTOL * scale,   RTOL = 1e-5
# errors below this are float32 rounding, not logic bugs (those are >= 1e-3, see TOLERANCE)
ROUNDING_CEILING = 1e-3
where ref is float64 on the same float32 inputs and `scale` is the level's condition-aware
magnitude (sum|x| for a sum, |a||x|+|b| for affine, ...), so cancellation in the true result
doesn't make the test impossible. Why 1e-5:
  * floor: float32 has eps = 1.19e-7; rounding the output alone costs <= 6e-8, and float32
    accumulation over our longest rows (N <= 2053) has typical error sqrt(N)*eps ~ 5e-6.
    A correct float32 kernel lands under 1e-5 on these cases (checked in tests/).
  * ceiling: every bug class we care about (dropped partial tile, wrong identity, overflow, missing
    max-subtraction, E[x^2]-E[x]^2 cancellation) produces errors >= 1e-3 relative to scale, two
    orders of magnitude above the threshold. Looser tolerances only buy room for silent bugs.
  * exact ops (max, transpose) are held to the same bound; any real bug there is O(1).
Non-finite output where the reference is finite always fails.
"""
import argparse
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np

from . import guard, rules, static
from .levels import LEVELS, TINY, Case, Level

RTOL = 1e-5
# errors below this are float32 rounding, not logic bugs (those are >= 1e-3, see TOLERANCE)
ROUNDING_CEILING = 1e-3
CASE_TIMEOUT_S = 20.0


@dataclass
class Violation:
    line: int | None
    kind: str
    what: str
    source_line: str = ""

    def fix(self):
        return rules.fix_for(self.kind, self.what)

    def text(self):
        loc = f"line {self.line} `{self.source_line}`" if self.line else "kernel"
        return f"{loc}: {self.what} [{self.kind}]"


@dataclass
class CaseResult:
    case: Case
    status: str                 # pass | wrong | nonfinite | shape | error | timeout
    msg: str = ""
    worst: float = 0.0          # max |got-ref| / scale
    facts: tuple = ()           # location/value facts, compared across cases to find the pattern


@dataclass
class Report:
    level: Level
    violations: list = field(default_factory=list)
    results: list = field(default_factory=list)
    load_error: str | None = None

    @property
    def n_pass(self):
        return sum(r.status == "pass" for r in self.results)

    @property
    def numerics_ok(self):
        return self.load_error is None and bool(self.results) and self.n_pass == len(self.results)

    @property
    def passed(self):
        return self.numerics_ok and not self.violations

    def failure_classes(self):
        """Coarse labels for the taxonomy; the agent refines them."""
        c = sorted({v.kind for v in self.violations})
        if self.load_error:
            c.append("load-error")
        c += sorted({r.status for r in self.results if r.status != "pass"})
        return c

    def summary(self):
        lv = self.level
        if self.load_error:
            return f"L{lv.num} {lv.name}: FAIL, kernel did not load ({self.load_error})"
        verdict = "PASS" if self.passed else "FAIL"
        return (f"L{lv.num} {lv.name}: {verdict}, {len(self.violations)} rule violation(s), "
                f"{self.n_pass}/{len(self.results)} cases correct")

    def numeric_feedback(self, max_cases=2):
        """Distilled numerical failure: the cross-case pattern first, then the worst cases."""
        if self.load_error:
            return self.load_error
        fails = [r for r in self.results if r.status != "pass"]
        if not fails:
            return f"All {len(self.results)} cases correct (tolerance {RTOL:g} x scale)."
        lines = [f"{len(fails)}/{len(self.results)} cases wrong."]
        for pat in filter(None, [_pattern(self.results), _common_facts(fails)]):
            lines.append(f"Pattern: {pat}.")
        errs = {}
        for r in fails:
            if r.status in ("error", "timeout"):
                errs.setdefault(r.msg, []).append(r.case.name)
        for msg, names in errs.items():
            lines.append(f"Crash in {len(names)} case(s) (e.g. '{names[0]}'): {msg}")
        shown = [r for r in sorted(fails, key=lambda r: -r.worst) if r.status not in ("error", "timeout")]
        for r in _diverse(shown, max_cases):
            lines.append(f"- '{r.case.name}': {r.msg}")
        return "\n".join(lines)

    def text(self, max_cases=2):
        out = [self.summary()]
        if self.violations:
            out.append("Rule violations (a violating kernel scores zero):")
            out += [f"- {v.text()}" for v in self.violations]
        out.append(self.numeric_feedback(max_cases))
        return "\n".join(out)


def _diverse(results, k):
    """Worst-first, but prefer cases that differ in tags so two examples aren't the same bug twice."""
    picked, seen = [], set()
    for r in results:
        key = frozenset(r.case.tags)
        if key not in seen:
            picked.append(r)
            seen.add(key)
        if len(picked) == k:
            break
    return picked


def _pattern(results):
    """Find the case tag that best separates failing from passing cases."""
    fails = [set(r.case.tags) for r in results if r.status != "pass"]
    passes = [set(r.case.tags) for r in results if r.status == "pass"]
    if not passes:
        return "every case fails, including small single-tile ones" if fails else None
    tags = set().union(*fails)
    best = None
    for t in sorted(tags):
        nf = sum(t in f for f in fails)
        np_ = sum(t in p for p in passes)
        if nf == len(fails) and np_ == 0:
            return f"every failing case is '{t}' and no passing case is"
        score = (nf / len(fails)) * (nf / (nf + np_))
        if best is None or score > best[0]:
            best = (score, t, nf, np_)
    # a tag shared with many passing cases is a coincidence, not a pattern; saying nothing beats
    # pointing the model at the wrong cause
    if best:
        _, t, nf, np_ = best
        if nf / len(fails) >= 0.9 and nf / (nf + np_) >= 0.75:
            return f"{nf}/{len(fails)} failing cases are '{t}' (only {np_} passing case(s) are)"
    return None


def _common_facts(fails):
    """Facts true of (nearly) every failing case, e.g. 'every wrong output is exactly 0'."""
    measured = [set(r.facts) for r in fails if r.status in ("wrong", "nonfinite")]
    if len(measured) < 2:
        return None
    counts = {}
    for fs in measured:
        for f in fs:
            counts[f] = counts.get(f, 0) + 1
    common = [f for f, c in sorted(counts.items()) if c >= 0.9 * len(measured)]
    if not common:
        return None
    n = len(measured)
    return f"in {'all' if all(counts[f] == n for f in common) else 'nearly all'} {n} wrong cases, " + "; ".join(common)


def _ranges(coords, limit=3):
    coords = list(coords)
    runs, start = [], coords[0]
    for a, b in zip(coords, coords[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}..{a}")
            if b is not None:
                start = b
    more = f" +{len(runs) - limit} more" if len(runs) > limit else ""
    return ", ".join(runs[:limit]) + more


def _locate(bad, tiles, names):
    """Returns (text, facts). Facts are phrased to read well when shared across cases."""
    parts, facts = [], []
    for ax, T in enumerate(tiles):
        D = bad.shape[ax]
        others = tuple(i for i in range(bad.ndim) if i != ax)
        coords = np.nonzero(bad.any(axis=others) if others else bad)[0]
        nm = names[ax]
        if len(coords) == D:
            parts.append(f"every {nm} (all {D})")
            continue
        ntiles = -(-D // T)
        tiles_bad = sorted({int(c) // T for c in coords})
        last_start = (ntiles - 1) * T
        if list(coords) == [D - 1]:
            parts.append(f"only the last {nm} ({nm} {D - 1} of {D})")
            facts.append(f"only the last {nm} is wrong")
        elif ntiles == 1:
            parts.append(f"{len(coords)}/{D} {nm}s ({_ranges(coords)})")
        elif tiles_bad == [ntiles - 1] and D % T:
            parts.append(f"only {nm}s {_ranges(coords)}, all inside the last partial {nm} tile "
                         f"({nm}s {last_start}..{D - 1}, width {D - last_start})")
            facts.append(f"wrong outputs lie only in the last partial {nm} tile")
        else:
            parts.append(f"{len(coords)}/{D} {nm}s ({_ranges(coords)}), in {nm} tile(s) "
                         f"{tiles_bad} of {ntiles} (tile = {T} {nm}s)")
    return "; ".join(parts), facts


def _value_facts(g, ref, bad):
    gb, rb = g[bad], ref[bad]
    facts = []
    if gb.size >= 2 and np.isfinite(gb[0]) and (gb == gb[0]).all():
        facts.append(f"every wrong output is exactly {gb[0]:g}")
    if rb.size >= 2 and (rb < 0).all():
        facts.append("every wrong output has a negative expected value")
    elif rb.size >= 2 and (rb > 0).all() and (gb < rb).all():
        facts.append("every wrong output has a positive expected value and is too small")
    return facts


def compare(level, case, got, ref64, scale):
    if not isinstance(got, np.ndarray):
        return CaseResult(case, "shape", f"returned {type(got).__name__}, expected an array of "
                                         f"shape {ref64.shape}", np.inf)
    if got.shape != ref64.shape:
        return CaseResult(case, "shape", f"returned shape {got.shape}, expected {ref64.shape}", np.inf)
    g = got.astype(np.float64)
    scale = np.broadcast_to(np.maximum(scale, TINY), ref64.shape)
    with np.errstate(all="ignore"):
        err = np.abs(g - ref64) / scale
    err[np.isnan(err)] = np.inf
    err[(g == ref64)] = 0.0  # equal infinities
    bad = err > RTOL
    worst = float(err.max()) if err.size else 0.0
    if not bad.any():
        return CaseResult(case, "pass", "", worst)
    nonfinite = ~np.isfinite(g) & np.isfinite(ref64)
    idx = np.unravel_index(int(np.argmax(err)), err.shape)
    d = (g - ref64)[bad & np.isfinite(g)]
    direction = ("too large" if (d > 0).all() else "too small" if (d < 0).all() else "mixed signs") if d.size else "non-finite"
    istr = ", ".join(map(str, idx))
    msg = (f"out[{istr}] expected {ref64[idx]:.6g} got {g[idx]:.6g} "
           f"(error {err[idx]:.2g} x scale, limit {RTOL:g}); {bad.mean():.0%} of outputs wrong, "
           f"{direction}")
    if nonfinite.any():
        msg += f", {int(nonfinite.sum())} non-finite (nan/inf) where the answer is finite"
    where, facts = _locate(bad, level.out_tile, level.out_axes)
    facts += _value_facts(g, ref64, bad)
    msg += f"; wrong at: {where}"
    if facts:
        msg += "; " + "; ".join(f for f in facts if not f.startswith(("only the last", "wrong outputs lie")))
    if not nonfinite.any() and worst < ROUNDING_CEILING:
        msg += (f". Error is rounding-sized (< {ROUNDING_CEILING:g} x scale), not a logic bug: "
                f"accumulate in float64 (np.zeros(n, dtype=np.float64)) and cast only the final result")
    return CaseResult(case, "nonfinite" if nonfinite.any() else "wrong", msg, worst, tuple(facts))


def _crash_msg(exc, source_lines):
    tb = traceback.extract_tb(exc.__traceback__)
    frames = [f for f in tb if f.filename == guard.KERNEL_FILENAME]
    where = ""
    if frames:
        ln = frames[-1].lineno
        src = source_lines[ln - 1].strip() if 0 < ln <= len(source_lines) else ""
        where = f" at line {ln} `{src}`"
    return f"{type(exc).__name__}: {exc}{where}"


def verify(level, source):
    lines = source.splitlines()
    src = lambda ln: lines[ln - 1].strip() if ln and 0 < ln <= len(lines) else ""
    rep = Report(level)
    viol = {}
    for ln, kind, what in static.check(source):
        viol[(ln, kind)] = Violation(ln, kind, what, src(ln))
    if any(k in ("syntax", "no-kernel") for _, k in viol):
        rep.violations = list(viol.values())
        rep.load_error = "; ".join(v.fix() for v in rep.violations if v.kind in ("syntax", "no-kernel"))
        return rep
    try:
        fn, rec = guard.load_kernel(source)
    except Exception as e:  # kernel module raised at import time
        rep.violations = list(viol.values())
        rep.load_error = _crash_msg(e, lines)
        return rep

    for case in level.get_cases():
        args64 = tuple(a.astype(np.float64) if isinstance(a, np.ndarray) else a for a in case.args)
        ref = np.asarray(level.ref(*args64), dtype=np.float64)
        scale = level.scale(args64, ref)
        try:
            got = guard.call_kernel(fn, rec, case.args, CASE_TIMEOUT_S)
        except guard.KernelTimeout:
            rep.results.append(CaseResult(case, "timeout", f"timed out after {CASE_TIMEOUT_S:g}s", np.inf))
            continue
        except Exception as e:
            rep.results.append(CaseResult(case, "error", _crash_msg(e, lines), np.inf))
            continue
        rep.results.append(compare(level, case, got, ref, scale))

    for (ln, kind), what in rec.violations.items():
        viol.setdefault((ln, kind), Violation(ln, kind, what, src(ln)))
    rep.violations = _dedupe(viol.values())
    return rep


# on one line, a specific violation explains the generic one (np.sum also trips "add.reduce")
_SUBSUMES = {"sum-like": {"reduction"}, "max-like": {"reduction"}, "linalg": {"matmul-op", "reduction"},
             "fancy-index": {"whole-array"}, "newaxis": {"broadcast"}, "layout": {"whole-array"}}


def _dedupe(violations):
    by_line = {}
    for v in violations:
        by_line.setdefault(v.line, []).append(v)
    out = []
    for vs in by_line.values():
        kinds = {v.kind for v in vs}
        drop = set().union(*(_SUBSUMES.get(k, set()) for k in kinds))
        out += [v for v in vs if v.kind not in drop or v.kind in _SUBSUMES]
    return sorted(out, key=lambda v: (v.line or 0, v.kind))


def main(argv=None):
    p = argparse.ArgumentParser(description="Verify a kernel file against a ladder level.")
    p.add_argument("level", type=int)
    p.add_argument("kernel_file")
    p.add_argument("-v", "--verbose", action="store_true", help="print every case")
    p.add_argument("--fixes", action="store_true", help="print the repair instruction per violation")
    a = p.parse_args(argv)
    rep = verify(LEVELS[a.level], open(a.kernel_file).read())
    print(rep.text())
    if a.fixes:
        for v in rep.violations:
            print(f"FIX line {v.line}: {v.fix()}")
    if a.verbose:
        for r in rep.results:
            print(f"  {r.status:9s} worst={r.worst:.2e}  {r.case.name}")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
