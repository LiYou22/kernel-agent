"""Verdict -> instruction. Turns a harness Report into ONE change for the model to make.

The worked example in the challenge: feeding back `line 16: calls banned max` reproduced the same
violation; "replace the np.max and np.sum calls with explicit python loops over the columns ..."
fixed it in one round. So every directive names an action, and only one, chosen in this order:
the kernel must load, then run, then return the right shape, then obey the rules, then be right.

`key` is a coarse failure class. The loop uses it to notice repeats (the numbers in a message
change every round; the class doesn't) and the taxonomy groups by it.
"""
from dataclasses import dataclass

from .harness import ROUNDING_CEILING, common_facts, pattern_tag
from .rules import TILE_C, TILE_R

# a kernel can break several rules; fix the ones that change its structure first
RULE_ORDER = ["syntax", "no-kernel", "import", "builtin", "escape", "whole-array", "linalg",
              "matmul-op", "sum-like", "max-like", "reduction", "builtin-reduce", "layout",
              "fancy-index", "newaxis", "broadcast", "keepdims", "indexing-fn"]

TAG_MEANING = {
    "partial-col-tile": f"N is not a multiple of {TILE_C}, so the last column tile is partial",
    "partial-row-tile": f"M is not a multiple of {TILE_R}, so the last row tile is partial",
    "multi-col-tile": f"N > {TILE_C}, so there is more than one column tile",
    "multi-row-tile": f"M > {TILE_R}, so there is more than one row tile",
    "dim-1": "M or N is exactly 1",
    "prime-dim": "a dimension is prime",
    "all-negative": "every value is negative",
    "huge-negative": "values are around -1e30",
    "large-magnitude": "values are around +-1e4",
    "zeros": "every value is 0",
    "constant-rows": "every row holds one repeated value",
    "large-mean": "values are around 1e4 with spread 1",
    "cancelling": "values of +-1e6 cancel to a small result",
    "negative-scale": "the scale a is negative",
    "huge-magnitude": "values are around +-1e18",
    "w-ge-S": "the window w is at least the sequence length",
    "w-0": "the window w is 0",
    "receptive-field-eq-L": "the receptive field equals L (one output)",
    "dilation-gt1": "dilation > 1",
    "stride-gt1": "stride > 1",
    "index-coded": "values are 1, 2, 3, ... (they encode their position)",
    "partial-k-tile": f"K is not a multiple of {TILE_R}",
    "multi-k-tile": f"K > {TILE_R}",
    "k-1": "K is exactly 1",
}


# when the failures are exactly the cases with one tag, some tags have a known fix
TAG_FIX = {
    "huge-magnitude": "Values around 1e18 overflow float32 when squared (float32 max is 3.4e38). "
                      "Accumulate the squares in float64 (cast each column with "
                      ".astype(np.float64) before squaring), or divide by the row's largest "
                      "magnitude before squaring.",
    "partial-k-tile": "Results are wrong exactly when K is not a multiple of the K tile. Loop "
                      "`for k0 in range(0, K, step)` with `k1 = min(k0 + step, K)` so the last, "
                      "partial K tile is included, and use its real size k1 - k0.",
    "multi-row-tile": "Results are wrong exactly when there is more than one row tile. Every tile "
                      "must use its own origin: read and write global row i0 + r, never the local "
                      "r, so later tiles do not overwrite earlier ones.",
    "multi-col-tile": "Results are wrong exactly when there is more than one column tile. Every "
                      "tile must use its own origin: read and write global column j0 + c, never "
                      "the local c, so later tiles do not overwrite earlier ones.",
    "w-ge-S": "Results are wrong exactly when the window w is at least the sequence length. Clip "
              "the window to j in [max(0, i - w), min(S - 1, i + w)].",
    "w-0": "Results are wrong exactly when w = 0. Then the window is just j = i and out[i] = v[i].",
    "receptive-field-eq-L": "Results are wrong exactly when the receptive field dilation*(K-1)+1 "
                            "equals L. Then L_out = 1: compute L_out = (L - dilation*(K-1) - 1) "
                            "// stride + 1 and let every loop run over range(L_out).",
    "dilation-gt1": "Results are wrong exactly when dilation > 1. Tap k of output t reads "
                    "x[c, t*stride + k*dilation].",
    "stride-gt1": "Results are wrong exactly when stride > 1. Output t starts at input position "
                  "t*stride.",
    "k-1": "Results are wrong exactly when K = 1. Make every K loop and slice work for K = 1.",
    "dim-1": "Results are wrong exactly when M or N is 1. Make every loop and slice work for a "
             "dimension of exactly 1.",
}


# the same tag can mean different bugs at different levels (large-mean: exp overflow in softmax,
# cancellation in layernorm), so these take precedence over the generic table and over nonfinite
LEVEL_TAG_FIX = {
    ("layernorm", "large-mean"):
        "Rows with a large mean (around 1e4) and spread 1 lose all precision: computing the "
        "variance as mean(x**2) - mean(x)**2 cancels catastrophically (and can go negative). "
        "Compute the mean first, then the variance as the mean of (x - mean)**2 in a second pass, "
        "accumulating in float64.",
}


@dataclass
class Directive:
    key: str            # failure class, e.g. "rule:sum-like", "numeric:partial-col"
    instruction: str    # the one change to make
    evidence: str = ""  # short distilled evidence shown with it

    def short(self, n=110):
        s = self.instruction.split(". ")[0]
        return s if len(s) <= n else s[: n - 3] + "..."


def _first_sentence_of(msg):
    return msg.split(";")[0]


def translate(rep):
    if rep.load_error:
        kinds = {v.kind for v in rep.violations}
        if "syntax" in kinds:
            return Directive("syntax", f"The code does not parse. {rep.load_error} Send the "
                                       f"complete function.")
        if "no-kernel" in kinds:
            return Directive("no-kernel", f"Start the function with exactly this line: "
                                          f"def {rep.level.signature}:")
        return Directive("load-crash", f"The module fails when it is loaded: {rep.load_error}. "
                                       f"Fix that line.")

    crashes = [r for r in rep.results if r.status in ("error", "timeout")]
    if crashes:
        r = crashes[0]
        if r.status == "timeout":
            return Directive("timeout", f"The kernel is too slow on '{r.case.name}'. Loop over "
                                        f"tiles and, inside a tile, over columns only, operating "
                                        f"on whole column slices; do not loop over single elements.")
        exc = r.msg.split(":")[0]
        names = ", ".join(f"'{c.case.name}'" for c in crashes[:3])
        return Directive(f"crash:{exc}",
                         f"The kernel raises {r.msg}. It crashes on {len(crashes)} of "
                         f"{len(rep.results)} cases, e.g. {names}. Fix it so it works for every "
                         f"shape, including dimensions of 1 and sizes that are not multiples of "
                         f"the tile size.")

    shape = [r for r in rep.results if r.status == "shape"]
    if shape:
        return Directive("shape", f"Wrong output shape: {shape[0].msg} on '{shape[0].case.name}'. "
                                  f"Return exactly that shape.")

    if rep.violations:
        by_kind = {}
        for v in rep.violations:
            by_kind.setdefault(v.kind, []).append(v)
        kind = min(by_kind, key=lambda k: RULE_ORDER.index(k) if k in RULE_ORDER else 99)
        vs = by_kind[kind]
        where = ", ".join(f"line {v.line} `{v.source_line}`" for v in vs if v.line)
        rest = len(rep.violations) - len(vs)
        later = f" ({rest} other rule problem(s) will be handled after this one.)" if rest else ""
        return Directive(f"rule:{kind}", vs[0].fix() + later, f"Rule broken at {where}." if where else "")

    return _numeric(rep)


def _numeric(rep):
    fails = [r for r in rep.results if r.status != "pass"]
    evidence = rep.numeric_feedback(max_cases=1)
    facts, _ = common_facts(fails)
    if any("mean is large compared to their spread" in f for f in facts):
        return Directive("numeric:cancellation",
                         "Only rows whose mean is large next to their spread are wrong: the "
                         "variance computed as mean(x**2) - mean(x)**2 cancels catastrophically "
                         "(and can go negative). Compute the mean first, then the variance as the "
                         "mean of (x - mean)**2 in a second pass, accumulating in float64.",
                         evidence)
    pt = pattern_tag(rep.results)
    if pt and (rep.level.name, pt[0]) in LEVEL_TAG_FIX:
        return Directive(f"numeric:tag:{pt[0]}", LEVEL_TAG_FIX[rep.level.name, pt[0]], evidence)
    if any(r.status == "nonfinite" for r in fails):
        return Directive("numeric:nonfinite",
                         "Some outputs are nan or inf although the true answer is finite. Find "
                         "the step that overflows or divides by zero for these inputs (for "
                         "example exp of a large number, or squaring a large number) and rewrite "
                         "it to stay finite, e.g. subtract the row maximum before exp, or divide "
                         "by the largest magnitude before squaring.", evidence)
    if all(r.worst < ROUNDING_CEILING for r in fails):
        return Directive("numeric:precision",
                         "The logic is right but float32 rounding error is too large. Accumulate "
                         "in float64: create accumulators with dtype=np.float64 and convert only "
                         "the final result.", evidence)

    fact = lambda s: any(s in f for f in facts)
    const = next((f.rsplit(" ", 1)[1] for f in facts if f.startswith("every wrong output is exactly")), None)
    tile_only = next((f for f in facts if "column tile only" in f), None)
    if tile_only:
        which = tile_only.split("over the ")[1].split(" column")[0]
        return Directive("numeric:tile-overwrite",
                         f"Each wrong row's result is the result over the {which} column tile "
                         f"only: the per-row result is overwritten for every column tile instead "
                         f"of combined across tiles. Keep one running value per row across ALL "
                         f"column tiles: initialize it before the column-tile loop, combine each "
                         f"tile into it inside the loop, and write it to the output only after "
                         f"the loop.", evidence)
    if const is not None and fact("negative expected value"):
        return Directive("numeric:identity",
                         f"Outputs whose true value is negative come out as exactly {const}: the "
                         f"running result starts from {const}. Start it from -np.inf instead "
                         f"(np.full(n, -np.inf)), the identity of max.", evidence)
    if fact("only the last row is wrong") or fact("only the last col is wrong"):
        axis = "row" if fact("only the last row is wrong") else "column"
        return Directive(f"numeric:last-{axis}",
                         f"Only the last {axis} is wrong: an off-by-one in the {axis} bounds. "
                         f"The end of each tile must be min(start + tile, size), and loops must "
                         f"run up to the size itself, not size - 1.", evidence)
    if fact("tile alone, not of the whole row"):
        return Directive("numeric:per-tile-stat",
                         "Each column tile is normalised with a statistic (max, sum or mean of "
                         "squares) of that tile alone. The statistic must cover the WHOLE row: "
                         "first loop over all column tiles to accumulate it, then loop over the "
                         "tiles again to write the output.", evidence)
    if fact("last partial K tile is missing") or fact("first") and fact("values of K only"):
        return Directive("numeric:partial-k",
                         "The last, partial K tile is left out of the sum. Loop "
                         "`for k0 in range(0, K, step)` with `k1 = min(k0 + step, K)` and use "
                         "its real size k1 - k0, so every k in 0..K-1 contributes.", evidence)
    if fact("result for window w - 1") or fact("result for window w + 1"):
        return Directive("numeric:band-edge",
                         "The window has the wrong width: it must include both edges, j from "
                         "max(0, i - w) to min(S - 1, i + w) inclusive (2w + 1 keys away from the "
                         "sequence ends).", evidence)
    if fact("one-sided (causal) window"):
        return Directive("numeric:one-sided",
                         "The window is two-sided: include keys after the query (j up to "
                         "min(S - 1, i + w)) as well as before it (j down to max(0, i - w)).",
                         evidence)
    if fact("the dilation is ignored"):
        return Directive("numeric:dilation",
                         "The dilation is ignored. Tap k of output t reads "
                         "x[c, t*stride + k*dilation], so the slice for tap k starts at "
                         "t0*stride + k*dilation.", evidence)
    if fact("copied without transposing"):
        return Directive("numeric:not-transposed",
                         "Wrong outputs hold x[r, c] at out[r, c]: the data was copied, not "
                         "transposed. Write row i of each input tile into column i of the "
                         "output: out[j0:j1, i] = x[i, j0:j1].", evidence)
    off = next((f for f in facts if "instead of x[c, r]" in f), None)
    if off:
        return Directive("numeric:tile-origin",
                         f"Wrong outputs are read from the wrong place: {off.split('; ')[0]}. "
                         f"Inside a tile, add the tile origin to every index: the input row is "
                         f"i0 + r and the output row is j0 + c.", evidence)
    lv = rep.level
    for ax, name in enumerate(lv.out_axes):
        T = lv.out_tile[ax]
        if fact(f"only in the last partial {name} tile"):
            return Directive(f"numeric:partial-{name}",
                             f"Only the last, partial {T}-wide tile along the output's {name} "
                             f"axis is wrong (missing or mis-sized). Loop the tile start up to "
                             f"the full size with end = min(start + {T}, size), and use the "
                             f"tile's real size end - start (never a fixed {T}) in every inner "
                             f"loop and slice.", evidence)
    if pt and pt[0] in TAG_FIX:
        return Directive(f"numeric:tag:{pt[0]}", TAG_FIX[pt[0]], evidence)
    if const is not None:
        what = "never computed or written" if float(const) == 0 else "left at an initial value"
        return Directive("numeric:constant",
                         f"Wrong outputs are all exactly {const}, i.e. {what}. Make sure every "
                         f"tile, including partial ones, is computed and written, and that "
                         f"accumulators start from the identity of the operation.", evidence)

    if pt:
        tag = pt[0]
        meaning = TAG_MEANING.get(tag, tag)
        return Directive(f"numeric:tag:{tag}",
                         f"The kernel is right except when {meaning}. Find the assumption in the "
                         f"code that these inputs break and remove it.", evidence)
    return Directive("numeric:other",
                     "The results are wrong. Re-derive the formula from the task statement and "
                     "check each step against it.", evidence)
