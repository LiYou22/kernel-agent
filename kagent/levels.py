"""The operation ladder: reference, error scale, and eval cases per level.

Inputs are float32 (what an accelerator would hold); references run in float64 on those exact
float32 values, so the only error measured is the kernel's own.

Error bound: each level defines `scale`, the magnitude a backward-stable float32 algorithm's error
is proportional to (sum|x| for a row sum, not |sum x|, which can cancel to ~0), and `bound_k`, the
number of roundings on the path to one output. A kernel passes when |got - ref| <= gamma_k * scale
everywhere; see TOLERANCE in harness.py.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .rules import TILE_C, TILE_R

TINY = 1e-30


@dataclass
class Case:
    name: str
    args: tuple                 # positional kernel args: float32 arrays and python scalars
    tags: tuple = ()


@dataclass
class Level:
    num: int
    name: str
    signature: str              # "kernel(x, a, b)"
    task: str                   # one-sentence statement, the only thing attempt 1 needs
    ref: Callable               # float64 args -> float64 output
    scale: Callable             # (float64 args, ref) -> array broadcastable to ref
    bound_k: Callable           # float64 args -> k in the float32 bound gamma_k * scale
    min_flops: Callable         # float64 args -> FLOPs of the direct algorithm, for efficiency
    make_cases: Callable[[bool], list]   # holdout -> cases

    out_tile: tuple = (TILE_R, TILE_C)   # tile size per output axis, for locating errors
    out_axes: tuple = ("row", "col")     # what each output axis means, for error messages
    # (args64, got, ref, bad) -> facts: level-specific hypotheses about a wrong output, e.g. "equals
    # the reference over the last column tile only". Facts are phrased to be shared across cases.
    explain: Callable | None = None
    abs_floor: float = 0.0               # absolute slack added to the bound (e.g. underflow to 0)
    notes: str = ""
    _cache: dict = field(default_factory=dict, repr=False)

    def get_cases(self, holdout=False):
        """Dev cases drive the repair loop. Holdout cases (other shapes, other seed) are never
        shown to the model; they only decide how much to trust a kernel that passed dev."""
        if holdout not in self._cache:
            self._cache[holdout] = self.make_cases(holdout)
        return self._cache[holdout]


def _is_prime(n):
    return n > 2 and all(n % p for p in range(2, int(n ** 0.5) + 1))


def shape_tags(M, N):
    t = []
    if M % TILE_R:
        t.append("partial-row-tile")
    if N % TILE_C:
        t.append("partial-col-tile")
    if M > TILE_R:
        t.append("multi-row-tile")
    if N > TILE_C:
        t.append("multi-col-tile")
    if M == 1 or N == 1:
        t.append("dim-1")
    if _is_prime(M) or _is_prime(N):
        t.append("prime-dim")
    return t


# (M, N): exact tile, tile+1, primes spanning several tiles, dims of exactly 1
SHAPES = [(1, 1), (1, 700), (7, 13), (128, 512), (129, 513), (131, 1031), (300, 1), (37, 2053)]
HOSTILE_SHAPES = [(131, 1031), (129, 513), (1, 700), (300, 1)]


def values(kind, M, N, rng):
    """Value generators. Each name is a tag; failures that concentrate on one are diagnostic."""
    if kind == "normal":
        x = rng.standard_normal((M, N))
    elif kind == "large-magnitude":
        x = rng.standard_normal((M, N)) * 1e4
    elif kind == "all-negative":
        x = -1.0 - np.abs(rng.standard_normal((M, N))) * 5
    elif kind == "huge-negative":
        x = -1e30 * (1.0 + rng.random((M, N)))
    elif kind == "zeros":
        x = np.zeros((M, N))
    elif kind == "constant-rows":
        x = np.repeat(rng.standard_normal((M, 1)) * 3, N, axis=1)
    elif kind == "large-mean":
        x = 1e4 + rng.standard_normal((M, N))
    elif kind == "huge-magnitude":
        x = rng.standard_normal((M, N)) * 1e18
    elif kind == "index-coded":
        x = np.arange(1, M * N + 1, dtype=np.float64).reshape(M, N)   # value encodes its position
    elif kind == "cancelling":
        x = np.where(rng.random((M, N)) < 0.5, 1e6, -1e6) + rng.standard_normal((M, N))
    else:
        raise ValueError(kind)
    return x.astype(np.float32)


def _holdout_shapes(rng, n=6):
    """Random shapes, always including a dim of 1 and a multi-tile prime."""
    shapes = [(int(rng.integers(1, 400)), int(rng.integers(1, 2200))) for _ in range(n - 2)]
    return shapes + [(int(rng.integers(130, 260)), 1), (1, 1543)]


def _cases_2d(value_kinds, extra=lambda x, rng, holdout: (), seed=0, holdout=False):
    """Every shape with normal values, plus each hostile value kind on the awkward shapes."""
    rng = np.random.default_rng(seed + (1000 if holdout else 0))
    cases = []
    if holdout:
        shapes = _holdout_shapes(rng)
        hostile = [shapes[0], shapes[-1], shapes[-2]]
    else:
        shapes, hostile = SHAPES, HOSTILE_SHAPES
    plan = [("normal", s) for s in shapes] + [(k, s) for k in value_kinds for s in hostile]
    for kind, (M, N) in plan:
        x = values(kind, M, N, rng)
        for suffix, rest, tags in extra(x, rng, holdout) or [("", (), ())]:
            cases.append(Case(f"{kind} {M}x{N}{suffix}", (x, *rest),
                              tuple(shape_tags(M, N)) + (kind,) + tuple(tags)))
    return cases


# ---- tile-level hypotheses (explain hooks) -------------------------------------------------------

def _col_tiles(N):
    return [(j0, min(j0 + TILE_C, N)) for j0 in range(0, N, TILE_C)]


def reduction_tile_facts(ref, args, g, bad):
    """Row reduction over several column tiles: does each wrong row hold ONE tile's result?
    (The 'accumulator overwritten per tile' bug, E9.)"""
    x = args[0]
    tiles = _col_tiles(x.shape[1])
    if len(tiles) < 2:
        return []
    rows = np.nonzero(bad)[0]
    for t, (j0, j1) in enumerate(tiles):
        sub = np.asarray(ref(x[:, j0:j1], *args[1:]))
        if np.allclose(g[rows], sub[rows], rtol=1e-5, atol=0):
            which = "last" if t == len(tiles) - 1 else "first" if t == 0 else f"number {t}"
            return [f"each wrong output equals the result over the {which} column tile only"]
    return []


def rowwise_tile_facts(ref_slice, args, g, bad):
    """Row-wise normalisation over several column tiles: was each tile normalised with its own
    statistic (max, sum, mean square) instead of the whole row's?"""
    x = args[0]
    tiles = _col_tiles(x.shape[1])
    if len(tiles) < 2:
        return []
    per_tile = np.concatenate([np.asarray(ref_slice(args, j0, j1)) for j0, j1 in tiles], axis=1)
    if np.allclose(g[bad], per_tile[bad], rtol=1e-4, atol=1e-30):
        return ["each column tile was normalised with a statistic of that tile alone, not of the whole row"]
    return []


def transpose_facts(args, g, bad):
    """Inputs are index-coded (x[i, j] = i*N + j + 1), so a wrong output says where it came from."""
    x = args[0]
    M, N = x.shape
    rr, cc = np.nonzero(bad)
    v = g[rr, cc]
    if not len(v) or not np.all(v == np.round(v)):
        return []
    if np.any(v == 0):
        n0 = int(np.sum(v == 0))
        facts0 = [f"~{n0} of the wrong outputs are 0, i.e. never written"]
        keep = v != 0
        rr, cc, v = rr[keep], cc[keep], v[keep]
        if not len(v):
            return facts0
    else:
        facts0 = []
    if not np.all((v >= 1) & (v <= M * N)):
        return facts0
    src_i, src_j = (v.astype(np.int64) - 1) // N, (v.astype(np.int64) - 1) % N
    facts = []
    k = int(np.argmax(bad[rr, cc])) if len(rr) else 0
    r0, c0 = int(rr[k]), int(cc[k])
    facts.append(f"~e.g. out[{r0}, {c0}] holds x[{src_i[k]}, {src_j[k]}], it should hold x[{c0}, {r0}]")
    if np.all((src_i == rr) & (src_j == cc)):
        facts.append("each wrong out[r, c] holds x[r, c], i.e. it was copied without transposing")
    di, dj = np.unique(src_i - cc), np.unique(src_j - rr)
    if len(di) == 1 and len(dj) == 1 and (di[0], dj[0]) != (0, 0):
        facts.append(f"each wrong out[r, c] holds x[c + {di[0]}, r + {dj[0]}] instead of x[c, r] "
                     f"(an offset in the tile origin)")
    return facts0 + facts


def matmul_facts(args, g, bad):
    """Is the product missing the last partial K tile (or using only the first)?"""
    a, b = args
    K = a.shape[1]
    for step in (128, 512):
        cut = (K // step) * step
        if 0 < cut < K and np.allclose(g[bad], (a[:, :cut] @ b[:cut])[bad], rtol=1e-4, atol=1e-30):
            return [f"each wrong output is the product over K rounded down to a multiple of {step}: "
                    f"the last partial K tile is missing"]
        if step < K and np.allclose(g[bad], (a[:, :step] @ b[:step])[bad], rtol=1e-4, atol=1e-30):
            return [f"each wrong output is the product over the first {step} values of K only"]
    return []


# ---- level 1: y = relu(a*x + b) ---------------------------------------------------------------

def _l1_extra(x, rng, holdout):
    if holdout:
        a, b = float(rng.uniform(0.1, 4)), float(rng.uniform(-2, 2))
        return [(f" a={a:.3g} b={b:.3g}", (a, b), ()),
                (f" a={-a:.3g} b={-b:.3g}", (-a, -b), ("negative-scale",))]
    return [(" a=1.7 b=-0.25", (1.7, -0.25), ()),
            (" a=-3 b=2", (-3.0, 2.0), ("negative-scale",))]


L1 = Level(
    1, "relu_affine", "kernel(x, a, b)",
    "Given a 2-D float32 array x and python floats a, b, return relu(a*x + b) elementwise, "
    "same shape as x.",
    ref=lambda x, a, b: np.maximum(a * x + b, 0.0),
    scale=lambda args, r: np.abs(args[1]) * np.abs(args[0]) + abs(args[2]),
    bound_k=lambda args: 3,
    min_flops=lambda args: 3 * args[0].size,     # mul, add, max
    make_cases=lambda h: _cases_2d(["large-magnitude", "zeros", "all-negative"], _l1_extra,
                                   seed=1, holdout=h),
)

# ---- level 2: row-wise sum ----------------------------------------------------------------------

L2 = Level(
    2, "row_sum", "kernel(x)",
    "Given a 2-D float32 array x of shape (M, N), return the 1-D array of length M whose i-th "
    "entry is the sum of row i.",
    ref=lambda x: x.sum(axis=1),
    scale=lambda args, r: np.abs(args[0]).sum(axis=1),
    bound_k=lambda args: args[0].shape[1],
    min_flops=lambda args: args[0].size,
    make_cases=lambda h: _cases_2d(["all-negative", "constant-rows", "large-mean", "cancelling",
                                    "zeros"], seed=2, holdout=h),
    out_tile=(TILE_R,),
    out_axes=("row",),
    explain=lambda args, g, r, bad: reduction_tile_facts(L2.ref, args, g, bad),
)

# ---- level 3: row-wise max ----------------------------------------------------------------------

L3 = Level(
    3, "row_max", "kernel(x)",
    "Given a 2-D float32 array x of shape (M, N), return the 1-D array of length M whose i-th "
    "entry is the maximum of row i.",
    ref=lambda x: x.max(axis=1),
    scale=lambda args, r: np.abs(args[0]).max(axis=1),
    bound_k=lambda args: 0,
    min_flops=lambda args: args[0].size,
    make_cases=lambda h: _cases_2d(["all-negative", "huge-negative", "constant-rows", "zeros"],
                                   seed=3, holdout=h),
    out_tile=(TILE_R,),
    out_axes=("row",),
    explain=lambda args, g, r, bad: reduction_tile_facts(L3.ref, args, g, bad),
    notes="max is exact in floating point, so any error at all is a bug.",
)

# ---- level 4: RMSNorm ---------------------------------------------------------------------------

def _rmsnorm(x, g, eps):
    return x / np.sqrt((x * x).mean(axis=1, keepdims=True) + eps) * g


def _l4_extra(x, rng, holdout):
    g = rng.uniform(0.5, 1.5, x.shape[1]).astype(np.float32)
    return [(" eps=1e-6", (g, 1e-6), ())]


L4 = Level(
    4, "rmsnorm", "kernel(x, g, eps)",
    "Given a 2-D float32 array x of shape (M, N), a 1-D float32 array g of length N and a python "
    "float eps, return the (M, N) array y[i, j] = x[i, j] / sqrt(mean(x[i, :]**2) + eps) * g[j] "
    "(RMSNorm over the last axis).",
    ref=_rmsnorm,
    # mean square: N squares + N-1 adds + divide; +eps, sqrt halves it; divide, multiply by g
    scale=lambda args, r: np.abs(r),
    bound_k=lambda args: args[0].shape[1] + 4,
    min_flops=lambda args: 4 * args[0].size,
    make_cases=lambda h: _cases_2d(["huge-magnitude", "large-mean", "constant-rows", "zeros"],
                                   _l4_extra, seed=4, holdout=h),
    explain=lambda args, g, r, bad: rowwise_tile_facts(
        lambda a, j0, j1: _rmsnorm(a[0][:, j0:j1], a[1][j0:j1], a[2]), args, g, bad),
)

# ---- level 5: softmax ---------------------------------------------------------------------------

def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


L5 = Level(
    5, "softmax", "kernel(x)",
    "Given a 2-D float32 array x of shape (M, N), return a numerically stable softmax over the "
    "last axis: y[i, j] = exp(x[i, j]) / sum_k exp(x[i, k]), same shape as x.",
    ref=_softmax,
    # exp(x - m) carries the rounding of x - m (relative u*|x - m|); the row sum adds gamma_N;
    # outputs below the smallest normal may flush to 0
    scale=lambda args, r: r * (1 + np.abs(args[0] - args[0].max(axis=1, keepdims=True))),
    bound_k=lambda args: args[0].shape[1] + 4,
    min_flops=lambda args: 5 * args[0].size,
    make_cases=lambda h: _cases_2d(["large-magnitude", "large-mean", "huge-negative",
                                    "constant-rows"], seed=5, holdout=h),
    explain=lambda args, g, r, bad: rowwise_tile_facts(
        lambda a, j0, j1: _softmax(a[0][:, j0:j1]), args, g, bad),
    abs_floor=2.0 ** -126,
)

# ---- level 6: transpose -------------------------------------------------------------------------

def _cases_transpose(holdout):
    rng = np.random.default_rng(6 + (1000 if holdout else 0))
    shapes = _holdout_shapes(rng) if holdout else SHAPES + [(512, 128), (130, 640)]
    cases = []
    for M, N in shapes:
        cases.append(Case(f"index-coded {M}x{N}", (values("index-coded", M, N, rng),),
                          tuple(shape_tags(M, N)) + ("index-coded",)))
    for M, N in (shapes[:2] if holdout else HOSTILE_SHAPES[:2]):
        cases.append(Case(f"normal {M}x{N}", (values("normal", M, N, rng),),
                          tuple(shape_tags(M, N)) + ("normal",)))
    return cases


L6 = Level(
    6, "transpose", "kernel(x)",
    "Given a 2-D float32 array x of shape (M, N), return its transpose: the (N, M) array y with "
    "y[j, i] = x[i, j].",
    ref=lambda x: x.T.copy(),
    scale=lambda args, r: np.abs(r),
    bound_k=lambda args: 0,
    min_flops=lambda args: 0,
    make_cases=_cases_transpose,
    out_tile=(TILE_C, TILE_R),          # output rows are input columns
    explain=lambda args, g, r, bad: transpose_facts(args, g, bad),
    notes="pure data movement: exact.",
)

# ---- level 7: matmul ----------------------------------------------------------------------------

MM_SHAPES = [(1, 1, 1), (1, 7, 1), (7, 1, 13), (128, 128, 512), (129, 131, 513), (37, 521, 61),
             (64, 1031, 5), (130, 256, 1)]
MM_HOSTILE = [(129, 131, 513), (37, 521, 61)]


def mm_tags(M, K, N):
    t = shape_tags(M, N)
    if K % TILE_R:
        t.append("partial-k-tile")
    if K > TILE_R:
        t.append("multi-k-tile")
    if K == 1:
        t.append("k-1")
    return t


def _cases_matmul(holdout):
    rng = np.random.default_rng(7 + (1000 if holdout else 0))
    if holdout:
        shapes = [(int(rng.integers(1, 200)), int(rng.integers(1, 700)), int(rng.integers(1, 600)))
                  for _ in range(4)] + [(1, 389, 1), (150, 1, 70)]
        hostile = shapes[:2]
    else:
        shapes, hostile = MM_SHAPES, MM_HOSTILE
    cases = []
    plan = [("normal", s) for s in shapes] + [(k, s) for k in ("large-magnitude", "cancelling")
                                              for s in hostile]
    for kind, (M, K, N) in plan:
        a, b = values(kind, M, K, rng), values(kind, K, N, rng)
        cases.append(Case(f"{kind} {M}x{K} @ {K}x{N}", (a, b), tuple(mm_tags(M, K, N)) + (kind,)))
    return cases


L7 = Level(
    7, "matmul", "kernel(a, b)",
    "Given float32 arrays a of shape (M, K) and b of shape (K, N), return their matrix product, "
    "the (M, N) array c[i, j] = sum_k a[i, k] * b[k, j].",
    ref=lambda a, b: a @ b,
    scale=lambda args, r: np.abs(args[0]) @ np.abs(args[1]),
    bound_k=lambda args: args[0].shape[1] + 1,    # dot product of length K, then the store
    min_flops=lambda args: 2 * args[0].shape[0] * args[0].shape[1] * args[1].shape[1],
    make_cases=_cases_matmul,
    explain=lambda args, g, r, bad: matmul_facts(args, g, bad),
)

LEVELS = {lv.num: lv for lv in [L1, L2, L3, L4, L5, L6, L7]}
