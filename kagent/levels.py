"""The operation ladder: reference, error scale, and eval cases per level.

Inputs are float32 (what an accelerator would hold); references run in float64 on those exact
float32 values, so the only error measured is the kernel's own.

Error scale: each level defines `scale`, the magnitude a backward-stable float32 algorithm's error
is proportional to (e.g. sum|x| for a row sum, not |sum x|, which can cancel to ~0). A kernel
passes when |got - ref| <= RTOL * scale everywhere. See TOLERANCE in harness.py for RTOL.
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
    make_cases: Callable[[], list]
    out_tile: tuple = (TILE_R, TILE_C)   # tile size per output axis, for locating errors
    out_axes: tuple = ("row", "col")     # what each output axis means, for error messages
    notes: str = ""
    cases: list = field(default=None, repr=False)

    def get_cases(self):
        if self.cases is None:
            self.cases = self.make_cases()
        return self.cases


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
    elif kind == "cancelling":
        x = np.where(rng.random((M, N)) < 0.5, 1e6, -1e6) + rng.standard_normal((M, N))
    else:
        raise ValueError(kind)
    return x.astype(np.float32)


def _cases_2d(value_kinds, extra=lambda x, rng: (), seed=0):
    """Every shape with normal values, plus each hostile value kind on the awkward shapes."""
    rng = np.random.default_rng(seed)
    cases = []
    plan = [("normal", s) for s in SHAPES] + [(k, s) for k in value_kinds for s in HOSTILE_SHAPES]
    for kind, (M, N) in plan:
        x = values(kind, M, N, rng)
        for suffix, rest, tags in extra(x, rng) or [("", (), ())]:
            cases.append(Case(f"{kind} {M}x{N}{suffix}", (x, *rest),
                              tuple(shape_tags(M, N)) + (kind,) + tuple(tags)))
    return cases


# ---- level 1: y = relu(a*x + b) ---------------------------------------------------------------

def _l1_extra(x, rng):
    return [(" a=1.7 b=-0.25", (1.7, -0.25), ()),
            (" a=-3 b=2", (-3.0, 2.0), ("negative-scale",))]


L1 = Level(
    1, "relu_affine", "kernel(x, a, b)",
    "Given a 2-D float32 array x and python floats a, b, return relu(a*x + b) elementwise, "
    "same shape as x.",
    ref=lambda x, a, b: np.maximum(a * x + b, 0.0),
    scale=lambda args, r: np.abs(args[1]) * np.abs(args[0]) + abs(args[2]),
    make_cases=lambda: _cases_2d(["large-magnitude", "zeros", "all-negative"], _l1_extra, seed=1),
)

# ---- level 2: row-wise sum ----------------------------------------------------------------------

L2 = Level(
    2, "row_sum", "kernel(x)",
    "Given a 2-D float32 array x of shape (M, N), return the 1-D array of length M whose i-th "
    "entry is the sum of row i.",
    ref=lambda x: x.sum(axis=1),
    scale=lambda args, r: np.abs(args[0]).sum(axis=1),
    make_cases=lambda: _cases_2d(["all-negative", "constant-rows", "large-mean", "cancelling",
                                  "zeros"], seed=2),
    out_tile=(TILE_R,),
    out_axes=("row",),
)

# ---- level 3: row-wise max ----------------------------------------------------------------------

L3 = Level(
    3, "row_max", "kernel(x)",
    "Given a 2-D float32 array x of shape (M, N), return the 1-D array of length M whose i-th "
    "entry is the maximum of row i.",
    ref=lambda x: x.max(axis=1),
    scale=lambda args, r: np.abs(args[0]).max(axis=1),
    make_cases=lambda: _cases_2d(["all-negative", "huge-negative", "constant-rows", "zeros"],
                                 seed=3),
    out_tile=(TILE_R,),
    notes="max is exact in floating point, so any error at all is a bug.",
)

LEVELS = {lv.num: lv for lv in [L1, L2, L3]}
