"""Harness self-test: hand-written kernels must pass; each deliberate bug must be caught with a
message that names its cause. If the harness can't catch our bugs it won't catch the model's."""
from pathlib import Path

import pytest

from kagent.harness import verify
from kagent.levels import LEVELS

K = Path(__file__).parent.parent / "kernels"


def run(level, path):
    return verify(LEVELS[level], (K / path).read_text())


@pytest.mark.parametrize("level,path", [(1, "hand/l1_relu_affine.py"), (2, "hand/l2_row_sum.py"),
                                        (3, "hand/l3_row_max.py")])
def test_correct_kernels_pass(level, path):
    rep = run(level, path)
    assert rep.passed, rep.text()
    assert max(r.worst for r in rep.results) < 1e-7  # only output rounding


# (level, file, expected violation kinds, substring the feedback must contain)
BROKEN = [
    (1, "l1_whole_array.py", {"whole-array"}, None),
    (1, "l1_mask.py", {"fancy-index"}, None),
    (1, "l1_skip_partial_col.py", set(), "'partial-col-tile'"),
    (1, "l1_off_by_one_rows.py", set(), "only the last row"),
    (2, "l2_np_sum.py", {"sum-like"}, None),
    (2, "l2_tile_width_assumed.py", set(), "'partial-col-tile'"),
    (2, "l2_float32_accumulator.py", set(), "accumulate in float64"),
    (3, "l3_zero_init.py", set(), "every wrong output is exactly 0"),
    (3, "l3_sentinel_init.py", set(), "every failing case is 'huge-negative'"),
]


@pytest.mark.parametrize("level,path,kinds,needle", BROKEN)
def test_broken_kernels_caught(level, path, kinds, needle):
    rep = run(level, "broken/" + path)
    assert not rep.passed
    assert {v.kind for v in rep.violations} == kinds, rep.text()
    if needle:
        assert needle in rep.text(), rep.text()
    for v in rep.violations:
        assert v.line is not None, v  # every violation must point at a line


def test_static_catches_unreached_code():
    src = ("import numpy as np\n\ndef kernel(x):\n    if x.shape[0] > 10**9:\n"
           "        return np.sum(x, axis=1)\n    return x[:, 0]\n")
    rep = verify(LEVELS[2], src)
    assert any(v.kind == "sum-like" and v.line == 5 for v in rep.violations)


def test_banned_imports_and_syntax():
    assert verify(LEVELS[1], "import scipy\ndef kernel(x, a, b):\n    return x\n").violations[0].kind == "import"
    rep = verify(LEVELS[1], "def kernel(x, a, b)\n    return x\n")
    assert rep.load_error and "syntax" in rep.failure_classes()
