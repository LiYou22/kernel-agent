#!/usr/bin/env python3
"""
test_verifier.py — the verifier checked against kernels whose verdict is known.

    python test_verifier.py

The shipped references must pass the level's shapes AND the holdout, inside the proven error bound.
Each deliberately broken kernel must fail with the failure class that names its bug. A harness
whose explanation is wrong is as harmful as one whose verdict is wrong: it sends the model to the
wrong fix. Needs the Neuron SDK, because it simulates.
"""

import json
import os
import tempfile

import agent
import verifier

REF = {n: open(f"reference_level{n}.py").read() for n in (1, 2, 3, 4)}


def broken(level, old, new):
    assert old in REF[level], old
    return REF[level].replace(old, new)


PSUM = "        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)"

# (name, level, source, the class it must get, a phrase its feedback must contain)
BUGS = [
    ("PSUM tile reallocated for every K chunk", 4,
     broken(4, PSUM, "        res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)\n"
            + PSUM), "numeric:k-overwrite", "last K tile only"),
    ("only the first N tile is computed", 4,
     broken(4, "for n in nl.affine_range(N // TILE_N):", "for n in nl.affine_range(1):"),
     "numeric:unwritten", "every column tile but the first"),
    ("np.matmul on the host", 4,
     broken(4, "  result = nl.ndarray(", "  junk = np.matmul(lhsT, rhs)\n  result = nl.ndarray("),
     "rule:framework-call", "nisa.nc_matmul"),
    ("pooling sum never scaled", 1,
     broken(1, "operand0=1.0 / (pool_size * pool_size)", "operand0=1.0"),
     "numeric:sum-not-mean", "1.0 / (pool_size * pool_size)"),
    ("tile allocated at the limit, not the size", 1,
     broken(1, "in_tile = nl.ndarray(in_tensor.shape,", "in_tile = nl.ndarray((128, sz_hin, sz_win),"),
     "crash:copy-shape", "MAXIMUM"),
    ("copied without transposing", 2,
     broken(2, "dst=out_tile[:, nl.ds(i_f2*sz_f1+i_f1, 1)]",
            "dst=out_tile[:, nl.ds(i_f1*sz_f2+i_f2, 1)]"),
     "numeric:not-transposed", "f2*F1 + f1"),
    ("bfloat16 tile in a transpose", 2,
     broken(2, "out_tile = nl.ndarray(shape=(sz_p, sz_f2*sz_f1), dtype=in_tensor.dtype,",
            "out_tile = nl.ndarray(shape=(sz_p, sz_f2*sz_f1), dtype=nl.bfloat16,"),
     "numeric:precision", "dtype"),
    ("operands of two dtypes", 3,
     broken(3, "lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)",
            "lhs_tile = nl.ndarray(lhsT.shape, dtype=nl.bfloat16, buffer=nl.sbuf)"),
     "crash:dtype-mismatch", "line 64"),
    ("no @nki.jit", 2, broken(2, "@nki.jit\n", ""), "rule:no-jit", "@nki.jit"),
]


def test_references_pass_dev_and_holdout():
    for level, src in REF.items():
        for holdout in (False, True):
            v = agent.grade(src, level, holdout)
            assert v.solved and v.key == "pass", (level, holdout, v.feedback)
            assert v.bound_use <= 0.5, (level, holdout, v.bound_use)


def test_each_bug_gets_its_class():
    for name, level, src, key, phrase in BUGS:
        v = agent.grade(src, level)
        assert not v.solved, name
        assert v.key == key, (name, v.key, v.feedback)
        assert phrase in v.feedback, (name, v.feedback)


def test_a_kernel_that_fits_only_its_own_shapes_is_not_verified():
    src = broken(4, '  assert K == K_, "lhsT', '  assert K != 384\n  assert K == K_, "lhsT')
    assert agent.grade(src, 4).solved
    assert not agent.grade(src, 4, holdout=True).solved


def test_an_untiled_dimension_is_not_fixed_by_growing_the_tile():
    # N and M are tiled, K is taken whole with `:`. The fix is a K loop, not a 256-row tile.
    one_line = broken(4, """        nisa.dma_copy(dst=rhs_tile, 
                      src=rhs[k * TILE_K:(k + 1) * TILE_K,
                              n * TILE_N:(n + 1) * TILE_N])""",
                      "        nisa.dma_copy(dst=rhs_tile, src=rhs[:, n * TILE_N:(n + 1) * TILE_N])")
    several = broken(4, "src=rhs[k * TILE_K:(k + 1) * TILE_K,", "src=rhs[:,")
    for src, says in ((one_line, "`rhs_tile` is right as it is"), (several, "do not grow the tile")):
        v = agent.grade(src, 4)
        assert v.key == "crash:copy-shape", v.key
        assert says in v.feedback and "ONE psum tile" in v.feedback, v.feedback
        assert "exactly the shape of" not in v.feedback, v.feedback


def test_an_untiled_free_dimension_names_the_loop_to_add():
    src = broken(4, "for m in nl.affine_range(M // TILE_M):", "for m in nl.affine_range(1):")
    src = src.replace("lhsT_tile = nl.ndarray((TILE_K, TILE_M),", "lhsT_tile = nl.ndarray((TILE_K, M),")
    src = src.replace("m * TILE_M:(m + 1) * TILE_M])", ":])", 1)
    v = agent.grade(src, 4)
    assert v.key == "crash:free-limit", (v.key, v.feedback)
    assert "Add a loop over M in chunks of 128" in v.feedback, v.feedback


def test_every_untiled_dimension_is_named_at_once():
    # only N is looped over: K is taken whole on rhs, M is taken whole on lhsT
    src = broken(4, "for m in nl.affine_range(M // TILE_M):", "for m in nl.affine_range(1):")
    src = src.replace("lhsT_tile = nl.ndarray((TILE_K, TILE_M),", "lhsT_tile = nl.ndarray((TILE_K, M),")
    src = src.replace("m * TILE_M:(m + 1) * TILE_M])", ":])", 1)
    src = src.replace("src=rhs[k * TILE_K:(k + 1) * TILE_K,", "src=rhs[:,")
    v = agent.grade(src, 4)
    assert v.restructure, v.key
    assert "K and M are NOT tiled; N is tiled correctly, so keep that loop" in v.feedback, v.feedback
    assert "ONE psum tile before the K loop" in v.feedback and "columns m0:m0 + 128 of lhsT" in v.feedback
    assert "N, in chunks of" not in v.feedback
    # a kernel with one dimension missing is told about that one only
    only_k = agent.grade(broken(4, "src=rhs[k * TILE_K:(k + 1) * TILE_K,", "src=rhs[:,"), 4)
    assert "K is NOT tiled; M and N are tiled correctly, so keep those loops" in only_k.feedback
    assert agent.tiling_probe(None, 1) == {}                    # not a matmul: nothing to probe


def test_traffic_bar_is_its_own_class():
    src = REF[4].replace("def nki_matmul_tiled_", "def nki_matmul_hoist_load_")
    assert agent.grade(src, 5).key == "traffic"


def test_ambiguous_tags_are_all_named():
    v = agent.grade(BUGS[1][2], 4)
    assert "'multi-m-tile' and 'multi-n-tile'" in v.feedback, v.feedback


def test_unusable_replies_never_reach_the_checker():
    assert agent.Reply("```python\nx = 1", finish="length").unusable() == "truncated"
    assert agent.Reply("").unusable() == "empty"
    assert agent.Reply("The fix is easy.").unusable() == "no-code"
    assert agent.Reply("```python\nx = 1\n```").unusable() == ""


def test_context_profile_reports_usage_against_the_limit():
    prof = agent.ContextProfile(4096)
    row = prof.record(2, 0, agent.Reply("x", prompt_tokens=1000, completion_tokens=24),
                      dict(task=900, code=100))
    assert (row["used"], row["share"], row["largest"]) == (1024, 0.25, "task")
    prof.record(2, 1, agent.Reply("x", prompt_tokens=3700, completion_tokens=100),
                dict(code=3000, task=700))
    prof.record(3, 0, agent.Reply("x", prompt_tokens=10, completion_tokens=10), dict(task=10))
    assert "NEAR THE LIMIT" in prof.line(prof.rows[1])
    m = prof.summary(2)
    assert (m["calls"], m["peak"], m["peak_round"], m["peak_largest"]) == (2, 3800, 1, "code")
    assert prof.summary()["calls"] == 3 and prof.summary(2, start=2) is None


def test_the_bank_does_not_store_an_error_twice():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bank.json")
        err = "dma_copy requires src and dst to have the same number of elements"
        add = lambda **kw: agent.remember(path, 2, 5, **kw)
        assert add(failed=("`a`", f"line 18: {err} [dst (128, 12), src (32, 12)]")) == "failed"
        long = "changed `" + "x" * 150 + "` to `" + "y" * 150 + "` -> cleared: something else"
        assert add(worked=long) == "worked" and add(worked=long) == ""     # cut to 220, still one
        # the same error on another line and other shapes is the same error
        assert add(failed=("`b`", f"line 30: {err} [dst (128, 64), src (8, 35)]")) == ""
        assert add(failed=("`c`", "numeric:k-overwrite: 1 of 4 shapes passed; on X: 9% wrong")) \
            == "failed"
        assert add(failed=("`d`", "numeric:k-overwrite: 2 of 4 shapes passed; on Y: 5% wrong")) == ""
        # a fix for an error supersedes the failed pattern that recorded it ...
        assert add(worked=f"changed `a` to `z` -> cleared: {err} [dst ") == "worked"
        entry = json.load(open(path))["2"]
        assert [f["change"] for f in entry["failed_patterns"]] == ["`c`"]
        # ... and the error is not stored again afterwards
        assert add(failed=("`e`", f"line 9: {err} [dst (1, 2), src (3, 4)]")) == ""
        assert agent.memory_text(entry).count(err) == 1


def test_history_lists_a_repeated_error_once():
    text = agent.history_text([("first version", "line 5: boom [dst (1, 2)]"),
                               ("changed `a` to `b`", "line 7: boom [dst (3, 4)]"),
                               ("added `c`", "the checker's report above")])
    assert text.count("boom") == 1 and "(2 times)" in text and "changed `a` to `b`" in text


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
