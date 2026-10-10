"""The loop, driven by a scripted model: every path that a live run can take, without a server."""
from pathlib import Path

from kagent.agent import ScriptedModel, solve
from kagent.levels import LEVELS

K = Path(__file__).parent.parent / "kernels"
GOOD = (K / "hand/l1_relu_affine.py").read_text()
# what Qwen3-8B actually returned on the first live call (E6): right numbers, one violation
ASARRAY = GOOD.replace("    M, N = x.shape\n", "    x = np.asarray(x, dtype=np.float32)\n    M, N = x.shape\n")
MASK = (K / "broken/l1_mask.py").read_text()


def block(code):
    return f"Here you go:\n```python\n{code}```\n"


def run(replies, **kw):
    m = ScriptedModel([block(r) if r.startswith("import") else r for r in replies])
    res = solve(LEVELS[1], m, say=lambda *_: None, **kw)
    return res, m.prompts


def test_first_prompt_has_no_rules_list():
    _, prompts = run([GOOD])
    p = prompts[0]
    assert "kernel(x, a, b)" in p and "np.sum" not in p and "fancy" not in p


def test_violation_repaired_in_one_round():
    res, prompts = run([ASARRAY, GOOD])
    assert res.status == "verified" and res.attempts == 2 and res.confidence > 0.9
    assert "Remove `np.asarray`" in prompts[1] and "x = np.asarray" in prompts[1]
    assert "Already tried" not in prompts[1]  # no ledger until a failure class repeats


def test_no_code_never_reaches_checker_and_prompt_changes():
    res, prompts = run(["<TRUNCATED>```python\ndef kernel(x, a, b):\n    M, N", "", GOOD])
    assert res.status == "verified" and res.attempts == 3
    assert prompts[0] != prompts[1]  # a greedy model given the same prompt repeats itself
    assert res.keys[:2] == ["reply:truncated", "reply:empty"]


def test_repeat_brings_ledger_then_stops():
    res, prompts = run([MASK, MASK, MASK, GOOD], patience=3)
    assert res.status == "stuck" and res.attempts == 3 and res.final_key == "rule:fancy-index"
    assert "Already tried" in prompts[2] and "Do something different" in prompts[2]
    assert "repeated an earlier version unchanged" in prompts[2]


def test_passing_dev_but_failing_holdout_is_not_verified():
    # correct on every dev shape, wrong on a width the dev set never uses: overfitting to the eval set
    cheat = GOOD.replace("    return out\n", "    if N == 1543:\n        out[0, 0] = 1e9\n    return out\n")
    res, _ = run([cheat])
    assert res.status == "dev-only" and res.confidence < 0.5


def test_naive_mode_sends_the_report_verbatim_and_resends_on_no_code():
    m = ScriptedModel([block(ASARRAY), "", block(GOOD)])
    res = solve(LEVELS[1], m, say=lambda *_: None, mode="naive")
    assert res.status == "verified" and res.attempts == 3
    assert "A checker reports:" in m.prompts[1] and "np.asarray [escape]" in m.prompts[1]
    assert "Remove `np.asarray`" not in m.prompts[1]  # no translated instruction
    assert m.prompts[2] == m.prompts[1]               # no reply gate: the same prompt again
