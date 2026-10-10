# upstream-nki

Runs of the upstream NKI kernel agent (`trainium-agent-labs/projects/02-kernel-agent`, commit
`8f1ca41`) on the seat pod, plus four prompt/feedback experiments layered on it (v1..v4, each built from the previous one's failures).

| file | what |
|---|---|
| `FEEDBACK-EXPERIMENTS.md` | what the baseline showed, what each version changes, solve rates |
| `agent_feedback_v1.py` | sidecar: one targeted hint (invented `nisa.multiply` -> `nisa.tensor_scalar`) |
| `agent_feedback_v2.py` | sidecar: shape card, rewrite-not-patch repair prompt, per-level direction. **Solved level 4.** |
| `agent_feedback_v3.py` | v2 + `.ap()` stride rule, MemoryRegion rule, nested-loop level-4 text |
| `agent_feedback_v4.py` | v3 + matmul "shape doctor" (names which tensor each element count is) |
| `logs/baseline.log` | unchanged `agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5`, Qwen3-8B |
| `logs/attempts-baseline.jsonl` | every attempt of that run: level, round, reward, code, feedback (360 rows) |
| `logs/feedback-v*-L*.log` | per-version, per-level runs (added as they finish) |
| `02-kernel-agent/` | a full copy of the upstream project with `agent.py` itself reworked: verifier, holdout, memory bank, restructure rounds. See below |

v3 imports v2 and v4 imports v3, so all four files travel together. The sidecars import the pod's `agent.py` and rebind `enrich` / `first_prompt` / `repair_prompt`;
nothing upstream is edited. Both have a `--check-feedback` mode that runs without a model.

## `02-kernel-agent/`: the upstream agent, edited in place

Unlike the sidecars, this one changes `agent.py` rather than wrapping it, so the folder is
self-contained: it carries the upstream files it needs (`nkibench.py`, `kernelbench.py`,
`try_level.py`, the four reference kernels, `CHALLENGE-kernel-agent.md`), all unchanged from commit
`8f1ca41`, next to the changed and new ones.

| file | what |
|---|---|
| `agent.py` | the loop: reply gate, failure classes, holdout and confidence, memory bank, error memory, token split, context profiler, `--plan-tokens`, `--trace`, tiling guide, restructure rounds |
| `verifier.py` | new: proven float32 error bounds, holdout cases, cross-shape diagnosis, one directive per verdict |
| `test_verifier.py` | new: the verifier against the references and deliberately broken kernels |
| `loop_harness.py` | new: the feedback loop against a scripted model, 13 scenarios |
| `gpu_tiling_agent_guide.md`, `nki_kernel_essentials.md` | background the prompts draw on |
| `README.md` | the upstream README plus "How a kernel is verified" |

`python test_verifier.py` and `python loop_harness.py` need the Neuron SDK and no endpoint. Nothing
here has been measured against the live model yet: there are no solve rates for it.
