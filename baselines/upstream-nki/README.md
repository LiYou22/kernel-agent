# upstream-nki

Runs of the upstream NKI kernel agent (`trainium-agent-labs/projects/02-kernel-agent`, commit
`8f1ca41`) on the seat pod, plus two prompt/feedback experiments layered on it.

| file | what |
|---|---|
| `baseline.log` | unchanged `agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5`, Qwen3-8B |
| `attempts-baseline.jsonl` | every attempt of that run: level, round, reward, code, feedback (360 rows) |
| `agent_feedback_v1.py` | sidecar: one targeted hint (invented `nisa.multiply` -> `nisa.tensor_scalar`) |
| `agent_feedback_v2.py` | sidecar: shape card, rewrite-not-patch repair prompt, per-level direction |
| `FEEDBACK-EXPERIMENTS.md` | what the baseline showed, what each version changes, solve rates |

The sidecars import the pod's `agent.py` and rebind `enrich` / `first_prompt` / `repair_prompt`;
nothing upstream is edited. Both have a `--check-feedback` mode that runs without a model.
