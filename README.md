# kernel-agent

An agent that writes tiled NumPy kernels (Stage A) for a 10-level op ladder, with a model that sees
at most 8192 tokens per call. Rules live in the verifier, not the prompt; the agent turns each
verdict into one concrete repair instruction.

## Layout

| path | what |
|---|---|
| `kagent/rules.py` | banned constructs and the repair instruction for each violation kind |
| `kagent/static.py` | AST rule check (catches violations on unexecuted paths) |
| `kagent/guard.py` | runtime tile guard: numpy proxy + `ndarray` subclass that records violations by line |
| `kagent/levels.py` | per-level reference, error scale, eval cases (hostile shapes and values) |
| `kagent/harness.py` | verify a kernel: rules + numerics + distilled failure description |
| `kagent/client.py` | model client: exact token counts, budget enforcement, finish_reason checks, cache |
| `kernels/hand/` | hand-written correct kernels (must pass) |
| `kernels/broken/` | deliberately broken kernels (must fail, with a useful message) |
| `tests/` | harness self-tests |
| `notes/notes.tex` | design and experiment log |

## Usage

```bash
uv sync
uv run pytest -q                                              # harness self-test
uv run python -m kagent.harness 2 kernels/hand/l2_row_sum.py -v --fixes
cd notes && latexmk -pdf -outdir=build notes.tex              # build the notes
```

Tolerance: `|got - ref| <= 1e-5 * scale`, float64 reference on float32 inputs, per-level
condition-aware scale; reasoning in `kagent/harness.py` and `notes/notes.tex`.

## Model

Qwen3-8B on vLLM in the seat pod (`KERNEL_AGENT_BASE_URL`, `KERNEL_AGENT_MODEL` are set there).
Context 8192 is input + output. Port-forwarding is not permitted on the cluster, so agent runs
happen inside the pod.
