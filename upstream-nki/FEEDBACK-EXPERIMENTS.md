# Feedback experiments on the kernel agent (Qwen3-8B, local vLLM)

Two sidecar scripts that change only the prompt and feedback text of `agent.py`, so the effect
of each wording change can be measured against the unchanged baseline. Neither edits `agent.py`;
each imports it and rebinds `enrich`, `first_prompt` and `repair_prompt` before calling `main()`.

    python agent_feedback_v1.py --check-feedback     # routing self-check, no model
    python agent_feedback_v2.py --check-feedback     # prints the new prompts, no model
    python agent_feedback_v2.py --all --rounds 8 --samples 4 --context 8192 --repeat 3

## Setup

| | |
|---|---|
| model | Qwen/Qwen3-8B, thinking off, temperature 0.6, top_p 0.95 |
| rounds / samples / context | 8 / 4 / 8192 |
| grading | `nkibench.py` on the CPU simulator, reward = 0.1 parses + 0.2 rules + 0.2 runs + 0.5 correct (per shape) |

## What the baseline log showed

Every stuck level was stuck on the *same* failure for 4 rounds, with the checker's hint already
naming the fix. The model was not failing to read the hint; the prompt made it impossible to act on.

- **Level 3**: output allocated as `lhsT.shape[1:]`, a 1-D `(64,)` tile. The model did not know the
  result is `(M, N)`. It also reused the `(K, N)` operand tile as the PSUM copy target.
- **Level 4**: the kernel was the correct single-tile kernel (passes K=M=128, N=512, so 0.62) and
  never changed. `repair_prompt` says *"change exactly what the checker names and keep everything
  else identical"* while the checker says *"loop over the partition dimension in chunks"*. The two
  instructions contradict, and the model obeyed the first.
- **Level 1**: `nc_matmul` for a pooling reduction, one `dma_copy` per 2x2 window, invented
  `nisa.multiply` / `scalar_mul` / `transpose_moving=`. The reference uses a strided `.ap()` view
  and `nl.sum(axis=[3, 4])`, and nothing in the prompt said how `.ap()` works.

## Changes

**v1** (one targeted hint): when the error is `nki.isa has no attribute multiply|scalar_mul`,
replace the "closest real names" list with the actual call,
`nisa.tensor_scalar(dst=, data=, op0=nl.multiply, operand0=scale)` and how to allocate `dst`.

**v2** (three changes, v1 kept):
1. *Shape card.* Both prompts list every test shape the checker runs, with the output shape.
2. *Restructure instead of patch.* When feedback asks for a loop-structure change (partition
   exceeds 128, K exceeds pmax, out-of-bounds, partial coverage) the repair prompt asks for a
   rewrite of the function body instead of a one-line patch.
3. *Per-level direction*, prose only, no reference code. Level 1: it is a reduction, load the whole
   input once, how `.ap([[stride, count], ...])` works with a worked example on a *different* shape,
   then `nl.sum` and `tensor_scalar`. Level 3: the result is `(M, N)` and needs its own tiles. Level 4:
   which of M, N, K maps to which loop, and that PSUM accumulates across the K loop.

## Results: solve rate per level

<!-- fill from the "over N runs" blocks of baseline.log, feedback-v1.log, feedback-v2.log -->

| level | baseline (5 runs) | v1 (5 runs, level 1 only) | v2 (5 runs) |
|---|---|---|---|
| 1 avg pool | 0/5, reward 0.30 every run | pending | pending |
| 2 transpose | 5/5, round 0 every run | n/a | pending |
| 3 matmul single | 0/5, reward 0.30 every run | n/a | pending |
| 4 matmul tiled | 0/5, reward 0.62 every run | n/a | pending |

Baseline: 360 attempts (5 runs x 8 rounds x 4 samples, minus early stops), zero variance across runs.
Where the 360 attempts died, by level (first error of the first failing shape):

| level | attempts | failure | count |
|---|---|---|---|
| 1 | 160 | invented `nki.isa` name (`multiply`, `scalar_mul`) | 80 |
| 1 | | `dma_copy` element-count mismatch (one window per copy) | 40 |
| 1 | | `nc_matmul(... transpose_moving=)` | 20 |
| 1 | | `nc_matmul` dst in sbuf | 20 |
| 3 | 100 | `dma_copy` element-count mismatch | 39 |
| 3 | | reshape of result into `(1, 64)` | 32 |
| 3 | | 1-D tile (`out = lhsT.shape[1:]`) | 29 |
| 4 | 80 | `dma_copy` dst partition 256 > 128 (single-tile kernel, never tiled) | 72 |
| 4 | | psum `(M,N)` copied into the `(K,M)` operand tile | 8 |

Rewards per run (best, worst, mean) and the raw lists are in the logs; attempts with code and
feedback are in `attempts.jsonl`, `attempts-feedback-v1.jsonl`, `attempts-feedback-v2.jsonl`.

## Caveats

- Level 2 solves on round 0 in the baseline with a kernel that indexes the HBM input element by
  element inside `nl.affine_range`. The simulator accepts it; it was not checked on hardware.
- All numbers are simulator correctness (layer 1). No device timings.
