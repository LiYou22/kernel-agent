# kernel-agent

An agent that has a small LLM (Qwen3-8B, served by vLLM on AWS Trainium) write tiled NumPy kernels
for a 10-level operator ladder, with at most 8192 tokens per call (input + output). The rules live in
the verifier, not in the prompt: the model writes freely, the harness finds what is wrong, and the
agent translates that verdict into one concrete change. A second stage makes verified kernels cheaper
(FLOPs, HBM traffic, instructions) without making them wrong.

## Reproduce

### 1. Locally, no model (about 2 minutes)

```bash
uv sync
uv run pytest -q                                   # expect: 80 passed
export KERNELBENCH_DIR=/path/to/02-kernel-agent   # the organizers' kernelbench.py (final gate)
uv run python -m kagent.harness 4 kernels/hand/l4_rmsnorm.py          # PASS, within the float32 bound
uv run python -m kagent.harness 3 kernels/broken/l3_overwrite_per_tile.py --fixes   # names the bug
uv run python -m kagent.agent --levels 1 2 --scripted tests/scripts/demo.json --out runs/demo.jsonl
uv run python scripts/plot_tokens.py runs/demo.jsonl                  # input tokens per attempt, by section
uv run python scripts/final_kernels.py                                # FINAL.md: both checkers
```

Without `KERNELBENCH_DIR` a kernel is reported `verified-unjudged`, never `verified`.

### 2. With the model, on a seat pod

Requires the hackathon's EKS cluster, where each seat pod serves Qwen3-8B on vLLM
(`/workspace/serve.sh`).

```bash
# AWS credentials from the workshop channel in this terminal, then:
aws eks update-kubeconfig --name hack-hyd --region ap-south-2
export POD=seat-NNN
kubectl exec $POD -c app -- bash -c 'curl -s localhost:8000/health -o /dev/null -w "%{http_code}\n"'   # 200
scripts/sync.sh                                    # push tracked files to /workspace/stageA
scripts/launch.sh ours 1 2 3 4                     # background; 5 runs per level (REPEAT=n)
scripts/launch.sh naive 1 2 3 4                    # ablation: the checker's report sent back verbatim
kubectl exec $POD -c app -- bash -c 'tail -n 20 /workspace/stageA/runs/*.log'
scripts/fetch.sh                                   # copies the logs to results/$POD/
```

### 3. Analysis

```bash
R=results/seat-130
uv run python scripts/analyze.py ours=$R/ours-L1234-seat-130.jsonl naive=$R/naive-L1234-seat-130.jsonl --recheck
uv run python scripts/taxonomy.py ours=$R/ours-L1234-seat-130.jsonl naive=$R/naive-L1234-seat-130.jsonl --md TAXONOMY.md
uv run python scripts/plot_tokens.py $R/ours-L1234-seat-130.jsonl 4     # one run, all levels
uv run python scripts/reward.py $R/ours-L1234-seat-130.jsonl            # per-run reward
# efficiency stage on the verified kernels (on the pod), then the before/after report:
python3 scripts/optimize.py --from runs/ours-L1234-seat-130.jsonl --rounds 3 --out runs/opt.jsonl
uv run python scripts/opt_report.py $R/opt2-live.jsonl
```

Final-agent logs for all 10 levels are in `results/final-c5e98d5a75/`.
