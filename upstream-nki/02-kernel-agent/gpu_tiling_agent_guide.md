# GPU Tiling Optimization Guide

## Core Concept
Tiling divides large computations into blocks that fit in fast on-chip memory (GPU shared memory/registers or NKI SBUF/PSUM). **Goal:** minimize HBM traffic, maximize reuse, and maintain parallelism.

## Matmul Tiling
For `C = A @ B`: `A[M,K]`, `B[K,N]`, `C[M,N]`.
- `BM`: output rows per tile; `BN`: output columns; `BK`: reduction elements per step.

```python
# Conceptual pseudocode; independent output tiles run in parallel.
for m in range(0, M, BM):
    for n in range(0, N, BN):
        acc = zeros((BM, BN), dtype=float32)
        for k in range(0, K, BK):
            a = load(A[m:m+BM, k:k+BK], mask_edges=True)
            b = load(B[k:k+BK, n:n+BN], mask_edges=True)
            acc += a @ b
        store(C[m:m+BM, n:n+BN], acc, mask_edges=True)
```

## Why It Works
Reuse each A element across `BN` output columns and each B element across `BM` output rows. For FP16 inputs, ignoring output traffic and other overhead:
- `FLOPs = 2 * BM * BN * BK`
- `Input bytes = 2 * BK * (BM + BN)`
- `Arithmetic intensity = BM * BN / (BM + BN)` FLOPs/byte.

## Tile Tradeoffs
- Larger `BM`: more B reuse; larger accumulators.
- Larger `BN`: more A reuse; larger accumulators.
- Larger `BK`: fewer K-loop steps; larger input tiles; does **not** inherently increase ideal arithmetic intensity.
- FP16 input tile storage ≈ `2 * BK * (BM + BN)` bytes; double buffering roughly doubles it.
- Respect register/PSUM capacity, shared-memory/SBUF limits, hardware tile limits, occupancy, and edge masking. **Largest is not always fastest.**

## Optimization Rules
1. First establish a correct tiled baseline; test boundary shapes.
2. Measure latency and HBM traffic; identify memory vs. compute bottlenecks.
3. Hoist redundant loads; reuse operand tiles across multiple outputs when on-chip memory permits.
4. Block M/N to increase reuse; reorder output tiles for better cache locality.
5. Tune `BM`, `BN`, `BK` empirically; watch spills and occupancy.
6. Make one hypothesized change at a time; keep only correct, reproducible wins.

## NKI-Specific Rules
- SBUF holds operand/intermediate tiles; PSUM holds matmul accumulations.
- Partition axis is constrained (often ≤128, depending on hardware/instruction).
- `nisa.nc_matmul(A, B)` computes `A.T @ B`; conceptual input tiles: `A[BK,BM]`, `B[BK,BN]`.
- Respect instruction-specific tile limits and layout rules; transfer PSUM results to SBUF before HBM storage.
- Avoid redundant HBM loads, tiny DMA transfers, and SBUF/PSUM capacity overflows.

## Agent Feedback Loop
1. Generate → compile → correctness-test against reference.
2. Profile latency, HBM traffic, and resource usage.
3. Diagnose bottleneck and propose **one** optimization.
4. Rerun correctness and benchmark across all required shapes.
5. Keep if better; otherwise revert. Record result and best-known kernel.

**Principle:** Maximize useful operations per transferred byte without exhausting on-chip resources or reducing parallelism too much.
