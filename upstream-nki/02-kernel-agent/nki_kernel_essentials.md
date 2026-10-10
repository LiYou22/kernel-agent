# NKI Essentials: A Practical Guide for Writing Efficient Kernels

A concise reference for building the eight NKI kernels in the provided learning ladder. Based on the [AWS Neuron NKI documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html).

## 1. Programming and memory model

NKI (Neuron Kernel Interface) exposes explicit tile layouts, indexing and on-chip memory management for AWS Neuron hardware.

```text
HBM (large, off-chip memory)
  | nl.load
  v
SBUF (on-chip input/intermediate tiles) <--> compute engines
  |                                         |
  |                                  matmul accumulation
  |                                         v
  +<--------------------------------- PSUM (partial sums)
  |
  | nl.store
  v
HBM (output)
```

**Main rule:** Minimize off-chip transfers; reuse tiles in SBUF and accumulate matmul results in PSUM. Move PSUM results to SBUF as required before writing to HBM.

### Partition and free axes

- **Partition axis (P):** Maps to hardware partitions, commonly limited to 128 partitions per tile.
- **Free axis (F):** Describes elements within a partition; limits depend on the instruction and available memory.
- The axes are **not interchangeable**; layout determines which operations are legal and efficient.

For `nisa.nc_matmul`, the conceptual operation is `C = A.T @ B`, with `A: [K, M]`, `B: [K, N]`, and `C: [M, N]` accumulated in PSUM. On NeuronCore-v2/v3, commonly used instruction tile limits are `K <= 128`, `M <= 128`, and `N <= 512` (instruction-specific, not universal tensor limits).

## 2. Core APIs

| API | Use |
|---|---|
| `nl.load`, `nl.store` | Transfer tiles between HBM and on-chip memory |
| `nl.ndarray`, `nl.zeros` | Allocate tensors/tiles |
| `nl.arange` | Construct explicit tile indices |
| `nl.sum`, `nl.mean`, `nl.max` | Tile reductions |
| `nl.exp` | Exponentiation (e.g., softmax) |
| `nl.matmul`, `nisa.nc_matmul` | Matrix multiplication, with ISA-level control for the latter |
| `nisa.nc_transpose` | Hardware-supported layout transpose |
| `range()` | Loops in current NKI APIs; older examples may use `nl.affine_range`, `nl.sequential_range`, or `nl.static_range` |

Use explicit strides, masks and indexes to handle access patterns, edges and partial tiles.

## 3. Eight-level kernel ladder

| Level | Kernel | Main lesson |
|---|---|---|
| **1** | Average pooling 2D | Construct pooling-window access patterns and reduce with `nl.sum` / `nl.mean`; normalize when summing. |
| **2** | 2D transpose | Cross partition/free layouts with efficient tiled transfers; transfer *count* can dominate cost. |
| **3** | Single-tile matmul | Respect operand layout, tile limits and PSUM-to-SBUF result movement. |
| **4** | Tiled matmul | Iterate over M/N/K tiles, accumulate correctly, and measure redundant HBM reads. |
| **5** | Hoisted-load matmul | Move repeat loads out of inner loops and reuse tiles in SBUF. |
| **6** | M/N-blocked matmul | Explore M/N block sizes to exchange on-chip capacity for greater tile reuse. |
| **7** | Fully blocked matmul | Co-optimize M/N/K blocking, memory residency and accumulation. |
| **8** | Single-head attention | Fuse `QK^T`, stable softmax and multiplication by `V` without writing the score matrix to HBM. |

## 4. Performance priorities

### A. Reduce redundant HBM reads

A straightforward three-loop matmul may load the same A or B tile repeatedly:

```python
# Schematic pseudocode, not a complete NKI kernel
for m in range(M_tiles):
    for n in range(N_tiles):
        for k in range(K_tiles):
            A_tile = load_A(m, k)
            B_tile = load_B(k, n)
            accumulate_matmul(A_tile, B_tile)
```

Hoist repeated loads when legal; then block M and N to reuse loaded tiles across multiple outputs. Pick blocks that fit SBUF and PSUM constraints.

### B. Diagnose arithmetic intensity *before* optimizing

`Arithmetic intensity = FLOPs / HBM bytes transferred`.

Your level-4 description cites **16 MFlops per 160 KB**, approximately **100 FLOPs/byte** (97.7 if KB means KiB), versus a benchmark ridge of **222 FLOPs/byte**. Under that benchmark model, the kernel is **memory-bound**: first reduce data transfer, not arithmetic.

### C. Transfer overhead and spills

- **Transpose:** Many tiny transfers may be expensive even if total bytes moved are modest.
- **Too-small blocks:** Leave reuse opportunities untapped.
- **Too-large blocks:** Exceed useful on-chip capacity and may cause spills or other slowdowns.
- **Test large shapes:** Single-tile benchmarks can conceal redundant traffic that becomes severe with multiple tiles on each axis.

## 5. Attention essentials

Scaled dot-product attention:

```text
Scores = (Q @ K.T) / sqrt(d)
Output = softmax(Scores) @ V
```

For a numerically stable softmax, subtract the maximum score per row before exponentiating:

```text
m_i = max_j(S_ij)
P_ij = exp(S_ij - m_i) / sum_k(exp(S_ik - m_i))
```

**Avoid materializing the full `sequence_length × sequence_length` scores matrix in HBM.** Keep intermediates on-chip where possible; for larger sequences, use tiled/online softmax with correct running normalization. Check the supported reduction axes for each NKI operation.

## 6. Kernel correctness and optimization checklist

- [ ] Match the reference function's shapes, dtype, boundary behavior and numeric tolerances.
- [ ] Assign partition and free axes deliberately.
- [ ] Verify operation-specific tile-size limits.
- [ ] Mask partial/edge tiles and avoid invalid accesses.
- [ ] Keep reusable inputs in SBUF and handle PSUM accumulation/result copies correctly.
- [ ] Estimate FLOPs, HBM bytes and arithmetic intensity.
- [ ] Eliminate redundant HBM loads and excessive tiny transfers.
- [ ] Benchmark every listed shape, especially large multi-tile cases.

## Key takeaway

**Correct layout → legal tiles → correct accumulation → lower HBM traffic → greater on-chip reuse → fused operations.**

The most important optimization principle is **perform more useful computation per byte transferred from HBM**.

## References

- [NKI documentation home](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html)
- [NKI performance guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki_perf_guide.html)
- [`nisa.nc_matmul` reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html)
