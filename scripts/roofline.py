"""Roofline time model shared by the optimiser and the analysis.

t = max(bytes moved to/from HBM, FLOPs / RIDGE), in units of HBM bytes. RIDGE is the Trainium
figure nkibench uses (222 FLOP/byte, published for bf16, indicative for float32). time_x = t / t_min,
where t_min uses the op's minimal traffic (each input read once, output written once) and minimal
FLOPs: 1.00 means at the roofline.
"""
RIDGE = 222.0


def metrics(rep):
    c = rep.cost
    moved = c["hbm_read"] + c["hbm_write"]
    t = max(moved, c["flops"] / RIDGE)
    t_min = max(c["min_read"] + c["min_write"], c["min_flops"] / RIDGE)
    return dict(flops_x=c["flops"] / c["min_flops"] if c["min_flops"] else None,
                reads_x=c["hbm_read"] / c["min_read"], writes_x=c["hbm_write"] / c["min_write"],
                intensity=c["flops"] / moved if moved else 0.0, time_x=t / t_min,
                bound="compute" if c["flops"] / RIDGE > moved else "memory",
                ops=c["ops"], elems_per_op=c["flops"] / max(c["ops"], 1), case=c["case"])
