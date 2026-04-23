#!/usr/bin/env python3
"""Stage 2(a): characterize whether decode matmul shapes are memory-bound.

For each shape (M=1, K, N), report:
- cold latency (L2-flushed) and warm latency
- achieved bandwidth vs peak HBM/DRAM bandwidth
- compute-bound lower bound (FLOPs / peak FLOPs)
- memory-bound lower bound (bytes / peak BW)
- regime classification

Also measures online-activation-quant overhead standalone, since the spec
warns end-to-end latency can lose once that cost is included.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


GPU_SPECS = {
    "NVIDIA GeForce RTX 4060 Laptop GPU": {
        "bw_GBs": 256.0,
        "bf16_tflops": 30.0,
        "int8_tflops": 60.0,
        "l2_MB": 24.0,
    },
}


def _bench_ms(torch, fn, iters: int, flush_l2: bool = False, flush_buf=None) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    times = []
    if flush_l2:
        for _ in range(iters):
            flush_buf.zero_()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return sum(times[: max(1, iters // 2)]) / max(1, iters // 2)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _classify(memb_us: float, compb_us: float, actual_us: float) -> str:
    if compb_us > memb_us * 4:
        return "compute-bound"
    if memb_us > compb_us * 4:
        if actual_us < memb_us * 0.7:
            return "L2-resident (not HBM-bound)"
        return "memory-bound"
    return "mixed"


def main() -> int:
    import torch
    import triton

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kernels.llm_like import exact_matmul_kernel
    from kernels.quant_kernels import (
        int8_matmul_kernel,
        quantize_a_i8_per_row,
        quantize_b_i8_per_col,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device_name = torch.cuda.get_device_name(0)
    specs = GPU_SPECS.get(device_name)
    if specs is None:
        print(f"# warning: unknown GPU {device_name}; roofline lower bounds disabled")
        specs = {"bw_GBs": 0.0, "bf16_tflops": 0.0, "int8_tflops": 0.0, "l2_MB": 0.0}

    dtype = torch.bfloat16
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    flush_bytes = 128 * 1024 * 1024
    flush_buf = torch.empty(flush_bytes // 4, device="cuda", dtype=torch.float32)

    shapes = [
        ("q_proj",         1, 2048, 2048),
        ("k_proj",         1, 1024, 2048),
        ("v_proj",         1, 1024, 2048),
        ("o_proj",         1, 2048, 2048),
        ("in_proj_qkv",    1, 4096, 2048),
        ("gate_proj",      1, 6144, 2048),
        ("up_proj",        1, 6144, 2048),
        ("gate_up_fused",  1, 12288, 2048),
        ("down_proj",      1, 2048, 6144),
        ("spec_badcase",   1, 6144, 2048),
    ]

    rows = []
    for tag, M, N, K in shapes:
        weight_bytes = K * N * 2
        act_bytes = M * K * 2
        out_bytes = M * N * 2
        flops = 2 * M * N * K
        memb_us = (weight_bytes + act_bytes + out_bytes) / (specs["bw_GBs"] * 1e9) * 1e6 if specs["bw_GBs"] else float("nan")
        compb_us = flops / (specs["bf16_tflops"] * 1e12) * 1e6 if specs["bf16_tflops"] else float("nan")

        torch.manual_seed(0)
        a = torch.randn((M, K), device="cuda", dtype=dtype)
        b = torch.randn((K, N), device="cuda", dtype=dtype)
        c = torch.empty((M, N), device="cuda", dtype=dtype)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        def exact():
            exact_matmul_kernel[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )

        exact()
        warm_ms = _bench_ms(torch, exact, iters=200)
        cold_ms = _bench_ms(torch, exact, iters=30, flush_l2=True, flush_buf=flush_buf)

        a_q, a_scale = quantize_a_i8_per_row(a)
        b_q, b_scale = quantize_b_i8_per_col(b)
        c_i32 = torch.empty((M, N), device="cuda", dtype=torch.int32)

        def i8():
            int8_matmul_kernel[grid](
                a_q, b_q, c_i32, M, N, K,
                a_q.stride(0), a_q.stride(1),
                b_q.stride(0), b_q.stride(1),
                c_i32.stride(0), c_i32.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )

        i8()
        i8_warm_ms = _bench_ms(torch, i8, iters=200)
        i8_cold_ms = _bench_ms(torch, i8, iters=30, flush_l2=True, flush_buf=flush_buf)

        def act_quant_only():
            _aq, _as = quantize_a_i8_per_row(a)

        act_quant_only()
        actq_ms = _bench_ms(torch, act_quant_only, iters=200)

        actual_us_warm = warm_ms * 1000
        actual_us_cold = cold_ms * 1000
        achieved_bw_warm = weight_bytes / (warm_ms * 1e6) if warm_ms > 0 else 0.0
        achieved_bw_cold = weight_bytes / (cold_ms * 1e6) if cold_ms > 0 else 0.0
        regime = _classify(memb_us, compb_us, actual_us_cold)

        row = {
            "tag": tag, "M": M, "K": K, "N": N,
            "weight_MB": weight_bytes / 1024 / 1024,
            "fits_in_L2": weight_bytes / 1024 / 1024 < specs["l2_MB"],
            "mem_lb_us": round(memb_us, 2),
            "comp_lb_us": round(compb_us, 2),
            "exact_warm_us": round(warm_ms * 1000, 2),
            "exact_cold_us": round(cold_ms * 1000, 2),
            "achieved_GBs_warm": round(achieved_bw_warm, 1),
            "achieved_GBs_cold": round(achieved_bw_cold, 1),
            "i8_warm_us": round(i8_warm_ms * 1000, 2),
            "i8_cold_us": round(i8_cold_ms * 1000, 2),
            "actq_us": round(actq_ms * 1000, 2),
            "regime": regime,
            "w8a16_lb_us": round((weight_bytes / 2 + act_bytes + out_bytes) / (specs["bw_GBs"] * 1e9) * 1e6, 2) if specs["bw_GBs"] else None,
            "w8a16_ceiling_vs_cold": round(cold_ms * 1000 / ((weight_bytes / 2 + act_bytes + out_bytes) / (specs["bw_GBs"] * 1e9) * 1e6), 2) if specs["bw_GBs"] else None,
        }
        rows.append(row)
        print(
            f"{tag:30s} MxKxN={M}x{K}x{N:<6d} "
            f"w={row['weight_MB']:5.1f}MB "
            f"cold={row['exact_cold_us']:7.2f}us warm={row['exact_warm_us']:7.2f}us "
            f"BW_cold={row['achieved_GBs_cold']:5.0f}GB/s "
            f"i8_cold={row['i8_cold_us']:6.2f}us actq={row['actq_us']:5.2f}us "
            f"-> {regime}"
        )

    out = Path("/tmp/stage2a_regime.json")
    out.write_text(json.dumps({"device": device_name, "specs": specs, "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
