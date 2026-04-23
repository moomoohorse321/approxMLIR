#!/usr/bin/env python3
"""Probe whether a real decode shape reaches the INT8 fast path."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def _bench_ms(torch_mod, fn, iters: int) -> float:
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    torch_mod.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch_mod.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _first_mma_line(ptx: str) -> str:
    for line in ptx.splitlines():
        if "mma.sync" in line.lower() or "dp4a" in line.lower() or "imma" in line.lower():
            return line.strip()
    return "none"


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
        raise RuntimeError("CUDA is required")

    M = int(os.environ.get("PROBE_M", "1"))
    K = int(os.environ.get("PROBE_K", "2048"))
    N = int(os.environ.get("PROBE_N", "6144"))
    dtype_name = os.environ.get("PROBE_DTYPE", "bf16").lower()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    block_m, block_n, block_k = 64, 64, 32

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    c_exact = torch.empty((M, N), device="cuda", dtype=dtype)
    exact_handle = exact_matmul_kernel[grid](
        a, b, c_exact,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_exact.stride(0), c_exact.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )

    a_q, a_scale = quantize_a_i8_per_row(a)
    b_q, b_scale = quantize_b_i8_per_col(b)
    c_i32 = torch.empty((M, N), device="cuda", dtype=torch.int32)
    fast_handle = int8_matmul_kernel[grid](
        a_q, b_q, c_i32,
        M, N, K,
        a_q.stride(0), a_q.stride(1),
        b_q.stride(0), b_q.stride(1),
        c_i32.stride(0), c_i32.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    c_fast = (c_i32.float() * a_scale[:, None] * b_scale[None, :]).to(dtype)

    fast_max_diff = float((c_fast - c_exact).abs().max().item())
    fast_mean_diff = float((c_fast - c_exact).abs().mean().item())

    def exact_call() -> None:
        exact_matmul_kernel[grid](
            a, b, c_exact,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c_exact.stride(0), c_exact.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )

    def fast_kernel_only() -> None:
        int8_matmul_kernel[grid](
            a_q, b_q, c_i32,
            M, N, K,
            a_q.stride(0), a_q.stride(1),
            b_q.stride(0), b_q.stride(1),
            c_i32.stride(0), c_i32.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )

    def fast_end_to_end() -> None:
        aq, ascale = quantize_a_i8_per_row(a)
        int8_matmul_kernel[grid](
            aq, b_q, c_i32,
            M, N, K,
            aq.stride(0), aq.stride(1),
            b_q.stride(0), b_q.stride(1),
            c_i32.stride(0), c_i32.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )
        _ = (c_i32.float() * ascale[:, None] * b_scale[None, :]).to(dtype)

    exact_ms = _bench_ms(torch, exact_call, iters=200)
    fast_kernel_ms = _bench_ms(torch, fast_kernel_only, iters=200)
    fast_e2e_ms = _bench_ms(torch, fast_end_to_end, iters=200)

    print(f"shape:                     A=({M},{K}) B=({K},{N}) dtype={dtype}")
    print(f"exact latency (ms):        {exact_ms:.5f}")
    print(f"fast int8 kernel (ms):     {fast_kernel_ms:.5f}")
    print(f"fast int8 end2end (ms):    {fast_e2e_ms:.5f}")
    print(f"fast kernel speedup:       {exact_ms / fast_kernel_ms:.3f}x")
    print(f"fast end2end speedup:      {exact_ms / fast_e2e_ms:.3f}x")
    print(f"fast max/mean diff:        {fast_max_diff:.6f} / {fast_mean_diff:.6f}")
    print(f"exact mma:                 {_first_mma_line(exact_handle.asm.get('ptx', ''))}")
    print(f"fast int8 mma:             {_first_mma_line(fast_handle.asm.get('ptx', ''))}")
    print(f"fast path uses s8 mma:     {'s8.s8.s32' in fast_handle.asm.get('ptx', '')}")

    Path("/tmp/stage2a_fastpath_fast.ptx").write_text(fast_handle.asm.get("ptx", ""))
    Path("/tmp/stage2a_fastpath_fast.ttgir").write_text(fast_handle.asm.get("ttgir", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
