#!/usr/bin/env python3
"""Stage 2(a): func_substitute demo on the decode attn_qo_proj matmul.

Substitutes ``matmul_kernel`` with ``approx_matmul_kernel_1`` via the hooked
Triton runtime. The approximate path consumes a host-prepacked int8 weight
buffer stored behind the original ``b_ptr`` argument, so this demo exercises
both callee substitution and caller-side data-path switching.

Env: ``TRITON_PASS_PLUGIN_PATH`` (required), ``APPROX_TRITON_VERBOSE`` (opt),
``APPROX_GROUP_SIZES`` (comma-separated K-group sizes, default ``896,128,32``),
``APPROX_TARGET`` (``attn_qo_proj``, ``attn_kv_proj``, ``mlp_gate_up``, ``mlp_down``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _setup_dlopen_flags() -> None:
    try:
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    except AttributeError:
        pass


def _bench_ms(torch, fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    _setup_dlopen_flags()

    plugin = os.environ.get("TRITON_PASS_PLUGIN_PATH", "")
    if not plugin:
        raise RuntimeError(
            "Set TRITON_PASS_PLUGIN_PATH to libApproxTritonPlugin.so before running"
        )
    # Triton C++ reads TRITON_PLUGIN_PATHS once at import; add ours without clobbering others.
    _existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
    _paths = _existing.split(":") if _existing else []
    if plugin not in _paths:
        _paths.insert(0, plugin)
        os.environ["TRITON_PLUGIN_PATHS"] = ":".join(p for p in _paths if p)

    import torch
    import triton
    from triton import knobs

    import approx_runtime as ar

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kernels.llm_like import matmul_kernel
    from kernels.quant_kernels import approx_matmul_kernel_1, pack_b_i8_per_group_fp16_storage

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 2(a) demo")

    target = os.environ.get("APPROX_TARGET", "attn_qo_proj")
    shapes = {
        "attn_qo_proj": (1, 896, 896),
        "attn_kv_proj": (1, 128, 896),
        "mlp_gate_up": (1, 4864, 896),
        "mlp_down": (1, 896, 4864),
    }
    if target not in shapes:
        raise RuntimeError(f"Unknown APPROX_TARGET: {target}")
    M, N, K = shapes[target]
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    group_sizes = [
        int(v) for v in os.environ.get("APPROX_GROUP_SIZES", "896,128,32").split(",") if v.strip()
    ]
    if not group_sizes:
        raise RuntimeError("APPROX_GROUP_SIZES must not be empty")

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c_exact = torch.empty((M, N), device="cuda", dtype=torch.float16)
    c_approx_ref = torch.empty_like(c_exact)
    c_hooked = torch.empty_like(c_exact)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    knobs.runtime.add_stages_inspection_hook = None

    matmul_kernel[grid](
        a, b, c_exact, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_exact.stride(0), c_exact.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    config = {
        "decision_tree": None,
        "safety_contract": None,
        "static_transform": ar.StaticTransform(
            transform_type="func_substitute",
            knob_val=1,
            approx_kernel=approx_matmul_kernel_1,
        ),
    }
    pipeline = ar.get_pipeline_for_config(config, workload=ar.WorkloadType.TRITON)
    exact_ms = None
    exact_weight_bytes = b.numel() * b.element_size()
    print(f"target:                    {target}")
    print(f"exact weight bytes:        {exact_weight_bytes}")

    def exact_call() -> None:
        matmul_kernel[grid](
            a, b, c_exact, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c_exact.stride(0), c_exact.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

    exact_call()
    exact_ms = _bench_ms(torch, exact_call, iters=200)
    print(f"exact latency (ms):        {exact_ms:.5f}")

    for group_size in group_sizes:
        if K % group_size != 0 and group_size != K:
            print(f"group_size_k={group_size}: warning: last group is ragged")
        b_packed = pack_b_i8_per_group_fp16_storage(b, group_size)
        approx_handle = approx_matmul_kernel_1[grid](
            a, b_packed, c_approx_ref, M, N, K,
            a.stride(0), a.stride(1),
            group_size, 0,
            c_approx_ref.stride(0), c_approx_ref.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        extra_ttir = [approx_handle.asm["ttir"]] if "ttir" in approx_handle.asm else []
        if not extra_ttir:
            raise RuntimeError("approx kernel did not expose ttir asm; cannot merge")

        hook = ar.make_triton_stages_hook(
            passes=pipeline,
            plugin_path=plugin,
            stage_name="make_ttir_approx",
            func_name="matmul_kernel",
            config=config,
            extra_ttir_texts=extra_ttir,
            verbose=os.environ.get("APPROX_TRITON_VERBOSE", "0") == "1",
        )

        knobs.runtime.add_stages_inspection_hook = hook
        handle = matmul_kernel[grid](
            a, b_packed, c_hooked, M, N, K,
            a.stride(0), a.stride(1),
            group_size, 0,
            c_hooked.stride(0), c_hooked.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        knobs.runtime.add_stages_inspection_hook = None

        ttir_text = handle.asm.get("ttir", "")
        substitution_active = "approx_matmul_kernel_1" in ttir_text
        diff_exact = torch.max(torch.abs(c_hooked - c_exact)).item()
        diff_approx = torch.max(torch.abs(c_hooked - c_approx_ref)).item()
        norm_exact = torch.max(torch.abs(c_exact)).item() + 1e-8
        rel_err = diff_exact / norm_exact

        for _ in range(10):
            approx_matmul_kernel_1[grid](
                a, b_packed, c_approx_ref, M, N, K,
                a.stride(0), a.stride(1),
                group_size, 0,
                c_approx_ref.stride(0), c_approx_ref.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
        torch.cuda.synchronize()

        def approx_call() -> None:
            approx_matmul_kernel_1[grid](
                a, b_packed, c_approx_ref, M, N, K,
                a.stride(0), a.stride(1),
                group_size, 0,
                c_approx_ref.stride(0), c_approx_ref.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )

        def hooked_call() -> None:
            matmul_kernel[grid](
                a, b_packed, c_hooked, M, N, K,
                a.stride(0), a.stride(1),
                group_size, 0,
                c_hooked.stride(0), c_hooked.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )

        approx_ms = _bench_ms(torch, approx_call, iters=200)
        knobs.runtime.add_stages_inspection_hook = hook
        hooked_ms = _bench_ms(torch, hooked_call, iters=100)
        knobs.runtime.add_stages_inspection_hook = None

        print(f"\ngroup_size_k:             {group_size}")
        print(f"hooked ttir length:        {len(ttir_text)}")
        print(f"substitution active:       {substitution_active}")
        print(f"max|hooked - exact|:       {diff_exact:.4f}")
        print(f"max|hooked - approx_ref|:  {diff_approx:.6f}  (should be ~0)")
        print(f"relative err vs exact:     {rel_err:.4f}")
        print(f"packed weight bytes:       {b_packed.numel() * b_packed.element_size()}")
        print(f"approx latency (ms):       {approx_ms:.5f}")
        print(f"hooked latency (ms):       {hooked_ms:.5f}")
        print(f"approx speedup vs exact:   {exact_ms / approx_ms:.3f}x")
        print(f"hooked speedup vs exact:   {exact_ms / hooked_ms:.3f}x")
        print("note: approx path reuses host-prepacked weights; online kernel only unpacks/dequants")

    if not substitution_active:
        print("WARNING: approx_matmul_kernel_1 not found in hooked TTIR")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
