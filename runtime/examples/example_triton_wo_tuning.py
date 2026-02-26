#!/usr/bin/env python3
"""Triton counterpart of example_wo_tuning.py using plugin pass manager.

This script demonstrates the Triton path:
1. Define exact + approximate Triton kernels
2. Reuse ApproxMLIR config objects
3. Install TTIR stage hook that injects annotations and runs approx plugin passes
4. Launch kernel through Triton JIT (no approx-opt dependency)
"""

import os

import approx_runtime as ar
import torch
import triton
import triton.language as tl
from triton import knobs


PLUGIN = os.environ.get("TRITON_PASS_PLUGIN_PATH", "")
if not PLUGIN:
    raise RuntimeError(
        "Set TRITON_PASS_PLUGIN_PATH to libApproxTritonPlugin.so before running"
    )

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def approx_add_kernel_1(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    y_approx = y * 0.98
    tl.store(out_ptr + offsets, x + y_approx, mask=mask)

config = {
    "decision_tree": None,
    "safety_contract": None,
    "static_transform": ar.StaticTransform(
        transform_type="func_substitute",
        knob_val=1,
        approx_kernel=approx_add_kernel_1,
    ),
}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton example")

    size = 4096
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    out_exact = torch.empty_like(x)
    out_approx_ref = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)

    knobs.runtime.add_stages_inspection_hook = None
    add_kernel[grid](x, y, out_exact, size, BLOCK_SIZE=256)
    approx_handle = approx_add_kernel_1[grid](x, y, out_approx_ref, size, BLOCK_SIZE=256)
    extra_ttir = [approx_handle.asm["ttir"]] if "ttir" in approx_handle.asm else []

    pipeline = ar.get_pipeline_for_config(config, workload=ar.WorkloadType.TRITON)
    hook = ar.make_triton_stages_hook(
        passes=pipeline,
        plugin_path=PLUGIN,
        stage_name="make_ttir_approx",
        func_name="add_kernel",
        config=config,
        extra_ttir_texts=extra_ttir,
        verbose=os.environ.get("APPROX_TRITON_VERBOSE", "0") == "1",
    )
    knobs.runtime.add_stages_inspection_hook = hook
    handle = add_kernel[grid](x, y, out, size, BLOCK_SIZE=256)
    print("kernel launch ok")
    if "ttir" in handle.asm:
        print("ttir length:", len(handle.asm["ttir"]))
        print("substitution active:", "approx_add_kernel_1" in handle.asm["ttir"])
    diff_exact = torch.max(torch.abs(out - out_exact)).item()
    diff_approx = torch.max(torch.abs(out - out_approx_ref)).item()
    print("max|hooked - exact|:", diff_exact)
    print("max|hooked - approx_ref|:", diff_approx)
    print("sample:", out[:4].cpu())
    knobs.runtime.add_stages_inspection_hook = None


if __name__ == "__main__":
    main()
