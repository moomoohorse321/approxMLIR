#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


_EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_ROOT))

from kernels.quant_kernels import (  # noqa: E402
    _quantize_a_i8_per_row_kernel,
    approx_int8_dequant_matmul_kernel_1,
    approx_int8_dual_dequant_matmul_kernel_1,
    int8_dequant_matmul_kernel,
)


def _dump_kernel(handle, prefix: Path) -> dict[str, object]:
    asm = handle.asm
    out: dict[str, object] = {"prefix": str(prefix)}
    for key in ("ttir", "ttgir", "ptx"):
        text = asm.get(key)
        if text:
            path = prefix.with_suffix(f".{key}")
            path.write_text(text)
            out[f"{key}_path"] = str(path)
            out[f"{key}_bytes"] = len(text.encode())
    ptx = asm.get("ptx", "")
    ttgir = asm.get("ttgir", "")
    out["uses_int8_mma"] = "mma.sync" in ptx and ".s8.s8." in ptx
    out["ttgir_has_tt_dot"] = "tt.dot" in ttgir
    return out


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    out_dir = Path(os.environ.get("OUT_DIR", "/tmp/approx_quant_mlir_validation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    a = torch.randn((1, 2048), device=device, dtype=dtype)
    q = torch.empty((1, 2048), device=device, dtype=torch.int8)
    a_scale = torch.empty((1,), device=device, dtype=torch.float32)
    actq_handle = _quantize_a_i8_per_row_kernel[(1,)](
        a, q, a_scale,
        1, 2048,
        a.stride(0), a.stride(1),
        q.stride(0), q.stride(1),
        BLOCK_K=1024,
    )

    a_q = torch.randint(-127, 128, (1, 2048), device=device, dtype=torch.int8)
    b_q = torch.randint(-127, 128, (2048, 6144), device=device, dtype=torch.int8)
    b_scale = torch.ones((6144,), device=device, dtype=torch.float32)
    c = torch.empty((1, 6144), device=device, dtype=dtype)
    mm_handle = int8_dequant_matmul_kernel[(1, 96)](
        a_q, b_q, a_scale, b_scale, c,
        1, 6144, 2048,
        a_q.stride(0), a_q.stride(1),
        b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    approx_mm_handle = approx_int8_dequant_matmul_kernel_1[(1, 96)](
        a_q, b_q, a_scale, b_scale, c,
        1, 6144, 2048,
        a_q.stride(0), a_q.stride(1),
        b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )

    c2 = torch.empty_like(c)
    dual_handle = approx_int8_dual_dequant_matmul_kernel_1[(1, 96)](
        a_q, b_q, b_q, a_scale, b_scale, b_scale, c, c2,
        1, 6144, 2048,
        a_q.stride(0), a_q.stride(1),
        b_q.stride(0), b_q.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )

    summary = {
        "activation_quant_kernel": _dump_kernel(actq_handle, out_dir / "activation_quant"),
        "int8_dequant_matmul_kernel": _dump_kernel(mm_handle, out_dir / "int8_dequant_matmul"),
        "approx_int8_dequant_matmul_kernel_1": _dump_kernel(
            approx_mm_handle, out_dir / "approx_int8_dequant_matmul"
        ),
        "approx_int8_dual_dequant_matmul_kernel_1": _dump_kernel(
            dual_handle, out_dir / "approx_int8_dual_dequant_matmul"
        ),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote validation artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
