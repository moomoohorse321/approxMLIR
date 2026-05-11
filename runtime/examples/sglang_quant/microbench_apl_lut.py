#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from approx_apl_lut import APL_LAYOUT_NATURAL, apl_lut_dequantize_weight, apl_lut_quantize_weight  # noqa: E402
from approx_kernels import sglang_apl_lut_linear_kernel  # noqa: E402


def _latency_ms(fn, *, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def _summary(values: list[float]) -> dict:
    if not values:
        return {"latencies_ms": [], "median_ms": None, "iqr_ms": None}
    qs = statistics.quantiles(values, n=4) if len(values) >= 4 else [values[0], values[len(values) // 2], values[-1]]
    return {
        "latencies_ms": values,
        "median_ms": statistics.median(values),
        "iqr_ms": qs[2] - qs[0],
    }


def _default_block_k(bits: int) -> int:
    return 128 if int(bits) == 7 else 256


def _shape_from_stats(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    best: tuple[int, int] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") not in ("apply_approx", "prepare_weight"):
            continue
        n = rec.get("N") or rec.get("N_orig") or (rec.get("weight_shape") or [None])[0]
        k = rec.get("K") or rec.get("K_orig") or (rec.get("weight_shape") or [None, None])[1]
        if n and k:
            candidate = (int(n), int(k))
            if best is None or candidate[0] * candidate[1] > best[0] * best[1]:
                best = candidate
    return best


def _bench_case(args, *, bits: int, variant: str, n: int, k: int) -> dict:
    if variant != "natural_tl_dot":
        return {
            "bits": bits,
            "kernel_variant": variant,
            "supported": False,
            "reason": "candidate is tracked in the matrix but not implemented in this Triton path yet",
        }

    import triton

    torch.manual_seed(args.seed)
    weight = torch.randn(n, k, device="cpu", dtype=getattr(torch, args.weight_dtype))
    q_cpu, lut_cpu, meta = apl_lut_quantize_weight(
        weight,
        bits=bits,
        layout_version=APL_LAYOUT_NATURAL,
        quantizer_version=args.quantizer_version,
    )
    x = torch.randn(args.m, k, device="cuda", dtype=getattr(torch, args.input_dtype))
    qweight = q_cpu.cuda()
    lut = lut_cpu.cuda()
    dequant = apl_lut_dequantize_weight(q_cpu, lut_cpu, bits, k_orig=k, n_orig=n).cuda()
    out = torch.empty((args.m, n), device="cuda", dtype=x.dtype)
    block_n = args.block_n
    block_k = args.block_k if args.block_k > 0 else _default_block_k(bits)
    grid = (args.m, triton.cdiv(n, block_n))

    def run_exact():
        return x.float() @ dequant.t().float()

    def run_triton():
        sglang_apl_lut_linear_kernel[grid](
            x,
            qweight,
            lut,
            out,
            args.m,
            n,
            k,
            int(meta["K_padded"]),
            x.stride(0),
            x.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            qweight.stride(2),
            lut.stride(0),
            lut.stride(1),
            out.stride(0),
            out.stride(1),
            BITS=bits,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=args.num_warps,
        )
        return out

    exact_values = _latency_ms(run_exact, warmup=args.warmup, repeat=args.repeat)
    triton_values = _latency_ms(run_triton, warmup=args.warmup, repeat=args.repeat)
    exact = _summary(exact_values)
    triton_rec = _summary(triton_values)
    return {
        "bits": bits,
        "kernel_variant": variant,
        "supported": True,
        "layout_version": meta["layout_version"],
        "quantizer_version": meta["quantizer_version"],
        "source_dtype": meta["source_dtype"],
        "M": args.m,
        "N": n,
        "K": k,
        "K_padded": meta["K_padded"],
        "N_padded": meta["N_padded"],
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": args.num_warps,
        "exact": exact,
        "triton": triton_rec,
        "hard_gate_pass": triton_rec["median_ms"] < exact["median_ms"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode microbench for the SGLang APL LUT kernel")
    parser.add_argument("--stats-jsonl", type=Path, help="quant_stats.jsonl used to infer the largest target shape")
    parser.add_argument("--n", type=int, default=9728)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--kernel-variants",
        nargs="+",
        default=["natural_tl_dot", "cuda_permuted_row_block", "ksplit_decode"],
    )
    parser.add_argument("--quantizer-version", default="row_uniform_lut_v1")
    parser.add_argument("--weight-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--input-dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=0, help="0 selects the tested per-bit default")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this microbench")
    if args.block_k and args.block_k % 32 != 0:
        raise SystemExit("--block-k must be divisible by 32")

    shape = _shape_from_stats(args.stats_jsonl) if args.stats_jsonl else None
    n, k = shape or (args.n, args.k)
    results = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "shape_source": str(args.stats_jsonl) if shape and args.stats_jsonl else "args_or_fallback",
        "results": [
            _bench_case(args, bits=bits, variant=variant, n=n, k=k)
            for bits in args.bits
            for variant in args.kernel_variants
        ],
    }
    text = json.dumps(results, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
