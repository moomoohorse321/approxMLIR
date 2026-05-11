#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F


def _time_cuda(fn, warmup: int, runs: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(float(start.elapsed_time(end)) / 1000.0)
    return out


def _bench_group(entry: dict, hidden_size: int, dtype: torch.dtype, warmup: int, runs: int) -> dict:
    device = torch.device("cuda")
    m = int(entry["M"])
    intermediate = int(entry["intermediate_size"])
    kept = int(entry["kept_intermediate"])
    x = torch.randn((m, hidden_size), device=device, dtype=dtype)
    gate_up = torch.randn((2 * intermediate, hidden_size), device=device, dtype=dtype)
    down = torch.randn((hidden_size, intermediate), device=device, dtype=dtype)
    gate_up_c = gate_up[: 2 * kept].contiguous()
    down_c = down[:, :kept].contiguous()

    def dense():
        y2 = F.linear(x, gate_up)
        y = F.silu(y2[:, :intermediate]) * y2[:, intermediate:]
        return F.linear(y, down)

    def compact():
        y2 = F.linear(x, gate_up_c)
        y = F.silu(y2[:, :kept]) * y2[:, kept:]
        return F.linear(y, down_c)

    dense_times = _time_cuda(dense, warmup, runs)
    compact_times = _time_cuda(compact, warmup, runs)
    dense_median = statistics.median(dense_times)
    compact_median = statistics.median(compact_times)
    return {
        **entry,
        "dense_times": dense_times,
        "compact_times": compact_times,
        "dense_median": dense_median,
        "compact_median": compact_median,
        "speedup_vs_dense": dense_median / compact_median if compact_median else None,
        "compact_faster": compact_median < dense_median,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU microbenchmark for RTN MLP compact path")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this microbenchmark")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    rows = []
    for group, entry in artifact.get("groups", {}).items():
        rows.append(
            {
                "group": group,
                "M": args.m,
                "intermediate_size": int(entry["intermediate_size"]),
                "kept_intermediate": int(entry["kept_intermediate"]),
                "actual_sparsity": float(entry["actual_sparsity"]),
                "block_i": int(entry["block_i"]),
            }
        )
    if args.hidden_size <= 0 and artifact.get("groups"):
        args.hidden_size = int(next(iter(artifact["groups"].values())).get("hidden_size", 2048))
    if args.hidden_size <= 0:
        args.hidden_size = 2048
    results = [_bench_group(row, args.hidden_size, dtype, args.warmup, args.runs) for row in rows]
    payload = {
        "artifact": str(args.artifact),
        "hidden_size": args.hidden_size,
        "M": args.m,
        "dtype": args.dtype,
        "results": results,
        "all_compact_faster": all(row["compact_faster"] for row in results) if results else False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_compact_faster": payload["all_compact_faster"]}, sort_keys=True))
    return 0 if payload["all_compact_faster"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
