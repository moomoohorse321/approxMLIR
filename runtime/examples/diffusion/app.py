#!/usr/bin/env python3
"""CLI runner for the exact diffusion application benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = THIS_FILE.parent
APPROXMLIR_ROOT = THIS_FILE.parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from dataset import (
    DEFAULT_LATENCY_MEMORY_BOUND_REQUESTS,
    DEFAULT_SMOKE_REQUESTS,
    dataset_fingerprint,
    load_requests,
)
from exact_backend import (
    DEFAULT_MODEL_ID,
    DiffusersExactBackend,
    ExactBackendConfig,
    inspect_environment,
)
from qos import build_exact_record, compute_image_stats, summarize_qos, utc_now, write_json, write_jsonl


DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exact diffusion application benchmark.")
    parser.add_argument("--mode", choices=["inspect", "run"], default="inspect")
    parser.add_argument("--execution-mode", choices=["exact", "approx"], default="exact")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label", default="exact-laptop")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--profile",
        choices=["laptop_smoke", "laptop_eval", "latency_memory_bound"],
        default="laptop_smoke",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("APPROX_DIFFUSION_MODEL", DEFAULT_MODEL_ID),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--precision",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model downloads. By default the exact benchmark is local-files-only.",
    )
    parser.add_argument("--enable-attention-slicing", action="store_true")
    parser.add_argument("--enable-vae-slicing", action="store_true")
    parser.add_argument("--enable-xformers", action="store_true")
    parser.add_argument(
        "--sequential-cpu-offload",
        action="store_true",
        help="Use only for a separate laptop_offload-style run; it changes latency.",
    )
    parser.add_argument("--warmup-runs", type=int, default=int(os.environ.get("APPROX_DIFFUSION_WARMUP_RUNS", "0")))
    parser.add_argument(
        "--memory-bound-repeats",
        type=int,
        default=int(os.environ.get("APPROX_DIFFUSION_MEMORY_BOUND_REPEATS", "0")),
    )
    parser.add_argument(
        "--memory-bound-candidates",
        type=int,
        default=int(os.environ.get("APPROX_DIFFUSION_MEMORY_BOUND_CANDIDATES", "32")),
    )
    parser.add_argument(
        "--memory-bound-in-features",
        type=int,
        default=int(os.environ.get("APPROX_DIFFUSION_MEMORY_BOUND_IN_FEATURES", "4096")),
    )
    parser.add_argument(
        "--memory-bound-out-features",
        type=int,
        default=int(os.environ.get("APPROX_DIFFUSION_MEMORY_BOUND_OUT_FEATURES", "32768")),
    )
    parser.add_argument("--quant-manifest", type=Path, default=None)
    parser.add_argument(
        "--quant-target",
        action="append",
        default=[],
        help="Substring or regex selecting quantized sites; repeatable. Defaults to unet.",
    )
    parser.add_argument("--quant-inventory-out", type=Path, default=None)
    parser.add_argument("--quant-generated-manifest-out", type=Path, default=None)
    parser.add_argument("--quant-stats-path", type=Path, default=None)
    parser.add_argument(
        "--quant-plan",
        choices=["w8a16", "w4a16_awq"],
        default=os.environ.get("APPROX_DIFFUSION_QUANT_PLAN", "w8a16"),
    )
    parser.add_argument(
        "--quant-block-m",
        type=int,
        default=0,
        help="M tile for W8A16 linear kernels; 0 chooses an input-shape-dependent tile.",
    )
    parser.add_argument("--quant-block-n", type=int, default=128)
    parser.add_argument("--quant-block-k", type=int, default=64)
    parser.add_argument("--quant-group-k", type=int, default=64)
    parser.add_argument("--approx-use-substitute", action="store_true")
    parser.add_argument("--drop-original-weights", action="store_true")
    parser.add_argument("--strict-quant", action="store_true")
    parser.add_argument("--trace-quant-events", action="store_true")
    parser.add_argument(
        "--triton-pass-plugin",
        default=os.environ.get("TRITON_PASS_PLUGIN_PATH", ""),
    )
    args = parser.parse_args()
    if args.input_jsonl is None:
        args.input_jsonl = (
            DEFAULT_LATENCY_MEMORY_BOUND_REQUESTS
            if args.profile == "latency_memory_bound"
            else DEFAULT_SMOKE_REQUESTS
        )
    if args.profile == "latency_memory_bound":
        if args.memory_bound_repeats <= 0:
            args.memory_bound_repeats = 128
        if args.warmup_runs <= 0:
            args.warmup_runs = 1
    return args


def build_run_dir(output_root: Path, run_label: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{run_label}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def inspect_payload(input_jsonl: Path) -> dict[str, Any]:
    return {
        "benchmark": "diffusion_exact_application",
        "default_input_jsonl": str(input_jsonl),
        "latency_memory_bound_input_jsonl": str(DEFAULT_LATENCY_MEMORY_BOUND_REQUESTS),
        "default_output_dir": str(DEFAULT_OUTPUT_ROOT),
        "default_model_id": os.environ.get("APPROX_DIFFUSION_MODEL", DEFAULT_MODEL_ID),
        "default_local_files_only": True,
        "application_not_model_microbenchmark": True,
        "substitution_modes": ["exact", "approx_w8a16_linear", "approx_w4a16_awq_linear"],
        "environment": inspect_environment(),
    }


def manifest_payload(args: argparse.Namespace, run_dir: Path, requests_count: int) -> dict[str, Any]:
    execution_mode = getattr(args, "execution_mode", "exact")
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "benchmark": "diffusion_exact_application",
        "mode": execution_mode,
        "profile": args.profile,
        "approxmlir_root": str(APPROXMLIR_ROOT),
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": dataset_fingerprint(args.input_jsonl),
        "num_requests": requests_count,
        "run_dir": str(run_dir),
        "backend": {
            "model_id": args.model_id,
            "execution_mode": execution_mode,
            "device": args.device,
            "precision": args.precision,
            "local_files_only": not args.allow_download,
            "enable_attention_slicing": args.enable_attention_slicing,
            "enable_vae_slicing": args.enable_vae_slicing,
            "enable_xformers": args.enable_xformers,
            "sequential_cpu_offload": args.sequential_cpu_offload,
            "safety_checker": "disabled_prompt_policy",
            "warmup_runs": args.warmup_runs,
            "memory_bound_conditioner": {
                "enabled": args.memory_bound_repeats > 0,
                "repeats": args.memory_bound_repeats,
                "candidates": args.memory_bound_candidates,
                "in_features": args.memory_bound_in_features,
                "out_features": args.memory_bound_out_features,
            },
        },
        "quantization": {
            "manifest": str(getattr(args, "quant_manifest", None))
            if getattr(args, "quant_manifest", None)
            else None,
            "target": getattr(args, "quant_target", None) or ["unet"],
            "use_substitute": getattr(args, "approx_use_substitute", False),
            "drop_original_weights": getattr(args, "drop_original_weights", False),
            "strict": getattr(args, "strict_quant", False),
            "quant_plan": getattr(args, "quant_plan", "w8a16"),
            "block_m": getattr(args, "quant_block_m", 0),
            "block_n": getattr(args, "quant_block_n", 128),
            "block_k": getattr(args, "quant_block_k", 64),
            "group_k": getattr(args, "quant_group_k", 64),
            "trace_apply_events": getattr(args, "trace_quant_events", False),
            "triton_pass_plugin": getattr(args, "triton_pass_plugin", "") or None,
        }
        if execution_mode == "approx"
        else None,
        "environment": inspect_environment(),
        "qos_contract": {
            "completion_rate": "must equal 1.0",
            "image_valid_rate": "must equal 1.0",
            "safety_pass_rate": "must equal 1.0",
            "optional_metrics_missing_policy": "missing optional metrics are reported unavailable",
        },
    }


def run_exact(
    args: argparse.Namespace,
    *,
    backend: DiffusersExactBackend | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    requests = load_requests(args.input_jsonl, limit=args.limit)
    run_dir = run_dir or build_run_dir(args.output_dir, args.run_label)
    manifest_path = run_dir / "manifest.json"
    manifest = manifest_payload(args, run_dir, len(requests))
    write_json(manifest_path, manifest)

    backend_config = ExactBackendConfig(
        model_id=args.model_id,
        device=args.device,
        precision=args.precision,
        local_files_only=not args.allow_download,
        enable_attention_slicing=args.enable_attention_slicing,
        enable_vae_slicing=args.enable_vae_slicing,
        enable_xformers=args.enable_xformers,
        sequential_cpu_offload=args.sequential_cpu_offload,
        memory_bound_repeats=args.memory_bound_repeats,
        memory_bound_candidates=args.memory_bound_candidates,
        memory_bound_in_features=args.memory_bound_in_features,
        memory_bound_out_features=args.memory_bound_out_features,
    )
    if backend is None and getattr(args, "execution_mode", "exact") == "approx":
        if getattr(args, "triton_pass_plugin", ""):
            os.environ["TRITON_PASS_PLUGIN_PATH"] = args.triton_pass_plugin
            existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
            paths = existing.split(":") if existing else []
            if args.triton_pass_plugin not in paths:
                paths.insert(0, args.triton_pass_plugin)
                os.environ["TRITON_PLUGIN_PATHS"] = ":".join(path for path in paths if path)
        from approx_backend import ApproxBackendConfig, DiffusersApproxBackend

        backend = DiffusersApproxBackend(
            backend_config,
            ApproxBackendConfig(
                manifest_path=getattr(args, "quant_manifest", None),
                target_patterns=tuple(getattr(args, "quant_target", None) or ["unet"]),
                use_substitute=getattr(args, "approx_use_substitute", False),
                strict=getattr(args, "strict_quant", False),
                drop_original_weights=getattr(args, "drop_original_weights", False),
                stats_path=getattr(args, "quant_stats_path", None) or (run_dir / "quant_stats.jsonl"),
                inventory_path=getattr(args, "quant_inventory_out", None) or (run_dir / "site_inventory.json"),
                generated_manifest_path=getattr(args, "quant_generated_manifest_out", None)
                or (run_dir / "quant_manifest.json"),
                quant_plan=getattr(args, "quant_plan", "w8a16"),
                block_m=getattr(args, "quant_block_m", 0),
                block_n=getattr(args, "quant_block_n", 128),
                block_k=getattr(args, "quant_block_k", 64),
                group_k=getattr(args, "quant_group_k", 64),
                trace_apply_events=getattr(args, "trace_quant_events", False),
                plugin_path=getattr(args, "triton_pass_plugin", ""),
            ),
        )
    backend = backend or DiffusersExactBackend(backend_config)

    records: list[dict[str, Any]] = []
    try:
        for warmup_index in range(max(0, int(getattr(args, "warmup_runs", 0)))):
            warmup_request = requests[warmup_index % len(requests)]
            warmup_path = run_dir / "warmup" / f"{warmup_index:04d}-{warmup_request.request_id}.png"
            backend.generate(warmup_request, warmup_path)

        start = time.perf_counter()
        for request in requests:
            image_path = run_dir / "images" / f"{request.request_id}.png"
            errors: list[str] = []
            substitution = None
            application_metrics: dict[str, Any] = {}
            try:
                result = backend.generate(request, image_path)
                image_stats = compute_image_stats(result.image_path)
                latency_ms = result.latency_ms
                memory = result.memory
                final_image_path = result.image_path
                substitution = result.substitution
                application_metrics = result.application_metrics or {}
            except Exception as exc:
                image_stats = compute_image_stats(image_path)
                latency_ms = {
                    "total": 0.0,
                    "text_encoder": 0.0,
                    "denoise": 0.0,
                    "vae_decode": 0.0,
                    "postprocess": 0.0,
                    "scoring": 0.0,
                }
                memory = {"peak_cuda_bytes": None, "model_bytes": None}
                final_image_path = image_path if image_path.exists() else None
                errors = [f"{type(exc).__name__}: {exc}"]
            records.append(
                build_exact_record(
                    request_payload=request.to_json(),
                    image_path=final_image_path,
                    latency_ms=latency_ms,
                    memory=memory,
                    image_stats=image_stats,
                    errors=errors,
                    manifest_path=manifest_path,
                    substitution=substitution,
                    application_metrics=application_metrics,
                )
            )
    finally:
        backend.close()

    qos_summary = summarize_qos(records, profile=args.profile)
    wall_clock_ms = (time.perf_counter() - start) * 1000.0
    run_summary = {
        "benchmark": "diffusion_exact_application",
        "mode": getattr(args, "execution_mode", "exact"),
        "profile": args.profile,
        "run_dir": str(run_dir),
        "records_path": str(run_dir / "records.jsonl"),
        "qos_summary_path": str(run_dir / "qos_summary.json"),
        "num_requests": len(records),
        "accepted": qos_summary["accepted"],
        "wall_clock_ms": wall_clock_ms,
    }
    write_jsonl(run_dir / "records.jsonl", records)
    write_json(run_dir / "qos_summary.json", qos_summary)
    write_json(run_dir / "run_summary.json", run_summary)
    return run_summary


def main() -> None:
    args = parse_args()
    if args.mode == "inspect":
        print(json.dumps(inspect_payload(args.input_jsonl), indent=2, sort_keys=True))
        return
    summary = run_exact(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
