#!/usr/bin/env python3
"""Sequential exact-vs-approx sweep for diffusion quantization."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from compare_outputs import compare_record_sets


THIS_FILE = Path(__file__).resolve()
APP = THIS_FILE.parent / "app.py"


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in ("0", "false", "False", "")


def _base_args(output_dir: Path, run_label: str) -> list[str]:
    args = [
        sys.executable,
        str(APP),
        "--mode",
        "run",
        "--output-dir",
        str(output_dir),
        "--run-label",
        run_label,
        "--profile",
        os.environ.get("APPROX_DIFFUSION_PROFILE", "latency_memory_bound"),
        "--model-id",
        os.environ.get("APPROX_DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5"),
        "--device",
        os.environ.get("APPROX_DIFFUSION_DEVICE", "cuda"),
        "--precision",
        os.environ.get("APPROX_DIFFUSION_PRECISION", "float16"),
    ]
    input_jsonl = os.environ.get("APPROX_DIFFUSION_INPUT_JSONL")
    if input_jsonl:
        args.extend(["--input-jsonl", input_jsonl])
    limit = os.environ.get("APPROX_DIFFUSION_LIMIT")
    if limit:
        args.extend(["--limit", limit])
    if _bool_env("APPROX_DIFFUSION_ALLOW_DOWNLOAD", False):
        args.append("--allow-download")
    if _bool_env("APPROX_DIFFUSION_ATTENTION_SLICING", True):
        args.append("--enable-attention-slicing")
    if _bool_env("APPROX_DIFFUSION_VAE_SLICING", False):
        args.append("--enable-vae-slicing")
    if _bool_env("APPROX_DIFFUSION_XFORMERS", False):
        args.append("--enable-xformers")
    warmup_runs = os.environ.get("APPROX_DIFFUSION_WARMUP_RUNS")
    if warmup_runs:
        args.extend(["--warmup-runs", warmup_runs])
    repeats = os.environ.get("APPROX_DIFFUSION_MEMORY_BOUND_REPEATS")
    if repeats:
        args.extend(["--memory-bound-repeats", repeats])
    candidates = os.environ.get("APPROX_DIFFUSION_MEMORY_BOUND_CANDIDATES")
    if candidates:
        args.extend(["--memory-bound-candidates", candidates])
    in_features = os.environ.get("APPROX_DIFFUSION_MEMORY_BOUND_IN_FEATURES")
    if in_features:
        args.extend(["--memory-bound-in-features", in_features])
    out_features = os.environ.get("APPROX_DIFFUSION_MEMORY_BOUND_OUT_FEATURES")
    if out_features:
        args.extend(["--memory-bound-out-features", out_features])
    return args


def _run_case(output_dir: Path, name: str, extra_args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        [*_base_args(output_dir, name), *extra_args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    case_dir = output_dir / f"{name}-stdout"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    payload = _extract_last_json(proc.stdout)
    return {
        "name": name,
        "returncode": proc.returncode,
        "stdout_path": str(case_dir / "stdout.log"),
        "summary": payload,
    }


def _extract_last_json(text: str) -> dict[str, Any] | None:
    for match in reversed(list(re.finditer(r"(?m)^\{", text))):
        candidate = text[match.start() :].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_qos(case: dict[str, Any]) -> dict[str, Any]:
    summary = case.get("summary") or {}
    path = summary.get("qos_summary_path")
    return _load_json(Path(path)) if path else {}


def _speedup(exact_qos: dict[str, Any], approx_qos: dict[str, Any], key: str) -> float | None:
    exact = (exact_qos.get("latency_ms") or {}).get(key)
    approx = (approx_qos.get("latency_ms") or {}).get(key)
    if exact and approx:
        return float(exact) / float(approx)
    return None


def _build_approx_args(plan: str, targets: list[str]) -> list[str]:
    default_block_m = "32"
    default_block_n = "64" if plan == "w4a16_awq" else "128"
    default_block_k = "32" if plan == "w4a16_awq" else "64"
    default_group_k = "128" if plan == "w4a16_awq" else "64"
    approx_args = ["--execution-mode", "approx", "--quant-plan", plan]
    for target in targets:
        approx_args.extend(["--quant-target", target])
    approx_args.extend(
        [
            "--quant-block-m",
            os.environ.get(f"APPROX_DIFFUSION_{plan.upper()}_BLOCK_M", os.environ.get("APPROX_DIFFUSION_BLOCK_M", default_block_m)),
            "--quant-block-n",
            os.environ.get(f"APPROX_DIFFUSION_{plan.upper()}_BLOCK_N", os.environ.get("APPROX_DIFFUSION_BLOCK_N", default_block_n)),
            "--quant-block-k",
            os.environ.get(f"APPROX_DIFFUSION_{plan.upper()}_BLOCK_K", os.environ.get("APPROX_DIFFUSION_BLOCK_K", default_block_k)),
            "--quant-group-k",
            os.environ.get(f"APPROX_DIFFUSION_{plan.upper()}_GROUP_K", os.environ.get("APPROX_DIFFUSION_GROUP_K", default_group_k)),
        ]
    )
    if _bool_env("APPROX_DIFFUSION_USE_SUBSTITUTE", True):
        approx_args.append("--approx-use-substitute")
    if _bool_env("APPROX_DIFFUSION_DROP_ORIGINAL_WEIGHTS", True):
        approx_args.append("--drop-original-weights")
    if _bool_env("APPROX_DIFFUSION_STRICT_QUANT", False):
        approx_args.append("--strict-quant")
    plugin = os.environ.get("TRITON_PASS_PLUGIN_PATH")
    if plugin:
        approx_args.extend(["--triton-pass-plugin", plugin])
    if _bool_env("APPROX_DIFFUSION_TRACE_QUANT_EVENTS", False):
        approx_args.append("--trace-quant-events")
    return approx_args


def _frontier_for_case(base_out: Path, exact_case: dict[str, Any], approx_case: dict[str, Any]) -> dict[str, Any] | None:
    if not exact_case.get("summary") or not approx_case.get("summary"):
        return None
    exact_summary = exact_case["summary"]
    approx_summary = approx_case["summary"]
    comparison = compare_record_sets(
        Path(exact_summary["records_path"]),
        Path(approx_summary["records_path"]),
    )
    comparison_path = base_out / f"exact_vs_{approx_case['name']}.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    exact_qos = _case_qos(exact_case)
    approx_qos = _case_qos(approx_case)
    p50_speedup = _speedup(exact_qos, approx_qos, "p50")
    p95_speedup = _speedup(exact_qos, approx_qos, "p95")
    exact_mem = (exact_qos.get("memory") or {}).get("peak_cuda_bytes_max")
    approx_mem = (approx_qos.get("memory") or {}).get("peak_cuda_bytes_max")
    mem_reduction = None
    if exact_mem and approx_mem:
        mem_reduction = (float(exact_mem) - float(approx_mem)) / float(exact_mem)
    accepted = (
        approx_case["returncode"] == 0
        and bool(approx_qos.get("accepted"))
        and bool(comparison.get("accepted"))
        and (
            (p50_speedup is not None and p50_speedup >= 1.05)
            or (p95_speedup is not None and p95_speedup >= 1.05)
        )
    )
    return {
        "accepted": accepted,
        "case": approx_case["name"],
        "comparison_path": str(comparison_path),
        "p50_speedup": p50_speedup,
        "p95_speedup": p95_speedup,
        "peak_memory_reduction": mem_reduction,
        "exact_qos": exact_qos,
        "approx_qos": approx_qos,
        "quality": comparison.get("metrics"),
    }


def main() -> int:
    base_out = Path(os.environ.get("OUT_DIR", "/tmp/approx_diffusion_sweep"))
    base_out.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    exact_case = _run_case(base_out, "exact", ["--execution-mode", "exact"])
    results.append(exact_case)
    print("[diffusion-sweep] exact " + json.dumps(exact_case["summary"], sort_keys=True), flush=True)
    if exact_case["returncode"] != 0:
        (base_out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        return 1

    targets = [
        target.strip()
        for target in os.environ.get("APPROX_DIFFUSION_SWEEP_TARGETS", "memory_bound_conditioner").split(",")
        if target.strip()
    ]
    plans = [
        plan.strip()
        for plan in os.environ.get("APPROX_DIFFUSION_SWEEP_PLANS", "w8a16,w4a16_awq").split(",")
        if plan.strip()
    ]
    frontier_summaries: list[dict[str, Any]] = []
    for plan in plans:
        case_name = f"{plan}_latency_frontier"
        approx_case = _run_case(base_out, case_name, _build_approx_args(plan, targets))
        results.append(approx_case)
        print(f"[diffusion-sweep] {case_name} " + json.dumps(approx_case["summary"], sort_keys=True), flush=True)
        frontier = _frontier_for_case(base_out, exact_case, approx_case)
        if frontier is not None:
            frontier_summaries.append(frontier)
            results.append({"name": f"{case_name}_frontier", "summary": frontier, "returncode": 0 if frontier["accepted"] else 1})
            print(f"[diffusion-sweep] {case_name}_frontier " + json.dumps(frontier, sort_keys=True), flush=True)

    accepted_frontiers = [frontier for frontier in frontier_summaries if frontier["accepted"]]
    selected = max(
        accepted_frontiers,
        key=lambda frontier: frontier.get("p50_speedup") or 0.0,
    ) if accepted_frontiers else None
    final = {
        "accepted": selected is not None,
        "selected_case": selected["case"] if selected else None,
        "frontiers": frontier_summaries,
    }
    results.append({"name": "frontier", "summary": final, "returncode": 0 if selected else 1})
    print("[diffusion-sweep] frontier " + json.dumps(final, sort_keys=True), flush=True)

    (base_out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if final["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
