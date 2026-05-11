#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROBE_SCRIPT = THIS_FILE.parent / "probe_sglang_triton_dump.py"
ACCURACY_SCRIPT = THIS_FILE.parent / "compare_sglang_logprobs.py"
BUILD_ARTIFACT_SCRIPT = THIS_FILE.parent / "build_rtn_mlp_artifact.py"
MICROBENCH_SCRIPT = THIS_FILE.parent / "microbench_rtn_mlp.py"
DEFAULT_PLUGIN = Path(
    "/storage/yuchen/approx_triton_work/build/approx-triton-plugin-faechlo/lib/libApproxTritonPlugin.so"
)

SPARSITIES = (0.01, 0.02, 0.04, 0.06, 0.08, 0.10)
BLOCKS = (64, 32)
ACCURACY_LIMITS = {
    "teacher_forced_perplexity_ratio": 1.01,
    "top1_agreement_rate": 0.99,
}
MIN_PROMPT_TOKENS = 4096
MIN_ACCURACY_TOKENS = 512
MIN_PREFILL_APPLY_M = 1


def _base_env(out_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OUT_DIR": str(out_dir),
            "MODEL_PATH": os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct"),
            "BATCH_SIZE": os.environ.get("BATCH_SIZE", "1"),
            "MAX_NEW_TOKENS": os.environ.get("MAX_NEW_TOKENS", "4"),
            "LONG_PROMPT_REPEAT": os.environ.get("LONG_PROMPT_REPEAT", "256"),
            "WARMUP_RUNS": os.environ.get("WARMUP_RUNS", "2"),
            "MEASURE_RUNS": os.environ.get("MEASURE_RUNS", "10"),
            "ATTENTION_BACKEND": os.environ.get("ATTENTION_BACKEND", "triton"),
            "SAMPLING_BACKEND": os.environ.get("SAMPLING_BACKEND", "pytorch"),
            "SGLANG_MEM_FRACTION_STATIC": os.environ.get("SGLANG_MEM_FRACTION_STATIC", "0.80"),
            "SGLANG_DISABLE_CUDA_GRAPH": "1",
            "SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH": "1",
            "SGLANG_DISABLE_OVERLAP_SCHEDULE": "1",
            "SGLANG_DISABLE_RADIX_CACHE": "1",
            "APPROX_SGLANG_USE_SUBSTITUTE": "0",
            "APPROX_SGLANG_DECODE_ONLY": "0",
            "REQUIRE_MIN_PROMPT_TOKENS": str(MIN_PROMPT_TOKENS),
            "TRITON_PASS_PLUGIN_PATH": os.environ.get("TRITON_PASS_PLUGIN_PATH", str(DEFAULT_PLUGIN)),
        }
    )
    if "TRITON_PLUGIN_PATHS" not in env:
        env["TRITON_PLUGIN_PATHS"] = env["TRITON_PASS_PLUGIN_PATH"]
    return env


def _parse_probe_stdout(stdout: str) -> dict:
    parsed = {}
    for line in stdout.splitlines():
        if not line.startswith("[sglang-probe] "):
            continue
        try:
            label, payload = line[len("[sglang-probe] "):].split(": ", 1)
            parsed[label] = json.loads(payload)
        except (ValueError, json.JSONDecodeError):
            continue
    return parsed


def _run_probe(base_out: Path, name: str, extra: dict[str, str], run_id: int) -> dict:
    out_dir = base_out / name / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _base_env(out_dir)
    env.update(extra)
    proc = subprocess.run(
        [sys.executable, str(PROBE_SCRIPT)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    rec = {
        "name": name,
        "run_id": run_id,
        "returncode": proc.returncode,
        "out_dir": str(out_dir),
        "env": {
            k: env[k]
            for k in sorted(env)
            if k.startswith("APPROX_SGLANG")
            or k
            in (
                "MODEL_PATH",
                "BATCH_SIZE",
                "MAX_NEW_TOKENS",
                "WARMUP_RUNS",
                "MEASURE_RUNS",
                "CUDA_VISIBLE_DEVICES",
                "REQUIRE_MIN_PROMPT_TOKENS",
            )
        },
    }
    rec.update(_parse_probe_stdout(proc.stdout))
    return rec


def _run_accuracy(base_out: Path, name: str, extra: dict[str, str]) -> dict:
    out_dir = base_out / name / "accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _base_env(out_dir)
    env.update(extra)
    env["OUT_DIR"] = str(out_dir)
    env["MAX_NEW_TOKENS"] = os.environ.get("ACCURACY_MAX_NEW_TOKENS", "256")
    per_prompt_tokens = max(1, int(env["MAX_NEW_TOKENS"]))
    prompt_copies = (MIN_ACCURACY_TOKENS + per_prompt_tokens - 1) // per_prompt_tokens
    env["ACCURACY_PROMPT_COPIES"] = os.environ.get("ACCURACY_PROMPT_COPIES", str(prompt_copies))
    env["MIN_ACCURACY_TARGET_TOKENS"] = str(MIN_ACCURACY_TOKENS)
    env["ACCURACY_OUTPUT_PATH"] = str(out_dir / "accuracy.json")
    proc = subprocess.run(
        [sys.executable, str(ACCURACY_SCRIPT)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    payload = None
    accuracy_path = out_dir / "accuracy.json"
    if accuracy_path.exists():
        payload = json.loads(accuracy_path.read_text(encoding="utf-8"))
    metrics = (payload or {}).get("metrics", {})
    passed = (
        proc.returncode == 0
        and metrics.get("num_scored_tokens", 0) >= MIN_ACCURACY_TOKENS
        and metrics.get("teacher_forced_perplexity_ratio", float("inf"))
        <= ACCURACY_LIMITS["teacher_forced_perplexity_ratio"]
        and metrics.get("top1_agreement_rate", 0.0) >= ACCURACY_LIMITS["top1_agreement_rate"]
    )
    return {
        "name": name,
        "returncode": proc.returncode,
        "out_dir": str(out_dir),
        "passed": passed,
        "payload": payload,
    }


def _build_artifact(base_out: Path, sparsity: float, block_i: int) -> Path:
    out = base_out / "artifacts" / f"rtn_mlp_s{sparsity:.2f}_b{block_i}.json".replace(".", "p")
    if out.exists():
        return out
    cmd = [
        sys.executable,
        str(BUILD_ARTIFACT_SCRIPT),
        "--model",
        os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct"),
        "--output",
        str(out),
        "--sparsity",
        f"{sparsity:.4f}",
        "--block-i",
        str(block_i),
    ]
    layers = os.environ.get("APPROX_SGLANG_LAYERS", "")
    if layers:
        cmd.extend(["--layers", layers])
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (out.with_suffix(".build.log")).write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"artifact build failed for sparsity={sparsity} block_i={block_i}: {proc.stdout}")
    return out


def _candidate_artifacts(base_out: Path) -> list[Path]:
    explicit = os.environ.get("APPROX_SGLANG_ARTIFACTS", "")
    if explicit:
        return [Path(x) for x in explicit.split(":") if x]
    single = os.environ.get("APPROX_SGLANG_PRUNE_ARTIFACT", "")
    if single:
        return [Path(single)]
    return [_build_artifact(base_out, sparsity, block_i) for block_i in BLOCKS for sparsity in SPARSITIES]


def _median(rec: dict) -> float | None:
    return rec.get("latency_summary", {}).get("median")


def _prompt_ok(rec: dict) -> bool:
    return int(rec.get("prompt_tokens", {}).get("min_prompt_tokens", 0)) >= MIN_PROMPT_TOKENS


def _stats_cover_artifact(rec: dict, artifact_path: Path) -> tuple[bool, list[str]]:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    stats = rec.get("pruning_stats", {})
    by_prefix = stats.get("by_prefix", {})
    max_m = stats.get("max_m_by_prefix", {})
    missing = []
    for group, entry in artifact.get("groups", {}).items():
        roles = set(entry.get("roles", []))
        if "gate_up_proj" in roles:
            required_sets = (("gate_up_proj", "down_proj"),)
        else:
            required_sets = (("gate_proj", "up_proj", "down_proj"),)
        covered = False
        local_missing = []
        for required in required_sets:
            local_missing = []
            for role in required:
                prefix = f"{group}.{role}" if group else role
                if by_prefix.get(prefix, 0) <= 0:
                    local_missing.append(prefix)
            if int(max_m.get(prefix, 0)) < MIN_PREFILL_APPLY_M:
                local_missing.append(f"{prefix}:no_prefill_apply")
            if not local_missing:
                covered = True
                break
        if covered:
            continue
        for role in sorted(roles):
            prefix = f"{group}.{role}" if group else role
            if by_prefix.get(prefix, 0) <= 0:
                missing.append(prefix)
            if int(max_m.get(prefix, 0)) < MIN_PREFILL_APPLY_M:
                missing.append(f"{prefix}:no_prefill_apply")
    return not missing and stats.get("apply_approx", 0) > 0, missing


def _case_env(artifact: Path) -> dict[str, str]:
    return {
        "APPROX_SGLANG_PRUNING": "1",
        "APPROX_SGLANG_MODE": "approx",
        "APPROX_SGLANG_TARGET": "all",
        "APPROX_SGLANG_BACKEND": "rtn_mlp_prune",
        "APPROX_SGLANG_PRUNE_BACKEND": "rtn_mlp_prune",
        "APPROX_SGLANG_PRUNE_ARTIFACT": str(artifact),
    }


def _run_microbench(base_out: Path, artifact: Path) -> dict:
    out_dir = base_out / "microbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{artifact.stem}.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.run(
        [
            sys.executable,
            str(MICROBENCH_SCRIPT),
            "--artifact",
            str(artifact),
            "--output",
            str(output),
            "--m",
            str(MIN_PROMPT_TOKENS),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (out_dir / f"{artifact.stem}.stdout.log").write_text(proc.stdout, encoding="utf-8")
    payload = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    return {
        "artifact": str(artifact),
        "returncode": proc.returncode,
        "out_dir": str(out_dir),
        "payload": payload,
        "passed": proc.returncode == 0 and bool((payload or {}).get("all_compact_faster")),
    }


def main() -> int:
    base_out = Path(os.environ.get("OUT_DIR", "/storage/yuchen/approx_sglang_pruning_runs"))
    base_out.mkdir(parents=True, exist_ok=True)

    results = []
    accuracies = []
    summary = {
        "accepted": False,
        "best": None,
        "criteria": {
            "latency": "approx_median <= exact_median * 0.98 and two confirmation runs pass",
            "min_prompt_tokens": MIN_PROMPT_TOKENS,
            "min_accuracy_target_tokens": MIN_ACCURACY_TOKENS,
            **ACCURACY_LIMITS,
        },
    }

    exact_runs = [
        _run_probe(base_out, "exact", {"APPROX_SGLANG_PRUNING": "0", "APPROX_SGLANG_MODE": "exact"}, run_id=i)
        for i in range(2)
    ]
    results.extend(exact_runs)
    exact_medians = [_median(x) for x in exact_runs]
    if any(x["returncode"] != 0 or _median(x) is None or not _prompt_ok(x) for x in exact_runs):
        summary["failed_reason"] = "exact baseline failed, missing latency, or prompt length < 4096"
        (base_out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        (base_out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return 1
    exact_combined = statistics.median(float(x) for x in exact_medians)
    summary["exact_medians"] = exact_medians

    best = None
    for artifact in _candidate_artifacts(base_out):
        name = artifact.stem
        microbench = None
        if os.environ.get("SKIP_RTN_MLP_MICROBENCH", "0") != "1":
            microbench = _run_microbench(base_out, artifact)
            if not microbench["passed"]:
                results.append({"name": f"microbench/{name}", **microbench})
                (base_out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
                continue
        env = _case_env(artifact)
        pruned_runs = [_run_probe(base_out, f"pruned/{name}", env, run_id=i) for i in range(2)]
        results.extend(pruned_runs)
        pruned_medians = [_median(x) for x in pruned_runs]
        coverage = [_stats_cover_artifact(x, artifact) for x in pruned_runs]
        speed_pass = all(
            pruned_medians[i] is not None
            and float(pruned_medians[i]) <= float(exact_medians[i]) * 0.98
            for i in range(2)
        )
        valid_pruned_medians = [float(x) for x in pruned_medians if x is not None]
        combined_pruned = statistics.median(valid_pruned_medians) if valid_pruned_medians else None
        combined_pass = combined_pruned is not None and combined_pruned <= exact_combined * 0.98
        prelim = {
            "name": name,
            "artifact": str(artifact),
            "returncodes": [x["returncode"] for x in pruned_runs],
            "pruned_medians": pruned_medians,
            "speedup_vs_exact": exact_combined / combined_pruned if combined_pruned else None,
            "coverage": [{"passed": ok, "missing": missing} for ok, missing in coverage],
            "speed_pass": speed_pass,
            "combined_pass": combined_pass,
            "microbench": microbench,
        }
        print("[sglang-pruning-sweep] candidate " + json.dumps(prelim, sort_keys=True), flush=True)
        (base_out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        if (
            any(x["returncode"] != 0 or _median(x) is None or not _prompt_ok(x) for x in pruned_runs)
            or not all(ok for ok, _ in coverage)
            or not speed_pass
            or not combined_pass
        ):
            continue
        acc = _run_accuracy(base_out, f"pruned/{name}", env)
        accuracies.append(acc)
        (base_out / "accuracy.json").write_text(json.dumps(accuracies, indent=2, sort_keys=True), encoding="utf-8")
        if acc["passed"] and (best is None or prelim["speedup_vs_exact"] > best["speedup_vs_exact"]):
            best = {**prelim, "accuracy": acc}

    summary["accepted"] = best is not None
    summary["best"] = best
    if best is None:
        summary["failed_reason"] = "no candidate satisfied coverage, strict accuracy, and >=2% confirmed speedup"
    (base_out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    (base_out / "accuracy.json").write_text(json.dumps(accuracies, indent=2, sort_keys=True), encoding="utf-8")
    (base_out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("[sglang-pruning-sweep] summary " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if best is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
