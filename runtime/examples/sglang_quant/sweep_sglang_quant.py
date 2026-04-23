#!/usr/bin/env python3
"""Sequential SGLang quantization sweep for the approxMLIR mainline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
PLUGIN = REPO_ROOT / "approxMLIR" / "external-tools" / "approx-triton-plugin" / "build" / "lib" / "libApproxTritonPlugin.so"


def _run_case(base_out: Path, name: str, extra_env: dict[str, str]) -> dict:
    out_dir = base_out / name
    env = os.environ.copy()
    env.update(
        {
            "OUT_DIR": str(out_dir),
            "MODEL_PATH": os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct"),
            "BATCH_SIZE": os.environ.get("BATCH_SIZE", "1"),
            "MAX_NEW_TOKENS": os.environ.get("MAX_NEW_TOKENS", "16"),
            "WARMUP_RUNS": os.environ.get("WARMUP_RUNS", "1"),
            "MEASURE_RUNS": os.environ.get("MEASURE_RUNS", "3"),
            "ATTENTION_BACKEND": os.environ.get("ATTENTION_BACKEND", "triton"),
            "SAMPLING_BACKEND": os.environ.get("SAMPLING_BACKEND", "pytorch"),
            "SGLANG_MEM_FRACTION_STATIC": os.environ.get("SGLANG_MEM_FRACTION_STATIC", "0.45"),
            "SGLANG_DISABLE_CUDA_GRAPH": os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "0"),
            "TRITON_PASS_PLUGIN_PATH": str(PLUGIN),
        }
    )
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, str(THIS_FILE.parent / "probe_sglang_triton_dump.py")],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    rec: dict = {"name": name, "returncode": proc.returncode, "out_dir": str(out_dir)}
    for line in proc.stdout.splitlines():
        if not line.startswith("[sglang-probe] "):
            continue
        try:
            label, payload = line[len("[sglang-probe] "):].split(": ", 1)
            rec[label] = json.loads(payload)
        except ValueError:
            continue
    (out_dir / "stdout.log").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    return rec


def main() -> int:
    base_out = Path(os.environ.get("OUT_DIR", "/tmp/approx_sglang_quant_sweep"))
    base_out.mkdir(parents=True, exist_ok=True)
    cases = [
        ("exact", {"APPROX_SGLANG_QUANT": "0", "APPROX_SGLANG_MODE": "exact"}),
        (
            "gate_up_w8a16",
            {
                "APPROX_SGLANG_QUANT": "1",
                "APPROX_SGLANG_MODE": "approx",
                "APPROX_SGLANG_TARGET": "gate_up_proj",
                "APPROX_SGLANG_BACKEND": "triton_w8a16",
                "APPROX_SGLANG_USE_SUBSTITUTE": "1",
            },
        ),
        (
            "qkv_w8a16",
            {
                "APPROX_SGLANG_QUANT": "1",
                "APPROX_SGLANG_MODE": "approx",
                "APPROX_SGLANG_TARGET": "qkv_proj",
                "APPROX_SGLANG_BACKEND": "triton_w8a16",
                "APPROX_SGLANG_USE_SUBSTITUTE": "1",
            },
        ),
        (
            "gate_up_qkv_w8a16_drop",
            {
                "APPROX_SGLANG_QUANT": "1",
                "APPROX_SGLANG_MODE": "approx",
                "APPROX_SGLANG_TARGET": "gate_up_proj,qkv_proj",
                "APPROX_SGLANG_BACKEND": "triton_w8a16",
                "APPROX_SGLANG_DROP_ORIGINAL_WEIGHT": "1",
                "APPROX_SGLANG_USE_SUBSTITUTE": "1",
            },
        ),
        (
            "gate_up_down_w8a16_drop",
            {
                "APPROX_SGLANG_QUANT": "1",
                "APPROX_SGLANG_MODE": "approx",
                "APPROX_SGLANG_TARGET": "gate_up_proj,down_proj",
                "APPROX_SGLANG_BACKEND": "triton_w8a16",
                "APPROX_SGLANG_DROP_ORIGINAL_WEIGHT": "1",
                "APPROX_SGLANG_USE_SUBSTITUTE": "1",
            },
        ),
    ]
    results = []
    exact_median = None
    for name, env in cases:
        print(f"[sglang-sweep] running {name}", flush=True)
        rec = _run_case(base_out, name, env)
        results.append(rec)
        lat = rec.get("latency_summary", {})
        qstats = rec.get("quant_stats", {})
        median = lat.get("median")
        if name == "exact":
            exact_median = median
        speedup = exact_median / median if exact_median and median else None
        print(
            "[sglang-sweep] result "
            + json.dumps(
                {
                    "name": name,
                    "returncode": rec["returncode"],
                    "median": median,
                    "speedup_vs_exact": speedup,
                    "apply_approx": qstats.get("apply_approx"),
                    "substituted": qstats.get("substituted"),
                    "text": rec.get("output", {}).get("text"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if rec["returncode"] != 0:
            break
    (base_out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
