#!/usr/bin/env python3
# sweep_sglang_quant.py
#
# Sequential driver for the approxMLIR SGLang quantization mainline. Each
# "case" is one subprocess invocation of probe_sglang_triton_dump.py with a
# different APPROX_SGLANG_* env combination; we parse the probe's stdout for
# the stable "[sglang-probe] <label>: <json>" lines and collect per-case
# medians + substitution stats into results.json.
#
# Stops on the first failing case (preserves the partial results file) so
# that an OOM or plugin issue gets surfaced early instead of burying it in
# a full-run log.
#
# Reading order:
#   1) `_CASES` / `_case_env`       — declarative case list + flat env builder
#   2) `_run_case`                  — spawn probe, collect probe-line payloads
#   3) `_print_case_result`         — the one-line summary we print per case
#   4) `main`                       — loop cases, write results.json, return RC

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]
PLUGIN = REPO_ROOT / "approxMLIR" / "external-tools" / "approx-triton-plugin" / "build" / "lib" / "libApproxTritonPlugin.so"
PROBE_SCRIPT = THIS_FILE.parent / "probe_sglang_triton_dump.py"


# Declarative list of cases. Each entry is (case_name, extra_env_dict). The
# case_name also doubles as the per-case out_dir suffix and the key under
# which results.json stores its record. Keep `exact` first: the sweep uses
# it as the denominator for speedup_vs_exact.
_CASES: list[tuple[str, dict[str, str]]] = [
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


# Build the full env passed to one probe subprocess: the caller's env, then
# the "probe knobs" overrideable from the caller (MODEL_PATH etc), then the
# case-specific APPROX_SGLANG_* settings. Case-specific wins — that's the
# whole point of the sweep.
def _case_env(out_dir: Path, case_extra: dict[str, str]) -> dict[str, str]:
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
    env.update(case_extra)
    return env


# Spawn one probe subprocess and parse every "[sglang-probe] <label>: <json>"
# line into a dict keyed by label. Also persists full stdout to stdout.log
# inside the case's out_dir for post-mortem debugging.
def _run_case(base_out: Path, name: str, extra_env: dict[str, str]) -> dict:
    # out_dir: per-case directory; the probe writes TTIR dumps, quant_stats,
    #          and a triton_cache here. Separating by case makes diffs easy.
    # rec:     accumulated probe-line payloads; "name", "returncode", "out_dir"
    #          are always present; all other keys come from probe line labels.
    out_dir = base_out / name
    proc = subprocess.run(
        [sys.executable, str(PROBE_SCRIPT)],
        env=_case_env(out_dir, extra_env),
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    return rec


# Render one case's outcome as a single JSON line on stdout, computing
# speedup_vs_exact relative to the sticky baseline median passed in.
# Returns the case's median so the caller can promote it to `exact_median`
# when the case is named "exact".
def _print_case_result(rec: dict, exact_median: float | None) -> float | None:
    lat = rec.get("latency_summary", {})
    qstats = rec.get("quant_stats", {})
    median = lat.get("median")
    speedup = exact_median / median if exact_median and median else None
    print(
        "[sglang-sweep] result "
        + json.dumps(
            {
                "name": rec["name"],
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
    return median


# Run every case in order, stop on first non-zero return code, dump the
# partial results to results.json, and return an overall return code.
# The exit code is 0 iff every case returned 0 — the sweep script is only
# "successful" when the full matrix ran clean.
def main() -> int:
    base_out = Path(os.environ.get("OUT_DIR", "/tmp/approx_sglang_quant_sweep"))
    base_out.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    exact_median: float | None = None
    for name, env in _CASES:
        print(f"[sglang-sweep] running {name}", flush=True)
        rec = _run_case(base_out, name, env)
        results.append(rec)
        median = _print_case_result(rec, exact_median)
        if name == "exact":
            exact_median = median
        if rec["returncode"] != 0:
            break

    (base_out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
