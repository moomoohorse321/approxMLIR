#!/usr/bin/env python3
"""Mainline SGLang + Triton quantization driver."""

from __future__ import annotations

import json
import os
import shutil
import sys
import statistics
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
EXAMPLES_ROOT = THIS_FILE.parents[1]
APPROXMLIR_ROOT = THIS_FILE.parents[3]
REPO_ROOT = THIS_FILE.parents[4]
SOURCE_TRITON_PYTHON = REPO_ROOT / "triton" / "python"


def _prepare_paths() -> None:
    if str(THIS_FILE.parent) not in sys.path:
        sys.path.insert(0, str(THIS_FILE.parent))
    if str(EXAMPLES_ROOT) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_ROOT))
    if os.environ.get("SGLANG_FORCE_SOURCE_TRITON", "1") == "1":
        if str(SOURCE_TRITON_PYTHON) not in sys.path:
            sys.path.insert(0, str(SOURCE_TRITON_PYTHON))


def _setup_dlopen_flags() -> None:
    try:
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    except AttributeError:
        pass


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _print_probe(label: str, payload) -> None:
    print(f"[sglang-probe] {label}: {json.dumps(_jsonable(payload), sort_keys=True)}")


def main() -> int:
    _prepare_paths()
    _setup_dlopen_flags()

    out_dir = Path(os.environ.get("OUT_DIR", "/tmp/approx_sglang_probe"))
    cache_dir = out_dir / "triton_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    os.environ["APPROX_SGLANG_DUMP_OUT_DIR"] = str(out_dir)
    os.environ.setdefault("APPROX_SGLANG_STATS_PATH", str(out_dir / "quant_stats.jsonl"))
    os.environ["APPROX_SOURCE_TRITON_PYTHON"] = str(SOURCE_TRITON_PYTHON)
    os.environ["APPROX_EXAMPLES_ROOT"] = str(EXAMPLES_ROOT)
    bootstrap = THIS_FILE.parent / "bootstrap"
    child_pythonpath = [
        str(bootstrap),
        str(THIS_FILE.parent),
        str(SOURCE_TRITON_PYTHON),
        str(EXAMPLES_ROOT),
        *[p for p in os.environ.get("PYTHONPATH", "").split(":") if p],
    ]
    os.environ["PYTHONPATH"] = ":".join(dict.fromkeys(child_pythonpath))

    try:
        import triton
    except ImportError as exc:
        print(f"[sglang-probe] missing triton: {exc}", file=sys.stderr)
        return 2

    try:
        from triton import knobs
    except ImportError as exc:
        print(f"[sglang-probe] triton has no knobs module: {exc}", file=sys.stderr)
        return 2

    try:
        import approx_runtime as ar
    except ImportError as exc:
        print(f"[sglang-probe] missing approx_runtime: {exc}", file=sys.stderr)
        return 2

    try:
        import sglang as sgl
    except ImportError as exc:
        print(
            "[sglang-probe] missing sglang; this probe does not install it. "
            "Install SGLang in an isolated environment before running generation.",
            file=sys.stderr,
        )
        print(f"[sglang-probe] import error: {exc}", file=sys.stderr)
        return 2

    _print_probe(
        "versions",
        {
            "python": sys.executable,
            "triton_file": getattr(triton, "__file__", None),
            "triton_version": getattr(triton, "__version__", None),
            "sglang_file": getattr(sgl, "__file__", None),
            "sglang_version": getattr(sgl, "__version__", None),
            "has_stage_hook": hasattr(knobs.runtime, "add_stages_inspection_hook"),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        },
    )

    if not hasattr(knobs.runtime, "add_stages_inspection_hook"):
        print("[sglang-probe] incompatible Triton: missing stages inspection hook", file=sys.stderr)
        return 3

    knobs.runtime.add_stages_inspection_hook = ar.make_triton_dump_hook(
        out_dir=out_dir,
        source="sglang_triton_probe",
        verbose=os.environ.get("DUMP_VERBOSE", "0") == "1",
    )

    if os.environ.get("RUN_GENERATE", "1") == "0":
        print("[sglang-probe] RUN_GENERATE=0; import and hook checks only")
        return 0

    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    prompt = os.environ.get("PROMPT", "The capital of France is")
    prompts = json.loads(os.environ.get("PROMPTS_JSON", "null") or "null") or [prompt]
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "2"))
    warmup_runs = int(os.environ.get("WARMUP_RUNS", "0"))
    measure_runs = int(os.environ.get("MEASURE_RUNS", "1"))
    attention_backend = os.environ.get("ATTENTION_BACKEND", "triton")
    sampling_backend = os.environ.get("SAMPLING_BACKEND", "pytorch")

    engine_kwargs = {
        "model_path": model_path,
        "attention_backend": attention_backend,
        "sampling_backend": sampling_backend,
        "disable_cuda_graph": os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "1") == "1",
        "log_level": "error",
    }
    mem_fraction = os.environ.get("SGLANG_MEM_FRACTION_STATIC")
    if mem_fraction:
        engine_kwargs["mem_fraction_static"] = float(mem_fraction)

    _print_probe("engine_kwargs", engine_kwargs)
    llm = sgl.Engine(**engine_kwargs)
    try:
        for i in range(warmup_runs):
            batch = [prompts[(i + j) % len(prompts)] for j in range(batch_size)]
            llm.generate(
                batch,
                {"temperature": 0.0, "max_new_tokens": max_new_tokens},
            )
        outputs = []
        latencies = []
        for i in range(measure_runs):
            batch = [prompts[(i + j) % len(prompts)] for j in range(batch_size)]
            out = llm.generate(
                batch,
                {"temperature": 0.0, "max_new_tokens": max_new_tokens},
            )
            rec = out[0] if out else None
            outputs.extend(out)
            if rec:
                meta = rec.get("meta_info", {})
                if "e2e_latency" in meta:
                    latencies.append(float(meta["e2e_latency"]))
        _print_probe("output", outputs[0] if outputs else None)
        if latencies:
            _print_probe(
                "latency_summary",
                {
                    "warmup_runs": warmup_runs,
                    "measure_runs": measure_runs,
                    "batch_size": batch_size,
                    "latencies": latencies,
                    "median": statistics.median(latencies),
                    "mean": statistics.mean(latencies),
                },
            )
    finally:
        llm.shutdown()
        knobs.runtime.add_stages_inspection_hook = None

    jsons = sorted(p for p in out_dir.glob("*.json") if p.name != "launches.jsonl")
    by_name: dict[str, int] = {}
    for path in jsons:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        name = str(rec.get("func_name", "<unknown>"))
        by_name[name] = by_name.get(name, 0) + 1
    _print_probe("dump_summary", {"out_dir": str(out_dir), "num_ttir": len(jsons), "by_name": by_name})
    stats_path = Path(os.environ["APPROX_SGLANG_STATS_PATH"])
    stats = {"prepare_weight": 0, "apply_approx": 0, "substituted": 0, "by_prefix": {}}
    if stats_path.exists():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = rec.get("event")
            if event in ("prepare_weight", "apply_approx"):
                stats[event] += 1
            if event == "apply_approx":
                if rec.get("substituted"):
                    stats["substituted"] += 1
                prefix = str(rec.get("prefix", "<unknown>"))
                stats["by_prefix"][prefix] = stats["by_prefix"].get(prefix, 0) + 1
    _print_probe("quant_stats", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
