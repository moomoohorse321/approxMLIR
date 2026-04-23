#!/usr/bin/env python3
# probe_sglang_triton_dump.py
#
# Mainline SGLang + Triton quantization driver. A single run does one
# SGLang generation pass, collects Triton TTIR dumps via the approx_runtime
# stage-inspection hook, and emits JSONL/JSON summaries that the sweep
# driver (sweep_sglang_quant.py) parses for case-level medians and
# substitution stats.
#
# This file intentionally does only the "one measurement" job — no case
# orchestration, no env sweeping. The sweep script is the layer that
# calls this in a loop.
#
# Reading order for main():
#   1) `_configure_environment`  — wire OUT_DIR, TRITON_CACHE_DIR, PYTHONPATH
#   2) `_probe_imports`          — bail early with clear messages on missing
#                                  triton / knobs / approx_runtime / sglang
#   3) `_install_dump_hook`      — register the stages-inspection hook
#   4) `_run_generation`         — warmup + measurement loop over SGLang Engine
#   5) `_summarize_dumps`        — count TTIR JSONs by func_name
#   6) `_summarize_quant_stats`  — fold quant_stats.jsonl into a small dict

from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
from pathlib import Path


# Project-relative anchors. These are resolved once at import time so every
# downstream helper can reference them without re-walking parent directories.
THIS_FILE = Path(__file__).resolve()
EXAMPLES_ROOT = THIS_FILE.parents[1]
APPROXMLIR_ROOT = THIS_FILE.parents[3]
REPO_ROOT = THIS_FILE.parents[4]
SOURCE_TRITON_PYTHON = REPO_ROOT / "triton" / "python"


# Prepend our source paths so sgl and the approx helpers import from the
# expected source trees, not any PyPI wheel that may be lurking in the venv.
def _prepare_paths() -> None:
    if str(THIS_FILE.parent) not in sys.path:
        sys.path.insert(0, str(THIS_FILE.parent))
    if str(EXAMPLES_ROOT) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_ROOT))
    if os.environ.get("SGLANG_FORCE_SOURCE_TRITON", "1") == "1":
        if str(SOURCE_TRITON_PYTHON) not in sys.path:
            sys.path.insert(0, str(SOURCE_TRITON_PYTHON))


# Match the global-dlopen-flags trick that approxMLIR requires for the
# Triton pass plugin to resolve MLIR symbols across shared-object boundaries.
def _setup_dlopen_flags() -> None:
    try:
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    except AttributeError:
        pass


# Recursively coerce a value into something json.dumps will accept.
# Used for probe lines that include non-JSON leaves (e.g. Path objects).
def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


# All lines the sweep script parses follow this shape: "[sglang-probe] <label>: <json>".
# Keeping the format stable is load-bearing — don't reshape this without
# updating the regex-ish split on the sweep side.
def _print_probe(label: str, payload) -> None:
    print(f"[sglang-probe] {label}: {json.dumps(_jsonable(payload), sort_keys=True)}")


# Stand up the out dir, Triton cache, PYTHONPATH for child processes, and
# the env vars that downstream modules (approx_quant_patch, approx_runtime)
# read. Called once at the top of main(), before any heavy imports.
def _configure_environment(out_dir: Path) -> Path:
    # out_dir:    user-supplied or defaulted; receives TTIR dumps + stats.
    # cache_dir:  Triton compilation cache, wiped per run so compile-time
    #             effects show up instead of hiding behind a warm cache.
    # bootstrap:  child-process sitecustomize lives here; we prepend it to
    #             PYTHONPATH so SGLang workers pick it up on import.
    cache_dir = out_dir / "triton_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    os.environ["APPROX_SGLANG_DUMP_OUT_DIR"] = str(out_dir)
    os.environ.setdefault("APPROX_SGLANG_STATS_PATH", str(out_dir / "quant_stats.jsonl"))
    if os.environ.get("APPROX_SGLANG_SQ_COLLECT", "0") == "1":
        os.environ["SGLANG_DISABLE_CUDA_GRAPH"] = "1"
        sq_stats_dir = Path(os.environ.get("APPROX_SGLANG_SQ_STATS_DIR", str(out_dir / "sq_stats")))
        shutil.rmtree(sq_stats_dir, ignore_errors=True)
        sq_stats_dir.mkdir(parents=True, exist_ok=True)
        os.environ["APPROX_SGLANG_SQ_STATS_DIR"] = str(sq_stats_dir)
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
    return cache_dir


# Do the four gated imports (triton / knobs / approx_runtime / sglang) with
# distinct exit codes and error messages so that a sweep can tell *why* a
# case failed without rummaging through tracebacks.
# Returns (triton_module, knobs_module, approx_runtime_module, sglang_module)
# or raises SystemExit with the right code.
def _probe_imports():
    try:
        import triton
    except ImportError as exc:
        print(f"[sglang-probe] missing triton: {exc}", file=sys.stderr)
        raise SystemExit(2)
    try:
        from triton import knobs
    except ImportError as exc:
        print(f"[sglang-probe] triton has no knobs module: {exc}", file=sys.stderr)
        raise SystemExit(2)
    try:
        import approx_runtime as ar
    except ImportError as exc:
        print(f"[sglang-probe] missing approx_runtime: {exc}", file=sys.stderr)
        raise SystemExit(2)
    try:
        import sglang as sgl
    except ImportError as exc:
        print(
            "[sglang-probe] missing sglang; this probe does not install it. "
            "Install SGLang in an isolated environment before running generation.",
            file=sys.stderr,
        )
        print(f"[sglang-probe] import error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    return triton, knobs, ar, sgl


# Attach the approx_runtime stages-inspection hook to Triton so every
# compile produces a TTIR dump in out_dir. No-op uninstall happens in the
# generation caller's finally clause.
def _install_dump_hook(knobs, ar, out_dir: Path) -> None:
    if not hasattr(knobs.runtime, "add_stages_inspection_hook"):
        print("[sglang-probe] incompatible Triton: missing stages inspection hook", file=sys.stderr)
        raise SystemExit(3)
    knobs.runtime.add_stages_inspection_hook = ar.make_triton_dump_hook(
        out_dir=out_dir,
        source="sglang_triton_probe",
        verbose=os.environ.get("DUMP_VERBOSE", "0") == "1",
    )


# Assemble the dict of kwargs for sgl.Engine(...) from env. Pure function,
# no side effects — caller is the one that starts the engine.
def _build_engine_kwargs() -> dict:
    engine_kwargs = {
        "model_path": os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct"),
        "attention_backend": os.environ.get("ATTENTION_BACKEND", "triton"),
        "sampling_backend": os.environ.get("SAMPLING_BACKEND", "pytorch"),
        "disable_cuda_graph": os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "1") == "1",
        "log_level": "error",
    }
    mem_fraction = os.environ.get("SGLANG_MEM_FRACTION_STATIC")
    if mem_fraction:
        engine_kwargs["mem_fraction_static"] = float(mem_fraction)
    return engine_kwargs


# Run warmup_runs unmeasured passes, then measure_runs timed passes, and
# return (outputs_flat, latencies). Rotates prompts by run index so CUDA
# graph capture doesn't collapse onto a single prompt cache key.
def _run_generation(
    llm,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    warmup_runs: int,
    measure_runs: int,
) -> tuple[list, list[float]]:
    # sampling:  temperature=0 for deterministic output so the sweep can
    #            string-compare text across cases for a trivial correctness check.
    # latencies: sourced from SGLang's own meta_info, not wall clock; skipped
    #            silently when the engine doesn't populate e2e_latency.
    sampling = {"temperature": 0.0, "max_new_tokens": max_new_tokens}

    for i in range(warmup_runs):
        batch = [prompts[(i + j) % len(prompts)] for j in range(batch_size)]
        llm.generate(batch, sampling)

    outputs: list = []
    latencies: list[float] = []
    for i in range(measure_runs):
        batch = [prompts[(i + j) % len(prompts)] for j in range(batch_size)]
        out = llm.generate(batch, sampling)
        outputs.extend(out)
        rec = out[0] if out else None
        if rec:
            meta = rec.get("meta_info", {})
            if "e2e_latency" in meta:
                latencies.append(float(meta["e2e_latency"]))
    return outputs, latencies


# Walk every *.json (excluding the launches.jsonl log) in out_dir and
# count dumps by func_name. Fed back to the sweep as "which kernels showed
# up in this case" — useful for sanity-checking targeting.
def _summarize_dumps(out_dir: Path) -> None:
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


# Fold the quant_stats.jsonl events emitted by approx_quant_patch._record
# into a single compact summary dict. Schema here must match what the
# sweep driver reads off the probe line.
def _summarize_quant_stats(stats_path: Path) -> None:
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


# Orchestrator. Each phase below is numbered so a reader can follow the flow
# without chasing definitions. See the file header for the same numbering.
def main() -> int:
    # 1) Paths + dlopen flags, then env scaffolding.
    _prepare_paths()
    _setup_dlopen_flags()
    out_dir = Path(os.environ.get("OUT_DIR", "/tmp/approx_sglang_probe"))
    _configure_environment(out_dir)

    # 2) Gated imports — bail early with targeted exit codes.
    triton, knobs, ar, sgl = _probe_imports()
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

    # 3) Install the TTIR dump hook. After this, every Triton compile dumps.
    _install_dump_hook(knobs, ar, out_dir)

    # Smoke-test early exit for CI / import-only sanity checks.
    if os.environ.get("RUN_GENERATE", "1") == "0":
        print("[sglang-probe] RUN_GENERATE=0; import and hook checks only")
        return 0

    # 4) Start the engine and run generation, guarded so we always clean up.
    prompt = os.environ.get("PROMPT", "The capital of France is")
    prompts = json.loads(os.environ.get("PROMPTS_JSON", "null") or "null") or [prompt]
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "2"))
    warmup_runs = int(os.environ.get("WARMUP_RUNS", "0"))
    measure_runs = int(os.environ.get("MEASURE_RUNS", "1"))

    engine_kwargs = _build_engine_kwargs()
    _print_probe("engine_kwargs", engine_kwargs)
    llm = sgl.Engine(**engine_kwargs)
    try:
        outputs, latencies = _run_generation(
            llm, prompts, batch_size, max_new_tokens, warmup_runs, measure_runs
        )
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

    # 5 + 6) Post-run summaries.
    _summarize_dumps(out_dir)
    _summarize_quant_stats(Path(os.environ["APPROX_SGLANG_STATS_PATH"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
