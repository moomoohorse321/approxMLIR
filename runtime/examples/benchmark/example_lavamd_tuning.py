#!/usr/bin/env python3
"""ApproxMLIR lavaMD auto-tuning example (C++ pipeline).

This example demonstrates:
1) Running cgeist on C/C++ source
2) Auto-generating annotations from @approx comments
3) Tuning thresholds/decisions with OpenTuner

Note: The evaluation function here measures compile time only and uses a
placeholder accuracy. Replace with real runtime + accuracy metrics for
meaningful tuning.
"""

from pathlib import Path
import os
import struct
import tempfile
import time
import sys
import math
import re

import approx_runtime as ar
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_common import (
    default_source_path,
    compile_mlir_to_native_exec,
    result_logger,
    exact_decision_config,
)


def _default_lavamd_path() -> Path:
    return default_source_path("approx_lavaMD.c")


def main():
    cpp_path = Path(
        os.environ.get("LAVAMD_CPP_PATH", str(_default_lavamd_path()))
    )
    if not cpp_path.exists():
        raise FileNotFoundError(f"lavaMD source not found: {cpp_path}")

    cpp_source = cpp_path.read_text(encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[4]
    cgeist_config = ar.CgeistConfig(
        cgeist_path=os.environ.get("CGEIST_PATH", "cgeist"),
        resource_dir=os.environ.get(
            "CGEIST_RESOURCE_DIR",
            str(repo_root / "llvm-project" / "build" / "lib" / "clang" / "18"),
        ),
        include_dirs=[
            os.environ.get(
                "CGEIST_INCLUDE_DIR",
                str(repo_root / "tools" / "cgeist" / "Test" / "polybench" / "utilities"),
            )
        ],
    )
    toolchain = ar.ToolchainConfig(
        ml_opt_path=os.environ.get("APPROX_OPT_ML", "approx-opt"),
        cpp_opt_path=os.environ.get("APPROX_OPT_CPP", "polygeist-opt"),
    )
    
    annotated_mlir = ar.compile_cpp_source(
        cpp_source, emit="annotated", cgeist_config=cgeist_config, toolchain=toolchain
    )

    manager = ar.MLIRConfigManager()
    params = manager.parse_annotations(annotated_mlir)

    def _parse_time_ms(data_string: str) -> float | None:
        for line in data_string.splitlines():
            s = line.strip()
            m = re.compile(
                r"^\s*Total execution time:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:ms)?\s*$"
            ).match(s)
            if m:
                return float(m.group(1))
        return None

    def _read_size_t(f):
        pos = f.tell()
        raw = f.read(8)
        if len(raw) == 8:
            (n,) = struct.unpack("Q", raw)
            if n < 10_000_000_000:
                return n
        f.seek(pos)
        raw = f.read(4)
        if len(raw) != 4:
            raise RuntimeError("Failed to read size_t header")
        (n4,) = struct.unpack("I", raw)
        return n4

    def _read_result_bin(path: Path) -> dict:
        with path.open("rb") as f:
            N = _read_size_t(f)
            rec_size = 8 * 4
            blob = f.read(N * rec_size)
            if len(blob) < N * rec_size:
                raise RuntimeError("result file truncated")
        V = [0.0] * N
        F = [0.0] * (3 * N)
        off = 0
        for i in range(N):
            v, x, y, z = struct.unpack_from("dddd", blob, off)
            V[i] = v
            j = 3 * i
            F[j] = x
            F[j + 1] = y
            F[j + 2] = z
            off += rec_size
        return {"V": V, "F": F}

    def _l2(vec):
        return math.sqrt(sum(x * x for x in vec))

    def _dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def get_gt(fname: Path):
        return _read_result_bin(fname)

    def compute_similarity(gt_values, cand_values):
        if not isinstance(gt_values, dict) or not isinstance(cand_values, dict):
            return 0.0
        if "V" not in gt_values or "F" not in gt_values:
            return 0.0
        if "V" not in cand_values or "F" not in cand_values:
            return 0.0
        Vb, Fb = gt_values["V"], gt_values["F"]
        Va, Fa = cand_values["V"], cand_values["F"]
        if len(Va) == 0 and len(Vb) == 0:
            return 1.0
        if len(Va) != len(Vb) or len(Fa) != len(Fb):
            return 0.0
        eps = 1e-12
        diffV = [a - b for a, b in zip(Va, Vb)]
        diffF = [a - b for a, b in zip(Fa, Fb)]
        relL2_V = _l2(diffV) / max(_l2(Vb), eps)
        relL2_F = _l2(diffF) / max(_l2(Fb), eps)
        denom = max(_l2(Fa) * _l2(Fb), eps)
        cos_F = _dot(Fa, Fb) / denom
        base_sim = math.exp(-(relL2_V + relL2_F))
        cos_term = (cos_F + 1.0) / 2.0
        sim = 0.5 * base_sim + 0.5 * max(0.0, min(1.0, cos_term))
        return max(0.0, min(1.0, sim))

    gt_path = Path(os.environ.get("LAVAMD_GT_PATH", str(Path(__file__).resolve().parent / "gt.txt")))
    exact_cfg = exact_decision_config(params)
    exact_mlir = manager.apply_config(annotated_mlir, exact_cfg)
    exact_exec = compile_mlir_to_native_exec(
        exact_mlir, cgeist_config=cgeist_config, toolchain=toolchain, tag="lavamd"
    )
    import subprocess as sp
    exact_run = sp.run(
        [str(exact_exec), "-boxes1d", "8"],
        cwd=exact_exec.parent,
        capture_output=True,
        text=True,
    )
    if exact_run.returncode != 0:
        raise RuntimeError(exact_run.stderr)
    result_path = exact_exec.parent / "result.txt"
    if not result_path.exists():
        raise RuntimeError("lavaMD did not produce result.txt")
    gt_path.write_bytes(result_path.read_bytes())
    gt_values = get_gt(gt_path)
    exact_time_ms = _parse_time_ms(exact_run.stdout)
    if exact_time_ms is None:
        raise RuntimeError("lavamd: failed to parse Total execution time from stdout (gt run)")

    def _compile_exec(mlir_text: str) -> Path:
        return compile_mlir_to_native_exec(
            mlir_text,
            cgeist_config=cgeist_config,
            toolchain=toolchain,
            tag="lavamd",
        )

    def _run_exec(exec_path: Path) -> tuple[float, dict]:
        import subprocess as sp
        result = sp.run(
            [str(exec_path), "-boxes1d", "8"],
            cwd=exec_path.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"lavaMD run failed:\\n{result.stderr}")
        result_path = exec_path.parent / "result.txt"
        if not result_path.exists():
            raise RuntimeError("lavaMD did not produce result.txt")
        values = _read_result_bin(result_path)
        time_ms = _parse_time_ms(result.stdout)
        if time_ms is None:
            raise RuntimeError("lavamd: failed to parse Total execution time from stdout")
        return time_ms, values

    def evaluate_fn(config: dict) -> tuple[float, float]:
        modified = manager.apply_config(annotated_mlir, config)
        exec_path = _compile_exec(modified)
        elapsed_ms, values = _run_exec(exec_path)
        accuracy = compute_similarity(gt_values, values)
        return elapsed_ms, accuracy

    result = ar.tune(
        mlir_source=annotated_mlir,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=float(os.environ.get("ACCURACY_THRESHOLD", "0.95")),
        time_budget=20,
        result_callback=result_logger("lavamd"),
    )

    print(f"Baseline time: {exact_time_ms:.2f}ms")
    print(f"Best config: {result['best_config']}")
    print(f"Best time: {result['best_time']:.2f}ms")
    print("Results logged to exec/lavamd/results.csv")


if __name__ == "__main__":
    main()
