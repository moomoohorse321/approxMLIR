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

import approx_runtime as ar
import numpy as np


def _default_lavamd_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "tools" / "cgeist" / "Test" / "approxMLIR" / "approx_lavaMD.c"


def main():
    cpp_path = Path(
        os.environ.get("LAVAMD_CPP_PATH", str(_default_lavamd_path()))
    )
    if not cpp_path.exists():
        raise FileNotFoundError(f"lavaMD source not found: {cpp_path}")

    cpp_source = cpp_path.read_text(encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[3]
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

    def _compile_exec(mlir_text: str) -> bytes:
        transformed = ar.compile(
            mlir_text,
            workload=ar.WorkloadType.CPP,
            toolchain=toolchain,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mlir_path = tmpdir_path / "lavaMD.mlir"
            mlir_path.write_text(transformed, encoding="utf-8")
            exec_path = tmpdir_path / "lavaMD.exec"

            import subprocess as sp
            cmd = [
                cgeist_config.cgeist_path,
                f"-resource-dir={cgeist_config.resource_dir}",
            ]
            for inc in cgeist_config.include_dirs:
                cmd.extend(["-I", inc])
            cmd.extend(["-lm", str(mlir_path), "-import-mlir", "-o", str(exec_path)])
            result = sp.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"cgeist import failed:\n{result.stderr}")

            return exec_path.read_bytes()

    def _run_exec(exec_bytes: bytes) -> tuple[float, np.ndarray]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            exec_path = tmpdir_path / "lavaMD.exec"
            exec_path.write_bytes(exec_bytes)
            exec_path.chmod(0o755)
            import subprocess as sp
            start = time.perf_counter()
            result = sp.run(
                [str(exec_path), "-boxes1d", "6"],
                cwd=tmpdir_path,
                capture_output=True,
                text=True,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if result.returncode != 0:
                raise RuntimeError(f"lavaMD run failed:\n{result.stderr}")

            result_path = tmpdir_path / "result.txt"
            if not result_path.exists():
                raise RuntimeError("lavaMD did not produce result.txt")

            data = result_path.read_bytes()
            if len(data) < struct.calcsize("Q"):
                raise RuntimeError("result.txt is too small")
            count = struct.unpack_from("Q", data, 0)[0]
            vec = np.frombuffer(
                data,
                dtype=np.float64,
                offset=struct.calcsize("Q"),
                count=count * 4,
            )
            vec = vec.reshape((count, 4))
            return elapsed_ms, vec

    exact_config = {}
    for name, param in params.items():
        if param.param_type == "decision":
            exact_config[name] = 1
    exact_mlir = manager.apply_config(annotated_mlir, exact_config)
    exact_exec = _compile_exec(exact_mlir)
    exact_time_ms, exact_vec = _run_exec(exact_exec)

    def evaluate_fn(config: dict) -> tuple[float, float]:
        modified = manager.apply_config(annotated_mlir, config)
        exec_bytes = _compile_exec(modified)
        elapsed_ms, vec = _run_exec(exec_bytes)

        diff = vec - exact_vec
        num = np.linalg.norm(diff.ravel())
        denom = np.linalg.norm(exact_vec.ravel())
        if denom == 0:
            accuracy = 1.0 if num == 0 else 0.0
        else:
            accuracy = 1.0 - (num / denom)
        return elapsed_ms, accuracy

    all_results = []

    def result_callback(cfg, time_ms, accuracy):
        all_results.append(
            {"config": dict(cfg), "time_ms": time_ms, "accuracy": accuracy}
        )

    result = ar.tune(
        mlir_source=annotated_mlir,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=0.95,
        time_budget=20,
        result_callback=result_callback,
    )

    print(f"Baseline time: {exact_time_ms:.2f}ms")
    print(f"Best config: {result['best_config']}")
    print(f"Best time: {result['best_time']:.2f}ms")
    print(f"Total configs evaluated: {len(all_results)}")
    print("All configs:")
    for entry in all_results:
        print(entry)


if __name__ == "__main__":
    main()
