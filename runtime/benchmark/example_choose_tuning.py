#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import sys

import approx_runtime as ar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_common import (  # noqa: E402
    default_cgeist_config,
    default_source_path,
    default_toolchain,
    compile_cpp_path_to_annotated_mlir,
    compile_mlir_to_native_exec,
    run_exec,
    result_logger,
)


def main() -> None:
    cpp_path = Path(os.environ.get("CHOOSE_CPP_PATH", str(default_source_path("approx_choose.c"))))
    if not cpp_path.exists():
        raise FileNotFoundError(f"choose source not found: {cpp_path}")

    toolchain = default_toolchain()
    cgeist_config = default_cgeist_config()
    annotated_mlir = compile_cpp_path_to_annotated_mlir(
        cpp_path,
        cgeist_config=cgeist_config,
        toolchain=toolchain,
    )

    manager = ar.MLIRConfigManager()
    params = manager.parse_annotations(annotated_mlir)

    def evaluate_fn(config: dict) -> tuple[float, float]:
        modified = manager.apply_config(annotated_mlir, config)
        exec_path = compile_mlir_to_native_exec(
            modified, cgeist_config=cgeist_config, toolchain=toolchain, tag="choose"
        )
        result = run_exec(exec_path, ["2"])
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.elapsed_ms, 1.0

    result = ar.tune(
        mlir_source=annotated_mlir,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=1.0,
        time_budget=20,
        result_callback=result_logger("choose"),
    )

    print(f"Found {len(params)} tunable params")
    print(f"Best config: {result['best_config']}")


if __name__ == "__main__":
    main()
