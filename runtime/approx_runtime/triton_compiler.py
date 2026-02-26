"""Triton plugin-based pass runner for ApproxMLIR pipelines."""

from __future__ import annotations

import importlib
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from .toolchain import WorkloadType, get_toolchain

__all__ = [
    "TRITON_AVAILABLE",
    "TritonCompilationError",
    "compile_with_triton_plugin",
]

TRITON_AVAILABLE = (
    importlib.util.find_spec("triton") is not None
    and importlib.util.find_spec("triton._C.libtriton") is not None
)


class TritonCompilationError(Exception):
    """Raised when Triton plugin pass execution fails."""


def compile_with_triton_plugin(
    mlir_text: str,
    passes: Optional[List[str]] = None,
    plugin_path: Optional[str] = None,
    stage_name: str = "approx_triton_pipeline",
) -> str:
    """Run ApproxMLIR passes through Triton's plugin pass manager.

    Args:
        mlir_text: Input MLIR module text.
        passes: Pass names exposed by the plugin API.
        plugin_path: Path to plugin shared object; falls back to env/toolchain.
        stage_name: Label used in pass manager run.

    Returns:
        Transformed MLIR text.
    """
    toolchain = get_toolchain()
    pipeline = list(passes or toolchain.get_pipeline(WorkloadType.TRITON))
    selected_plugin = plugin_path or toolchain.triton_plugin_path or os.environ.get(
        "TRITON_PASS_PLUGIN_PATH", ""
    )
    if not selected_plugin:
        raise TritonCompilationError(
            "TRITON_PASS_PLUGIN_PATH is required for TRITON workload"
        )

    os.environ["TRITON_PASS_PLUGIN_PATH"] = selected_plugin

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(mlir_text)
        input_path = f.name

    if not TRITON_AVAILABLE:
        raise TritonCompilationError("Triton Python bindings are required for TRITON workload")

    triton_lib = importlib.import_module("triton._C.libtriton")
    triton_ir = triton_lib.ir
    triton_passes = triton_lib.passes

    context = triton_ir.context()
    triton_ir.load_dialects(context)
    module = triton_ir.parse_mlir_module(input_path, context)
    pass_manager = triton_ir.pass_manager(context)

    available = set(dir(triton_passes.plugin))
    for pass_name in pipeline:
        pass_fn = getattr(triton_passes.plugin, pass_name, None)
        if pass_fn is None:
            raise TritonCompilationError(
                f"Plugin pass '{pass_name}' not found. "
                f"Available plugin entries: {sorted(available)}"
            )
        pass_fn(pass_manager)

    pass_manager.run(module, stage_name)
    Path(input_path).unlink(missing_ok=True)
    return str(module)
