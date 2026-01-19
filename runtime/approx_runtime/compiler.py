"""approx-opt compiler integration."""

import subprocess
import tempfile
from typing import List, Optional
from pathlib import Path
import warnings

from .toolchain import WorkloadType, ToolchainConfig, get_toolchain

__all__ = [
    'compile', 
    'compile_file', 
    'PASS_PIPELINE',
    'FUNC_SUBSTITUTE_PIPELINE', 
    'CompilationError',
    'get_pipeline_for_config',
    'WorkloadType',
    'ToolchainConfig',
]


# Default pass pipeline for full approximation lowering (ML toolchain).
PASS_PIPELINE = ToolchainConfig().ml_pipeline

# Pipeline when func_substitute is used (needs pre-emit-transform first).
FUNC_SUBSTITUTE_PIPELINE = ["pre-emit-transform"] + PASS_PIPELINE


def get_pipeline_for_config(config: dict) -> List[str]:
    """Select appropriate pass pipeline based on configuration.
    
    Returns FUNC_SUBSTITUTE_PIPELINE if func_substitute transform is used,
    otherwise returns standard PASS_PIPELINE.
    """
    toolchain = get_toolchain()
    return toolchain.get_pipeline(WorkloadType.ML, config)


class CompilationError(Exception):
    """Raised when approx-opt compilation fails."""
    pass


def _resolve_toolchain(
    passes: Optional[List[str]],
    approx_opt_path: Optional[str],
    workload: WorkloadType,
    toolchain: Optional[ToolchainConfig],
) -> tuple[str, List[str]]:
    toolchain = toolchain or get_toolchain()
    if approx_opt_path is not None:
        warnings.warn(
            "approx_opt_path is deprecated. Use toolchain config instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        opt_path = approx_opt_path
    else:
        opt_path = toolchain.get_opt_path(workload)

    pipeline = passes or toolchain.get_pipeline(workload)
    return opt_path, pipeline


def compile(
    mlir_text: str,
    passes: Optional[List[str]] = None,
    approx_opt_path: Optional[str] = None,
    workload: WorkloadType = WorkloadType.ML,
    toolchain: Optional[ToolchainConfig] = None,
) -> str:
    """Run approx-opt on MLIR text.
    
    Args:
        mlir_text: Input MLIR module text
        passes: List of passes to run (default: PASS_PIPELINE)
        approx_opt_path: Deprecated: Path to approx-opt binary
        workload: Type of workload (ML or CPP)
        toolchain: Toolchain configuration (default: global config)
    
    Returns:
        Transformed MLIR text
    
    Raises:
        CompilationError: If approx-opt fails
    """
    opt_path, passes = _resolve_toolchain(
        passes, approx_opt_path, workload, toolchain
    )
    
    with tempfile.NamedTemporaryFile(
        suffix='.mlir', mode='w', delete=False
    ) as f:
        f.write(mlir_text)
        f.flush()
        input_path = f.name
        print(input_path)
    try:
        cmd = [opt_path, input_path] + [f'--{p}' for p in passes]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise CompilationError(
                f"approx-opt failed with code {result.returncode}:\n{result.stderr}"
            )
        
        return result.stdout
    finally:
        Path(input_path).unlink(missing_ok=True)


def compile_file(
    input_path: str,
    output_path: Optional[str] = None,
    passes: Optional[List[str]] = None,
    approx_opt_path: Optional[str] = None,
    workload: WorkloadType = WorkloadType.ML,
    toolchain: Optional[ToolchainConfig] = None,
) -> str:
    """Run approx-opt on an MLIR file.
    
    Args:
        input_path: Path to input MLIR file
        output_path: Path for output (if None, returns stdout)
        passes: List of passes to run (default: PASS_PIPELINE)
        approx_opt_path: Deprecated: Path to approx-opt binary
        workload: Type of workload (ML or CPP)
        toolchain: Toolchain configuration (default: global config)
    
    Returns:
        Transformed MLIR text (or output_path if specified)
    
    Raises:
        CompilationError: If approx-opt fails
    """
    opt_path, passes = _resolve_toolchain(
        passes, approx_opt_path, workload, toolchain
    )
    
    cmd = [opt_path, input_path] + [f'--{p}' for p in passes]
    if output_path:
        cmd.extend(['-o', output_path])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise CompilationError(
            f"approx-opt failed with code {result.returncode}:\n{result.stderr}"
        )
    
    return output_path if output_path else result.stdout
