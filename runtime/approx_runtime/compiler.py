"""approx-opt compiler integration."""

import subprocess
import tempfile
from typing import List, Optional
from pathlib import Path

__all__ = [
    'compile', 
    'compile_file', 
    'PASS_PIPELINE',
    'FUNC_SUBSTITUTE_PIPELINE', 
    'CompilationError',
    'get_pipeline_for_config',
]


# Default pass pipeline for full approximation lowering
PASS_PIPELINE = [
    "emit-approx",
    "emit-management",
    "config-approx",
    "transform-approx",
    "finalize-approx",
    "legalize-to-stablehlo"
]

# Pipeline when func_substitute is used (needs pre-emit-transform first)
# pre-emit-transform converts: func @f { body } 
#   -> func @__internal_f { body } + func @f { call @__internal_f }
FUNC_SUBSTITUTE_PIPELINE = [
    "pre-emit-transform",  # Refactor target func to wrapper + __internal_
    "emit-approx",
    "emit-management", 
    "config-approx",
    "transform-approx",
    "finalize-approx",
    "legalize-to-stablehlo"
]


def get_pipeline_for_config(config: dict) -> List[str]:
    """Select appropriate pass pipeline based on configuration.
    
    Returns FUNC_SUBSTITUTE_PIPELINE if func_substitute transform is used,
    otherwise returns standard PASS_PIPELINE.
    """
    if not config:
        return PASS_PIPELINE
    
    dt = config.get('decision_tree')
    if dt and dt.transform_type == "func_substitute":
        return FUNC_SUBSTITUTE_PIPELINE
    
    st = config.get('static_transform')
    if st and st.transform_type == "func_substitute":
        return FUNC_SUBSTITUTE_PIPELINE
    
    sc = config.get('safety_contract')
    if sc:
        return PASS_PIPELINE[:-1]
    
    return PASS_PIPELINE


class CompilationError(Exception):
    """Raised when approx-opt compilation fails."""
    pass


def compile(
    mlir_text: str,
    passes: Optional[List[str]] = None,
    approx_opt_path: str = "approx-opt",
) -> str:
    """Run approx-opt on MLIR text.
    
    Args:
        mlir_text: Input MLIR module text
        passes: List of passes to run (default: PASS_PIPELINE)
        approx_opt_path: Path to approx-opt binary
    
    Returns:
        Transformed MLIR text
    
    Raises:
        CompilationError: If approx-opt fails
    """
    passes = passes or PASS_PIPELINE
    
    with tempfile.NamedTemporaryFile(
        suffix='.mlir', mode='w', delete=False
    ) as f:
        f.write(mlir_text)
        f.flush()
        input_path = f.name
        print(input_path)
    try:
        cmd = [approx_opt_path, input_path] + [f'--{p}' for p in passes]
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
    approx_opt_path: str = "approx-opt",
) -> str:
    """Run approx-opt on an MLIR file.
    
    Args:
        input_path: Path to input MLIR file
        output_path: Path for output (if None, returns stdout)
        passes: List of passes to run (default: PASS_PIPELINE)
        approx_opt_path: Path to approx-opt binary
    
    Returns:
        Transformed MLIR text (or output_path if specified)
    
    Raises:
        CompilationError: If approx-opt fails
    """
    passes = passes or PASS_PIPELINE
    
    cmd = [approx_opt_path, input_path] + [f'--{p}' for p in passes]
    if output_path:
        cmd.extend(['-o', output_path])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise CompilationError(
            f"approx-opt failed with code {result.returncode}:\n{result.stderr}"
        )
    
    return output_path if output_path else result.stdout
