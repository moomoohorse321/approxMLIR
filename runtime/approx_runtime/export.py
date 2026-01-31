"""JAX function export to MLIR."""

from typing import Callable, Any, List, Tuple, Optional, Dict
import functools
import re

__all__ = [
    'export_to_mlir', 
    'export_module',
    'get_function_name',
    'JAX_AVAILABLE',
]

# JAX import with graceful fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import ShapeDtypeStruct
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    ShapeDtypeStruct = None
    JAX_AVAILABLE = False


def export_to_mlir(func: Callable, *example_args: Any) -> str:
    """Export a JAX function to StableHLO MLIR.
    
    Args:
        func: JAX function to export (can have @Knob decorator)
        *example_args: Example arguments to trace the function
    
    Returns:
        MLIR module text (StableHLO dialect)
    
    Raises:
        ImportError: If JAX is not available
        RuntimeError: If export fails
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for export_to_mlir")
    
    # jit the function if not already
    jitted = jax.jit(func)
    
    # Export to StableHLO
    import jax.export as jax_export
    exported = jax_export.export(jitted)(*example_args)
    
    # Get MLIR text
    return str(exported.mlir_module())


def export_module(
    functions: Dict[str, Tuple[Callable, Tuple]],
    module_name: Optional[str] = None,
) -> str:
    """Export multiple JAX functions to a single MLIR module.
    
    This handles the complexity of:
    1. Exporting each function via JAX
    2. Renaming JAX's @main to the actual function name
    3. Merging all functions into one module
    
    Args:
        functions: Dict mapping function name -> (callable, example_args)
                   Example: {"my_kernel": (kernel_fn, (x, y)), 
                             "get_state": (state_fn, (n,))}
        module_name: Optional module name (default: None)
    
    Returns:
        Combined MLIR module text with all functions
    
    Raises:
        ImportError: If JAX is not available
    
    Example:
        import jax
        import jax.numpy as jnp
        
        functions = {
            "my_kernel": (kernel_fn, (jax.ShapeDtypeStruct((1024,), jnp.float32),)),
            "get_state": (state_fn, (jax.ShapeDtypeStruct((), jnp.int32),)),
            "approx_my_kernel_1": (approx_v1, (jax.ShapeDtypeStruct((1024,), jnp.float32),)),
        }
        mlir = export_module(functions)
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for export_module")
    
    if not functions:
        return "module {}\n"
    
    # Export each function and extract the renamed func.func
    func_definitions = []
    
    for target_name, (func, example_args) in functions.items():
        # Export single function via JAX
        jitted = jax.jit(func)
        import jax.export as jax_export
        exported = jax_export.export(jitted)(*example_args)
        mlir_text = str(exported.mlir_module())
        
        # Extract and rename the function
        func_def = _extract_and_rename_func(mlir_text, target_name)
        if func_def:
            func_definitions.append(func_def)
    
    # Build the combined module
    if module_name:
        header = f"module @{module_name} {{\n"
    else:
        header = "module {\n"
    
    combined = header + "\n".join(func_definitions) + "\n}\n"
    return combined


def _extract_and_rename_func(mlir_text: str, new_name: str) -> Optional[str]:
    """Extract function from JAX-exported module and rename it.
    
    JAX exports functions wrapped in a module with the function typically
    named @main. This extracts the function body and renames it.
    
    Returns the func.func definition as a string, or None if extraction fails.
    """
    lines = mlir_text.split('\n')
    
    # Find the func.func definition
    brace_count = 0
    func_lines = []
    in_function = False
    
    for line in lines:
        # Look for func.func declaration
        if not in_function:
            # Match: func.func public @main(...) or func.func @main(...)
            match = re.match(r'(\s*)(func\.func\s+(?:public\s+)?@)(\w+)(\(.*)', line)
            if match:
                in_function = True
                # Build renamed function header
                func_lines.append(f"  func.func @{new_name}{match.group(4)}")
                brace_count = line.count('{') - line.count('}')
                continue
        
        if in_function:
            # Track braces to find function end
            brace_count += line.count('{') - line.count('}')
            
            # Clean and add the line
            clean_line = _strip_location_info(line)
            if clean_line.strip():
                # Preserve indentation but ensure minimum
                stripped = clean_line.lstrip()
                if stripped == '}':
                    func_lines.append("  }")
                else:
                    func_lines.append("    " + stripped)
            
            # Function ends when braces balance
            if brace_count == 0:
                break
    
    if not func_lines:
        return None
    
    return '\n'.join(func_lines)


def _strip_location_info(line: str) -> str:
    """Remove location attributes from MLIR line for cleaner output.
    
    Removes patterns like: loc(...) or loc(#loc...)
    """
    # Remove loc(...) patterns
    line = re.sub(r'\s*loc\([^)]*\)', '', line)
    return line


def get_function_name(func: Callable) -> str:
    """Get the name to use for a function in MLIR.
    
    Handles wrapped functions and returns the underlying name.
    """
    # Unwrap functools.wraps decorated functions
    while hasattr(func, '__wrapped__'):
        func = func.__wrapped__
    
    return func.__name__
