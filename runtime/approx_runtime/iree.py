"""IREE compilation and deployment."""

from typing import Optional, Any, Tuple

__all__ = ['compile_to_iree', 'load_module', 'IREE_AVAILABLE']

# IREE imports with graceful fallback (per jax-mlir-iree skill)
try:
    from iree.compiler import compile_str
    from iree import runtime as ireert
    IREE_AVAILABLE = True
except ImportError:
    compile_str = None
    ireert = None
    IREE_AVAILABLE = False


def compile_to_iree(mlir_text: str, backend: str = "llvm-cpu", **kwargs) -> bytes:
    """Compile MLIR to IREE VM flatbuffer.
    
    Args:
        mlir_text: MLIR module text
        backend: Target backend ("llvm-cpu", "cuda")
        **kwargs: Additional args passed to iree.compiler.compile_str
                  (e.g. input_type="stablehlo", extra_args=[...])
    
    Returns:
        Compiled VM flatbuffer bytes
    
    Raises:
        ImportError: If IREE is not available
    """
    if not IREE_AVAILABLE:
        raise ImportError("IREE compiler is required: pip install iree-base-compiler")
    
    return compile_str(mlir_text, target_backends=[backend], **kwargs)


def load_module(vmfb: bytes, backend: str = "cpu") -> Tuple[Any, Any]:
    """Load compiled IREE module for execution.
    
    Args:
        vmfb: Compiled VM flatbuffer bytes
        backend: Runtime backend ("cpu", "llvm-cpu" -> local-sync, "cuda" -> cuda)
    
    Returns:
        Tuple of (modules, device) for execution
    
    Raises:
        ImportError: If IREE runtime is not available
    """
    if not IREE_AVAILABLE:
        raise ImportError("IREE runtime is required: pip install iree-base-runtime")
    
    # Map backend to runtime device URI
    if backend in ("cpu", "llvm-cpu"):
        device_uri = "local-sync"
    else:
        device_uri = backend
    
    config = ireert.Config(device_uri)
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)
    
    return ctx.modules, ctx.config.device