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


def compile_to_iree(
    mlir_text: str,
    backend: str = "llvm-cpu",
    input_type: str = "stablehlo",
    extra_args: Optional[list] = None,
) -> bytes:
    """Compile MLIR to IREE VM flatbuffer.
    
    Args:
        mlir_text: MLIR module text
        backend: Target backend ("llvm-cpu", "cuda")
        input_type: Input dialect type ("stablehlo", "auto", "tosa", "none")
        extra_args: Additional compiler args (e.g. ["--iree-cuda-target=sm_80"])
    
    Returns:
        Compiled VM flatbuffer bytes
    
    Raises:
        ImportError: If IREE is not available
    """
    if not IREE_AVAILABLE:
        raise ImportError("IREE compiler is required: pip install iree-base-compiler")
    
    if not input_type:
        print(mlir_text)
        return compile_str(
            mlir_text,
            target_backends=[backend],
        )
    
    return compile_str(
        mlir_text,
        target_backends=[backend],
        input_type=input_type,
        extra_args=extra_args or [],
    )


def load_module(vmfb: bytes, backend: str = "cpu") -> Tuple[Any, Any]:
    """Load compiled IREE module for execution.
    
    Args:
        vmfb: Compiled VM flatbuffer bytes
        backend: Runtime backend ("cpu" -> local-sync, "cuda" -> cuda)
    
    Returns:
        Tuple of (modules, device) for execution
    
    Raises:
        ImportError: If IREE runtime is not available
    """
    if not IREE_AVAILABLE:
        raise ImportError("IREE runtime is required: pip install iree-base-runtime")
    
    # Map backend to runtime device URI (per skill)
    device_uri = "local-sync" if backend == "llvm-cpu" else backend
    
    config = ireert.Config(device_uri)
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, vmfb)
    ctx.add_vm_module(vm_module)
    
    return ctx.modules, ctx.config.device
