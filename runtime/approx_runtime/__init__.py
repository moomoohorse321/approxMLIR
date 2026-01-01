"""approx-runtime: Python interface for ApproxMLIR compiler toolchain."""

from .knob import (
    DecisionTree,
    SafetyContract, 
    StaticTransform,
    Knob,
    get_config,
)

from .mlir_gen import (
    generate_annotations,
    inject_annotations,
    collect_helper_functions,
    needs_pre_emit_transform,
)

from .compiler import (
    compile,
    compile_file,
    PASS_PIPELINE,
    FUNC_SUBSTITUTE_PIPELINE,
    CompilationError,
    get_pipeline_for_config,
)

from .export import (
    export_to_mlir,
    export_module,
    get_function_name,
    JAX_AVAILABLE,
)

from .iree import (
    compile_to_iree,
    load_module,
    IREE_AVAILABLE,
)

__version__ = "0.1.0"

__all__ = [
    # Knob configuration
    'DecisionTree',
    'SafetyContract',
    'StaticTransform',
    'Knob',
    'get_config',
    # MLIR generation
    'generate_annotations',
    'inject_annotations',
    'collect_helper_functions',
    'needs_pre_emit_transform',
    # Compiler
    'compile',
    'compile_file',
    'PASS_PIPELINE',
    'FUNC_SUBSTITUTE_PIPELINE',
    'CompilationError',
    'get_pipeline_for_config',
    # JAX export
    'export_to_mlir',
    'export_module',
    'get_function_name',
    'JAX_AVAILABLE',
    # IREE
    'compile_to_iree',
    'load_module',
    'IREE_AVAILABLE',
]
