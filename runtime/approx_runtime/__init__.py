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
    inject_annotations_text,
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
    WorkloadType,
    ToolchainConfig,
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

# Auto-tuning support (requires: pip install opentuner)

from .tuner_config import (
    TunableParam,
    MLIRConfigManager,
)

from .tuner import (
    ApproxTunerInterface,
    tune,
    create_tuner_arg_parser,
    tune_jax,
    tune_cpp,
)

from .toolchain import (
    get_toolchain,
    set_toolchain,
)

from .cpp_annotation import (
    CppAnnotation,
    AnnotationSyntaxError,
    parse_cpp_annotations,
    generate_cpp_annotation_mlir,
    parse_and_generate,
)

from .cpp_pipeline import (
    CgeistConfig,
    compile_cpp_source,
)

from .annotate import (
    with_config,
    export_with_config,
    export_module_with_configs,
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
    'inject_annotations_text',
    'collect_helper_functions',
    'needs_pre_emit_transform',
    # Compiler
    'compile',
    'compile_file',
    'PASS_PIPELINE',
    'FUNC_SUBSTITUTE_PIPELINE',
    'CompilationError',
    'get_pipeline_for_config',
    'WorkloadType',
    'ToolchainConfig',
    # JAX export
    'export_to_mlir',
    'export_module',
    'get_function_name',
    'JAX_AVAILABLE',
    # IREE
    'compile_to_iree',
    'load_module',
    'IREE_AVAILABLE',
    # Auto-tuning
    'TUNER_AVAILABLE',
    'TunableParam',
    'MLIRConfigManager',
    'ApproxTunerInterface',
    'tune',
    'create_tuner_arg_parser',
    'tune_jax',
    'tune_cpp',
    # Toolchain
    'get_toolchain',
    'set_toolchain',
    # C++ annotations
    'CppAnnotation',
    'AnnotationSyntaxError',
    'parse_cpp_annotations',
    'generate_cpp_annotation_mlir',
    'parse_and_generate',
    # C++ pipeline
    'CgeistConfig',
    'compile_cpp_source',
    # Non-invasive JAX API
    'with_config',
    'export_with_config',
    'export_module_with_configs',
]
