"""approx-runtime: Python interface for ApproxMLIR compiler toolchain."""

import os as _os

# Triton C++ reads TRITON_PLUGIN_PATHS once at import; add ours to the list without clobbering others.
_plugin_path = _os.environ.get("TRITON_PASS_PLUGIN_PATH")
if _plugin_path:
    _existing = _os.environ.get("TRITON_PLUGIN_PATHS", "")
    _paths = _existing.split(":") if _existing else []
    if _plugin_path not in _paths:
        _paths.insert(0, _plugin_path)
        _os.environ["TRITON_PLUGIN_PATHS"] = ":".join(p for p in _paths if p)

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
from .triton_compiler import (
    TRITON_AVAILABLE,
    TritonCompilationError,
    compile_with_triton_plugin,
)
from .triton_hook import (
    make_triton_stages_hook,
)
from .triton_dump import (
    make_triton_dump_hook,
    make_launch_recorder,
    LaunchRecorder,
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
    'TRITON_AVAILABLE',
    'TritonCompilationError',
    'compile_with_triton_plugin',
    'make_triton_stages_hook',
    'make_triton_dump_hook',
    'make_launch_recorder',
    'LaunchRecorder',
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
