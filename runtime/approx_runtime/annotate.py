"""Non-invasive annotation APIs for JAX workflows."""

from typing import Callable, Dict, Optional, Tuple, Any
import functools

from .knob import DecisionTree, SafetyContract, StaticTransform, get_config
from .export import export_module, export_to_mlir, get_function_name
from .mlir_gen import inject_annotations, collect_helper_functions

__all__ = [
    "with_config",
    "export_with_config",
    "export_module_with_configs",
]


def with_config(
    func: Callable,
    decision_tree: Optional[DecisionTree] = None,
    safety_contract: Optional[SafetyContract] = None,
    static_transform: Optional[StaticTransform] = None,
) -> Callable:
    """Return a new callable with approximation config attached."""
    if decision_tree and static_transform:
        raise ValueError("Cannot specify both decision_tree and static_transform")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._approx_config = {
        "decision_tree": decision_tree,
        "safety_contract": safety_contract,
        "static_transform": static_transform,
    }
    return wrapper


def export_with_config(
    func: Callable,
    example_args: Tuple[Any, ...],
    config: dict,
    func_name: Optional[str] = None,
) -> str:
    """Export JAX function to MLIR with config applied in one step."""
    if not config:
        return export_to_mlir(func, *example_args)

    name = func_name or get_function_name(func)

    sc = config.get("safety_contract")
    if sc is not None:
        raise ValueError(
            "export_with_config does not infer shapes for safety_contract helpers. "
            "Use export_module_with_configs instead."
        )

    helpers = collect_helper_functions(name, config)
    if not helpers:
        mlir = export_to_mlir(func, *example_args)
        return inject_annotations(mlir, name, config)

    functions = {name: (func, example_args)}
    dt = config.get("decision_tree")

    for helper_name, helper_fn in helpers:
        if dt and helper_name == dt.state_function.__name__:
            state_args = tuple(example_args[i] for i in dt.state_indices)
            functions[helper_name] = (helper_fn, state_args)
        else:
            functions[helper_name] = (helper_fn, example_args)

    mlir = export_module(functions)
    return inject_annotations(mlir, name, config)


def export_module_with_configs(
    functions: Dict[str, Tuple[Callable, Tuple[Any, ...], Optional[dict]]],
    module_name: Optional[str] = None,
) -> str:
    """Export multiple JAX functions with optional configs to single module."""
    export_dict = {name: (fn, args) for name, (fn, args, _) in functions.items()}
    mlir = export_module(export_dict, module_name=module_name)

    for name, (_, _, config) in functions.items():
        if config:
            mlir = inject_annotations(mlir, name, config)

    return mlir
