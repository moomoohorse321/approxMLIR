"""Generate approx.util.annotation.* MLIR ops from knob configuration."""

from typing import Optional, List, Callable, Set
import re

__all__ = [
    'generate_annotations', 
    'inject_annotations',
    'inject_annotations_text',
    'collect_helper_functions',
    'needs_pre_emit_transform',
]


def collect_helper_functions(func_name: str, config: dict) -> List[tuple]:
    """Collect all helper functions that need to be exported to MLIR.
    
    Returns list of (mlir_name, callable) tuples for functions that must
    be included in the MLIR module alongside the main kernel.
    
    For func_substitute, generates names like: approx_<func_name>_<knob_val>
    """
    helpers = []
    
    dt = config.get('decision_tree')
    if dt:
        # State function: exported with its original name
        helpers.append((dt.state_function.__name__, dt.state_function))
        
        # Approx kernels for func_substitute
        if dt.transform_type == "func_substitute":
            for knob_val, kernel in dt.approx_kernels.items():
                mlir_name = f"approx_{func_name}_{knob_val}"
                helpers.append((mlir_name, kernel))
    
    sc = config.get('safety_contract')
    if sc:
        # Checker and recover functions
        helpers.append((sc.checker.__name__, sc.checker))
        helpers.append((sc.recover.__name__, sc.recover))
    
    st = config.get('static_transform')
    if st and st.transform_type == "func_substitute" and st.approx_kernel:
        mlir_name = f"approx_{func_name}_{st.knob_val}"
        helpers.append((mlir_name, st.approx_kernel))
    
    return helpers


def needs_pre_emit_transform(config: dict) -> bool:
    """Check if func_substitute is used, requiring pre-emit-transform pass.
    
    When func_substitute is the transform type, we need:
    1. The convert_to_call annotation to refactor main func -> wrapper + __internal_
    2. The pre-emit-transform pass to run before emit-approx
    """
    dt = config.get('decision_tree')
    if dt and dt.transform_type == "func_substitute":
        return True
    
    st = config.get('static_transform')
    if st and st.transform_type == "func_substitute":
        return True
    
    return False


def generate_annotations(func_name: str, config: dict) -> str:
    """Generate approx.util.annotation.* ops as MLIR text.
    
    Args:
        func_name: Name of the function to annotate
        config: Approximation config from @Knob decorator
    
    Returns:
        MLIR text containing annotation ops
    """
    ops = []
    
    # For func_substitute, emit convert_to_call annotation first
    # This triggers pre-emit-transform to refactor: func -> wrapper calling __internal_func
    if needs_pre_emit_transform(config):
        ops.append(f'''  "approx.util.annotation.convert_to_call"() <{{
    func_name = "{func_name}"
  }}> : () -> ()''')
    
    # Decision tree annotation
    dt = config.get('decision_tree')
    if dt:
        state_indices_str = ', '.join(str(i) for i in dt.state_indices)
        thresholds_str = ', '.join(str(t) for t in dt.thresholds)
        decisions_str = ', '.join(str(d) for d in dt.decisions)
        thresholds_lower_str = ', '.join(str(t) for t in dt.thresholds_lower)
        thresholds_upper_str = ', '.join(str(t) for t in dt.thresholds_upper)
        decision_values_str = ', '.join(str(d) for d in dt.decisions)
        
        ops.append(f'''  "approx.util.annotation.decision_tree"() <{{
    func_name = "{func_name}",
    transform_type = "{dt.transform_type}",
    state_indices = array<i64: {state_indices_str}>,
    state_function = "{dt.state_function.__name__}",
    num_thresholds = {len(dt.thresholds)} : i32,
    thresholds_uppers = array<i32: {thresholds_upper_str}>,
    thresholds_lowers = array<i32: {thresholds_lower_str}>,
    decision_values = array<i32: {decision_values_str}>,
    thresholds = array<i32: {thresholds_str}>,
    decisions = array<i32: {decisions_str}>
  }}> : () -> ()''')
    
    # Safety contract annotation
    sc = config.get('safety_contract')
    if sc:
        checker_name = sc.checker.__name__
        recover_name = sc.recover.__name__
        
        ops.append(f'''  "approx.util.annotation.try"() <{{
    func_name = "{func_name}",
    recover = "{recover_name}",
    checker = "{checker_name}"
  }}> : () -> ()''')
    
    # Static transform annotation
    st = config.get('static_transform')
    if st:
        ops.append(f'''  "approx.util.annotation.transform"() <{{
    func_name = "{func_name}",
    transform_type = "{st.transform_type}",
    knob_val = {st.knob_val} : i32
  }}> : () -> ()''')
    
    return '\n'.join(ops)


def inject_annotations(mlir_text: str, func_name: str, config: dict) -> str:
    """Insert annotation ops at the start of the MLIR module.
    
    Args:
        mlir_text: Original MLIR module text
        func_name: Name of the function to annotate
        config: Approximation config from @Knob decorator
    
    Returns:
        MLIR text with annotations inserted after 'module ... {'
    """
    if not config:
        return mlir_text
    
    annotations = generate_annotations(func_name, config)
    if not annotations:
        return mlir_text
    
    # Find module opening and insert annotations
    # Handle patterns:
    #   module {
    #   module @name {
    #   module @name attributes {...} {
    module_pattern = r'(module\s*(?:@[\w]+\s*)?(?:attributes\s*\{[^}]*\}\s*)?\{)'
    match = re.search(module_pattern, mlir_text)
    
    if match:
        insert_pos = match.end()
        return mlir_text[:insert_pos] + '\n' + annotations + mlir_text[insert_pos:]
    
    # Fallback: prepend to file
    return annotations + '\n' + mlir_text


def inject_annotations_text(mlir_text: str, annotations_text: str) -> str:
    """Insert raw annotation ops at the start of the MLIR module.

    Args:
        mlir_text: Original MLIR module text
        annotations_text: MLIR ops to insert (no module wrapper)

    Returns:
        MLIR text with annotations inserted after 'module ... {'
    """
    if not annotations_text.strip():
        return mlir_text

    annotations = annotations_text
    if annotations.startswith("\ufeff"):
        annotations = annotations.lstrip("\ufeff")
    if not annotations.endswith("\n"):
        annotations += "\n"

    module_pattern = r'(module\s*(?:@[\w]+\s*)?(?:attributes\s*\{[^}]*\}\s*)?\{)'
    match = re.search(module_pattern, mlir_text)
    if not match:
        return annotations + mlir_text

    insert_pos = match.end()

    # Detect indentation from the first non-empty line after module opener.
    indent = "  "
    tail = mlir_text[insert_pos:]
    for line in tail.splitlines():
        if line.strip() and line.strip() != "}":
            indent = re.match(r"[ \t]*", line).group(0)
            break

    indented = "\n".join(
        (indent + ln) if ln.strip() else ""
        for ln in annotations.splitlines()
    )
    if not indented.endswith("\n"):
        indented += "\n"

    return mlir_text[:insert_pos] + "\n" + indented + mlir_text[insert_pos:]
