"""Parse C/C++ @approx comments and generate MLIR annotation ops."""

from dataclasses import dataclass, field
from typing import List, Optional
import re

__all__ = [
    "CppAnnotation",
    "AnnotationSyntaxError",
    "parse_cpp_annotations",
    "generate_cpp_annotation_mlir",
    "parse_and_generate",
]


class AnnotationSyntaxError(ValueError):
    """Raised when C++ annotation syntax is invalid."""


@dataclass
class CppAnnotation:
    """Parsed annotation from C/C++ source comment."""

    func_name: str
    transform_type: str
    state_indices: List[int]
    state_function: str
    thresholds: List[int]
    decisions: List[int]
    decision_values: List[int] = field(default_factory=list)
    thresholds_lower: List[int] = field(default_factory=list)
    thresholds_upper: List[int] = field(default_factory=list)
    line_number: int = 0


def parse_cpp_annotations(source: str) -> List[CppAnnotation]:
    """Parse all @approx annotations from C/C++ source text."""
    lines = source.splitlines()
    annotations: List[CppAnnotation] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "//" in line and "@approx:" in line:
            raw = line[line.index("@approx:") + len("@approx:"):].strip()
            if raw.startswith("decision_tree"):
                raw = raw[len("decision_tree"):].strip()

            if "{" in raw:
                block_lines, end_idx = _collect_block(lines, i)
                data = _parse_block(block_lines, i + 1)
                i = end_idx
            else:
                data = _parse_single_line(raw, i + 1)
                i += 1

            func_name, arg_count = _find_next_function(lines, i)
            ann = _build_annotation(data, func_name, arg_count, i + 1)
            annotations.append(ann)
        else:
            i += 1
    return annotations


def generate_cpp_annotation_mlir(
    annotations: List[CppAnnotation],
    module_name: Optional[str] = None,
) -> str:
    """Generate MLIR annotation ops from parsed C++ annotations."""
    lines: List[str] = []

    identity_funcs = []
    for ann in annotations:
        if ann.state_function.startswith("__identity_"):
            identity_funcs.append(ann.state_function)

    for name in sorted(set(identity_funcs)):
        lines.append(f"func.func @{name}(%arg0: i32) -> i32 {{")
        lines.append("  return %arg0 : i32")
        lines.append("}")
        lines.append("")

    for ann in annotations:
        thresholds = ann.thresholds
        decisions = ann.decisions
        num_thresholds = len(thresholds)
        lowers = ann.thresholds_lower or [0] * num_thresholds
        uppers = ann.thresholds_upper or [100] * num_thresholds
        decision_values = ann.decision_values or decisions

        state_indices_str = ", ".join(str(i) for i in ann.state_indices)
        thresholds_str = ", ".join(str(v) for v in thresholds)
        decisions_str = ", ".join(str(v) for v in decisions)
        lowers_str = ", ".join(str(v) for v in lowers)
        uppers_str = ", ".join(str(v) for v in uppers)
        decision_values_str = ", ".join(str(v) for v in decision_values)

        lines.append('"approx.util.annotation.decision_tree"() <{')
        lines.append(f'  func_name = "{ann.func_name}",')
        lines.append(f'  transform_type = "{ann.transform_type}",')
        lines.append(f"  state_indices = array<i64: {state_indices_str}>,")
        lines.append(f'  state_function = "{ann.state_function}",')
        lines.append(f"  num_thresholds = {num_thresholds} : i32,")
        lines.append(f"  thresholds_uppers = array<i32: {uppers_str}>,")
        lines.append(f"  thresholds_lowers = array<i32: {lowers_str}>,")
        lines.append(f"  decision_values = array<i32: {decision_values_str}>,")
        lines.append(f"  thresholds = array<i32: {thresholds_str}>,")
        lines.append(f"  decisions = array<i32: {decisions_str}>")
        lines.append("}> : () -> ()")
        lines.append("")

        if ann.transform_type == "func_substitute":
            lines.append('"approx.util.annotation.convert_to_call"() <{')
            lines.append(f'  func_name = "{ann.func_name}"')
            lines.append("}> : () -> ()")
            lines.append("")

    body = "\n".join(lines).rstrip() + "\n"
    if module_name:
        return f"module @{module_name} {{\n{body}}}\n"
    return body


def parse_and_generate(source: str, module_name: Optional[str] = None) -> str:
    """Parse C/C++ source and generate MLIR annotations."""
    annotations = parse_cpp_annotations(source)
    return generate_cpp_annotation_mlir(annotations, module_name)


def _collect_block(lines: List[str], start: int) -> tuple[list[str], int]:
    block_lines = []
    i = start
    opened = False
    while i < len(lines):
        line = lines[i]
        if "//" not in line:
            break
        comment = line.split("//", 1)[1].strip()
        if "@approx:" in comment:
            comment = comment[comment.index("@approx:") + len("@approx:"):].strip()
            if comment.startswith("decision_tree"):
                comment = comment[len("decision_tree"):].strip()
        if "{" in comment:
            opened = True
            comment = comment.split("{", 1)[1].strip()
        if "}" in comment:
            comment = comment.split("}", 1)[0].strip()
            if comment:
                block_lines.append(comment)
            return block_lines, i + 1
        if opened and comment:
            block_lines.append(comment)
        i += 1
    raise AnnotationSyntaxError("Unterminated @approx block")


def _parse_block(lines: List[str], line_number: int) -> dict:
    data = {}
    for line in lines:
        if not line:
            continue
        if line.endswith(","):
            line = line[:-1]
        if ":" not in line:
            raise AnnotationSyntaxError(
                f"Invalid annotation entry at line {line_number}: {line}"
            )
        key, value = [p.strip() for p in line.split(":", 1)]
        data[key] = _parse_value(value)
    return _normalize_keys(data, line_number)


def _parse_single_line(raw: str, line_number: int) -> dict:
    tokens = raw.split()
    if not tokens:
        raise AnnotationSyntaxError(f"Empty @approx annotation at line {line_number}")
    data = {"transform_type": tokens[0]}
    for tok in tokens[1:]:
        if "=" not in tok:
            continue
        key, value = tok.split("=", 1)
        data[key.strip()] = _parse_value(value.strip())
    return _normalize_keys(data, line_number)


def _normalize_keys(data: dict, line_number: int) -> dict:
    aliases = {
        "state": "state_indices",
        "state_fn": "state_function",
        "thresh": "thresholds",
        "dec": "decisions",
        "decision_vals": "decision_values",
        "thresholds_lower": "thresholds_lower",
        "thresholds_upper": "thresholds_upper",
    }
    normalized = {}
    for key, value in data.items():
        k = aliases.get(key, key)
        normalized[k] = value
    if "transform_type" not in normalized:
        raise AnnotationSyntaxError(
            f"Missing transform_type at line {line_number}"
        )
    return normalized


def _parse_value(value: str):
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [int(v.strip()) for v in inner.split(",") if v.strip()]
    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
    return value


def _find_next_function(lines: List[str], start: int) -> tuple[str, int]:
    signature = ""
    i = start
    depth = 0
    started = False
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("//"):
            i += 1
            continue
        signature += " " + line
        for ch in line:
            if ch == "(":
                depth += 1
                started = True
            elif ch == ")":
                depth -= 1
        if started and depth == 0 and ("{" in line or ";" in line):
            break
        i += 1

    signature = signature.strip()
    if not signature:
        raise AnnotationSyntaxError("Missing function signature after @approx")

    lparen = signature.find("(")
    if lparen == -1:
        raise AnnotationSyntaxError(f"Unable to parse function signature: {signature}")

    prefix = signature[:lparen]
    name_matches = list(re.finditer(r"\b([A-Za-z_][\w]*)\b", prefix))
    if not name_matches:
        raise AnnotationSyntaxError(f"Unable to parse function name: {signature}")
    func_name = name_matches[-1].group(1)

    depth = 0
    rparen = -1
    for idx in range(lparen, len(signature)):
        ch = signature[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                rparen = idx
                break
    if rparen == -1:
        raise AnnotationSyntaxError(f"Unable to parse function args: {signature}")

    args_str = signature[lparen + 1:rparen].strip()
    arg_count = _count_args(args_str)

    return func_name, arg_count


def _count_args(args_str: str) -> int:
    if not args_str or args_str == "void":
        return 0
    parts = []
    buf = []
    depth = 0
    for ch in args_str:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return len([p for p in parts if p])


def _build_annotation(
    data: dict, func_name: str, arg_count: int, line_number: int
) -> CppAnnotation:
    transform_type = data["transform_type"]

    state_indices = data.get("state_indices", [-1])
    if not isinstance(state_indices, list):
        state_indices = [state_indices]
    state_indices = [
        _normalize_index(idx, arg_count, line_number) for idx in state_indices
    ]

    thresholds = data.get("thresholds")
    decisions = data.get("decisions")
    if thresholds is None or decisions is None:
        raise AnnotationSyntaxError(
            f"Missing thresholds/decisions for {func_name} at line {line_number}"
        )
    if len(decisions) != len(thresholds) + 1:
        raise AnnotationSyntaxError(
            f"decisions length must be thresholds + 1 for {func_name} at line {line_number}"
        )

    decision_values = data.get("decision_values", [])
    if decision_values and len(decision_values) < len(decisions):
        raise AnnotationSyntaxError(
            f"decision_values length must be >= decisions for {func_name} at line {line_number}"
        )

    state_function = data.get("state_function")
    if not state_function or state_function == "identity":
        state_function = f"__identity_{func_name}"

    thresholds_lower = data.get("thresholds_lower", [])
    thresholds_upper = data.get("thresholds_upper", [])
    if thresholds_lower and len(thresholds_lower) != len(thresholds):
        raise AnnotationSyntaxError(
            f"thresholds_lower length must match thresholds for {func_name} at line {line_number}"
        )
    if thresholds_upper and len(thresholds_upper) != len(thresholds):
        raise AnnotationSyntaxError(
            f"thresholds_upper length must match thresholds for {func_name} at line {line_number}"
        )

    return CppAnnotation(
        func_name=func_name,
        transform_type=transform_type,
        state_indices=state_indices,
        state_function=state_function,
        thresholds=thresholds,
        decisions=decisions,
        decision_values=decision_values,
        thresholds_lower=thresholds_lower,
        thresholds_upper=thresholds_upper,
        line_number=line_number,
    )


def _normalize_index(idx: int, arg_count: int, line_number: int) -> int:
    if idx < 0:
        idx = arg_count + idx
    if idx < 0 or idx >= arg_count:
        raise AnnotationSyntaxError(
            f"state_indices out of bounds at line {line_number}"
        )
    return idx
