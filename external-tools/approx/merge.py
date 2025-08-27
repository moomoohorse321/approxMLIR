#!/usr/bin/env python3
# merge.py — merge MLIR file with a separate annotations file.
# Usage:
#   python merge.py input.mlir annotations.mlir -o merged.mlir
#   # or print to stdout:
#   python merge.py input.mlir annotations.mlir

import argparse
import io
import os
import re
import sys
from typing import Tuple

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

def find_module_open_brace_line_span(src: str) -> Tuple[int, int]:
    """
    Returns (line_start_index, line_end_index) for the line that contains
    the module body opening '{'. We then insert AFTER this line.
    Strategy:
      1) Find first line starting with 'module'
      2) Scan forward line-by-line until a line containing '{' is found
         (the last '{' on that line is treated as the body opener).
    """
    m = re.search(r'(?m)^\s*module\b', src)
    if not m:
        raise ValueError("Could not find 'module' in input MLIR.")
    i = m.start()
    n = len(src)
    while i < n:
        j = src.find("\n", i)
        if j == -1:
            j = n
        line = src[i:j]
        if "{" in line:
            # We consider the last '{' on this line to be the module body opener.
            # We still insert AFTER the line, so caller needs line end.
            return (i, j)
        i = j + 1
    raise ValueError("Could not find an opening '{' for the module body.")

def detect_body_indent(src: str, after_line_end: int) -> str:
    """
    Detect indentation used for operations inside the module body by
    looking at the first non-empty, non-'}' line after the opening brace line.
    Fallback to two spaces if nothing found.
    """
    n = len(src)
    i = after_line_end
    while i < n:
        j = src.find("\n", i)
        if j == -1:
            j = n
        line = src[i:j]
        if line.strip() and line.strip() != "}":
            m = re.match(r"[ \t]*", line)
            return m.group(0) if m else "  "
        i = j + 1
    return "  "

def normalize_annotations_block(annotations: str) -> str:
    # Normalize line endings and strip a potential BOM
    if annotations.startswith("\ufeff"):
        annotations = annotations.lstrip("\ufeff")
    # Ensure trailing newline so our concatenation stays clean.
    if not annotations.endswith("\n"):
        annotations += "\n"
    return annotations

def indent_block(block: str, indent: str) -> str:
    # Prefix each non-empty line with the module body indent.
    # Keep blank lines truly blank (no trailing spaces).
    out_lines = []
    for ln in block.splitlines():
        if ln.strip() == "":
            out_lines.append("")  # preserve blank line
        else:
            out_lines.append(f"{indent}{ln}")
    return "\n".join(out_lines) + "\n"

def main():
    ap = argparse.ArgumentParser(description="Merge MLIR file with annotations.")
    ap.add_argument("mlir", help="Path to the input MLIR file (with 'module { ... }').")
    ap.add_argument("annotations", help="Path to the annotations MLIR snippet (ops/comments).")
    ap.add_argument("-o", "--out", help="Output path (default: stdout).")
    args = ap.parse_args()

    mlir_text = read_text(args.mlir)
    ann_text = read_text(args.annotations)

    ann_text = normalize_annotations_block(ann_text)

    # Find insertion point: after the line that contains the module body '{'
    line_start, line_end = find_module_open_brace_line_span(mlir_text)
    insert_pos = line_end  # end of that line
    # If there's no newline at this point, add one for clean insertion
    if insert_pos < len(mlir_text) and mlir_text[insert_pos:insert_pos+1] == "\n":
        insert_after = insert_pos + 1
    else:
        # no newline; insert one
        mlir_text = mlir_text[:insert_pos] + "\n" + mlir_text[insert_pos:]
        insert_after = insert_pos + 1

    # Detect indentation used by the module body
    indent = detect_body_indent(mlir_text, insert_after)
    indented_ann = indent_block(ann_text, indent)

    merged = mlir_text[:insert_after] + indented_ann + mlir_text[insert_after:]

    if args.out:
        write_text(args.out, merged)
    else:
        sys.stdout.write(merged)

if __name__ == "__main__":
    main()
