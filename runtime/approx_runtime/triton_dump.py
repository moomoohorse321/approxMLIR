"""Triton TTIR dump hook.

Observe-only counterpart to :mod:`triton_hook`. Installs itself via the same
``add_stages_inspection_hook`` entry point, but never modifies the module — it
just writes each compiled kernel's TTIR plus correlation metadata to disk so
the kernels can later be mapped back to the model layer they came from.

Per-kernel output layout::

    <out_dir>/<func_name>__<shape_hash>.ttir
    <out_dir>/<func_name>__<shape_hash>.json

The ``.json`` sidecar carries ``func_name``, ``kernel_id``, ``shape_hash``,
``signature`` (raw ``tt.func`` header), parsed ``args`` (shape + dtype per
argument), Triton ``metadata`` / ``options`` (best-effort serialised),
``call_count``, ``source`` tag, and timestamps.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


__all__ = ["make_triton_dump_hook", "LaunchRecorder", "make_launch_recorder"]


class LaunchRecorder:
    """Append-only JSONL log of kernel launches, keyed by logical tag.

    Triton compiles one TTIR per ``(arg-type signature, constexpr)`` equivalence
    class, so multiple logical launches (e.g. four Qwen projections with
    different runtime N/K) can share a single TTIR dump. The recorder captures
    the per-launch semantic context the hook cannot see from MLIR alone, so
    downstream target selection can still answer "which projection does this
    compiled kernel serve."

    Sample-only helper; not part of the runtime API.
    """

    def __init__(self, out_dir: "os.PathLike[str] | str", source: str = "unknown") -> None:
        self.path = Path(out_dir) / "launches.jsonl"
        self.source = source
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        kernel_name: str,
        tag: Optional[str],
        params: Dict[str, Any],
    ) -> None:
        line = json.dumps(
            {
                "ts": time.time(),
                "kernel": kernel_name,
                "tag": tag,
                "params": _jsonable(params),
                "source": self.source,
            },
            sort_keys=True,
        )
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def make_launch_recorder(
    out_dir: "os.PathLike[str] | str", source: str = "unknown"
) -> LaunchRecorder:
    return LaunchRecorder(out_dir, source)


_TT_FUNC_HEADER_RE = re.compile(
    r"tt\.func\s+(?:(?:public|private)\s+)?@([A-Za-z_.$][\w.$]*)\s*\("
)
_TENSOR_RE = re.compile(r"tensor<([^>]+)>")
_PTR_RE = re.compile(r"!tt\.ptr<([^>]+)>")


def _matching_paren(text: str, open_idx: int) -> int:
    """Index of the ``)`` matching ``text[open_idx] == '('``. ``-1`` if none."""
    depth = 0
    i = open_idx
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _first_func_signature(
    mlir_text: str, preferred_name: Optional[str]
) -> Optional[tuple[str, str]]:
    """Return ``(func_name, signature_header)`` for the most relevant tt.func.

    The header includes everything from ``tt.func`` up to and including the
    closing ``)`` of the argument list, accounting for nested parens inside
    attribute dicts and ``loc(...)`` annotations that Triton emits.
    """
    candidates: list[tuple[str, str]] = []
    for m in _TT_FUNC_HEADER_RE.finditer(mlir_text):
        paren_open = m.end() - 1  # position of the '('
        paren_close = _matching_paren(mlir_text, paren_open)
        if paren_close < 0:
            continue
        header = mlir_text[m.start() : paren_close + 1]
        candidates.append((m.group(1), header))
    if not candidates:
        return None
    if preferred_name:
        for name, header in candidates:
            if name == preferred_name:
                return name, header
    return candidates[0]


def _split_top_level_commas(text: str) -> list[str]:
    """Split ``text`` on commas that sit at paren/brace/bracket depth 0."""
    out: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(text):
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _extract_arg_type(arg_text: str) -> Optional[str]:
    """Given one argument like ``%x: !tt.ptr<f32> {...} loc(...)``, return the type text."""
    colon = arg_text.find(":")
    if colon < 0:
        return None
    rest = arg_text[colon + 1 :]
    # Walk until we hit a top-level boundary: ' {' (attr dict) or ' loc(' (location).
    depth = 0
    end = len(rest)
    i = 0
    while i < len(rest):
        c = rest[i]
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        elif depth == 0 and c.isspace():
            tail = rest[i:].lstrip()
            if tail.startswith("{") or tail.startswith("loc("):
                end = i
                break
        i += 1
    return rest[:end].strip()


def _parse_arg_type(type_str: str) -> Dict[str, Any]:
    """Best-effort classification of an MLIR arg type into shape / dtype fields."""
    t = type_str.strip()
    tensor_m = _TENSOR_RE.match(t)
    if tensor_m:
        inner = tensor_m.group(1)
        parts = inner.split("x")
        if len(parts) == 1:
            return {"kind": "tensor", "shape": [], "dtype": parts[0]}
        *dims, dtype = parts
        shape: list = []
        for d in dims:
            try:
                shape.append(int(d))
            except ValueError:
                shape.append(d)
        return {"kind": "tensor", "shape": shape, "dtype": dtype}
    ptr_m = _PTR_RE.match(t)
    if ptr_m:
        return {"kind": "ptr", "shape": None, "dtype": ptr_m.group(1).strip()}
    return {"kind": "scalar", "shape": None, "dtype": t}


def _parse_signature_args(signature_text: str) -> list:
    paren_start = signature_text.find("(")
    paren_end = signature_text.rfind(")")
    if paren_start < 0 or paren_end < 0 or paren_end <= paren_start + 1:
        return []
    inside = signature_text[paren_start + 1 : paren_end]
    args = []
    for piece in _split_top_level_commas(inside):
        type_text = _extract_arg_type(piece)
        if type_text is None:
            continue
        args.append(_parse_arg_type(type_text))
    return args


def _shape_hash(args: list) -> str:
    """Hash the structural arg list so loc/attr jitter doesn't perturb it."""
    payload = json.dumps(args, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _kernel_id(mlir_text: str) -> str:
    return hashlib.sha256(mlir_text.encode("utf-8")).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-serialisable data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    for attr in ("_asdict", "__dict__"):
        member = getattr(value, attr, None)
        if callable(member):
            try:
                return _jsonable(member())
            except Exception:  # pragma: no cover
                break
        if isinstance(member, dict) and member:
            return _jsonable(member)
    return repr(value)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def make_triton_dump_hook(
    out_dir: os.PathLike | str,
    source: str = "unknown",
    stage_name: str = "make_ttir_dump",
    func_name_filter: Optional[Callable[[str], bool]] = None,
    verbose: bool = False,
) -> Callable:
    """Create a stages-inspection hook that dumps each kernel's TTIR.

    Parameters
    ----------
    out_dir:
        Directory to write ``.ttir`` / ``.json`` pairs into. Created if needed.
    source:
        Free-form tag stored in every JSON sidecar (e.g. ``"hf_torch_compile"``)
        so dumps from different input paths can be merged later without
        ambiguity.
    func_name_filter:
        Optional predicate over the kernel's ``tt.func`` name. Kernels for
        which it returns ``False`` are skipped (no files written, no counter
        bumped). Defaults to accepting everything.
    verbose:
        If true, print one line per dumped kernel.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    settings = {
        "out_dir": str(out_path.resolve()),
        "source": source,
        "stage_name": stage_name,
        "has_filter": func_name_filter is not None,
    }
    key = f"approx_triton_dump_hook::{json.dumps(settings, sort_keys=True)}"
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()

    seen_lock = threading.Lock()
    seen_counts: Dict[tuple[str, str], int] = {}

    def _write_dump(mlir_text: str, metadata: Any, options: Any) -> None:
        preferred = None
        if isinstance(metadata, dict):
            preferred = metadata.get("name")
        sig = _first_func_signature(mlir_text, preferred)
        if sig is None:
            if verbose:
                print("[approx-dump] no tt.func found, skipping", flush=True)
            return
        func_name, signature_text = sig
        if func_name_filter is not None and not func_name_filter(func_name):
            return
        args = _parse_signature_args(signature_text)
        shash = _shape_hash(args)
        file_stem = f"{_safe_filename(func_name)}__{shash}"
        ttir_path = out_path / f"{file_stem}.ttir"
        json_path = out_path / f"{file_stem}.json"

        key_tuple = (func_name, shash)
        with seen_lock:
            seen_counts[key_tuple] = seen_counts.get(key_tuple, 0) + 1
            call_count = seen_counts[key_tuple]

        now = time.time()
        if call_count == 1:
            ttir_path.write_text(mlir_text, encoding="utf-8")
            record = {
                "func_name": func_name,
                "kernel_id": _kernel_id(mlir_text),
                "shape_hash": shash,
                "signature": signature_text,
                "args": args,
                "metadata": _jsonable(metadata),
                "options": _jsonable(options),
                "source": source,
                "call_count": 1,
                "first_seen_ts": now,
                "last_seen_ts": now,
            }
        else:
            try:
                record = json.loads(json_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                record = {
                    "func_name": func_name,
                    "kernel_id": _kernel_id(mlir_text),
                    "shape_hash": shash,
                    "signature": signature_text,
                    "args": args,
                    "metadata": _jsonable(metadata),
                    "options": _jsonable(options),
                    "source": source,
                    "first_seen_ts": now,
                }
            record["call_count"] = call_count
            record["last_seen_ts"] = now

        tmp_path = json_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(json_path)

        if verbose:
            print(
                f"[approx-dump] {func_name} shape_hash={shash} call_count={call_count} -> {ttir_path}",
                flush=True,
            )

    def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
        if all(arg is None for arg in (stages, options, language, capability)):
            return key, key_hash

        def make_ttir_wrapper(mod, metadata, opt, cap):
            mod = self.make_ttir(mod, metadata, opt, cap)
            try:
                _write_dump(str(mod), metadata, opt)
            except Exception as exc:  # pragma: no cover
                # Dumping is observational: never let it break compilation.
                if verbose:
                    print(f"[approx-dump] dump failed: {exc!r}", flush=True)
            return mod

        stages["ttir"] = lambda src, metadata: make_ttir_wrapper(
            src, metadata, options, capability
        )
        return key, key_hash

    return inspect_stages_hook
