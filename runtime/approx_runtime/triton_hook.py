"""Triton stages hook integration for ApproxMLIR pass plugins."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from .mlir_gen import inject_annotations, needs_pre_emit_transform
from .toolchain import WorkloadType, get_toolchain
from .triton_compiler import TritonCompilationError

__all__ = [
    "make_triton_stages_hook",
]

_FUNC_HEADER_RE = re.compile(r"^\s*tt\.func\b[^@]*@([A-Za-z_.$][\w.$]*)")
_TT_FUNC_VIS_RE = re.compile(r"(\btt\.func)\s+(public|private)\s+@([A-Za-z_.$][\w.$]*)")
_LOC_ATTR_RE_SIMPLE = re.compile(r"\bloc\(#loc[0-9A-Za-z_.$]*\)")
_LOC_ATTR_RE_NAMED = re.compile(r'\bloc\("[^"]*"\(#loc[0-9A-Za-z_.$]*\)\)')
_LOC_ALIAS_DEF_RE = re.compile(r"^\s*#loc[0-9A-Za-z_.$]*\s*=.*$", re.MULTILINE)


def _has_ttir_function(mlir_text: str, func_name: str) -> bool:
    pattern = re.compile(rf"^\s*tt\.func\b[^@]*@{re.escape(func_name)}\b", re.MULTILINE)
    return pattern.search(mlir_text) is not None


def _append_module_defs(mlir_text: str, defs: List[str]) -> str:
    if not defs:
        return mlir_text
    idx = mlir_text.rfind("}")
    if idx < 0:
        raise TritonCompilationError("Invalid TTIR module text: missing closing '}'")
    prefix = mlir_text[:idx].rstrip()
    suffix = mlir_text[idx:]
    return prefix + "\n" + "\n\n".join(defs) + "\n" + suffix


def _extract_all_ttir_function_blocks(mlir_text: str) -> List[tuple[str, str]]:
    lines = mlir_text.splitlines()
    collecting = False
    depth = 0
    current_name = ""
    out: List[str] = []
    blocks: List[tuple[str, str]] = []

    for line in lines:
        if not collecting:
            m = _FUNC_HEADER_RE.match(line)
            if not m:
                continue
            collecting = True
            current_name = m.group(1)
            out = [line]
            depth = line.count("{") - line.count("}")
            if depth <= 0:
                blocks.append((current_name, "\n".join(out)))
                collecting = False
            continue

        out.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            blocks.append((current_name, "\n".join(out)))
            collecting = False
            current_name = ""
            out = []

    return blocks


def _merge_extra_ttir_functions(
    mlir_text: str, extra_ttir_texts: List[str]
) -> str:
    defs: List[str] = []
    seen = set()
    for extra in extra_ttir_texts:
        for name, block in _extract_all_ttir_function_blocks(extra):
            if name in seen or _has_ttir_function(mlir_text, name):
                continue
            seen.add(name)
            defs.append(block)
    return _append_module_defs(mlir_text, defs)


def _strip_location_attributes(mlir_text: str) -> str:
    # Balanced-paren walk: flat regex can't match nested loc(callsite(...)) / loc("name"(...)).
    out: list[str] = []
    i = 0
    n = len(mlir_text)
    while i < n:
        if (
            mlir_text.startswith("loc(", i)
            and (i == 0 or not (mlir_text[i - 1].isalnum() or mlir_text[i - 1] == "_"))
        ):
            depth = 1
            j = i + 4
            while j < n and depth > 0:
                ch = mlir_text[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                j += 1
            if out and out[-1] == " ":
                out.pop()
            i = j
            continue
        out.append(mlir_text[i])
        i += 1
    stripped = "".join(out)
    stripped = _LOC_ALIAS_DEF_RE.sub("", stripped)
    return stripped


def _normalize_ttir_visibility(mlir_text: str, entry_func_name: str) -> str:
    def repl(match):
        func_name = match.group(3)
        vis = "public" if func_name == entry_func_name else "private"
        return f"{match.group(1)} {vis} @{func_name}"

    return _TT_FUNC_VIS_RE.sub(repl, mlir_text)


def _parse_mlir_text(triton_ir, mlir_text: str, context):
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False) as f:
        f.write(mlir_text)
        path = f.name
    module = triton_ir.parse_mlir_module(path, context)
    Path(path).unlink(missing_ok=True)
    if not hasattr(module, "context"):
        module.context = context
    return module


def _required_func_substitute_helpers(config: dict, target_func_name: str) -> List[str]:
    needed_knobs = set()
    dt = config.get("decision_tree")
    if dt and getattr(dt, "transform_type", None) == "func_substitute":
        for value in getattr(dt, "decisions", []):
            if value:
                needed_knobs.add(int(value))
    st = config.get("static_transform")
    if st and getattr(st, "transform_type", None) == "func_substitute":
        knob_val = int(getattr(st, "knob_val", 0))
        if knob_val:
            needed_knobs.add(knob_val)
    return [f"approx_{target_func_name}_{knob}" for knob in sorted(needed_knobs)]


def _validate_required_helpers(mlir_text: str, config: dict, target_func_name: str) -> None:
    dt = config.get("decision_tree")
    if dt:
        state_name = dt.state_function.__name__
        if not _has_ttir_function(mlir_text, state_name):
            raise TritonCompilationError(
                f"DecisionTree state function '{state_name}' not found in TTIR module"
            )
    sc = config.get("safety_contract")
    if sc:
        checker_name = sc.checker.__name__
        recover_name = sc.recover.__name__
        missing = [
            name for name in (checker_name, recover_name) if not _has_ttir_function(mlir_text, name)
        ]
        if missing:
            raise TritonCompilationError(
                f"SafetyContract helper functions not found in TTIR module: {missing}"
            )
    missing_approx_helpers = [
        name
        for name in _required_func_substitute_helpers(config, target_func_name)
        if not _has_ttir_function(mlir_text, name)
    ]
    if missing_approx_helpers:
        raise TritonCompilationError(
            "func_substitute requires explicit helper kernels in TTIR; "
            f"missing: {missing_approx_helpers}"
        )


def make_triton_stages_hook(
    passes: Optional[List[str]] = None,
    plugin_path: Optional[str] = None,
    stage_name: str = "make_ttir_approx",
    enable_debug: bool = False,
    func_name: Optional[str] = None,
    config: Optional[dict] = None,
    extra_ttir_texts: Optional[List[str]] = None,
    run_passes_sequentially: bool = True,
    verbose: bool = False,
) -> Callable:
    """Create a Triton stages-inspection hook that injects approx passes at TTIR.

    The returned callable matches Triton's `add_stages_inspection_hook` contract.
    """
    settings = {
        "passes": list(passes) if passes is not None else None,
        "plugin_path": plugin_path,
        "stage_name": stage_name,
        "enable_debug": enable_debug,
        "func_name": func_name,
        "config": config,
        "extra_ttir_texts": extra_ttir_texts,
        "run_passes_sequentially": run_passes_sequentially,
        "verbose": verbose,
    }
    key = f"approx_triton_hook::{json.dumps(settings, sort_keys=True, default=repr)}"
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    keepalive_contexts = []

    def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
        if all(arg is None for arg in (stages, options, language, capability)):
            return key, key_hash
        def make_ttir_wrapper(mod, metadata, opt, cap):
            mod = self.make_ttir(mod, metadata, opt, cap)
            original_context = getattr(mod, "context", None)
            target_func_name = None
            configured_passes = list(passes or get_toolchain().get_pipeline(
                WorkloadType.TRITON, config
            ))
            if (
                config
                and needs_pre_emit_transform(config)
                and "pre-emit-transform" not in configured_passes
            ):
                configured_passes = ["pre-emit-transform"] + configured_passes
            selected_plugin = (
                plugin_path
                or get_toolchain().triton_plugin_path
                or os.environ.get("TRITON_PASS_PLUGIN_PATH", "")
            )
            if not selected_plugin:
                raise TritonCompilationError(
                    "TRITON_PASS_PLUGIN_PATH is required for Triton stage hook"
                )

            os.environ["TRITON_PASS_PLUGIN_PATH"] = selected_plugin
            # Triton C++ reads TRITON_PLUGIN_PATHS (`:`-separated); add ours without clobbering others.
            _existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
            _paths = _existing.split(":") if _existing else []
            if selected_plugin not in _paths:
                _paths.insert(0, selected_plugin)
                os.environ["TRITON_PLUGIN_PATHS"] = ":".join(p for p in _paths if p)
            from triton._C.libtriton import ir as triton_ir
            from triton._C.libtriton import passes as triton_passes
            context = original_context
            needs_reparse = context is not None
            if context is None and hasattr(triton_ir, "context"):
                context = triton_ir.context()
                keepalive_contexts.append(context)
                needs_reparse = True
            if context is None:
                raise TritonCompilationError(
                    "Cannot determine Triton IR context for stage hook"
                )
            # Plugin dialects (approx.*) must be registered into whichever
            # context we ended up with — including a context Triton built
            # before the plugin was loaded. load_dialects is additive, so
            # calling it on an already-loaded context is safe.
            if hasattr(triton_ir, "load_dialects"):
                triton_ir.load_dialects(context)
            if config:
                metadata_name = metadata.get("name") if isinstance(metadata, dict) else None
                target_func_name = func_name or metadata_name
                if not target_func_name:
                    raise TritonCompilationError(
                        "func_name is required when config is provided and metadata has no name"
                    )
                if not _has_ttir_function(str(mod), target_func_name):
                    return mod
                annotated_mlir = inject_annotations(str(mod), target_func_name, config)
                if extra_ttir_texts:
                    annotated_mlir = _merge_extra_ttir_functions(
                        annotated_mlir, extra_ttir_texts
                    )
                _validate_required_helpers(annotated_mlir, config, target_func_name)
                annotated_mlir = _strip_location_attributes(annotated_mlir)
                annotated_mlir = _normalize_ttir_visibility(
                    annotated_mlir, target_func_name
                )
                if verbose:
                    Path("/tmp/approx_triton_hook_input.mlir").write_text(
                        annotated_mlir, encoding="utf-8"
                    )
                    print(
                        "[approx-hook] wrote /tmp/approx_triton_hook_input.mlir",
                        flush=True,
                    )
                    print(
                        f"[approx-hook] passes={configured_passes}",
                        flush=True,
                    )
                mod = _parse_mlir_text(triton_ir, annotated_mlir, context)
            elif needs_reparse:
                mod = _parse_mlir_text(triton_ir, str(mod), context)

            if run_passes_sequentially:
                for index, pass_name in enumerate(configured_passes):
                    pass_fn = getattr(triton_passes.plugin, pass_name, None)
                    if pass_fn is None:
                        raise TritonCompilationError(
                            f"Plugin pass '{pass_name}' not found in Triton stages hook"
                        )
                    if verbose:
                        print(
                            f"[approx-hook] running pass {index + 1}/{len(configured_passes)}: {pass_name}",
                            flush=True,
                        )
                    pass_manager = triton_ir.pass_manager(context)
                    if enable_debug:
                        pass_manager.enable_debug()
                    pass_fn(pass_manager)
                    pass_manager.run(mod, f"{stage_name}:{pass_name}")
                    if verbose:
                        dump_path = Path(
                            f"/tmp/approx_triton_hook_after_{index:02d}_{pass_name}.mlir"
                        )
                        dump_path.write_text(str(mod), encoding="utf-8")
                        print(f"[approx-hook] wrote {dump_path}", flush=True)
            else:
                pass_manager = triton_ir.pass_manager(context)
                if enable_debug:
                    pass_manager.enable_debug()
                for pass_name in configured_passes:
                    pass_fn = getattr(triton_passes.plugin, pass_name, None)
                    if pass_fn is None:
                        raise TritonCompilationError(
                            f"Plugin pass '{pass_name}' not found in Triton stages hook"
                        )
                    pass_fn(pass_manager)
                pass_manager.run(mod, stage_name)

            if target_func_name:
                normalized_mlir = _normalize_ttir_visibility(str(mod), target_func_name)
                mod = _parse_mlir_text(triton_ir, normalized_mlir, context)
                if verbose:
                    dump_path = Path("/tmp/approx_triton_hook_after_final_normalize.mlir")
                    dump_path.write_text(str(mod), encoding="utf-8")
                    print(f"[approx-hook] wrote {dump_path}", flush=True)
            return mod

        stages["ttir"] = lambda src, metadata: make_ttir_wrapper(
            src, metadata, options, capability
        )
        return key, key_hash

    return inspect_stages_hook
