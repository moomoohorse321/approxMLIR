"""Install approxMLIR's Triton dump hook inside SGLang worker processes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys


def _prepend(path: str | None) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _has_ttir_function(mlir_text: str, func_name: str) -> bool:
    pattern = re.compile(rf"^\s*tt\.func\b[^@]*@{re.escape(func_name)}\b", re.MULTILINE)
    return pattern.search(mlir_text) is not None


def _make_composite_hook(hooks_by_func: dict[str, object]):
    hook_keys = {func: hook()[0] for func, hook in sorted(hooks_by_func.items())}
    key_payload = {"version": 2, "hooks": hook_keys}
    key = f"approx_sglang_multi_hook::{json.dumps(key_payload, sort_keys=True)}"
    key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()

    class _PrecomputedTTIR:
        def __init__(self, mod):
            self._mod = mod

        def make_ttir(self, src, metadata, options, capability):
            return self._mod

    def composite_hook(
        self=None,
        stages=None,
        options=None,
        language=None,
        capability=None,
    ):
        if all(arg is None for arg in (stages, options, language, capability)):
            return key, key_hash

        def make_ttir(src, metadata):
            mod = self.make_ttir(src, metadata, options, capability)
            selected = None
            metadata_name = metadata.get("name") if isinstance(metadata, dict) else None
            if metadata_name is not None:
                selected = hooks_by_func.get(metadata_name)
            if selected is None:
                mlir_text = str(mod)
                for func_name, hook in hooks_by_func.items():
                    if _has_ttir_function(mlir_text, func_name):
                        selected = hook
                        break
            if selected is None:
                return mod
            local_stages = {}
            selected(
                self=_PrecomputedTTIR(mod),
                stages=local_stages,
                options=options,
                language=language,
                capability=capability,
            )
            return local_stages["ttir"](src, metadata)

        stages["ttir"] = make_ttir
        return key, key_hash

    return composite_hook


def _install_dump_hook() -> None:
    out_dir = os.environ.get("APPROX_SGLANG_DUMP_OUT_DIR")
    if not out_dir:
        return

    _prepend(os.environ.get("APPROX_SOURCE_TRITON_PYTHON"))
    _prepend(os.environ.get("APPROX_EXAMPLES_ROOT"))

    import approx_runtime as ar
    from mixed_backend_config import (
        NO_SUBSTITUTE_BACKENDS,
        backend_or_none,
        parse_mixed_backend_rules,
        unique_in_order,
    )
    from triton import knobs

    def install_dump_hook() -> None:
        knobs.runtime.add_stages_inspection_hook = ar.make_triton_dump_hook(
            out_dir=out_dir,
            source="sglang_triton_worker",
            verbose=os.environ.get("DUMP_VERBOSE", "0") == "1",
        )

    if os.environ.get("APPROX_SGLANG_USE_SUBSTITUTE", "0") == "1":
        import approx_substitution_state as subst_state
        from approx_kernels import (
            approx_sglang_dynamic_w8a8_linear_kernel_1,
            approx_sglang_prequant_w8a8_linear_kernel_1,
            approx_sglang_sq_w4a16_linear_kernel_1,
            approx_sglang_w8a16_linear_kernel_1,
        )

        backend_targets = {
            "triton": (
                "sglang_dynamic_w8a8_linear_kernel",
                approx_sglang_dynamic_w8a8_linear_kernel_1,
            ),
            "triton_prequant": (
                "sglang_prequant_w8a8_linear_kernel",
                approx_sglang_prequant_w8a8_linear_kernel_1,
            ),
            "triton_sq_w4a16": (
                "sglang_sq_w4a16_linear_kernel",
                approx_sglang_sq_w4a16_linear_kernel_1,
            ),
            "triton_awq_w4a16": (
                "sglang_sq_w4a16_linear_kernel",
                approx_sglang_sq_w4a16_linear_kernel_1,
            ),
            "triton_w8a16": (
                "sglang_w8a16_linear_kernel",
                approx_sglang_w8a16_linear_kernel_1,
            ),
        }

        mixed_rules = parse_mixed_backend_rules(
            os.environ.get("APPROX_SGLANG_MIXED_BACKENDS", "")
        )
        if mixed_rules:
            backends = unique_in_order(
                backend
                for _, backend in mixed_rules
                if backend not in NO_SUBSTITUTE_BACKENDS
            )
        else:
            raw_backend = os.environ.get("APPROX_SGLANG_BACKEND", "triton")
            backend = backend_or_none(raw_backend)
            backends = (
                []
                if backend is None or raw_backend in NO_SUBSTITUTE_BACKENDS
                else [raw_backend]
            )

        hooks_by_func = {}
        for backend in backends:
            target_func, approx_kernel = backend_targets[backend]
            config = {
                "decision_tree": None,
                "safety_contract": None,
                "static_transform": ar.StaticTransform(
                    transform_type="func_substitute",
                    knob_val=1,
                    approx_kernel=approx_kernel,
                ),
            }
            hooks_by_func[target_func] = ar.make_triton_stages_hook(
                passes=ar.get_pipeline_for_config(
                    config,
                    workload=ar.WorkloadType.TRITON,
                ),
                plugin_path=os.environ["TRITON_PASS_PLUGIN_PATH"],
                stage_name="make_ttir_approx",
                func_name=target_func,
                config=config,
                extra_ttir_texts=subst_state.extra_ttir_texts,
                verbose=os.environ.get("DUMP_VERBOSE", "0") == "1",
            )

        if hooks_by_func:
            knobs.runtime.add_stages_inspection_hook = _make_composite_hook(
                hooks_by_func
            )
        else:
            install_dump_hook()
    else:
        install_dump_hook()
    print(
        f"[approx-sglang-sitecustomize] installed Triton hook: {out_dir}",
        file=sys.stderr,
    )


_install_dump_hook()

import approx_quant_patch  # noqa: F401
