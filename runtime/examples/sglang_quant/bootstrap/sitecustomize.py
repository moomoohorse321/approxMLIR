"""Install approxMLIR's Triton dump hook inside SGLang worker processes."""

from __future__ import annotations

import os
import sys


def _prepend(path: str | None) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _install_dump_hook() -> None:
    out_dir = os.environ.get("APPROX_SGLANG_DUMP_OUT_DIR")
    if not out_dir:
        return

    _prepend(os.environ.get("APPROX_SOURCE_TRITON_PYTHON"))
    _prepend(os.environ.get("APPROX_EXAMPLES_ROOT"))

    try:
        import approx_runtime as ar
        from triton import knobs
    except Exception as exc:
        print(f"[approx-sglang-sitecustomize] hook install skipped: {exc}", file=sys.stderr)
        return

    if os.environ.get("APPROX_SGLANG_USE_SUBSTITUTE", "0") == "1":
        try:
            import approx_substitution_state as subst_state
            from approx_kernels import (
                approx_sglang_dynamic_w8a8_linear_kernel_1,
                approx_sglang_prequant_w8a8_linear_kernel_1,
                approx_sglang_sq_w4a16_linear_kernel_1,
                approx_sglang_w8a16_linear_kernel_1,
            )

            backend = os.environ.get("APPROX_SGLANG_BACKEND", "triton")
            if backend == "triton_prequant":
                target_func = "sglang_prequant_w8a8_linear_kernel"
                approx_kernel = approx_sglang_prequant_w8a8_linear_kernel_1
            elif backend == "triton_sq_w4a16":
                target_func = "sglang_sq_w4a16_linear_kernel"
                approx_kernel = approx_sglang_sq_w4a16_linear_kernel_1
            elif backend == "triton_w8a16":
                target_func = "sglang_w8a16_linear_kernel"
                approx_kernel = approx_sglang_w8a16_linear_kernel_1
            else:
                target_func = "sglang_dynamic_w8a8_linear_kernel"
                approx_kernel = approx_sglang_dynamic_w8a8_linear_kernel_1

            config = {
                "decision_tree": None,
                "safety_contract": None,
                "static_transform": ar.StaticTransform(
                    transform_type="func_substitute",
                    knob_val=1,
                    approx_kernel=approx_kernel,
                ),
            }
            knobs.runtime.add_stages_inspection_hook = ar.make_triton_stages_hook(
                passes=ar.get_pipeline_for_config(config, workload=ar.WorkloadType.TRITON),
                plugin_path=os.environ["TRITON_PASS_PLUGIN_PATH"],
                stage_name="make_ttir_approx",
                func_name=target_func,
                config=config,
                extra_ttir_texts=subst_state.extra_ttir_texts,
                verbose=os.environ.get("DUMP_VERBOSE", "0") == "1",
            )
        except Exception as exc:
            print(f"[approx-sglang-sitecustomize] substitute hook failed, falling back to dump: {exc}", file=sys.stderr)
            knobs.runtime.add_stages_inspection_hook = ar.make_triton_dump_hook(
                out_dir=out_dir,
                source="sglang_triton_worker",
                verbose=os.environ.get("DUMP_VERBOSE", "0") == "1",
            )
    else:
        knobs.runtime.add_stages_inspection_hook = ar.make_triton_dump_hook(
            out_dir=out_dir,
            source="sglang_triton_worker",
            verbose=os.environ.get("DUMP_VERBOSE", "0") == "1",
        )
    print(
        f"[approx-sglang-sitecustomize] installed Triton dump hook: {out_dir}",
        file=sys.stderr,
    )


_install_dump_hook()

try:
    import approx_quant_patch  # noqa: F401
except Exception as exc:
    if os.environ.get("APPROX_SGLANG_QUANT", "0") == "1":
        print(f"[approx-sglang-sitecustomize] quant patch failed: {exc}", file=sys.stderr)
