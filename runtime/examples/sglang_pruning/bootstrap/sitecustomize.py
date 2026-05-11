"""Install approxMLIR's Triton dump hook inside SGLang pruning workers."""

from __future__ import annotations

import os
import sys


def _prepend(path: str | None) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _patch_sglang_jit_compiler() -> None:
    try:
        from pathlib import Path

        import sglang.jit_kernel.utils as jit_utils
    except Exception:
        return

    python_prefix = Path(sys.executable).resolve().parents[1]
    conda_gxx = python_prefix / "bin" / "x86_64-conda-linux-gnu-g++"
    if not conda_gxx.exists() or getattr(jit_utils, "_approx_ccbin_patched", False):
        return

    original = jit_utils._get_default_target_flags

    def with_conda_ccbin():
        flags = list(original())
        if "-ccbin" not in flags:
            flags.extend(["-ccbin", str(conda_gxx)])
        return flags

    jit_utils._get_default_target_flags = with_conda_ccbin
    jit_utils._approx_ccbin_patched = True


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
                approx_sglang_block_prune_linear_kernel_1,
                approx_sglang_compact_k_prune_linear_kernel_1,
            )

            backend = os.environ.get("APPROX_SGLANG_PRUNE_BACKEND", os.environ.get("APPROX_SGLANG_BACKEND", "triton_block_prune"))
            if backend == "triton_compact_k_prune":
                target_func = "sglang_compact_k_prune_linear_kernel"
                approx_kernel = approx_sglang_compact_k_prune_linear_kernel_1
            else:
                target_func = "sglang_block_prune_linear_kernel"
                approx_kernel = approx_sglang_block_prune_linear_kernel_1

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


_patch_sglang_jit_compiler()
_install_dump_hook()

try:
    import approx_pruning_patch  # noqa: F401
except Exception as exc:
    if os.environ.get("APPROX_SGLANG_PRUNING", "0") == "1":
        print(f"[approx-sglang-sitecustomize] pruning patch failed: {exc}", file=sys.stderr)
