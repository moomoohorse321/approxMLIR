#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import approx_runtime as ar


#!/usr/bin/env python3
import argparse
import difflib
import os
import sys
import time
from typing import Dict, List, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

# Make approx_runtime importable without installing it.
RUNTIME_ROOT = os.environ.get(
    "APPROX_RUNTIME_ROOT",
    "/projects/beqg/approxMLIR/iree/third_party/approxMLIR/runtime",
)
if RUNTIME_ROOT not in sys.path:
    sys.path.insert(0, RUNTIME_ROOT)

import approx_runtime as ar



def _read_prompts(path: str) -> List[str]:
    prompts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def _string_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _reset_gemma_stats() -> None:
    if hasattr(gemma_iree, "_STATS"):
        gemma_iree._STATS.update(
            {"total_ms": 0.0, "transfer_ms": 0.0, "post_ms": 0.0, "count": 0}
        )


def _get_avg_compute_ms() -> float:
    stats = getattr(gemma_iree, "_STATS", None)
    if not stats or stats.get("count", 0) == 0:
        raise RuntimeError("No Gemma IREE stats recorded for avg_compute")
    avg_total = stats["total_ms"] / stats["count"]
    avg_transfer = stats["transfer_ms"] / stats["count"]
    avg_post = stats["post_ms"] / stats["count"]
    return avg_total - avg_transfer - avg_post


def configure_toolchain(approx_opt: str | None, polygeist_opt: str | None) -> None:
    toolchain = ar.ToolchainConfig()
    if approx_opt:
        toolchain.ml_opt_path = os.path.expanduser(approx_opt)
    if polygeist_opt:
        toolchain.cpp_opt_path = polygeist_opt
    ar.set_toolchain(toolchain)


def get_state(prompt_len: jax.Array) -> jax.Array:
    return prompt_len


@ar.Knob(
    decision_tree=ar.DecisionTree(
        state_function=get_state,
        state_indices=[0],
        thresholds=[128],
        decisions=[0, 1],
        transform_type="task_skipping",
        thresholds_lower=[8],
        thresholds_upper=[2048],
        decision_values=[0, 1],
    )
)
def selector_kernel(prompt_len: jax.Array) -> jax.Array:
    # Two branches so task_skipping can select which one to keep.
    return jax.lax.cond(
        prompt_len < 128,
        lambda _: jnp.int32(0),
        lambda _: jnp.int32(1),
        operand=None,
    )


def build_selector_mlir() -> Tuple[str, List[str], Dict]:
    config = ar.get_config(selector_kernel)
    prompt_shape = jax.ShapeDtypeStruct((), jnp.int32)
    functions = {
        "selector_kernel": (selector_kernel, (prompt_shape,), config),
        "get_state": (get_state, (prompt_shape,), None),
    }
    mlir = ar.export_module_with_configs(functions)
    pipeline = ar.get_pipeline_for_config(config)
    return mlir, pipeline, config


def compile_selector(
    mlir_annotated: str,
    pipeline: List[str],
    backend: str,
    out_dir: str,
    extra_args: List[str],
) -> Tuple[object, str]:
    os.makedirs(out_dir, exist_ok=True)
    compiled = ar.compile(mlir_annotated, passes=pipeline)
    vmfb = ar.compile_to_iree(
        compiled,
        backend=backend,
        input_type="stablehlo",
        extra_args=extra_args,
    )
    vmfb_path = os.path.join(out_dir, "selector.vmfb")
    with open(vmfb_path, "wb") as f:
        f.write(vmfb)
    modules, _ = ar.load_module(vmfb, backend=backend)
    return modules, vmfb_path


def select_variant(modules, prompt_len: int) -> int:
    result = modules.module["selector_kernel"](jnp.array(prompt_len, dtype=jnp.int32))
    if hasattr(result, "to_host"):
        result = result.to_host()
    return int(result)



def _run_opt(opt_path: str, input_mlir: str, passes: list[str], output_path: Path) -> str:
    cmd = [opt_path, str(input_mlir)] + [f"--{p}" for p in passes]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr}"
        )
    output_path.write_text(result.stdout)
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump MLIR and run passes stage-by-stage.")
    parser.add_argument(
        "--opt",
        default="./approx-opt",
        help="Path to approx-opt/iree-opt (default: ./approx-opt)",
    )
    parser.add_argument("--out-dir", default="selector_artifacts", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlir, pipeline, config = build_selector_mlir()
    base_path = out_dir / "selector_base.mlir"
    base_path.write_text(mlir)

    opt_path = args.opt
    if not Path(opt_path).is_file():
        fallback = out_dir.parent / "iree-opt"
        if fallback.is_file():
            opt_path = str(fallback.resolve())

    current_path = base_path
    for idx, p in enumerate(pipeline):
        stage_path = out_dir / f"selector_stage_{idx:02d}_{p}.mlir"
        try:
            _run_opt(opt_path, current_path, [p], stage_path)
        except RuntimeError as exc:
            err_path = out_dir / f"selector_stage_{idx:02d}_{p}.stderr.txt"
            err_path.write_text(str(exc))
            print(f"Stage {idx:02d} ({p}) failed. See {err_path}")
            return
        current_path = stage_path

    final_path = out_dir / "selector_final.mlir"
    final_path.write_text(current_path.read_text())

    print(f"Wrote base MLIR: {base_path}")
    for idx, p in enumerate(pipeline):
        print(f"Stage {idx:02d} ({p}): {out_dir / f'selector_stage_{idx:02d}_{p}.mlir'}")
    print(f"Final MLIR: {final_path}")


if __name__ == "__main__":
    main()
