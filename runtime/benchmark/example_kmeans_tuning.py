#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import itertools
import math
import os
import re
import sys

import approx_runtime as ar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_common import (  # noqa: E402
    default_cgeist_config,
    default_source_path,
    default_toolchain,
    compile_cpp_path_to_annotated_mlir,
    compile_mlir_to_native_exec,
    run_exec,
    result_logger,
    exact_decision_config,
)

def parse_kernel_out(
    data_string: str,
) -> tuple[float | None, dict[int, list[float]]]:
    time_ms: float | None = None
    centroids: dict[int, list[float]] = {}
    time_pattern = re.compile(r"\bK-means completed in ([\d.]+)\s*ms\b")
    centroid_pattern = re.compile(r"Centroid (\\d+): \\((.*?)\\)")
    for line in data_string.splitlines():
        time_match = time_pattern.search(line)
        if time_match:
            time_ms = float(time_match.group(1))
        centroid_match = centroid_pattern.search(line)
        if centroid_match:
            cluster_index = int(centroid_match.group(1))
            coords_str = centroid_match.group(2)
            coords = [float(c) for c in re.findall(r"[\d.-]+", coords_str)]
            centroids[cluster_index] = coords
    return time_ms, centroids


def compute_similarity(gt_out: dict[int, list[float]], approx_out: dict[int, list[float]]) -> float:
    if len(gt_out) != len(approx_out) or not gt_out:
        return 0.0
    gt_keys = sorted(gt_out.keys())
    approx_keys = sorted(approx_out.keys())
    best_mapping = {}
    min_total_dist = float("inf")
    for p in itertools.permutations(approx_keys):
        current_mapping = dict(zip(gt_keys, p))
        current_total_dist = 0.0
        for gt_key, approx_key in current_mapping.items():
            dist_sq = sum(
                (a - b) ** 2 for a, b in zip(gt_out[gt_key], approx_out[approx_key])
            )
            current_total_dist += math.sqrt(dist_sq)
        if current_total_dist < min_total_dist:
            min_total_dist = current_total_dist
            best_mapping = current_mapping
    y_exact = [coord for key in gt_keys for coord in gt_out[key]]
    y_approx = [coord for key in gt_keys for coord in approx_out[best_mapping[key]]]
    diff_sum_sq = sum((e - a) ** 2 for e, a in zip(y_exact, y_approx))
    norm_diff = math.sqrt(diff_sum_sq)
    exact_sum_sq = sum(e ** 2 for e in y_exact)
    norm_exact = math.sqrt(exact_sum_sq)
    if norm_exact == 0.0:
        return 1.0 if norm_diff == 0.0 else 0.0
    return 1.0 - (norm_diff / norm_exact)


def get_gt(fname: Path) -> dict[int, list[float]]:
    with fname.open("r", errors="ignore") as f:
        s = f.read()
    _, gt = parse_kernel_out(s)
    return gt


def main() -> None:
    cpp_path = Path(os.environ.get("KMEANS_CPP_PATH", str(default_source_path("approx_kmeans.c"))))
    if not cpp_path.exists():
        raise FileNotFoundError(f"kmeans source not found: {cpp_path}")

    toolchain = default_toolchain()
    cgeist_config = default_cgeist_config()
    annotated_mlir = compile_cpp_path_to_annotated_mlir(
        cpp_path,
        cgeist_config=cgeist_config,
        toolchain=toolchain,
    )

    manager = ar.MLIRConfigManager()
    params = manager.parse_annotations(annotated_mlir)

    argv = ["-n", "200", "-d", "2", "-k", "5", "-i", "5", "-s", "42"]
    gt_path = Path(os.environ.get("KMEANS_GT_PATH", str(Path(__file__).resolve().parent / "gt.txt")))
    if not gt_path.exists():
        exact_cfg = exact_decision_config(params)
        exact_mlir = manager.apply_config(annotated_mlir, exact_cfg)
        exact_exec = compile_mlir_to_native_exec(
            exact_mlir, cgeist_config=cgeist_config, toolchain=toolchain, tag="kmeans"
        )
        exact_result = run_exec(exact_exec, argv)
        if exact_result.returncode != 0:
            raise RuntimeError(exact_result.stderr)
        gt_path.write_text(exact_result.stdout, encoding="utf-8", errors="ignore")
    gt_centroids = get_gt(gt_path)

    def evaluate_fn(config: dict) -> tuple[float, float]:
        modified = manager.apply_config(annotated_mlir, config)
        exec_path = compile_mlir_to_native_exec(
            modified, cgeist_config=cgeist_config, toolchain=toolchain, tag="kmeans"
        )
        result = run_exec(exec_path, argv)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        time_ms, approx_centroids = parse_kernel_out(result.stdout)
        accuracy = compute_similarity(gt_centroids, approx_centroids)
        if time_ms is None:
            raise RuntimeError("kmeans: failed to parse computation time from stdout")
        return time_ms, accuracy

    result = ar.tune(
        mlir_source=annotated_mlir,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=float(os.environ.get("ACCURACY_THRESHOLD", "0.9")),
        time_budget=20,
        result_callback=result_logger("kmeans"),
    )

    print(f"Found {len(params)} tunable params")
    print(f"Best config: {result['best_config']}")


if __name__ == "__main__":
    main()
