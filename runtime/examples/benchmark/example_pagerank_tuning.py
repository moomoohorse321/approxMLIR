#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import random
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

def parse_pr_data(data_string: str) -> tuple[float | None, dict[int, float]]:
    pr_values: dict[int, float] = {}
    lines = data_string.strip().split("\\n")
    for i, line in enumerate(lines):
        if line.startswith("Time:"):
            lines = lines[i:]
            break
    execution_time = None
    if lines and lines[0].startswith("Time:"):
        time_line = lines[0]
        execution_time = float(time_line.split(":")[1].split()[0])
    for line in lines:
        if line.startswith("pr("):
            parts = line.split("=")
            index_str = parts[0].strip()
            index = int(index_str[3:-1])
            value = float(parts[1].strip())
            pr_values[index] = value
    return execution_time, pr_values


def get_ranks(pr_values: dict[int, float]) -> dict[int, float]:
    items = sorted(pr_values.items(), key=lambda x: (-x[1], x[0]))
    ranks: dict[int, float] = {}
    i = 0
    n = len(items)
    while i < n:
        j = i + 1
        while j < n and items[j][1] == items[i][1]:
            j += 1
        avg_rank = (i + (j - 1)) / 2.0
        for k in range(i, j):
            node_id = items[k][0]
            ranks[node_id] = avg_rank
        i = j
    return ranks


def compute_similarity(pr_values1: dict[int, float], pr_values2: dict[int, float]) -> float:
    ranks1 = get_ranks(pr_values1)
    ranks2 = get_ranks(pr_values2)
    common_keys = set(ranks1) & set(ranks2)
    N = len(common_keys)
    if N == 0:
        return 0.0
    if N == 1:
        return 1.0
    rank_diff_sum = sum(abs(ranks1[k] - ranks2[k]) for k in common_keys)
    if N % 2 == 0:
        max_diff = (N * N) / 2.0
    else:
        max_diff = (N * N - 1) / 2.0
    similarity = 1.0 - (rank_diff_sum / max_diff)
    if similarity < 0.0:
        return 0.0
    if similarity > 1.0:
        return 1.0
    return similarity


def get_gt(fname: Path) -> dict[int, float]:
    with fname.open("r", errors="ignore") as f:
        s = f.read()
    _, gt = parse_pr_data(s)
    return gt


def main() -> None:
    cpp_path = Path(
        os.environ.get("PAGERANK_CPP_PATH", str(default_source_path("approx_pagerank.c")))
    )
    if not cpp_path.exists():
        raise FileNotFoundError(f"pagerank source not found: {cpp_path}")

    toolchain = default_toolchain()
    cgeist_config = default_cgeist_config()
    annotated_mlir = compile_cpp_path_to_annotated_mlir(
        cpp_path,
        cgeist_config=cgeist_config,
        toolchain=toolchain,
    )

    manager = ar.MLIRConfigManager()
    params = manager.parse_annotations(annotated_mlir)

    argv_base = [
        "-m",
        "synthetic",
        "-t",
        "1",
        "-n",
        "2000",
        "-d",
        "10",
        "-i",
        "10",
        "-s",
        "1",
    ]
    conf_min = int(os.environ.get("PAGERANK_CONF_MIN", "0"))
    conf_max = int(os.environ.get("PAGERANK_CONF_MAX", "5"))
    conf_runs = int(os.environ.get("PAGERANK_CONF_RUNS", "3"))
    conf_seed = os.environ.get("PAGERANK_CONF_SEED")
    rng = random.Random(int(conf_seed)) if conf_seed is not None else random.Random()
    def sample_confidences() -> list[int]:
        return [rng.randint(conf_min, conf_max) for _ in range(conf_runs)]

    gt_dir = Path(os.environ.get("PAGERANK_GT_DIR", str(Path(__file__).resolve().parent)))
    gt_dir.mkdir(parents=True, exist_ok=True)
    def get_or_create_gt() -> dict[int, float]:
        gt_path_env = os.environ.get("PAGERANK_GT_PATH")
        if gt_path_env:
            gt_path = Path(gt_path_env)
        else:
            gt_path = gt_dir / "gt_pagerank.txt"
        if not gt_path.exists():
            exact_cfg = exact_decision_config(params)
            exact_mlir = manager.apply_config(annotated_mlir, exact_cfg)
            exact_exec = compile_mlir_to_native_exec(
                exact_mlir, cgeist_config=cgeist_config, toolchain=toolchain, tag="pagerank"
            )
            argv = argv_base + ["-e", str(conf_min)]
            exact_result = run_exec(exact_exec, argv)
            if exact_result.returncode != 0:
                raise RuntimeError(exact_result.stderr)
            gt_path.write_text(exact_result.stdout, encoding="utf-8", errors="ignore")
        return get_gt(gt_path)
    gt_values = get_or_create_gt()

    def evaluate_fn(config: dict) -> tuple[float, float]:
        modified = manager.apply_config(annotated_mlir, config)
        exec_path = compile_mlir_to_native_exec(
            modified, cgeist_config=cgeist_config, toolchain=toolchain, tag="pagerank"
        )
        times = []
        accuracies = []
        for confidence in sample_confidences():
            argv = argv_base + ["-e", str(confidence)]
            result = run_exec(exec_path, argv)
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            time_s, approx_values = parse_pr_data(result.stdout)
            if time_s is None:
                raise RuntimeError("pagerank: failed to parse Time: from stdout")
            accuracy = compute_similarity(gt_values, approx_values)
            times.append(time_s * 1000.0)
            accuracies.append(accuracy)
        return (sum(times) / len(times)), (sum(accuracies) / len(accuracies))

    result = ar.tune(
        mlir_source=annotated_mlir,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=float(os.environ.get("ACCURACY_THRESHOLD", "0.9")),
        time_budget=20,
        result_callback=result_logger("pagerank"),
    )

    print(f"Found {len(params)} tunable params")
    print(f"Best config: {result['best_config']}")


if __name__ == "__main__":
    main()
