#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import random
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

# import logging
# logging.basicConfig(level=logging.DEBUG, force=True)
# logging.getLogger("approx_tuner").setLevel(logging.DEBUG)

def parse_kernel_out(data_string: str) -> tuple[float | None, list[int]]:
    time_ms: float | None = None
    ranked_ids: list[int] = []
    time_pattern = re.compile(r"\bComputation time:\s*([\d.]+)\s*ms\b")
    doc_id_pattern = re.compile(r"\bRank\s+\d+:\s+Doc\s+(\d+)\b")
    for line in data_string.splitlines():
        doc_id_match = doc_id_pattern.search(line)
        if doc_id_match:
            ranked_ids.append(int(doc_id_match.group(1)))
            continue
        time_match = time_pattern.search(line)
        if time_match:
            time_ms = float(time_match.group(1))
    return time_ms, ranked_ids


def compute_similarity(gt_list: list[int], approx_list: list[int], p: float = 0.9) -> float:
    if not gt_list or not approx_list:
        return 0.0
    max_depth = min(1000, max(len(gt_list), len(approx_list)))
    rbo_score_raw = 0.0
    gt_set: set[int] = set()
    approx_set: set[int] = set()
    for d in range(1, max_depth + 1):
        if d <= len(gt_list):
            gt_set.add(gt_list[d - 1])
        if d <= len(approx_list):
            approx_set.add(approx_list[d - 1])
        overlap = len(gt_set.intersection(approx_set))
        agreement_at_d = overlap / d
        rbo_score_raw += (p ** (d - 1)) * agreement_at_d
    unnormalized_score = (1 - p) * rbo_score_raw
    max_possible_score_raw = sum((p ** (d - 1)) for d in range(1, max_depth + 1))
    max_possible_score = (1 - p) * max_possible_score_raw
    if max_possible_score == 0:
        return 0.0
    return unnormalized_score / max_possible_score


def get_gt(fname: Path) -> list[int]:
    with fname.open("r", errors="ignore") as f:
        s = f.read()
    _, gt_list = parse_kernel_out(s)
    return gt_list


def main() -> None:
    cpp_path = Path(os.environ.get("BM25_CPP_PATH", str(default_source_path("approx_bm25.c"))))
    if not cpp_path.exists():
        raise FileNotFoundError(f"bm25 source not found: {cpp_path}")

    toolchain = default_toolchain()
    cgeist_config = default_cgeist_config()
    annotated_mlir = compile_cpp_path_to_annotated_mlir(
        cpp_path,
        cgeist_config=cgeist_config,
        toolchain=toolchain,
    )

    manager = ar.MLIRConfigManager()
    params = manager.parse_annotations(annotated_mlir)

    query = os.environ.get("BM25_QUERY", "The great wall is such a nice building in China.")
    conf_min = int(os.environ.get("BM25_CONF_MIN", "0"))
    conf_max = int(os.environ.get("BM25_CONF_MAX", "5"))
    conf_runs = int(os.environ.get("BM25_CONF_RUNS", "3"))
    conf_seed = os.environ.get("BM25_CONF_SEED")
    rng = random.Random(int(conf_seed)) if conf_seed is not None else random.Random()
    def sample_confidences() -> list[int]:
        return [rng.randint(conf_min, conf_max) for _ in range(conf_runs)]
    docs_path = Path(
        os.environ.get("BM25_DOCS_PATH", str(Path(__file__).resolve().parent / "docs.txt"))
    )
    if not docs_path.exists():
        raise FileNotFoundError(f"bm25 docs not found: {docs_path}")
    gt_dir = Path(os.environ.get("BM25_GT_DIR", str(Path(__file__).resolve().parent)))
    gt_dir.mkdir(parents=True, exist_ok=True)
    def generate_gt() -> list[int]:
        gt_path_env = os.environ.get("BM25_GT_PATH")
        if gt_path_env:
            gt_path = Path(gt_path_env)
        else:
            gt_path = gt_dir / "gt_bm25.txt"
        exact_cfg = exact_decision_config(params)
        exact_mlir = manager.apply_config(annotated_mlir, exact_cfg)
        exact_exec = compile_mlir_to_native_exec(
            exact_mlir, cgeist_config=cgeist_config, toolchain=toolchain, tag="bm25"
        )
        exact_result = run_exec(exact_exec, [str(docs_path), query, str(conf_min)])
        if exact_result.returncode != 0:
            raise RuntimeError(exact_result.stderr)
        gt_path.write_text(exact_result.stdout, encoding="utf-8", errors="ignore")
        return get_gt(gt_path)
    gt_list = generate_gt()

    def evaluate_fn(config: dict) -> tuple[float, float]:
        modified = manager.apply_config(annotated_mlir, config)
        exec_path = compile_mlir_to_native_exec(
            modified, cgeist_config=cgeist_config, toolchain=toolchain, tag="bm25"
        )
        times = []
        accuracies = []
        for confidence in sample_confidences():
            result = run_exec(exec_path, [str(docs_path), query, str(confidence)])
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            time_ms, approx_list = parse_kernel_out(result.stdout)
            if time_ms is None:
                raise RuntimeError("bm25: failed to parse computation time from stdout")
            accuracy = compute_similarity(gt_list, approx_list, p=0.95)
            times.append(time_ms)
            accuracies.append(accuracy)
        return (sum(times) / len(times)), (sum(accuracies) / len(accuracies))

    result = ar.tune(
        mlir_source=annotated_mlir,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=float(os.environ.get("ACCURACY_THRESHOLD", "0.9")),
        time_budget=12000,
        result_callback=result_logger("bm25"),
    )

    print(f"Found {len(params)} tunable params")
    print(f"Best config: {result['best_config']}")


if __name__ == "__main__":
    main()
