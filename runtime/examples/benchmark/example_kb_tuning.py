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

def parse_kernel_out(
    data_string: str,
) -> tuple[float | None, list[tuple[int, str, float]]]:
    time_match = re.search(r"\bElapsed\s+([\d\.]+)\s*ms\b", data_string)
    time_ms = float(time_match.group(1)) if time_match else None
    pattern = re.compile(r"Rank\s+(\d+):\s+Doc\s+(\S+)\s+\(Score:\s+([\d\.]+)\)")
    parsed: list[tuple[int, str, float]] = []
    for line in data_string.strip().split("\\n"):
        match = pattern.match(line)
        if match:
            rank = int(match.group(1))
            doc_id = match.group(2)
            score = float(match.group(3))
            parsed.append((rank, doc_id, score))
    return time_ms, parsed


def compute_similarity(gt_string: str, approx_string: str, p: float = 0.9) -> float:
    _, gt_data = parse_kernel_out(gt_string)
    _, approx_data = parse_kernel_out(approx_string)
    gt_list = [doc_id for _, doc_id, _ in gt_data]
    approx_list = [doc_id for _, doc_id, _ in approx_data]
    if not gt_list or not approx_list:
        return 0.0
    max_depth = min(1000, max(len(gt_list), len(approx_list)))
    rbo_score_raw = 0.0
    gt_set: set[str] = set()
    approx_set: set[str] = set()
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


def get_gt(fname: Path) -> str:
    with fname.open("r", errors="ignore") as f:
        return f.read()


def _make_embedding_csv(value: float) -> str:
    return ",".join([f"{value:.4f}"] * 384)


def main() -> None:
    cpp_path = Path(os.environ.get("KB_CPP_PATH", str(default_source_path("approx_kb.c"))))
    if not cpp_path.exists():
        raise FileNotFoundError(f"kb source not found: {cpp_path}")

    toolchain = default_toolchain()
    cgeist_config = default_cgeist_config()
    annotated_mlir = compile_cpp_path_to_annotated_mlir(
        cpp_path,
        cgeist_config=cgeist_config,
        toolchain=toolchain,
    )

    manager = ar.MLIRConfigManager()
    params = manager.parse_annotations(annotated_mlir)

    query_csv = _make_embedding_csv(0.1)
    top_k = os.environ.get("KB_TOPK", "3")
    conf_min = int(os.environ.get("KB_CONF_MIN", "1"))
    conf_max = int(os.environ.get("KB_CONF_MAX", "5"))
    conf_runs = int(os.environ.get("KB_CONF_RUNS", "3"))
    conf_seed = os.environ.get("KB_CONF_SEED")
    rng = random.Random(int(conf_seed)) if conf_seed is not None else random.Random()
    def sample_confidences() -> list[int]:
        return [rng.randint(conf_min, conf_max) for _ in range(conf_runs)]
    gt_dir = Path(os.environ.get("KB_GT_DIR", str(Path(__file__).resolve().parent)))
    gt_dir.mkdir(parents=True, exist_ok=True)
    def get_or_create_gt() -> str:
        gt_path_env = os.environ.get("KB_GT_PATH")
        if gt_path_env:
            gt_path = Path(gt_path_env)
        else:
            gt_path = gt_dir / "gt_kb.txt"
        if not gt_path.exists():
            exact_cfg = exact_decision_config(params)
            exact_mlir = manager.apply_config(annotated_mlir, exact_cfg)
            exact_exec = compile_mlir_to_native_exec(
                exact_mlir, cgeist_config=cgeist_config, toolchain=toolchain, tag="kb"
            )
            exact_result = run_exec(
                exact_exec, [query_csv, top_k, str(conf_min)], stdin=docs_stdin
            )
            if exact_result.returncode != 0:
                raise RuntimeError(exact_result.stderr)
            gt_path.write_text(exact_result.stdout, encoding="utf-8", errors="ignore")
        return get_gt(gt_path)
    gt_string = get_or_create_gt()

    with open("input.txt", "r") as f:
        docs_stdin = f.read()

    def evaluate_fn(config: dict) -> tuple[float, float]:
        modified = manager.apply_config(annotated_mlir, config)
        exec_path = compile_mlir_to_native_exec(
            modified, cgeist_config=cgeist_config, toolchain=toolchain, tag="kb"
        )
        times = []
        accuracies = []
        for confidence in sample_confidences():
            result = run_exec(
                exec_path,
                [query_csv, top_k, str(confidence)],
                stdin=docs_stdin,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            time_ms, _ = parse_kernel_out(result.stdout)
            if time_ms is None:
                raise RuntimeError("kb: failed to parse execution time from stdout")
            accuracy = compute_similarity(gt_string, result.stdout, p=0.7)
            times.append(time_ms)
            accuracies.append(accuracy)
        return (sum(times) / len(times)), (sum(accuracies) / len(accuracies))

    result = ar.tune(
        mlir_source=annotated_mlir,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=float(os.environ.get("ACCURACY_THRESHOLD", "0.9")),
        time_budget=20,
        result_callback=result_logger("kb"),
    )

    print(f"Found {len(params)} tunable params")
    print(f"Best config: {result['best_config']}")


if __name__ == "__main__":
    main()
