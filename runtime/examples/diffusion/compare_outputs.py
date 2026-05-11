#!/usr/bin/env python3
"""Exact-vs-approx image comparison for diffusion runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[record["request_id"]] = record
    return records


def _image_arrays(path_a: Path, path_b: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path_a) as image_a, Image.open(path_b) as image_b:
        a = np.asarray(image_a.convert("RGB"), dtype=np.float32)
        b = np.asarray(image_b.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"image shape mismatch: {a.shape} vs {b.shape}")
    return a, b


def _ssim_luma(a, b) -> float:
    import numpy as np

    if a.ndim == 3:
        a = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
        b = 0.299 * b[:, :, 0] + 0.587 * b[:, :, 1] + 0.114 * b[:, :, 2]
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(a.var())
    var_b = float(b.var())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    denom = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if denom <= 0.0:
        return 0.0
    return float(((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)) / denom)


def _compare_memory_bound_conditioner(exact_record: dict[str, Any], approx_record: dict[str, Any]) -> dict[str, Any]:
    exact_metrics = (exact_record.get("application_metrics") or {}).get("memory_bound_conditioner")
    approx_metrics = (approx_record.get("application_metrics") or {}).get("memory_bound_conditioner")
    if not exact_metrics and not approx_metrics:
        return {"present": False, "accepted": True}
    if not exact_metrics or not approx_metrics:
        return {"present": True, "accepted": False, "error": "missing memory_bound_conditioner metrics"}

    exact_abs = float(exact_metrics.get("score_abs_mean") or 0.0)
    approx_abs = float(approx_metrics.get("score_abs_mean") or 0.0)
    exact_mean = float(exact_metrics.get("score_mean") or 0.0)
    approx_mean = float(approx_metrics.get("score_mean") or 0.0)
    scale = max(abs(exact_abs), 1.0e-6)
    abs_mean_rel_error = abs(exact_abs - approx_abs) / scale
    mean_scaled_error = abs(exact_mean - approx_mean) / scale
    accepted = abs_mean_rel_error <= 0.08 and mean_scaled_error <= 0.08
    return {
        "present": True,
        "accepted": accepted,
        "abs_mean_rel_error": abs_mean_rel_error,
        "mean_scaled_error": mean_scaled_error,
        "exact_score_abs_mean": exact_abs,
        "approx_score_abs_mean": approx_abs,
    }


def compare_record_sets(exact_records_path: Path, approx_records_path: Path) -> dict[str, Any]:
    exact = _load_records(exact_records_path)
    approx = _load_records(approx_records_path)
    missing = sorted(set(exact) - set(approx))
    extra = sorted(set(approx) - set(exact))
    records = []
    for request_id in sorted(set(exact) & set(approx)):
        exact_path = Path(exact[request_id].get("image_path") or "")
        approx_path = Path(approx[request_id].get("image_path") or "")
        try:
            a, b = _image_arrays(exact_path, approx_path)
            diff = a - b
            mse = float((diff * diff).mean())
            rmse = math.sqrt(mse)
            mae = float(abs(diff).mean())
            ssim = _ssim_luma(a, b)
            application = _compare_memory_bound_conditioner(exact[request_id], approx[request_id])
            ok = ssim >= 0.50 and rmse <= 90.0 and bool(application["accepted"])
            error = None
        except Exception as exc:
            rmse = None
            mae = None
            ssim = None
            application = {"present": False, "accepted": False}
            ok = False
            error = repr(exc)
        records.append(
            {
                "request_id": request_id,
                "accepted": ok,
                "rmse": rmse,
                "mae": mae,
                "ssim_luma": ssim,
                "application_metrics": application,
                "error": error,
            }
        )
    ssim_values = [float(record["ssim_luma"]) for record in records if record["ssim_luma"] is not None]
    rmse_values = [float(record["rmse"]) for record in records if record["rmse"] is not None]
    hard_guards = {
        "same_request_ids": not missing and not extra,
        "all_compared": len(records) == len(exact) and all(record["error"] is None for record in records),
        "ssim_mean_ge_0_50": bool(ssim_values) and statistics.mean(ssim_values) >= 0.50,
        "rmse_mean_le_90": bool(rmse_values) and statistics.mean(rmse_values) <= 90.0,
        "application_metrics_pass": all(
            record.get("application_metrics", {}).get("accepted") for record in records
        ),
    }
    return {
        "exact_records_path": str(exact_records_path),
        "approx_records_path": str(approx_records_path),
        "num_exact": len(exact),
        "num_approx": len(approx),
        "missing_request_ids": missing,
        "extra_request_ids": extra,
        "metrics": {
            "ssim_luma_mean": statistics.mean(ssim_values) if ssim_values else None,
            "ssim_luma_min": min(ssim_values) if ssim_values else None,
            "rmse_mean": statistics.mean(rmse_values) if rmse_values else None,
            "rmse_max": max(rmse_values) if rmse_values else None,
        },
        "hard_guards": hard_guards,
        "accepted": all(hard_guards.values()),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exact and approximate diffusion outputs.")
    parser.add_argument("--exact-records", type=Path, required=True)
    parser.add_argument("--approx-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = compare_record_sets(args.exact_records, args.approx_records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"accepted": summary["accepted"], "metrics": summary["metrics"]}, sort_keys=True))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
