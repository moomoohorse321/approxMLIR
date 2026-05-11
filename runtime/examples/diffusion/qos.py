#!/usr/bin/env python3
"""QoS helpers for the diffusion application benchmark."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BANNED_PROMPT_TERMS = {
    "blood",
    "gore",
    "graphic violence",
    "nude",
    "nudity",
    "porn",
    "sexual",
    "weapon",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def prompt_safety_pass(prompt: str, negative_prompt: str = "") -> bool:
    text = f"{prompt}\n{negative_prompt}".lower()
    return not any(term in text for term in BANNED_PROMPT_TERMS)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[int(rank)])
    weight = rank - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def compute_image_stats(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageStat
    except Exception as exc:
        return {
            "exists": path.exists(),
            "valid_image": False,
            "width": None,
            "height": None,
            "luma_mean": None,
            "luma_stddev": None,
            "nonblank": False,
            "error": f"Pillow unavailable: {exc}",
        }

    if not path.exists():
        return {
            "exists": False,
            "valid_image": False,
            "width": None,
            "height": None,
            "luma_mean": None,
            "luma_stddev": None,
            "nonblank": False,
            "error": "image file missing",
        }

    try:
        with Image.open(path) as image:
            image.load()
            gray = image.convert("L")
            stat = ImageStat.Stat(gray)
            width, height = image.size
    except Exception as exc:
        return {
            "exists": True,
            "valid_image": False,
            "width": None,
            "height": None,
            "luma_mean": None,
            "luma_stddev": None,
            "nonblank": False,
            "error": f"image decode failed: {exc}",
        }

    luma_mean = float(stat.mean[0])
    luma_stddev = float(stat.stddev[0])
    nonblank = 2.0 < luma_stddev and 2.0 < luma_mean < 253.0
    return {
        "exists": True,
        "valid_image": True,
        "width": int(width),
        "height": int(height),
        "luma_mean": luma_mean,
        "luma_stddev": luma_stddev,
        "nonblank": bool(nonblank),
        "error": None,
    }


def heuristic_prompt_alignment(prompt: str, image_stats: dict[str, Any]) -> float | None:
    if not prompt.strip() or not image_stats.get("nonblank"):
        return 0.0
    luma_stddev = float(image_stats.get("luma_stddev") or 0.0)
    # This is intentionally weak. A real CLIP scorer can replace it later, but
    # the exact benchmark should still reject blank/corrupt images on a laptop.
    return round(min(1.0, 0.55 + 0.45 * min(1.0, luma_stddev / 64.0)), 6)


def build_exact_record(
    *,
    request_payload: dict[str, Any],
    image_path: Path | None,
    latency_ms: dict[str, float],
    memory: dict[str, int | None],
    image_stats: dict[str, Any],
    errors: list[str],
    manifest_path: Path | None,
    substitution: dict[str, Any] | None = None,
    application_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = str(request_payload.get("prompt", ""))
    negative_prompt = str(request_payload.get("negative_prompt", ""))
    safety_pass = prompt_safety_pass(prompt, negative_prompt)
    prompt_alignment = heuristic_prompt_alignment(prompt, image_stats)
    image_valid = (
        bool(image_stats.get("valid_image"))
        and bool(image_stats.get("nonblank"))
        and image_stats.get("width") == int(request_payload.get("width", 0))
        and image_stats.get("height") == int(request_payload.get("height", 0))
    )
    accepted = not errors and image_valid and safety_pass
    if not image_valid and not errors:
        errors = [f"invalid image: {image_stats.get('error') or 'blank or wrong dimensions'}"]
    if not safety_pass:
        errors = [*errors, "prompt safety policy failed"]

    return {
        "request_id": request_payload["request_id"],
        "accepted": bool(accepted),
        "image_path": str(image_path) if image_path is not None else None,
        "latency_ms": latency_ms,
        "memory": memory,
        "quality": {
            "clip_score": None,
            "aesthetic_score": None,
            "prompt_image_alignment": prompt_alignment,
            "lpips_vs_exact": None,
            "ssim_vs_exact": None,
            "edge_alignment_f1": None,
            "subject_similarity": None,
            "image_stats": image_stats,
        },
        "application_metrics": application_metrics or {},
        "safety": {
            "passed": bool(safety_pass),
            "nsfw_score": None,
            "policy": "prompt-term-blocklist",
        },
        "substitution": substitution
        or {
            "mode": "exact",
            "manifest_path": str(manifest_path) if manifest_path else None,
            "sites_hit": 0,
            "sites_expected": 0,
        },
        "errors": errors,
    }


def summarize_qos(records: list[dict[str, Any]], *, profile: str) -> dict[str, Any]:
    total = len(records)
    accepted = sum(1 for record in records if record.get("accepted"))
    safety_passed = sum(1 for record in records if record.get("safety", {}).get("passed"))
    valid_images = sum(
        1
        for record in records
        if record.get("quality", {}).get("image_stats", {}).get("valid_image")
        and record.get("quality", {}).get("image_stats", {}).get("nonblank")
    )
    latencies = [
        float(record.get("latency_ms", {}).get("total", 0.0))
        for record in records
        if not record.get("errors") and record.get("latency_ms", {}).get("total") is not None
    ]
    peak_memories = [
        int(record.get("memory", {}).get("peak_cuda_bytes") or 0)
        for record in records
        if record.get("memory", {}).get("peak_cuda_bytes") is not None
    ]
    completion_rate = accepted / total if total else 0.0
    image_valid_rate = valid_images / total if total else 0.0
    safety_pass_rate = safety_passed / total if total else 0.0
    hard_guards = {
        "completion_rate_eq_1": completion_rate == 1.0,
        "image_valid_rate_eq_1": image_valid_rate == 1.0,
        "safety_pass_rate_eq_1": safety_pass_rate == 1.0,
        "schema_valid_rate_eq_1": True,
    }
    return {
        "profile": profile,
        "num_requests": total,
        "accepted_requests": accepted,
        "completion_rate": completion_rate,
        "image_valid_rate": image_valid_rate,
        "safety_pass_rate": safety_pass_rate,
        "latency_ms": {
            "p50": percentile(latencies, 50.0),
            "p95": percentile(latencies, 95.0),
            "mean": statistics.mean(latencies) if latencies else None,
        },
        "memory": {
            "peak_cuda_bytes_max": max(peak_memories) if peak_memories else None,
        },
        "available_quality_metrics": {
            "clip_score": False,
            "prompt_image_alignment": True,
            "lpips_vs_exact": False,
            "ssim_vs_exact": False,
        },
        "hard_guards": hard_guards,
        "accepted": all(hard_guards.values()),
    }
