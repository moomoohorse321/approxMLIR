#!/usr/bin/env python3
"""Dependency-light tests for the diffusion exact benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import app
from dataset import (
    DEFAULT_LATENCY_MEMORY_BOUND_REQUESTS,
    DEFAULT_SMOKE_REQUESTS,
    DatasetError,
    load_requests,
)
from exact_backend import GenerationResult
from qos import compute_image_stats, summarize_qos


class FakeBackend:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def generate(self, request, image_path: Path) -> GenerationResult:
        from PIL import Image

        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (request.width, request.height))
        pixels = image.load()
        for y in range(request.height):
            for x in range(request.width):
                pixels[x, y] = ((x * 13) % 256, (y * 17) % 256, ((x + y) * 7) % 256)
        image.save(image_path)
        return GenerationResult(
            image_path=image_path,
            latency_ms={
                "total": 12.5,
                "memory_bound_conditioner": 0.0,
                "text_encoder": 1.0,
                "denoise": 10.0,
                "vae_decode": 1.0,
                "postprocess": 0.4,
                "scoring": 0.1,
            },
            memory={"peak_cuda_bytes": 1234, "model_bytes": None},
            backend_metadata={"backend": "fake"},
        )


def write_request(path: Path, request_id: str = "unit-0001") -> None:
    payload = {
        "request_id": request_id,
        "task": "text_to_image",
        "prompt": "a real public benchmark style image prompt",
        "negative_prompt": "blurry, low quality, watermark",
        "width": 16,
        "height": 16,
        "num_inference_steps": 2,
        "guidance_scale": 7.0,
        "seed": 7,
        "conditioning": {"type": "none", "asset_path": None, "sha256": None},
        "policy": {"safety_required": True, "min_clip_score": 0.0, "max_nsfw_score": 0.0},
        "metadata": {"source_dataset": "unit"},
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def make_args(input_jsonl: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        mode="run",
        input_jsonl=input_jsonl,
        output_dir=output_dir,
        run_label="unit",
        limit=None,
        profile="laptop_smoke",
        model_id="unit-model",
        device="cpu",
        precision="float32",
        allow_download=False,
        enable_attention_slicing=False,
        enable_vae_slicing=False,
        enable_xformers=False,
        sequential_cpu_offload=False,
        warmup_runs=0,
        memory_bound_repeats=0,
        memory_bound_candidates=32,
        memory_bound_in_features=4096,
        memory_bound_out_features=32768,
    )


class DiffusionExactTests(unittest.TestCase):
    def test_smoke_requests_are_valid_real_profile_rows(self) -> None:
        requests = load_requests(DEFAULT_SMOKE_REQUESTS)

        self.assertEqual(len(requests), 4)
        self.assertTrue(all(request.task == "text_to_image" for request in requests))
        self.assertTrue(all(request.width == 512 and request.height == 512 for request in requests))
        self.assertTrue(all(request.metadata.get("source_dataset") == "nateraw/parti-prompts" for request in requests))

    def test_latency_memory_bound_profile_uses_conditioner_workload(self) -> None:
        requests = load_requests(DEFAULT_LATENCY_MEMORY_BOUND_REQUESTS)

        self.assertEqual(len(requests), 8)
        self.assertTrue(all(request.metadata.get("profile") == "latency_memory_bound" for request in requests))
        self.assertTrue(all((request.width, request.height) == (64, 64) for request in requests))
        self.assertTrue(all(request.num_inference_steps == 2 for request in requests))

    def test_dataset_rejects_duplicate_request_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.jsonl"
            write_request(path, "dup")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(path.read_text(encoding="utf-8"))

            with self.assertRaises(DatasetError):
                load_requests(path)

    def test_qos_accepts_valid_fake_image_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.png"
            from PIL import Image

            Image.effect_noise((16, 16), 64.0).convert("RGB").save(image_path)
            stats = compute_image_stats(image_path)
            record = {
                "accepted": True,
                "latency_ms": {"total": 10.0},
                "memory": {"peak_cuda_bytes": 42},
                "quality": {"image_stats": stats},
                "safety": {"passed": True},
                "errors": [],
            }

            summary = summarize_qos([record], profile="laptop_smoke")

            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["completion_rate"], 1.0)
            self.assertEqual(summary["image_valid_rate"], 1.0)
            self.assertEqual(summary["latency_ms"]["p50"], 10.0)

    def test_runner_writes_exact_output_schema_with_fake_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_jsonl = tmp_path / "requests.jsonl"
            write_request(input_jsonl)
            run_dir = tmp_path / "run"
            backend = FakeBackend()

            summary = app.run_exact(make_args(input_jsonl, tmp_path / "out"), backend=backend, run_dir=run_dir)

            self.assertTrue(backend.closed)
            self.assertTrue(summary["accepted"])
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "records.jsonl").exists())
            self.assertTrue((run_dir / "qos_summary.json").exists())
            self.assertTrue((run_dir / "run_summary.json").exists())

            records = [
                json.loads(line)
                for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["request_id"], "unit-0001")
            self.assertEqual(records[0]["substitution"]["mode"], "exact")
            self.assertEqual(records[0]["substitution"]["sites_hit"], 0)
            self.assertTrue(Path(records[0]["image_path"]).exists())


if __name__ == "__main__":
    unittest.main()
