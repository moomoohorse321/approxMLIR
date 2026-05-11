#!/usr/bin/env python3
"""Dependency-light tests for quant diffusion contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


class QuantDiffusionContractTests(unittest.TestCase):
    def test_inventory_and_manifest_are_deterministic(self) -> None:
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        from approx_manifest import PLAN_LINEAR_W4A16_AWQ, PLAN_LINEAR_W8A16, build_manifest, validate_manifest
        from site_inventory import inventory_model

        torch.manual_seed(0)

        class TinyPipe:
            def __init__(self) -> None:
                self.unet = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
                self.text_encoder = nn.Sequential(nn.Linear(4, 4))

        pipe = TinyPipe()
        sites_a = inventory_model(pipe, model_fingerprint="tiny")
        sites_b = inventory_model(pipe, model_fingerprint="tiny")

        self.assertEqual([site.stable_key for site in sites_a], [site.stable_key for site in sites_b])
        self.assertEqual(len([site for site in sites_a if site.op == "linear"]), 3)

        manifest = build_manifest(
            model="tiny",
            model_fingerprint="tiny",
            sites=sites_a,
            target_patterns=["unet"],
            use_substitute=False,
        )
        validate_manifest(manifest)
        bound = {binding["site_id"]: binding["default_plan"] for binding in manifest.bindings}
        self.assertEqual(len(bound), 2)
        self.assertTrue(all(plan == PLAN_LINEAR_W8A16 for plan in bound.values()))

        w4_manifest = build_manifest(
            model="tiny",
            model_fingerprint="tiny",
            sites=sites_a,
            target_patterns=["unet"],
            quant_plan="w4a16_awq",
            use_substitute=False,
        )
        validate_manifest(w4_manifest)
        w4_bound = {binding["site_id"]: binding["default_plan"] for binding in w4_manifest.bindings}
        self.assertTrue(all(plan == PLAN_LINEAR_W4A16_AWQ for plan in w4_bound.values()))
        self.assertTrue(
            any(artifact["artifact_type"] == "qweight_i4_t_packed" for artifact in w4_manifest.artifacts)
        )

    def test_manifest_validation_rejects_unknown_site_binding(self) -> None:
        from approx_manifest import QuantManifest, validate_manifest

        manifest = QuantManifest(
            schema_version=1,
            model="tiny",
            model_fingerprint="tiny",
            strict=True,
            sites=[],
            plans=[{"plan_id": "exact", "kind": "exact", "quantizer": "none", "layout": "fp", "op": "any", "params": {}, "artifact_types": [], "kernel_id": "torch", "kernel_abi_id": "torch.v1", "fallback": "exact"}],
            artifacts=[],
            bindings=[{"site_id": "missing", "default_plan": "exact", "regime_plans": {}}],
            tuning_evidence=[],
            negative_evidence=[],
        )
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_compare_outputs_accepts_identical_images(self) -> None:
        try:
            from PIL import Image
        except Exception as exc:
            self.skipTest(f"Pillow unavailable: {exc}")

        from compare_outputs import compare_record_sets

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exact_img = tmp_path / "exact.png"
            approx_img = tmp_path / "approx.png"
            Image.new("RGB", (16, 16), (80, 120, 160)).save(exact_img)
            Image.new("RGB", (16, 16), (80, 120, 160)).save(approx_img)
            exact_records = tmp_path / "exact.jsonl"
            approx_records = tmp_path / "approx.jsonl"
            exact_records.write_text(
                json.dumps({"request_id": "r1", "image_path": str(exact_img)}) + "\n",
                encoding="utf-8",
            )
            approx_records.write_text(
                json.dumps({"request_id": "r1", "image_path": str(approx_img)}) + "\n",
                encoding="utf-8",
            )

            summary = compare_record_sets(exact_records, approx_records)

            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["metrics"]["rmse_mean"], 0.0)
            self.assertAlmostEqual(summary["metrics"]["ssim_luma_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
