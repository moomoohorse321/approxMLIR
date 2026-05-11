#!/usr/bin/env python3
"""Approximate Diffusers backend with manifest-bound W8A16 substitutions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from approx_manifest import QuantManifest, build_manifest, load_manifest
from dataset import DiffusionRequest
from exact_backend import DiffusersExactBackend, ExactBackendConfig, GenerationResult
from site_inventory import inventory_model, inventory_summary, write_inventory
from substitute import DiffusionSubstitutionManager, SubstitutionConfig


@dataclass(frozen=True)
class ApproxBackendConfig:
    manifest_path: Path | None = None
    target_patterns: tuple[str, ...] = ("unet",)
    use_substitute: bool = True
    strict: bool = False
    drop_original_weights: bool = False
    stats_path: Path | None = None
    inventory_path: Path | None = None
    generated_manifest_path: Path | None = None
    quant_plan: str = "w8a16"
    block_m: int = 0
    block_n: int = 128
    block_k: int = 64
    group_k: int = 64
    trace_apply_events: bool = False
    plugin_path: str = ""


class DiffusersApproxBackend(DiffusersExactBackend):
    def __init__(self, config: ExactBackendConfig, approx_config: ApproxBackendConfig):
        super().__init__(config)
        self.approx_config = approx_config
        self._manager: DiffusionSubstitutionManager | None = None
        self._manifest: QuantManifest | None = None
        self._inventory_summary: dict[str, Any] | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        base = super().metadata
        base["backend"] = "diffusers_approx"
        base["quantization"] = {
            "manifest_path": str(self.approx_config.manifest_path) if self.approx_config.manifest_path else None,
            "target_patterns": list(self.approx_config.target_patterns),
            "use_substitute": self.approx_config.use_substitute,
            "strict": self.approx_config.strict,
            "drop_original_weights": self.approx_config.drop_original_weights,
            "quant_plan": self.approx_config.quant_plan,
            "block_m": self.approx_config.block_m,
            "block_n": self.approx_config.block_n,
            "block_k": self.approx_config.block_k,
            "group_k": self.approx_config.group_k,
            "trace_apply_events": self.approx_config.trace_apply_events,
            "inventory_summary": self._inventory_summary,
            "substitution": self._manager.total_summary() if self._manager else None,
        }
        return base

    def _load(self) -> None:
        if self._pipe is not None:
            return
        super()._load()
        assert self._pipe is not None
        sites = inventory_model(self._pipe, model_fingerprint=self.config.model_id)
        self._inventory_summary = inventory_summary(sites)
        if self.approx_config.inventory_path is not None:
            write_inventory(self.approx_config.inventory_path, sites)
        if self.approx_config.manifest_path is not None:
            manifest = load_manifest(self.approx_config.manifest_path)
        else:
            manifest = build_manifest(
                model=self.config.model_id,
                model_fingerprint=self.config.model_id,
                sites=sites,
                target_patterns=list(self.approx_config.target_patterns),
                strict=self.approx_config.strict,
                block_m=self.approx_config.block_m,
                block_n=self.approx_config.block_n,
                block_k=self.approx_config.block_k,
                group_k=self.approx_config.group_k,
                use_substitute=self.approx_config.use_substitute,
                quant_plan=self.approx_config.quant_plan,
            )
        self._manifest = manifest
        if self.approx_config.generated_manifest_path is not None:
            from approx_manifest import write_manifest

            write_manifest(self.approx_config.generated_manifest_path, manifest)
        self._manager = DiffusionSubstitutionManager(
            manifest,
            SubstitutionConfig(
                use_substitute=self.approx_config.use_substitute,
                block_m=self.approx_config.block_m,
                block_n=self.approx_config.block_n,
                block_k=self.approx_config.block_k,
                group_k=self.approx_config.group_k,
                strict=self.approx_config.strict,
                drop_original_weights=self.approx_config.drop_original_weights,
                trace_apply_events=self.approx_config.trace_apply_events,
                stats_path=self.approx_config.stats_path,
                plugin_path=self.approx_config.plugin_path,
            ),
        )
        self._manager.install(self._pipe)

    def generate(self, request: DiffusionRequest, image_path: Path) -> GenerationResult:
        self._load()
        before = self._manager.before_counts() if self._manager else {}
        result = super().generate(request, image_path)
        substitution = self._manager.delta_summary(before) if self._manager else None
        return GenerationResult(
            image_path=result.image_path,
            latency_ms=result.latency_ms,
            memory=result.memory,
            backend_metadata=self.metadata,
            substitution=substitution,
            application_metrics=result.application_metrics,
        )
