#!/usr/bin/env python3
"""Exact Diffusers backend for the diffusion application benchmark."""

from __future__ import annotations

import importlib.util
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset import DiffusionRequest


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"


@dataclass(frozen=True)
class ExactBackendConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    precision: str = "float16"
    local_files_only: bool = True
    enable_attention_slicing: bool = False
    enable_vae_slicing: bool = False
    enable_xformers: bool = False
    sequential_cpu_offload: bool = False
    safety_checker: str = "disabled_prompt_policy"
    memory_bound_repeats: int = 0
    memory_bound_candidates: int = 32
    memory_bound_in_features: int = 4096
    memory_bound_out_features: int = 32768


@dataclass(frozen=True)
class GenerationResult:
    image_path: Path
    latency_ms: dict[str, float]
    memory: dict[str, int | None]
    backend_metadata: dict[str, Any]
    substitution: dict[str, Any] | None = None
    application_metrics: dict[str, Any] | None = None


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def inspect_environment() -> dict[str, Any]:
    torch_importable = module_available("torch")
    cuda_available = None
    cuda_device_name = None
    if torch_importable:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_device_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_available = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_importable": torch_importable,
        "diffusers_importable": module_available("diffusers"),
        "pillow_importable": module_available("PIL"),
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
    }


class DiffusersExactBackend:
    """Lazy-loading exact Stable Diffusion pipeline.

    The benchmark app keeps this backend small on purpose: one local Diffusers
    pipeline, deterministic seeds, and no approximate substitutions.
    """

    def __init__(self, config: ExactBackendConfig):
        self.config = config
        self._torch = None
        self._pipe = None
        self._device = None
        self._dtype_name = None
        self._memory_bound_conditioner = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "diffusers_exact",
            "model_id": self.config.model_id,
            "device": self._device or self.config.device,
            "precision": self._dtype_name or self.config.precision,
            "local_files_only": self.config.local_files_only,
            "enable_attention_slicing": self.config.enable_attention_slicing,
            "enable_vae_slicing": self.config.enable_vae_slicing,
            "enable_xformers": self.config.enable_xformers,
            "sequential_cpu_offload": self.config.sequential_cpu_offload,
            "safety_checker": self.config.safety_checker,
            "memory_bound_conditioner": {
                "enabled": self.config.memory_bound_repeats > 0,
                "repeats": self.config.memory_bound_repeats,
                "candidates": self.config.memory_bound_candidates,
                "in_features": self.config.memory_bound_in_features,
                "out_features": self.config.memory_bound_out_features,
            },
        }

    def close(self) -> None:
        self._pipe = None
        self._memory_bound_conditioner = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except Exception as exc:
            raise RuntimeError(
                "Exact backend requires torch and diffusers. Install/stage them "
                "in the benchmark environment before running --mode run."
            ) from exc

        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, device)
        self._torch = torch
        self._device = device
        self._dtype_name = str(dtype).replace("torch.", "")

        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            local_files_only=self.config.local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if self.config.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if self.config.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if self.config.enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()

        if self.config.sequential_cpu_offload:
            if hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()
            else:
                raise RuntimeError("Pipeline does not support sequential CPU offload")
        else:
            pipe.to(device)
        if self.config.memory_bound_repeats > 0:
            from memory_bound import MemoryBoundConditioner, MemoryBoundConditionerConfig

            conditioner = MemoryBoundConditioner(
                torch,
                device,
                dtype,
                MemoryBoundConditionerConfig(
                    repeats=self.config.memory_bound_repeats,
                    candidates=self.config.memory_bound_candidates,
                    in_features=self.config.memory_bound_in_features,
                    out_features=self.config.memory_bound_out_features,
                ),
            )
            pipe.memory_bound_conditioner = conditioner
            self._memory_bound_conditioner = conditioner
        self._pipe = pipe

    def _resolve_device(self, torch_module) -> str:
        if self.config.device == "cuda":
            if not torch_module.cuda.is_available():
                raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is false")
            return "cuda"
        if self.config.device == "cpu":
            return "cpu"
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _resolve_dtype(self, torch_module, device: str):
        precision = self.config.precision
        if precision == "auto":
            precision = "float16" if device == "cuda" else "float32"
        if precision == "float16":
            return torch_module.float16 if device == "cuda" else torch_module.float32
        if precision == "bfloat16":
            return torch_module.bfloat16 if device == "cuda" else torch_module.float32
        if precision == "float32":
            return torch_module.float32
        raise ValueError(f"Unsupported precision: {self.config.precision}")

    def generate(self, request: DiffusionRequest, image_path: Path) -> GenerationResult:
        self._load()
        assert self._pipe is not None
        assert self._torch is not None
        assert self._device is not None

        image_path.parent.mkdir(parents=True, exist_ok=True)
        torch = self._torch
        if self._device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        generator_device = self._device if self._device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(request.seed)
        start = time.perf_counter()
        application_metrics: dict[str, Any] = {}
        conditioner_ms = 0.0
        if self._memory_bound_conditioner is not None:
            conditioner_metrics = self._memory_bound_conditioner.run(torch, request)
            application_metrics["memory_bound_conditioner"] = conditioner_metrics
            conditioner_ms = float(conditioner_metrics.get("latency_ms") or 0.0)
        image = self._pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        ).images[0]
        if self._device == "cuda":
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - start) * 1000.0
        image.save(image_path)

        peak_cuda_bytes = None
        if self._device == "cuda":
            peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
        return GenerationResult(
            image_path=image_path,
            latency_ms={
                "total": total_ms,
                "memory_bound_conditioner": conditioner_ms,
                "text_encoder": 0.0,
                "denoise": 0.0,
                "vae_decode": 0.0,
                "postprocess": 0.0,
                "scoring": 0.0,
            },
            memory={
                "peak_cuda_bytes": peak_cuda_bytes,
                "model_bytes": None,
            },
            backend_metadata=self.metadata,
            application_metrics=application_metrics,
        )
