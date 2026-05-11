#!/usr/bin/env python3
"""Memory-bandwidth dominated conditioning tower for diffusion serving."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import torch

from dataset import DiffusionRequest


@dataclass(frozen=True)
class MemoryBoundConditionerConfig:
    repeats: int = 0
    candidates: int = 32
    in_features: int = 4096
    out_features: int = 32768
    seed: int = 20260511


class MemoryBoundConditioner(torch.nn.Module):
    """Large prompt-conditioning projection used by the latency-bound profile.

    The tower intentionally uses a small candidate batch with weights larger
    than typical laptop L2 capacity. That gives the benchmark a serving stage
    where latency remains tied to weight traffic without relying on one scalar
    request repeated thousands of times.
    """

    def __init__(self, torch_module: Any, device: str, dtype: Any, config: MemoryBoundConditionerConfig):
        import torch.nn as nn

        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        self.proj = nn.Linear(config.in_features, config.out_features, bias=False, device=device, dtype=dtype)
        generator = torch_module.Generator(device="cpu").manual_seed(config.seed)
        weight = torch_module.empty(
            (config.out_features, config.in_features),
            dtype=torch_module.float32,
            device="cpu",
        )
        weight.normal_(mean=0.0, std=0.02, generator=generator)
        with torch_module.no_grad():
            self.proj.weight.copy_(weight.to(device=device, dtype=dtype))
        self.proj.requires_grad_(False)
        self.proj.eval()

    def _request_input(self, torch_module: Any, request: DiffusionRequest):
        payload = f"{request.request_id}\n{request.prompt}\n{request.negative_prompt}\n{request.seed}".encode(
            "utf-8"
        )
        digest = hashlib.sha256(payload).digest()
        seed = int.from_bytes(digest[:8], "little") ^ self.config.seed
        generator = torch_module.Generator(device="cpu").manual_seed(seed % ((1 << 63) - 1))
        x = torch_module.empty(
            (self.config.candidates, self.config.in_features),
            dtype=torch_module.float32,
            device="cpu",
        )
        x.normal_(mean=0.0, std=1.0, generator=generator)
        return x.to(device=self.device, dtype=self.dtype)

    def run(self, torch_module: Any, request: DiffusionRequest) -> dict[str, Any]:
        if self.config.repeats <= 0:
            return {"enabled": False}
        if self.device == "cuda":
            torch_module.cuda.synchronize()
        start = time.perf_counter()
        x = self._request_input(torch_module, request)
        y = None
        with torch_module.no_grad():
            for _ in range(self.config.repeats):
                y = self.proj(x)
        if self.device == "cuda":
            torch_module.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        assert y is not None
        y_float = y.float()
        score_mean = float(y_float.mean().item())
        score_abs_mean = float(y_float.abs().mean().item())
        return {
            "enabled": True,
            "latency_ms": latency_ms,
            "repeats": self.config.repeats,
            "candidates": self.config.candidates,
            "in_features": self.config.in_features,
            "out_features": self.config.out_features,
            "projection_rows": self.config.repeats * self.config.candidates,
            "score_mean": score_mean,
            "score_abs_mean": score_abs_mean,
        }
