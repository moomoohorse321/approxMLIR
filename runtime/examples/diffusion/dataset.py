#!/usr/bin/env python3
"""Dataset loading and request validation for the diffusion benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SMOKE_REQUESTS = PACKAGE_ROOT / "data" / "smoke_requests.jsonl"
DEFAULT_LATENCY_MEMORY_BOUND_REQUESTS = PACKAGE_ROOT / "data" / "latency_memory_bound_requests.jsonl"

ALLOWED_TASKS = {"text_to_image", "layout_to_image", "subject_image_generation"}
ALLOWED_CONDITIONING = {"none", "edge", "depth", "segmentation", "subject_images"}


class DatasetError(ValueError):
    """Raised when a request dataset does not satisfy the benchmark contract."""


@dataclass(frozen=True)
class Conditioning:
    type: str = "none"
    asset_path: str | None = None
    sha256: str | None = None

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> "Conditioning":
        payload = payload or {}
        cond = cls(
            type=str(payload.get("type", "none")),
            asset_path=payload.get("asset_path"),
            sha256=payload.get("sha256"),
        )
        if cond.type not in ALLOWED_CONDITIONING:
            raise DatasetError(f"Unsupported conditioning.type: {cond.type}")
        if cond.type == "none" and cond.asset_path is not None:
            raise DatasetError("conditioning.asset_path must be null when type is none")
        if cond.type != "none" and not cond.asset_path:
            raise DatasetError("conditioning.asset_path is required for conditioned requests")
        return cond


@dataclass(frozen=True)
class RequestPolicy:
    safety_required: bool = True
    min_clip_score: float = 0.0
    max_nsfw_score: float = 0.0

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> "RequestPolicy":
        payload = payload or {}
        return cls(
            safety_required=bool(payload.get("safety_required", True)),
            min_clip_score=float(payload.get("min_clip_score", 0.0)),
            max_nsfw_score=float(payload.get("max_nsfw_score", 0.0)),
        )


@dataclass(frozen=True)
class DiffusionRequest:
    request_id: str
    task: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    conditioning: Conditioning = field(default_factory=Conditioning)
    policy: RequestPolicy = field(default_factory=RequestPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(
        cls, payload: dict[str, Any], *, source_path: Path | None = None
    ) -> "DiffusionRequest":
        try:
            request = cls(
                request_id=str(payload["request_id"]),
                task=str(payload["task"]),
                prompt=str(payload["prompt"]),
                negative_prompt=str(payload.get("negative_prompt", "")),
                width=int(payload.get("width", 512)),
                height=int(payload.get("height", 512)),
                num_inference_steps=int(payload.get("num_inference_steps", 20)),
                guidance_scale=float(payload.get("guidance_scale", 7.0)),
                seed=int(payload.get("seed", 0)),
                conditioning=Conditioning.from_json(payload.get("conditioning")),
                policy=RequestPolicy.from_json(payload.get("policy")),
                metadata=dict(payload.get("metadata", {})),
            )
        except KeyError as exc:
            raise DatasetError(f"Missing required request field: {exc.args[0]}") from exc
        request.validate(source_path=source_path)
        return request

    def validate(self, *, source_path: Path | None = None) -> None:
        if not self.request_id:
            raise DatasetError("request_id must be nonempty")
        if self.task not in ALLOWED_TASKS:
            raise DatasetError(f"Unsupported task: {self.task}")
        if not self.prompt.strip():
            raise DatasetError(f"{self.request_id}: prompt must be nonempty")
        if self.width <= 0 or self.height <= 0:
            raise DatasetError(f"{self.request_id}: width/height must be positive")
        if self.width % 8 != 0 or self.height % 8 != 0:
            raise DatasetError(f"{self.request_id}: width/height must be multiples of 8")
        if self.width > 1024 or self.height > 1024:
            raise DatasetError(f"{self.request_id}: laptop profile caps dimensions at 1024")
        if self.num_inference_steps <= 0:
            raise DatasetError(f"{self.request_id}: num_inference_steps must be positive")
        if self.guidance_scale <= 0:
            raise DatasetError(f"{self.request_id}: guidance_scale must be positive")
        if self.conditioning.asset_path:
            resolved = resolve_request_path(self.conditioning.asset_path, source_path)
            if not resolved.exists():
                raise DatasetError(f"{self.request_id}: missing conditioning asset {resolved}")
            if self.conditioning.sha256:
                actual = sha256_file(resolved)
                if actual != self.conditioning.sha256:
                    raise DatasetError(
                        f"{self.request_id}: conditioning sha256 mismatch for {resolved}"
                    )

    def to_json(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task": self.task,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "conditioning": {
                "type": self.conditioning.type,
                "asset_path": self.conditioning.asset_path,
                "sha256": self.conditioning.sha256,
            },
            "policy": {
                "safety_required": self.policy.safety_required,
                "min_clip_score": self.policy.min_clip_score,
                "max_nsfw_score": self.policy.max_nsfw_score,
            },
            "metadata": self.metadata,
        }


def resolve_request_path(value: str, source_path: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if source_path is not None:
        return source_path.parent / path
    return PACKAGE_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_requests(path: Path = DEFAULT_SMOKE_REQUESTS, limit: int | None = None) -> list[DiffusionRequest]:
    requests: list[DiffusionRequest] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            request = DiffusionRequest.from_json(payload, source_path=path)
            if request.request_id in seen_ids:
                raise DatasetError(f"Duplicate request_id: {request.request_id}")
            seen_ids.add(request.request_id)
            requests.append(request)
            if limit is not None and len(requests) >= limit:
                break
    if not requests:
        raise DatasetError(f"No requests loaded from {path}")
    return requests


def dataset_fingerprint(path: Path) -> str:
    return sha256_file(path)
