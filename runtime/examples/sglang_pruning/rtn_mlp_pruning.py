from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MLP_SUFFIXES = (
    "gate_up_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class MLPModuleRef:
    prefix: str
    group: str
    role: str


def split_mlp_prefix(prefix: str) -> MLPModuleRef | None:
    for suffix in MLP_SUFFIXES:
        token = f".{suffix}"
        if prefix.endswith(token):
            return MLPModuleRef(prefix=prefix, group=prefix[: -len(token)], role=suffix)
        if prefix == suffix:
            return MLPModuleRef(prefix=prefix, group="", role=suffix)
    return None


def target_to_blocks(target_sparsity: float, total: int) -> tuple[int, float]:
    if total <= 0:
        return 0, 0.0
    target_sparsity = max(0.0, min(1.0, float(target_sparsity)))
    prune = int(round(total * target_sparsity))
    prune = max(0, min(total - 1, prune))
    return prune, prune / total


def indices_to_blocks(indices: list[int], block: int) -> list[int]:
    if block <= 1:
        return sorted({int(i) for i in indices})
    return sorted({int(i) // block for i in indices})


def load_artifact(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"pruning artifact not found: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("groups", {}), dict):
        raise ValueError("pruning artifact must contain a dict field named 'groups'")
    return payload


def group_entry_for_prefix(artifact: dict[str, Any], prefix: str) -> tuple[MLPModuleRef, dict[str, Any]] | None:
    ref = split_mlp_prefix(prefix)
    if ref is None:
        return None
    groups = artifact.get("groups", {})
    entry = groups.get(ref.group)
    if entry is None:
        return None
    if ref.role not in set(entry.get("roles", [])):
        return None
    return ref, entry


def summarize_artifact_targets(artifact: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for group, entry in artifact.get("groups", {}).items():
        roles = [str(x) for x in entry.get("roles", [])]
        out[str(group)] = [f"{group}.{role}" if group else role for role in roles]
    return out


def parse_layer_index(group: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", group)
    if not match:
        return None
    return int(match.group(1))


def make_keep_indices(scores, target_sparsity: float, block_i: int):
    import torch

    total = int(scores.numel())
    if total <= 0:
        raise ValueError("scores must be non-empty")
    block_i = max(1, int(block_i))
    num_blocks = int(math.ceil(total / block_i))
    prune_blocks, actual_sparsity = target_to_blocks(target_sparsity, num_blocks)
    if prune_blocks == 0:
        keep = torch.arange(total, device=scores.device, dtype=torch.long)
        return keep, 0.0, num_blocks, num_blocks

    block_scores = torch.empty((num_blocks,), device=scores.device, dtype=torch.float32)
    scores_f = scores.float()
    for block_idx in range(num_blocks):
        start = block_idx * block_i
        end = min(start + block_i, total)
        block_scores[block_idx] = scores_f[start:end].mean()
    keep_blocks = torch.topk(
        block_scores,
        k=num_blocks - prune_blocks,
        largest=True,
        sorted=True,
    ).indices.sort().values
    offsets = torch.arange(block_i, device=scores.device, dtype=torch.long)
    keep = (keep_blocks[:, None] * block_i + offsets[None, :]).reshape(-1)
    keep = keep[keep < total].contiguous()
    return keep, actual_sparsity, int(keep_blocks.numel()), num_blocks
