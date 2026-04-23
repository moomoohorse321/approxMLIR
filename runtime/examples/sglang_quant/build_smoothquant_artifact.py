#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _load_layers(path: Path) -> dict[str, torch.Tensor]:
    rec = torch.load(path, map_location="cpu")
    if isinstance(rec, dict) and isinstance(rec.get("layers"), dict):
        return {str(k): torch.as_tensor(v).float().cpu() for k, v in rec["layers"].items()}
    if isinstance(rec, dict):
        return {str(k): torch.as_tensor(v).float().cpu() for k, v in rec.items()}
    raise ValueError(f"unsupported stats shard format: {path}")


def main() -> int:
    stats_dir = Path(os.environ.get("APPROX_SGLANG_SQ_STATS_DIR", ""))
    artifact_path = Path(os.environ.get("APPROX_SGLANG_SQ_ARTIFACT_PATH", ""))
    if not str(stats_dir):
        raise SystemExit("APPROX_SGLANG_SQ_STATS_DIR is required")
    if not str(artifact_path):
        raise SystemExit("APPROX_SGLANG_SQ_ARTIFACT_PATH is required")

    shard_paths = sorted(stats_dir.glob("sq_stats_*.pt"))
    if not shard_paths:
        raise SystemExit(f"no sq_stats_*.pt files found in {stats_dir}")

    merged: dict[str, torch.Tensor] = {}
    for path in shard_paths:
        for prefix, tensor in _load_layers(path).items():
            prev = merged.get(prefix)
            if prev is None or prev.numel() != tensor.numel():
                merged[prefix] = tensor
            else:
                merged[prefix] = torch.maximum(prev, tensor)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"layers": merged, "num_shards": len(shard_paths)}, artifact_path)
    print(
        json.dumps(
            {
                "artifact_path": str(artifact_path),
                "num_layers": len(merged),
                "num_shards": len(shard_paths),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
