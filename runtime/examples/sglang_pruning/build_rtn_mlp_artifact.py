#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rtn_mlp_pruning import make_keep_indices, split_mlp_prefix


def _parse_layers(value: str) -> set[int] | None:
    if not value:
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(part))
    return out


def _layer_index(group: str) -> int | None:
    pieces = group.split(".")
    for i, piece in enumerate(pieces[:-1]):
        if piece == "layers":
            return int(pieces[i + 1])
    return None


def _linear_weights(model) -> dict[str, torch.Tensor]:
    out = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            out[name] = weight.detach().cpu()
    return out


def _collect_groups(weights: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    groups: dict[str, dict[str, torch.Tensor]] = {}
    for prefix, weight in weights.items():
        ref = split_mlp_prefix(prefix)
        if ref is None:
            continue
        groups.setdefault(ref.group, {})[ref.role] = weight
    return groups


def _scores_for_group(mods: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[str], str] | None:
    if "gate_up_proj" in mods and "down_proj" in mods:
        gate_up = mods["gate_up_proj"].float().abs()
        if gate_up.shape[0] % 2 != 0:
            return None
        intermediate = gate_up.shape[0] // 2
        scores = gate_up[:intermediate].mean(dim=1) + gate_up[intermediate:].mean(dim=1)
        down = mods["down_proj"].float().abs()
        if down.shape[1] == intermediate:
            scores = scores + down.mean(dim=0)
        return scores, ["gate_up_proj", "down_proj"], "fused_gate_up"
    if {"gate_proj", "up_proj", "down_proj"}.issubset(mods):
        gate = mods["gate_proj"].float().abs()
        up = mods["up_proj"].float().abs()
        down = mods["down_proj"].float().abs()
        if gate.shape[0] != up.shape[0] or down.shape[1] != gate.shape[0]:
            return None
        scores = gate.mean(dim=1) + up.mean(dim=1) + down.mean(dim=0)
        return scores, ["gate_up_proj", "gate_proj", "up_proj", "down_proj"], "split_gate_up"
    return None


def build_artifact(
    *,
    model_path: str,
    output: Path,
    target_sparsity: float,
    block_i: int,
    layers: set[int] | None,
    local_files_only: bool,
) -> dict:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        local_files_only=local_files_only,
    )
    model.to("cpu")
    weights = _linear_weights(model)
    groups = _collect_groups(weights)
    artifact_groups = {}
    skipped = {}
    for group, mods in sorted(groups.items()):
        layer_idx = _layer_index(group)
        if layers is not None and layer_idx not in layers:
            continue
        scored = _scores_for_group(mods)
        if scored is None:
            skipped[group] = sorted(mods)
            continue
        scores, roles, layout = scored
        keep, actual_sparsity, kept_blocks, total_blocks = make_keep_indices(
            scores,
            target_sparsity=target_sparsity,
            block_i=block_i,
        )
        artifact_groups[group] = {
            "layout": layout,
            "roles": roles,
            "layer_index": layer_idx,
            "block_i": int(block_i),
            "target_sparsity": float(target_sparsity),
            "actual_sparsity": float(actual_sparsity),
            "hidden_size": int(next(iter(mods.values())).shape[1]),
            "intermediate_size": int(scores.numel()),
            "kept_intermediate": int(keep.numel()),
            "kept_blocks": int(kept_blocks),
            "total_blocks": int(total_blocks),
            "kept_intermediate_indices": [int(x) for x in keep.tolist()],
        }
    payload = {
        "schema_version": 1,
        "method": "rtn_mlp_magnitude_prune",
        "model_path": model_path,
        "target_sparsity": float(target_sparsity),
        "block_i": int(block_i),
        "groups": artifact_groups,
        "skipped_groups": skipped,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RTN-style MLP pruning artifact")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--block-i", type=int, default=64)
    parser.add_argument("--layers", default="", help="comma list/ranges, e.g. 0,4,8-12")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    payload = build_artifact(
        model_path=args.model,
        output=args.output,
        target_sparsity=args.sparsity,
        block_i=args.block_i,
        layers=_parse_layers(args.layers),
        local_files_only=not args.allow_download,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_groups": len(payload["groups"]),
                "target_sparsity": payload["target_sparsity"],
                "block_i": payload["block_i"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
