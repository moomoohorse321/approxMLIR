#!/usr/bin/env python3
"""Stable candidate-site inventory for the diffusion benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SiteRecord:
    site_id: str
    stable_key: str
    op: str
    component: str
    module_path: str
    module_type: str
    shape: dict[str, int]
    dtype: dict[str, list[str]]
    weight_fingerprint: str
    matchers: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def tensor_fingerprint(tensor) -> str:
    import torch

    detached = tensor.detach().contiguous().cpu()
    payload = [
        str(detached.dtype).encode("utf-8"),
        json.dumps(list(detached.shape)).encode("utf-8"),
        detached.view(torch.uint8).numpy().tobytes(),
    ]
    return _sha256_bytes(payload)


def stable_site_key(
    *,
    model_fingerprint: str,
    module_path: str,
    op: str,
    shape: dict[str, int],
    dtype: dict[str, list[str]],
    weight_fingerprint: str,
) -> str:
    payload = {
        "model_fingerprint": model_fingerprint,
        "module_path": module_path,
        "op": op,
        "shape": shape,
        "dtype": dtype,
        "weight_fingerprint": weight_fingerprint,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _site_id(model_fingerprint: str, module_path: str, op: str) -> str:
    safe_model = model_fingerprint.replace(":", "_").replace("/", "_")
    return f"{safe_model}:{module_path}:{op}:replica:0/1"


def _iter_named_module_roots(obj) -> Iterable[tuple[str, Any]]:
    try:
        import torch.nn as nn
    except Exception:
        return

    if isinstance(obj, nn.Module):
        yield "model", obj
        return
    for name in (
        "memory_bound_conditioner",
        "text_encoder",
        "text_encoder_2",
        "unet",
        "transformer",
        "vae",
    ):
        component = getattr(obj, name, None)
        if isinstance(component, nn.Module):
            yield name, component


def inventory_model(model: Any, *, model_fingerprint: str = "unknown") -> list[SiteRecord]:
    """Return deterministic linear/conv candidate sites for a module or pipeline."""

    import torch.nn as nn

    sites: list[SiteRecord] = []
    for component_name, root in _iter_named_module_roots(model):
        for local_name, module in root.named_modules():
            if not local_name:
                continue
            module_path = f"{component_name}.{local_name}"
            if isinstance(module, nn.Linear):
                weight = module.weight
                shape = {
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                }
                dtype = {
                    "weight": [str(weight.dtype).replace("torch.", "")],
                    "activation": ["float16", "bfloat16", "float32"],
                }
                fingerprint = tensor_fingerprint(weight)
                stable_key = stable_site_key(
                    model_fingerprint=model_fingerprint,
                    module_path=module_path,
                    op="linear",
                    shape=shape,
                    dtype=dtype,
                    weight_fingerprint=fingerprint,
                )
                sites.append(
                    SiteRecord(
                        site_id=_site_id(model_fingerprint, module_path, "linear"),
                        stable_key=stable_key,
                        op="linear",
                        component=component_name,
                        module_path=module_path,
                        module_type=type(module).__name__,
                        shape=shape,
                        dtype=dtype,
                        weight_fingerprint=fingerprint,
                        matchers=[
                            {"type": "exact", "value": module_path, "priority": 100},
                            {"type": "contains", "value": local_name, "priority": 10},
                        ],
                    )
                )
            elif isinstance(module, nn.Conv2d):
                weight = module.weight
                shape = {
                    "in_channels": int(module.in_channels),
                    "out_channels": int(module.out_channels),
                    "kernel_h": int(module.kernel_size[0]),
                    "kernel_w": int(module.kernel_size[1]),
                    "groups": int(module.groups),
                }
                dtype = {
                    "weight": [str(weight.dtype).replace("torch.", "")],
                    "activation": ["float16", "bfloat16", "float32"],
                }
                fingerprint = tensor_fingerprint(weight)
                stable_key = stable_site_key(
                    model_fingerprint=model_fingerprint,
                    module_path=module_path,
                    op="conv2d",
                    shape=shape,
                    dtype=dtype,
                    weight_fingerprint=fingerprint,
                )
                sites.append(
                    SiteRecord(
                        site_id=_site_id(model_fingerprint, module_path, "conv2d"),
                        stable_key=stable_key,
                        op="conv2d",
                        component=component_name,
                        module_path=module_path,
                        module_type=type(module).__name__,
                        shape=shape,
                        dtype=dtype,
                        weight_fingerprint=fingerprint,
                        matchers=[
                            {"type": "exact", "value": module_path, "priority": 100},
                            {"type": "contains", "value": local_name, "priority": 10},
                        ],
                    )
                )
    sites.sort(key=lambda site: site.site_id)
    _validate_unique_sites(sites)
    return sites


def _validate_unique_sites(sites: list[SiteRecord]) -> None:
    site_ids = [site.site_id for site in sites]
    stable_keys = [site.stable_key for site in sites]
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("duplicate site_id in inventory")
    if len(stable_keys) != len(set(stable_keys)):
        raise ValueError("duplicate stable_key in inventory")


def inventory_summary(sites: list[SiteRecord]) -> dict[str, Any]:
    by_op: dict[str, int] = {}
    by_component: dict[str, int] = {}
    for site in sites:
        by_op[site.op] = by_op.get(site.op, 0) + 1
        by_component[site.component] = by_component.get(site.component, 0) + 1
    return {
        "num_sites": len(sites),
        "by_op": dict(sorted(by_op.items())),
        "by_component": dict(sorted(by_component.items())),
    }


def write_inventory(path: Path, sites: list[SiteRecord]) -> None:
    payload = {
        "schema_version": 1,
        "summary": inventory_summary(sites),
        "sites": [site.to_json() for site in sites],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_inventory(path: Path) -> list[SiteRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sites = [SiteRecord(**record) for record in payload.get("sites", [])]
    _validate_unique_sites(sites)
    return sites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect staged Diffusers candidate sites.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from diffusers import StableDiffusionPipeline

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.precision]
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        local_files_only=not args.allow_download,
        safety_checker=None,
        requires_safety_checker=False,
    )
    sites = inventory_model(pipe, model_fingerprint=args.model_id)
    write_inventory(args.output, sites)
    print(json.dumps(inventory_summary(sites), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
