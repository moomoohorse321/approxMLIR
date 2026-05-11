#!/usr/bin/env python3
"""Manifest contract for diffusion quantization candidates."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from site_inventory import SiteRecord, load_inventory


PLAN_EXACT = "exact"
PLAN_LINEAR_W8A16 = "linear_w8a16_triton"
PLAN_LINEAR_W4A16_AWQ = "linear_w4a16_awq_triton"


@dataclass(frozen=True)
class QuantPlan:
    plan_id: str
    kind: str
    quantizer: str
    layout: str
    op: str
    params: dict[str, Any]
    artifact_types: list[str]
    kernel_id: str
    kernel_abi_id: str
    fallback: str = PLAN_EXACT


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    site_id: str
    plan_id: str
    artifact_type: str
    source: str
    uri_scheme: str
    uri: str
    dtype: str
    shape: list[int]
    layout: str
    byte_size: int | None = None
    sha256: str | None = None
    build_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BindingRecord:
    site_id: str
    default_plan: str
    regime_plans: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRecord:
    tuning_id: str
    site_id: str
    candidate_plan: str
    selected: bool
    metrics: dict[str, Any]
    reason: str = ""


@dataclass(frozen=True)
class QuantManifest:
    schema_version: int
    model: str
    model_fingerprint: str
    strict: bool
    sites: list[dict[str, Any]]
    plans: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]
    bindings: list[dict[str, Any]]
    tuning_evidence: list[dict[str, Any]]
    negative_evidence: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def default_plans(
    *,
    block_m: int = 0,
    block_n: int = 128,
    block_k: int = 64,
    group_k: int = 64,
    use_substitute: bool = True,
) -> list[QuantPlan]:
    return [
        QuantPlan(
            plan_id=PLAN_EXACT,
            kind="exact",
            quantizer="none",
            layout="fp",
            op="any",
            params={},
            artifact_types=[],
            kernel_id="torch_exact",
            kernel_abi_id="torch_exact.v1",
            fallback=PLAN_EXACT,
        ),
        QuantPlan(
            plan_id=PLAN_LINEAR_W8A16,
            kind="w8a16",
            quantizer="int8_per_output_channel",
            layout="int8_col",
            op="linear",
            params={
                "block_m": int(block_m),
                "block_n": int(block_n),
                "block_k": int(block_k),
                "use_substitute": bool(use_substitute),
            },
            artifact_types=["qweight_i8_t", "qweight_scale"],
            kernel_id="diffusion_w8a16_linear",
            kernel_abi_id="triton_w8a16_linear.v1",
            fallback=PLAN_EXACT,
        ),
        QuantPlan(
            plan_id=PLAN_LINEAR_W4A16_AWQ,
            kind="w4a16_awq",
            quantizer="int4_groupwise_awq",
            layout="uint4_packed_col",
            op="linear",
            params={
                "block_m": int(block_m),
                "block_n": int(block_n),
                "block_k": int(block_k),
                "group_k": int(group_k),
                "use_substitute": bool(use_substitute),
            },
            artifact_types=["qweight_i4_t_packed", "qweight_scale_g", "act_scale_inv"],
            kernel_id="diffusion_w4a16_linear",
            kernel_abi_id="triton_w4a16_linear.v1",
            fallback=PLAN_EXACT,
        ),
    ]


def _site_matches(site: SiteRecord, patterns: list[str]) -> bool:
    if not patterns:
        return False
    haystack = f"{site.site_id}\n{site.module_path}\n{site.component}"
    for pattern in patterns:
        if pattern == "all":
            return True
        if pattern in haystack:
            return True
        try:
            if re.search(pattern, haystack):
                return True
        except re.error:
            continue
    return False


def build_manifest(
    *,
    model: str,
    model_fingerprint: str,
    sites: list[SiteRecord],
    target_patterns: list[str],
    strict: bool = False,
    block_m: int = 0,
    block_n: int = 128,
    block_k: int = 64,
    group_k: int = 64,
    use_substitute: bool = True,
    quant_plan: str = PLAN_LINEAR_W8A16,
) -> QuantManifest:
    if quant_plan in ("w8a16", "linear_w8a16"):
        quant_plan = PLAN_LINEAR_W8A16
    elif quant_plan in ("w4a16", "w4a16_awq", "linear_w4a16_awq"):
        quant_plan = PLAN_LINEAR_W4A16_AWQ
    if quant_plan not in (PLAN_LINEAR_W8A16, PLAN_LINEAR_W4A16_AWQ):
        raise ValueError(f"unsupported quant_plan: {quant_plan}")
    plans = default_plans(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_k=group_k,
        use_substitute=use_substitute,
    )
    bindings: list[BindingRecord] = []
    artifacts: list[ArtifactRecord] = []
    for site in sites:
        if site.op != "linear" or not _site_matches(site, target_patterns):
            continue
        bindings.append(BindingRecord(site_id=site.site_id, default_plan=quant_plan))
        k = int(site.shape["in_features"])
        n = int(site.shape["out_features"])
        if quant_plan == PLAN_LINEAR_W8A16:
            artifacts.extend(
                [
                    ArtifactRecord(
                        artifact_id=f"{site.stable_key}:qweight_i8_t",
                        site_id=site.site_id,
                        plan_id=PLAN_LINEAR_W8A16,
                        artifact_type="qweight_i8_t",
                        source="load_time",
                        uri_scheme="memory",
                        uri="module_buffer:_approx_diffusion_qweight_i8_t",
                        dtype="int8",
                        shape=[k, n],
                        layout="int8_col",
                        byte_size=k * n,
                        build_params={"quantizer": "int8_per_output_channel"},
                    ),
                    ArtifactRecord(
                        artifact_id=f"{site.stable_key}:qweight_scale",
                        site_id=site.site_id,
                        plan_id=PLAN_LINEAR_W8A16,
                        artifact_type="qweight_scale",
                        source="load_time",
                        uri_scheme="memory",
                        uri="module_buffer:_approx_diffusion_qweight_scale",
                        dtype="float32",
                        shape=[n],
                        layout="per_output_channel",
                        byte_size=4 * n,
                        build_params={"quantizer": "int8_per_output_channel"},
                    ),
                ]
            )
        else:
            num_groups = (k + group_k - 1) // group_k
            artifacts.extend(
                [
                    ArtifactRecord(
                        artifact_id=f"{site.stable_key}:qweight_i4_t_packed",
                        site_id=site.site_id,
                        plan_id=PLAN_LINEAR_W4A16_AWQ,
                        artifact_type="qweight_i4_t_packed",
                        source="load_time",
                        uri_scheme="memory",
                        uri="module_buffer:_approx_diffusion_qweight_i4_t_packed",
                        dtype="uint8",
                        shape=[(k + 1) // 2, n],
                        layout="uint4_packed_col",
                        byte_size=((k + 1) // 2) * n,
                        build_params={"quantizer": "int4_groupwise_awq", "group_k": group_k},
                    ),
                    ArtifactRecord(
                        artifact_id=f"{site.stable_key}:qweight_scale_g",
                        site_id=site.site_id,
                        plan_id=PLAN_LINEAR_W4A16_AWQ,
                        artifact_type="qweight_scale_g",
                        source="load_time",
                        uri_scheme="memory",
                        uri="module_buffer:_approx_diffusion_qweight_scale_g",
                        dtype="float16",
                        shape=[num_groups, n],
                        layout="groupwise_per_output_channel",
                        byte_size=2 * num_groups * n,
                        build_params={"quantizer": "int4_groupwise_awq", "group_k": group_k},
                    ),
                    ArtifactRecord(
                        artifact_id=f"{site.stable_key}:act_scale_inv",
                        site_id=site.site_id,
                        plan_id=PLAN_LINEAR_W4A16_AWQ,
                        artifact_type="act_scale_inv",
                        source="load_time",
                        uri_scheme="memory",
                        uri="module_buffer:_approx_diffusion_act_scale_inv",
                        dtype="float16",
                        shape=[k],
                        layout="per_input_channel",
                        byte_size=2 * k,
                        build_params={"quantizer": "int4_groupwise_awq", "group_k": group_k},
                    ),
                ]
            )
    manifest = QuantManifest(
        schema_version=1,
        model=model,
        model_fingerprint=model_fingerprint,
        strict=strict,
        sites=[site.to_json() for site in sites],
        plans=[asdict(plan) for plan in plans],
        artifacts=[asdict(artifact) for artifact in artifacts],
        bindings=[asdict(binding) for binding in bindings],
        tuning_evidence=[],
        negative_evidence=[],
    )
    validate_manifest(manifest)
    return manifest


def load_manifest(path: Path) -> QuantManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = QuantManifest(**payload)
    validate_manifest(manifest)
    return manifest


def write_manifest(path: Path, manifest: QuantManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json(), indent=2, sort_keys=True), encoding="utf-8")


def validate_manifest(manifest: QuantManifest) -> None:
    if manifest.schema_version != 1:
        raise ValueError(f"unsupported manifest schema_version: {manifest.schema_version}")
    site_ids = [site["site_id"] for site in manifest.sites]
    plan_ids = [plan["plan_id"] for plan in manifest.plans]
    artifact_ids = [artifact["artifact_id"] for artifact in manifest.artifacts]
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("duplicate site_id in manifest")
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("duplicate plan_id in manifest")
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("duplicate artifact_id in manifest")
    site_set = set(site_ids)
    plan_set = set(plan_ids)
    for binding in manifest.bindings:
        if binding["site_id"] not in site_set:
            raise ValueError(f"binding references unknown site: {binding['site_id']}")
        if binding["default_plan"] not in plan_set:
            raise ValueError(f"binding references unknown plan: {binding['default_plan']}")
        for regime, plan_id in binding.get("regime_plans", {}).items():
            if plan_id not in plan_set:
                raise ValueError(f"binding regime {regime} references unknown plan: {plan_id}")
    for artifact in manifest.artifacts:
        if artifact["site_id"] not in site_set:
            raise ValueError(f"artifact references unknown site: {artifact['site_id']}")
        if artifact["plan_id"] not in plan_set:
            raise ValueError(f"artifact references unknown plan: {artifact['plan_id']}")


def bindings_by_site(manifest: QuantManifest) -> dict[str, BindingRecord]:
    return {record["site_id"]: BindingRecord(**record) for record in manifest.bindings}


def plans_by_id(manifest: QuantManifest) -> dict[str, QuantPlan]:
    return {record["plan_id"]: QuantPlan(**record) for record in manifest.plans}


def sites_by_module_path(manifest: QuantManifest) -> dict[str, SiteRecord]:
    return {record["module_path"]: SiteRecord(**record) for record in manifest.sites}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a diffusion quantization manifest from inventory.")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-fingerprint", default=None)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--block-m", type=int, default=0)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--group-k", type=int, default=64)
    parser.add_argument("--quant-plan", choices=["w8a16", "w4a16_awq"], default="w8a16")
    parser.add_argument("--no-substitute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sites = load_inventory(args.inventory)
    manifest = build_manifest(
        model=args.model,
        model_fingerprint=args.model_fingerprint or args.model,
        sites=sites,
        target_patterns=args.target or ["unet"],
        strict=args.strict,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        group_k=args.group_k,
        quant_plan=args.quant_plan,
        use_substitute=not args.no_substitute,
    )
    write_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "num_sites": len(manifest.sites),
                "num_bindings": len(manifest.bindings),
                "num_artifacts": len(manifest.artifacts),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
