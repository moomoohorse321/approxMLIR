#!/usr/bin/env python3
"""Runtime Diffusers module substitution for diffusion quantization."""

from __future__ import annotations

import json
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from approx_kernels import (
    approx_diffusion_w4a16_linear_kernel_1,
    approx_diffusion_w8a16_linear_kernel_1,
    diffusion_w4a16_linear_kernel,
    diffusion_w8a16_linear_kernel,
    quantize_weight_i4_groupwise_packed,
    quantize_weight_i8_per_col,
)
from approx_manifest import (
    PLAN_LINEAR_W4A16_AWQ,
    PLAN_LINEAR_W8A16,
    QuantManifest,
    bindings_by_site,
    plans_by_id,
    sites_by_module_path,
)


@dataclass
class SubstitutionConfig:
    use_substitute: bool = True
    block_m: int = 0
    block_n: int = 128
    block_k: int = 64
    group_k: int = 64
    strict: bool = False
    drop_original_weights: bool = False
    trace_apply_events: bool = False
    stats_path: Path | None = None
    plugin_path: str = ""


@dataclass
class BoundSite:
    site_id: str
    module_path: str
    plan_id: str
    hits: int = 0
    misses: int = 0
    fallbacks: int = 0
    substituted_hits: int = 0


class DiffusionSubstitutionManager:
    def __init__(self, manifest: QuantManifest, config: SubstitutionConfig):
        self.manifest = manifest
        self.config = config
        self.bound_sites: dict[str, BoundSite] = {}
        self.events: list[dict[str, Any]] = []
        self._hook_installed = False
        self._hook_disabled_after_error = False

    def record(self, event: dict[str, Any]) -> None:
        payload = {"ts": time.time(), **event}
        self.events.append(payload)
        if self.config.stats_path is not None:
            self.config.stats_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.stats_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def install(self, pipe: Any) -> None:
        import torch.nn as nn

        site_by_path = sites_by_module_path(self.manifest)
        binding_by_site = bindings_by_site(self.manifest)
        plan_by_id = plans_by_id(self.manifest)
        selected_plan_ids = {
            binding.default_plan
            for binding in binding_by_site.values()
            if binding.default_plan in (PLAN_LINEAR_W8A16, PLAN_LINEAR_W4A16_AWQ)
        }
        if self.config.use_substitute:
            if len(selected_plan_ids) > 1 and (self.manifest.strict or self.config.strict):
                raise RuntimeError("mixed W8/W4 substitution hooks are not supported in one run")
            if selected_plan_ids:
                self._install_hook(sorted(selected_plan_ids)[0])

        for module_path, module in _iter_pipeline_modules(pipe):
            if not isinstance(module, nn.Linear):
                continue
            site = site_by_path.get(module_path)
            if site is None:
                continue
            binding = binding_by_site.get(site.site_id)
            if binding is None or binding.default_plan not in (PLAN_LINEAR_W8A16, PLAN_LINEAR_W4A16_AWQ):
                continue
            plan = plan_by_id[binding.default_plan]
            try:
                self._bind_linear(module_path, module, site.site_id, plan.plan_id, plan.params)
            except Exception as exc:
                self.record(
                    {
                        "event": "bind_failed",
                        "site_id": site.site_id,
                        "module_path": module_path,
                        "reason": repr(exc),
                    }
                )
                if self.manifest.strict or self.config.strict:
                    raise

    def _install_hook(self, plan_id: str) -> None:
        if self._hook_installed:
            return
        plugin_path = self.config.plugin_path or os.environ.get("TRITON_PASS_PLUGIN_PATH", "")
        if not plugin_path:
            if self.config.strict:
                raise RuntimeError("TRITON_PASS_PLUGIN_PATH is required for approxMLIR substitution")
            self.record({"event": "skip_hook", "reason": "missing_plugin_path"})
            self.config.use_substitute = False
            return

        runtime_dir = Path(__file__).resolve().parents[2]
        if str(runtime_dir) not in sys.path:
            sys.path.insert(0, str(runtime_dir))

        import approx_runtime as ar
        import approx_substitution_state as state
        from triton import knobs
        if plan_id == PLAN_LINEAR_W4A16_AWQ:
            func_name = "diffusion_w4a16_linear_kernel"
            approx_kernel = approx_diffusion_w4a16_linear_kernel_1
        else:
            func_name = "diffusion_w8a16_linear_kernel"
            approx_kernel = approx_diffusion_w8a16_linear_kernel_1

        config = {
            "decision_tree": None,
            "safety_contract": None,
            "static_transform": ar.StaticTransform(
                transform_type="func_substitute",
                knob_val=1,
                approx_kernel=approx_kernel,
            ),
        }
        hook = ar.make_triton_stages_hook(
            passes=ar.get_pipeline_for_config(config, workload=ar.WorkloadType.TRITON),
            plugin_path=plugin_path,
            stage_name="make_ttir_approx",
            func_name=func_name,
            config=config,
            extra_ttir_texts=state.extra_ttir_texts,
            verbose=os.environ.get("APPROX_DIFFUSION_VERBOSE", "0") == "1",
        )
        knobs.runtime.add_stages_inspection_hook = hook
        self._hook_installed = True
        self.record({"event": "install_hook", "plugin_path": plugin_path, "plan_id": plan_id})

    def disable_hook_after_error(self, reason: str) -> None:
        if self._hook_disabled_after_error:
            return
        try:
            from triton import knobs

            knobs.runtime.add_stages_inspection_hook = None
        except Exception:
            pass
        self.config.use_substitute = False
        self._hook_disabled_after_error = True
        self.record({"event": "disable_hook_after_error", "reason": reason})

    def _bind_linear(
        self,
        module_path: str,
        module: torch.nn.Linear,
        site_id: str,
        plan_id: str,
        params: dict[str, Any],
    ) -> None:
        weight = module.weight
        if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported linear weight dtype/shape: {weight.dtype} {tuple(weight.shape)}")
        if not weight.is_cuda:
            raise ValueError("quantized diffusion substitution currently requires CUDA weights")
        if plan_id == PLAN_LINEAR_W4A16_AWQ:
            qweight, scale, act_scale_inv = quantize_weight_i4_groupwise_packed(
                weight.data,
                group_size=int(params.get("group_k", self.config.group_k)),
            )
            module.register_buffer("_approx_diffusion_qweight_i4_t_packed", qweight, persistent=False)
            module.register_buffer("_approx_diffusion_qweight_scale_g", scale.to(device=weight.device), persistent=False)
            module.register_buffer("_approx_diffusion_act_scale_inv", act_scale_inv, persistent=False)
            qweight_shape = list(qweight.shape)
            scale_shape = list(scale.shape)
        else:
            qweight, scale = quantize_weight_i8_per_col(weight.data)
            module.register_buffer("_approx_diffusion_qweight_i8_t", qweight, persistent=False)
            module.register_buffer("_approx_diffusion_qweight_scale", scale.to(device=weight.device), persistent=False)
            qweight_shape = list(qweight.shape)
            scale_shape = list(scale.shape)
        module._approx_diffusion_original_forward = module.forward
        module._approx_diffusion_site_id = site_id
        module._approx_diffusion_module_path = module_path
        module._approx_diffusion_plan_id = plan_id
        module._approx_diffusion_manager = self
        module._approx_diffusion_block_m = int(params.get("block_m", self.config.block_m))
        module._approx_diffusion_block_n = int(params.get("block_n", self.config.block_n))
        module._approx_diffusion_block_k = int(params.get("block_k", self.config.block_k))
        module._approx_diffusion_group_k = int(params.get("group_k", self.config.group_k))
        if self.config.drop_original_weights:
            module.weight = torch.nn.Parameter(
                torch.empty(0, device=weight.device, dtype=weight.dtype),
                requires_grad=False,
            )
        module.forward = types.MethodType(_linear_forward, module)
        self.bound_sites[site_id] = BoundSite(
            site_id=site_id,
            module_path=module_path,
            plan_id=plan_id,
        )
        self.record(
            {
                "event": "bind_quant_site",
                "site_id": site_id,
                "module_path": module_path,
                "plan_id": plan_id,
                "weight_shape": list(weight.shape),
                "qweight_shape": qweight_shape,
                "scale_shape": scale_shape,
            }
        )

    def before_counts(self) -> dict[str, int]:
        return {
            "sites_hit": sum(site.hits for site in self.bound_sites.values()),
            "substituted_hits": sum(site.substituted_hits for site in self.bound_sites.values()),
            "fallbacks": sum(site.fallbacks for site in self.bound_sites.values()),
        }

    def delta_summary(self, before: dict[str, int]) -> dict[str, Any]:
        now = self.before_counts()
        plans = sorted({site.plan_id for site in self.bound_sites.values()})
        return {
            "mode": "approx",
            "manifest_path": None,
            "sites_expected": len(self.bound_sites),
            "sites_bound": len(self.bound_sites),
            "sites_hit": now["sites_hit"] - before.get("sites_hit", 0),
            "substituted_hits": now["substituted_hits"] - before.get("substituted_hits", 0),
            "fallbacks": now["fallbacks"] - before.get("fallbacks", 0),
            "plan": plans[0] if len(plans) == 1 else "mixed",
            "plans": plans,
        }

    def total_summary(self) -> dict[str, Any]:
        counts = self.before_counts()
        plans = sorted({site.plan_id for site in self.bound_sites.values()})
        return {
            "mode": "approx",
            "sites_expected": len(self.bound_sites),
            "sites_bound": len(self.bound_sites),
            **counts,
            "plan": plans[0] if len(plans) == 1 else "mixed",
            "plans": plans,
        }


def _iter_pipeline_modules(pipe: Any):
    import torch.nn as nn

    for component_name in (
        "memory_bound_conditioner",
        "text_encoder",
        "text_encoder_2",
        "unet",
        "transformer",
        "vae",
    ):
        component = getattr(pipe, component_name, None)
        if not isinstance(component, nn.Module):
            continue
        for local_name, module in component.named_modules():
            if local_name:
                yield f"{component_name}.{local_name}", module


def _linear_forward(self, input: torch.Tensor) -> torch.Tensor:
    manager: DiffusionSubstitutionManager = self._approx_diffusion_manager
    site = manager.bound_sites[self._approx_diffusion_site_id]
    if not input.is_cuda or input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        site.fallbacks += 1
        return self._approx_diffusion_original_forward(input)
    try:
        out = _run_quantized_linear(self, input, manager)
        site.hits += 1
        return out
    except Exception as exc:
        if manager.config.use_substitute and not (manager.config.strict or manager.manifest.strict):
            manager.disable_hook_after_error(repr(exc))
            try:
                out = _run_quantized_linear(self, input, manager)
                site.hits += 1
                return out
            except Exception as retry_exc:
                exc = retry_exc
        manager.record(
            {
                "event": "fallback_exact",
                "site_id": site.site_id,
                "module_path": site.module_path,
                "reason": repr(exc),
            }
        )
        site.fallbacks += 1
        if manager.config.strict or manager.manifest.strict:
            raise
        return self._approx_diffusion_original_forward(input)


def _run_quantized_linear(
    module: torch.nn.Linear,
    input: torch.Tensor,
    manager: DiffusionSubstitutionManager,
) -> torch.Tensor:
    if module._approx_diffusion_plan_id == PLAN_LINEAR_W4A16_AWQ:
        return _run_w4a16_linear(module, input, manager)
    return _run_w8a16_linear(module, input, manager)


def _run_w8a16_linear(module: torch.nn.Linear, input: torch.Tensor, manager: DiffusionSubstitutionManager) -> torch.Tensor:
    import triton
    import approx_substitution_state as state

    x_shape = input.shape
    if x_shape[-1] != module._approx_diffusion_qweight_i8_t.shape[0]:
        raise ValueError(
            f"input K mismatch: got {x_shape[-1]}, expected {module._approx_diffusion_qweight_i8_t.shape[0]}"
        )
    x2d = input.reshape(-1, x_shape[-1]).contiguous()
    bq = module._approx_diffusion_qweight_i8_t
    bs = module._approx_diffusion_qweight_scale
    m, k = x2d.shape
    n = bq.shape[1]
    block_m = _select_block_m(m, int(module._approx_diffusion_block_m))
    block_n = int(module._approx_diffusion_block_n)
    block_k = int(module._approx_diffusion_block_k)
    out = torch.empty((m, n), device=x2d.device, dtype=x2d.dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    args = (
        x2d,
        bq,
        bs,
        out,
        m,
        n,
        k,
        x2d.stride(0),
        x2d.stride(1),
        bq.stride(0),
        bq.stride(1),
        out.stride(0),
        out.stride(1),
    )
    kwargs = dict(BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    substituted = False
    if manager.config.use_substitute:
        shape_key = (
            "w8a16",
            str(x2d.dtype),
            m,
            n,
            k,
            x2d.stride(0),
            x2d.stride(1),
            bq.stride(0),
            bq.stride(1),
            out.stride(0),
            out.stride(1),
            block_m,
            block_n,
            block_k,
        )
        if shape_key not in state.ttir_by_shape:
            helper = approx_diffusion_w8a16_linear_kernel_1[grid](*args, **kwargs)
            state.ttir_by_shape[shape_key] = helper.asm["ttir"]
            state.seen_shapes.add(shape_key)
        state.extra_ttir_texts[:] = [state.ttir_by_shape[shape_key]]
    handle = diffusion_w8a16_linear_kernel[grid](*args, **kwargs)
    if "ttir" in handle.asm:
        substituted = "approx_diffusion_w8a16_linear_kernel_1" in handle.asm["ttir"]
    if substituted:
        manager.bound_sites[module._approx_diffusion_site_id].substituted_hits += 1
    if manager.config.trace_apply_events:
        manager.record(
            {
                "event": "apply_approx",
                "site_id": module._approx_diffusion_site_id,
                "module_path": module._approx_diffusion_module_path,
                "M": m,
                "N": n,
                "K": k,
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "substituted": substituted,
            }
        )
    if module.bias is not None:
        out = out + module.bias
    return out.reshape(*x_shape[:-1], n)


def _run_w4a16_linear(module: torch.nn.Linear, input: torch.Tensor, manager: DiffusionSubstitutionManager) -> torch.Tensor:
    import triton
    import approx_substitution_state as state

    x_shape = input.shape
    if x_shape[-1] != module._approx_diffusion_act_scale_inv.shape[0]:
        raise ValueError(
            f"input K mismatch: got {x_shape[-1]}, expected {module._approx_diffusion_act_scale_inv.shape[0]}"
        )
    x2d = input.reshape(-1, x_shape[-1]).contiguous()
    bq = module._approx_diffusion_qweight_i4_t_packed
    bs = module._approx_diffusion_qweight_scale_g
    act_scale_inv = module._approx_diffusion_act_scale_inv
    m, k = x2d.shape
    n = bq.shape[1]
    block_m = _select_block_m(m, int(module._approx_diffusion_block_m))
    block_n = int(module._approx_diffusion_block_n)
    block_k = int(module._approx_diffusion_block_k)
    group_k = int(module._approx_diffusion_group_k)
    if block_k <= 0 or block_k % 2 != 0:
        raise ValueError(f"invalid W4 block_k: {block_k}")
    if group_k <= 0 or block_k > group_k or group_k % block_k != 0:
        raise ValueError(f"invalid W4 block/group combination: block_k={block_k}, group_k={group_k}")

    out = torch.empty((m, n), device=x2d.device, dtype=x2d.dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    args = (
        x2d,
        bq,
        bs,
        act_scale_inv,
        out,
        m,
        n,
        k,
        x2d.stride(0),
        x2d.stride(1),
        bq.stride(0),
        bq.stride(1),
        bs.stride(0),
        bs.stride(1),
        out.stride(0),
        out.stride(1),
    )
    kwargs = dict(BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_K=group_k)
    substituted = False
    if manager.config.use_substitute:
        shape_key = (
            "w4a16",
            str(x2d.dtype),
            m,
            n,
            k,
            x2d.stride(0),
            x2d.stride(1),
            bq.stride(0),
            bq.stride(1),
            bs.stride(0),
            bs.stride(1),
            out.stride(0),
            out.stride(1),
            block_m,
            block_n,
            block_k,
            group_k,
        )
        if shape_key not in state.ttir_by_shape:
            helper = approx_diffusion_w4a16_linear_kernel_1[grid](*args, **kwargs)
            state.ttir_by_shape[shape_key] = helper.asm["ttir"]
            state.seen_shapes.add(shape_key)
        state.extra_ttir_texts[:] = [state.ttir_by_shape[shape_key]]
    handle = diffusion_w4a16_linear_kernel[grid](*args, **kwargs)
    if "ttir" in handle.asm:
        substituted = "approx_diffusion_w4a16_linear_kernel_1" in handle.asm["ttir"]
    if substituted:
        manager.bound_sites[module._approx_diffusion_site_id].substituted_hits += 1
    if manager.config.trace_apply_events:
        manager.record(
            {
                "event": "apply_approx",
                "site_id": module._approx_diffusion_site_id,
                "module_path": module._approx_diffusion_module_path,
                "M": m,
                "N": n,
                "K": k,
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "group_k": group_k,
                "substituted": substituted,
            }
        )
    if module.bias is not None:
        out = out + module.bias
    return out.reshape(*x_shape[:-1], n)


def _select_block_m(m: int, configured_block_m: int) -> int:
    if configured_block_m > 0:
        return configured_block_m
    if m <= 1:
        return 1
    if m <= 16:
        return 16
    if m <= 96:
        return 32
    if m <= 256:
        return 64
    return 32
