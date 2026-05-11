from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


def _enabled() -> bool:
    return os.environ.get("APPROX_SGLANG_PRUNING", "0") == "1"


def _record(event: dict) -> None:
    path = os.environ.get("APPROX_SGLANG_PRUNING_STATS_PATH", "")
    if not path:
        return
    event = {"ts": time.time(), "pid": os.getpid(), **event}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def _dump_ttir(event: dict, ttir: str) -> None:
    out_dir = os.environ.get("APPROX_SGLANG_DUMP_OUT_DIR", "")
    if not out_dir or not ttir:
        return
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    name = (
        f"pruning_{event.get('backend', 'unknown')}_"
        f"{os.getpid()}_{time.time_ns()}.json"
    )
    payload = {
        "source": "sglang_pruning_patch",
        "func_name": event.get("func_name"),
        "backend": event.get("backend"),
        "prefix": event.get("prefix"),
        "shape": {k: event.get(k) for k in ("M", "N", "K")},
        "substituted": event.get("substituted"),
        "ttir": ttir,
    }
    (path / name).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class _Config:
    mode: str
    target: str
    backend: str
    sparsity: float
    decode_only: bool
    use_substitute: bool
    block_k: int
    block_n: int
    block_m: int
    artifact_path: str

    @classmethod
    def from_env(cls) -> "_Config":
        return cls(
            mode=os.environ.get("APPROX_SGLANG_MODE", "exact"),
            target=os.environ.get("APPROX_SGLANG_TARGET", "gate_up_proj"),
            backend=os.environ.get(
                "APPROX_SGLANG_PRUNE_BACKEND",
                os.environ.get("APPROX_SGLANG_BACKEND", "triton_block_prune"),
            ),
            sparsity=float(os.environ.get("APPROX_SGLANG_PRUNE_SPARSITY", "0.2")),
            decode_only=os.environ.get("APPROX_SGLANG_DECODE_ONLY", "1") == "1",
            use_substitute=os.environ.get("APPROX_SGLANG_USE_SUBSTITUTE", "0") == "1",
            block_k=int(os.environ.get("APPROX_SGLANG_BLOCK_K", "64")),
            block_n=int(os.environ.get("APPROX_SGLANG_BLOCK_N", "128")),
            block_m=int(os.environ.get("APPROX_SGLANG_BLOCK_M", "16")),
            artifact_path=os.environ.get("APPROX_SGLANG_PRUNE_ARTIFACT", ""),
        )

    def target_match(self, layer: torch.nn.Module) -> bool:
        prefix = getattr(layer, "prefix", "")
        if self.target == "all":
            return True
        return any(tok and tok in prefix for tok in self.target.split(","))


def _run_substituted_kernel(
    *,
    backend_name: str,
    primary,
    substitute,
    substitute_name: str,
    grid,
    args: tuple,
    kwargs: dict,
    shape_key: tuple,
    use_substitute: bool,
    record_common: dict,
    layer: torch.nn.Module,
) -> None:
    import approx_substitution_state as state

    if use_substitute:
        if shape_key not in state.ttir_by_shape:
            helper = substitute[grid](*args, **kwargs)
            state.ttir_by_shape[shape_key] = helper.asm["ttir"]
            state.seen_shapes.add(shape_key)
            _record({"event": "prepare_substitute_ttir", "shape_key": list(shape_key)})
        state.extra_ttir_texts[:] = [state.ttir_by_shape[shape_key]]

    handle = primary[grid](*args, **kwargs)
    substituted = substitute_name in handle.asm.get("ttir", "")
    event = {
        "event": "apply_approx",
        "backend": backend_name,
        "func_name": getattr(primary, "__name__", backend_name),
        "substituted": substituted,
        **record_common,
    }
    _record(event)
    _dump_ttir(event, handle.asm.get("ttir", ""))
    if substituted:
        layer._approx_pruning_substitution_hits = getattr(layer, "_approx_pruning_substitution_hits", 0) + 1


def _install() -> None:
    if not _enabled():
        return

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from approx_kernels import (
        approx_sglang_block_prune_linear_kernel_1,
        approx_sglang_compact_k_prune_linear_kernel_1,
        block_magnitude_prune_mask,
        build_compact_k_prune_plan,
        sglang_block_prune_linear_kernel,
        sglang_compact_k_prune_linear_kernel,
    )
    from rtn_mlp_pruning import group_entry_for_prefix, load_artifact, summarize_artifact_targets
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if getattr(UnquantizedLinearMethod, "_approx_sglang_pruning_patched", False):
        return

    from sglang.srt.layers.linear import LinearBase

    if not getattr(LinearBase, "_approx_prefix_patched", False):
        original_linear_init = LinearBase.__init__

        def linear_init_with_prefix(self, *args, **kwargs):
            prefix = kwargs.get("prefix", args[5] if len(args) > 5 else "")
            original_linear_init(self, *args, **kwargs)
            self.prefix = prefix

        LinearBase.__init__ = linear_init_with_prefix
        LinearBase._approx_prefix_patched = True

    original_process = UnquantizedLinearMethod.process_weights_after_loading
    original_apply = UnquantizedLinearMethod.apply
    cfg = _Config.from_env()
    artifact = load_artifact(cfg.artifact_path) if cfg.backend == "rtn_mlp_prune" else {}
    if artifact:
        _record(
            {
                "event": "load_artifact",
                "backend": cfg.backend,
                "artifact_path": cfg.artifact_path,
                "targets": summarize_artifact_targets(artifact),
            }
        )

    def artifact_match(layer: torch.nn.Module):
        if cfg.backend != "rtn_mlp_prune":
            return None
        return group_entry_for_prefix(artifact, getattr(layer, "prefix", ""))

    def should_patch(layer: torch.nn.Module) -> bool:
        if cfg.backend == "rtn_mlp_prune":
            return artifact_match(layer) is not None
        return cfg.target_match(layer)

    def prepare_weight(layer: torch.nn.Module, event: str = "prepare_weight") -> bool:
        weight = getattr(layer, "weight", None)
        if (
            weight is None
            or weight.ndim != 2
            or not weight.is_cuda
            or weight.dtype not in (torch.float16, torch.bfloat16)
        ):
            return False
        if hasattr(layer, "_approx_prune_weight_t") or hasattr(layer, "_approx_rtn_mlp_weight"):
            return True

        artifact_item = artifact_match(layer)
        if artifact_item is not None:
            ref, entry = artifact_item
            keep = torch.tensor(
                [int(x) for x in entry.get("kept_intermediate_indices", [])],
                device=weight.device,
                dtype=torch.long,
            )
            if keep.numel() == 0:
                _record(
                    {
                        "event": "skip_prepare_weight",
                        "backend": cfg.backend,
                        "prefix": getattr(layer, "prefix", ""),
                        "reason": "empty_keep_indices",
                    }
                )
                return False
            compact_bias = None
            bias = getattr(layer, "bias", None)
            if ref.role == "gate_up_proj":
                out_features = int(weight.shape[0])
                if out_features % 2 != 0:
                    return False
                intermediate = out_features // 2
                row_keep = torch.cat([keep, keep + intermediate]).contiguous()
                compact_weight = weight.data.index_select(0, row_keep).contiguous()
                if bias is not None:
                    compact_bias = bias.data.index_select(0, row_keep).contiguous()
                compact_kind = "fused_gate_up_rows"
            elif ref.role in ("gate_proj", "up_proj"):
                compact_weight = weight.data.index_select(0, keep).contiguous()
                if bias is not None:
                    compact_bias = bias.data.index_select(0, keep).contiguous()
                compact_kind = f"{ref.role}_rows"
            elif ref.role == "down_proj":
                compact_weight = weight.data.index_select(1, keep).contiguous()
                if bias is not None:
                    compact_bias = bias.data.contiguous()
                compact_kind = "down_cols"
            else:
                return False

            layer.register_buffer("_approx_rtn_mlp_weight", compact_weight, persistent=False)
            layer.register_buffer("_approx_rtn_mlp_keep", keep.contiguous(), persistent=False)
            if compact_bias is not None:
                layer.register_buffer("_approx_rtn_mlp_bias", compact_bias, persistent=False)
            layer._approx_rtn_mlp_group = ref.group
            layer._approx_rtn_mlp_role = ref.role
            layer._approx_rtn_mlp_compact_kind = compact_kind
            layer._approx_rtn_mlp_actual_sparsity = float(entry.get("actual_sparsity", 0.0))
            layer._approx_rtn_mlp_target_sparsity = float(entry.get("target_sparsity", 0.0))
            layer._approx_rtn_mlp_block_i = int(entry.get("block_i", cfg.block_k))
            _record(
                {
                    "event": event,
                    "backend": cfg.backend,
                    "prefix": getattr(layer, "prefix", ""),
                    "group": ref.group,
                    "role": ref.role,
                    "compact_kind": compact_kind,
                    "weight_shape": list(weight.shape),
                    "compact_weight_shape": list(compact_weight.shape),
                    "target_sparsity": layer._approx_rtn_mlp_target_sparsity,
                    "actual_sparsity": layer._approx_rtn_mlp_actual_sparsity,
                    "block_i": layer._approx_rtn_mlp_block_i,
                    "kept_intermediate": int(keep.numel()),
                }
            )
            return True

        weight_t = weight.data.t().contiguous()
        mask = block_magnitude_prune_mask(
            weight_t,
            cfg.sparsity,
            block_k=cfg.block_k,
            block_n=cfg.block_n,
        ).contiguous()
        plan = build_compact_k_prune_plan(
            weight_t,
            cfg.sparsity,
            block_k=cfg.block_k,
            block_n=cfg.block_n,
            block_mask=mask,
        )
        num_k_blocks = int(mask.shape[0])
        keep_k_blocks = max(1, int(round(num_k_blocks * (1.0 - cfg.sparsity))))
        if cfg.sparsity <= 0.0:
            shared_keep_blocks = torch.arange(num_k_blocks, device=weight.device, dtype=torch.int64)
        else:
            shared_scores = torch.empty((num_k_blocks,), device=weight.device, dtype=torch.float32)
            weight_t_abs = weight_t.detach().float().abs()
            for kb in range(num_k_blocks):
                k0 = kb * cfg.block_k
                k1 = min(k0 + cfg.block_k, weight_t.shape[0])
                shared_scores[kb] = weight_t_abs[k0:k1, :].sum()
            shared_keep_blocks = torch.topk(shared_scores, k=keep_k_blocks, largest=True, sorted=True).indices.to(torch.int64)
            shared_keep_blocks = torch.sort(shared_keep_blocks).values
        k_offsets = torch.arange(cfg.block_k, device=weight.device, dtype=torch.int64)
        shared_keep_indices = (shared_keep_blocks[:, None] * cfg.block_k + k_offsets[None, :]).reshape(-1)
        shared_keep_indices = shared_keep_indices[shared_keep_indices < weight.shape[1]].contiguous()
        compact_weight = weight.data[:, shared_keep_indices].contiguous()
        layer.register_buffer("_approx_prune_weight_t", weight_t, persistent=False)
        layer.register_buffer("_approx_prune_block_mask", mask, persistent=False)
        layer.register_buffer("_approx_prune_kept_k_blocks", plan.kept_k_blocks.contiguous(), persistent=False)
        layer.register_buffer("_approx_prune_kept_counts", plan.kept_counts.contiguous(), persistent=False)
        layer.register_buffer("_approx_prune_shared_keep_indices", shared_keep_indices, persistent=False)
        layer.register_buffer("_approx_prune_native_compact_weight", compact_weight, persistent=False)

        total_blocks = int(mask.numel())
        kept_blocks = int(mask.sum().item()) if total_blocks else 0
        _record(
            {
                "event": event,
                "prefix": getattr(layer, "prefix", ""),
                "backend": cfg.backend,
                "weight_shape": list(weight.shape),
                "weight_t_shape": list(weight_t.shape),
                "sparsity": cfg.sparsity,
                "actual_sparsity": 1.0 - (kept_blocks / total_blocks) if total_blocks else 0.0,
                "block_k": cfg.block_k,
                "block_n": cfg.block_n,
                "kept_blocks": kept_blocks,
                "total_blocks": total_blocks,
                "native_compact_k": int(shared_keep_indices.numel()),
            }
        )
        return True

    def process_weights_after_loading(self, layer):
        original_process(self, layer)
        if cfg.mode == "approx" and should_patch(layer):
            prepare_weight(layer)

    def _apply_block(layer, x2d, x_shape, bias):
        import triton

        b = layer._approx_prune_weight_t
        mask = layer._approx_prune_block_mask
        m, k = x2d.shape
        n = b.shape[1]
        out = torch.empty((m, n), device=x2d.device, dtype=x2d.dtype)
        grid = (triton.cdiv(m, cfg.block_m), triton.cdiv(n, cfg.block_n))
        args = (
            x2d,
            b,
            out,
            mask,
            m,
            n,
            k,
            int(mask.shape[1]),
            x2d.stride(0),
            x2d.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
        )
        kwargs = dict(BLOCK_M=cfg.block_m, BLOCK_K=cfg.block_k, BLOCK_N=cfg.block_n)
        shape_key = (
            "triton_block_prune",
            str(x2d.dtype),
            m,
            n,
            k,
            x2d.stride(0),
            x2d.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            cfg.block_m,
            cfg.block_k,
            cfg.block_n,
        )
        _run_substituted_kernel(
            backend_name="triton_block_prune",
            primary=sglang_block_prune_linear_kernel,
            substitute=approx_sglang_block_prune_linear_kernel_1,
            substitute_name="approx_sglang_block_prune_linear_kernel_1",
            grid=grid,
            args=args,
            kwargs=kwargs,
            shape_key=shape_key,
            use_substitute=cfg.use_substitute,
            record_common={
                "prefix": getattr(layer, "prefix", ""),
                "M": m,
                "N": n,
                "K": k,
                "block_m": cfg.block_m,
                "block_k": cfg.block_k,
                "block_n": cfg.block_n,
                "sparsity": cfg.sparsity,
            },
            layer=layer,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*x_shape[:-1], n)

    def _apply_compact(layer, x2d, x_shape, bias):
        import triton

        b = layer._approx_prune_weight_t
        kept = layer._approx_prune_kept_k_blocks
        counts = layer._approx_prune_kept_counts
        m, k = x2d.shape
        n = b.shape[1]
        out = torch.empty((m, n), device=x2d.device, dtype=x2d.dtype)
        grid = (triton.cdiv(m, cfg.block_m), triton.cdiv(n, cfg.block_n))
        args = (
            x2d,
            b,
            out,
            kept,
            counts,
            m,
            n,
            k,
            int(kept.shape[1]),
            x2d.stride(0),
            x2d.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            kept.stride(0),
            kept.stride(1),
        )
        kwargs = dict(BLOCK_M=cfg.block_m, BLOCK_K=cfg.block_k, BLOCK_N=cfg.block_n)
        shape_key = (
            "triton_compact_k_prune",
            str(x2d.dtype),
            m,
            n,
            k,
            int(kept.shape[1]),
            x2d.stride(0),
            x2d.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            kept.stride(0),
            kept.stride(1),
            cfg.block_m,
            cfg.block_k,
            cfg.block_n,
        )
        _run_substituted_kernel(
            backend_name="triton_compact_k_prune",
            primary=sglang_compact_k_prune_linear_kernel,
            substitute=approx_sglang_compact_k_prune_linear_kernel_1,
            substitute_name="approx_sglang_compact_k_prune_linear_kernel_1",
            grid=grid,
            args=args,
            kwargs=kwargs,
            shape_key=shape_key,
            use_substitute=cfg.use_substitute,
            record_common={
                "prefix": getattr(layer, "prefix", ""),
                "M": m,
                "N": n,
                "K": k,
                "block_m": cfg.block_m,
                "block_k": cfg.block_k,
                "block_n": cfg.block_n,
                "max_keep": int(kept.shape[1]),
                "sparsity": cfg.sparsity,
            },
            layer=layer,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*x_shape[:-1], n)

    def _apply_native_compact(layer, x2d, x_shape, bias):
        keep = layer._approx_prune_shared_keep_indices
        compact_weight = layer._approx_prune_native_compact_weight
        compact_x = torch.index_select(x2d, dim=1, index=keep)
        out = F.linear(compact_x, compact_weight, bias)
        _record(
            {
                "event": "apply_approx",
                "backend": "native_compact_k_prune",
                "func_name": "torch.nn.functional.linear",
                "substituted": False,
                "prefix": getattr(layer, "prefix", ""),
                "M": int(x2d.shape[0]),
                "N": int(compact_weight.shape[0]),
                "K": int(x2d.shape[1]),
                "compact_K": int(compact_weight.shape[1]),
                "block_k": cfg.block_k,
                "block_n": cfg.block_n,
                "sparsity": cfg.sparsity,
            }
        )
        return out.reshape(*x_shape[:-1], compact_weight.shape[0])

    def _apply_rtn_mlp(method_self, layer, x2d, x_shape, bias):
        compact_weight = layer._approx_rtn_mlp_weight
        compact_bias = getattr(layer, "_approx_rtn_mlp_bias", bias)
        role = getattr(layer, "_approx_rtn_mlp_role", "")
        if role == "down_proj" and int(x2d.shape[1]) != int(compact_weight.shape[1]):
            _record(
                {
                    "event": "skip_apply_approx",
                    "backend": "rtn_mlp_prune",
                    "prefix": getattr(layer, "prefix", ""),
                    "group": getattr(layer, "_approx_rtn_mlp_group", ""),
                    "role": role,
                    "reason": "down_input_dim_mismatch",
                    "M": int(x2d.shape[0]),
                    "K": int(x2d.shape[1]),
                    "compact_K": int(compact_weight.shape[1]),
                }
            )
            return original_apply(method_self, layer, x2d.reshape(*x_shape), bias)
        out = F.linear(x2d, compact_weight, compact_bias)
        _record(
            {
                "event": "apply_approx",
                "backend": "rtn_mlp_prune",
                "func_name": "torch.nn.functional.linear",
                "substituted": False,
                "prefix": getattr(layer, "prefix", ""),
                "group": getattr(layer, "_approx_rtn_mlp_group", ""),
                "role": role,
                "compact_kind": getattr(layer, "_approx_rtn_mlp_compact_kind", ""),
                "M": int(x2d.shape[0]),
                "N": int(compact_weight.shape[0]),
                "K": int(x2d.shape[1]),
                "compact_K": int(compact_weight.shape[1]),
                "target_sparsity": getattr(layer, "_approx_rtn_mlp_target_sparsity", 0.0),
                "actual_sparsity": getattr(layer, "_approx_rtn_mlp_actual_sparsity", 0.0),
                "block_i": getattr(layer, "_approx_rtn_mlp_block_i", 0),
            }
        )
        return out.reshape(*x_shape[:-1], compact_weight.shape[0])

    def apply(self, layer, x, bias=None):
        if cfg.mode != "approx" or not should_patch(layer):
            return original_apply(self, layer, x, bias)
        if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_cuda:
            return original_apply(self, layer, x, bias)

        x_shape = tuple(x.shape)
        x2d = x.reshape(-1, x_shape[-1]).contiguous()
        if cfg.decode_only and x2d.shape[0] != 1:
            return original_apply(self, layer, x, bias)
        if not prepare_weight(layer, "prepare_weight_lazy"):
            return original_apply(self, layer, x, bias)
        if cfg.backend == "rtn_mlp_prune" and hasattr(layer, "_approx_rtn_mlp_weight"):
            return _apply_rtn_mlp(self, layer, x2d, x_shape, bias)
        if cfg.backend == "native_compact_k_prune":
            return _apply_native_compact(layer, x2d, x_shape, bias)
        if cfg.backend == "triton_compact_k_prune":
            return _apply_compact(layer, x2d, x_shape, bias)
        return _apply_block(layer, x2d, x_shape, bias)

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._approx_sglang_pruning_patched = True
    print("[approx-sglang-pruning] patched UnquantizedLinearMethod", file=sys.stderr)


_install()
