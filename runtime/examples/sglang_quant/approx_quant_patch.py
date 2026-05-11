# approx_quant_patch.py
#
# Runtime monkeypatch that injects approxMLIR quantization into SGLang's
# linear-layer dispatch path. When the APPROX_SGLANG_QUANT env flag is set
# and this module is imported (via sitecustomize.py inside SGLang worker
# subprocesses), it replaces `UnquantizedLinearMethod.apply` with a version
# that, on matching layers, routes through hand-written Triton quant backends.
#
# The current useful paths are:
#   * triton_w8a16      — load-time int8 weight-only path
#   * triton_sq_w4a16   — SmoothQuant-style calibrated group-wise int4 path
#   * triton_awq_w4a16  — AWQ-style calibrated group-wise int4 path
#
# For the SQ-W4 path, the flow is split deliberately:
#   * offline calibration collects activation absmax statistics per layer
#   * load/init time folds the smoothing factor into weights, then packs int4
#   * load/init time also precomputes a decode-only tiled layout for M=1
#   * online serving applies the activation-side inverse smoothing in-kernel

from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.parameter import Parameter


def _enabled() -> bool:
    return os.environ.get("APPROX_SGLANG_QUANT", "0") == "1"


def _record(event: dict) -> None:
    path = os.environ.get("APPROX_SGLANG_STATS_PATH", "")
    if not path:
        return
    event = {"ts": time.time(), "pid": os.getpid(), **event}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


@dataclass(frozen=True)
class _Config:
    mode: str
    target: str
    backend: str
    decode_only: bool
    use_substitute: bool
    drop_weight: bool
    sq_collect: bool
    sq_stats_dir: str
    sq_artifact_path: str
    sq_alpha: float
    sq_group_size: int
    awq_grid_size: int
    apl_bits: int
    apl_artifact_dir: str
    apl_layout_version: str
    apl_quantizer_version: str
    apl_kernel_variant: str

    @classmethod
    def from_env(cls) -> "_Config":
        return cls(
            mode=os.environ.get("APPROX_SGLANG_MODE", "exact"),
            target=os.environ.get("APPROX_SGLANG_TARGET", "proj"),
            backend=os.environ.get("APPROX_SGLANG_BACKEND", "triton"),
            decode_only=os.environ.get("APPROX_SGLANG_DECODE_ONLY", "1") == "1",
            use_substitute=os.environ.get("APPROX_SGLANG_USE_SUBSTITUTE", "0") == "1",
            drop_weight=os.environ.get("APPROX_SGLANG_DROP_ORIGINAL_WEIGHT", "0") == "1",
            sq_collect=os.environ.get("APPROX_SGLANG_SQ_COLLECT", "0") == "1",
            sq_stats_dir=os.environ.get("APPROX_SGLANG_SQ_STATS_DIR", ""),
            sq_artifact_path=os.environ.get("APPROX_SGLANG_SQ_ARTIFACT_PATH", ""),
            sq_alpha=float(os.environ.get("APPROX_SGLANG_SQ_ALPHA", "0.85")),
            sq_group_size=int(os.environ.get("APPROX_SGLANG_SQ_GROUP_SIZE", "128")),
            awq_grid_size=int(os.environ.get("APPROX_SGLANG_AWQ_GRID_SIZE", "20")),
            apl_bits=int(os.environ.get("APPROX_SGLANG_APL_BITS", "4")),
            apl_artifact_dir=os.environ.get("APPROX_SGLANG_APL_ARTIFACT_DIR", ""),
            apl_layout_version=os.environ.get("APPROX_SGLANG_APL_LAYOUT_VERSION", "natural_bitplane_v1"),
            apl_quantizer_version=os.environ.get("APPROX_SGLANG_APL_QUANTIZER_VERSION", "row_uniform_lut_v1"),
            apl_kernel_variant=os.environ.get("APPROX_SGLANG_APL_KERNEL_VARIANT", "natural_tl_dot"),
        )

    def target_match(self, layer: torch.nn.Module) -> bool:
        prefix = getattr(layer, "prefix", "")
        if self.target == "all":
            return True
        return any(tok and tok in prefix for tok in self.target.split(","))


def _block_dims(default_n: int, default_k: int = 64) -> tuple[int, int]:
    return (
        int(os.environ.get("APPROX_SGLANG_BLOCK_N", str(default_n))),
        int(os.environ.get("APPROX_SGLANG_BLOCK_K", str(default_k))),
    )


def _apl_launch_dims(bits: int) -> tuple[int, int, int]:
    block_n = int(
        os.environ.get(
            "APPROX_SGLANG_APL_BLOCK_N",
            os.environ.get("APPROX_SGLANG_BLOCK_N", "64"),
        )
    )
    block_k_env = os.environ.get("APPROX_SGLANG_APL_BLOCK_K", os.environ.get("APPROX_SGLANG_BLOCK_K", ""))
    block_k = int(block_k_env) if block_k_env else (128 if int(bits) == 7 else 256)
    num_warps = int(os.environ.get("APPROX_SGLANG_APL_NUM_WARPS", "4"))
    return block_n, block_k, num_warps


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
):
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
    _record({"event": "apply_approx", "backend": backend_name, "substituted": substituted, **record_common})
    if substituted:
        layer._approx_substitution_hits = getattr(layer, "_approx_substitution_hits", 0) + 1
    return substituted


def _install() -> None:
    if not _enabled():
        return

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from approx_apl_lut import (
        APL_LAYOUT_NATURAL,
        APL_QUANTIZER_IMPORTED_ANYPRECISION,
        APL_QUANTIZER_ROW_KMEANS,
        APL_QUANTIZER_ROW_UNIFORM,
        apl_lut_quantize_weight,
    )
    from approx_kernels import (
        approx_sglang_dynamic_w8a8_linear_kernel_1,
        approx_sglang_apl_lut_linear_kernel_1,
        approx_sglang_prequant_w8a8_linear_kernel_1,
        approx_sglang_sq_w4a16_linear_kernel_1,
        awq_quantize_weight_i4_groupwise_packed,
        approx_sglang_w8a16_linear_kernel_1,
        quantize_activation_i8_per_row,
        quantize_weight_i8_per_col,
        repack_i4_packed_for_decode_tile,
        sglang_dynamic_w8a8_linear_kernel,
        sglang_apl_lut_linear_kernel,
        sglang_prequant_w8a8_linear_kernel,
        sglang_sq_w4a16_linear_kernel,
        sglang_w8a16_linear_kernel,
        smoothquant_quantize_weight_i4_groupwise_packed,
    )
    try:
        from sgl_kernel import int8_scaled_mm
        from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8
    except Exception:
        int8_scaled_mm = None
        per_token_quant_int8 = None
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if getattr(UnquantizedLinearMethod, "_approx_sglang_quant_patched", False):
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
    if cfg.backend == "triton_apl_lut":
        if cfg.apl_kernel_variant != "natural_tl_dot":
            _record(
                {
                    "event": "unsupported_apl_kernel_variant",
                    "backend": "triton_apl_lut",
                    "kernel_variant": cfg.apl_kernel_variant,
                    "reason": "only natural_tl_dot is implemented in this runtime path",
                }
            )
        if cfg.apl_layout_version != APL_LAYOUT_NATURAL and cfg.apl_kernel_variant == "natural_tl_dot":
            _record(
                {
                    "event": "unsupported_apl_layout",
                    "backend": "triton_apl_lut",
                    "layout_version": cfg.apl_layout_version,
                    "kernel_variant": cfg.apl_kernel_variant,
                    "reason": "natural_tl_dot requires natural_bitplane_v1",
                }
            )

    sq_stats_by_prefix: dict[str, torch.Tensor] = {}
    sq_artifact_layers: dict[str, torch.Tensor] | None = None
    sq_artifact_attempted = False
    sq_stats_path = (
        Path(cfg.sq_stats_dir) / f"sq_stats_{os.getpid()}.pt"
        if cfg.sq_collect and cfg.sq_stats_dir
        else None
    )

    if cfg.sq_collect and cfg.sq_stats_dir:
        stats_dir = Path(cfg.sq_stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)

        def _persist_sq_stats() -> None:
            if sq_stats_path is None or not sq_stats_by_prefix:
                return
            torch.save({"pid": os.getpid(), "layers": sq_stats_by_prefix}, sq_stats_path)

        def _flush_sq_stats() -> None:
            if not sq_stats_by_prefix:
                return
            _persist_sq_stats()
            _record({"event": "sq_stats_flush", "path": str(sq_stats_path), "num_layers": len(sq_stats_by_prefix)})

        atexit.register(_flush_sq_stats)

    def _collect_sq_stats(layer: torch.nn.Module, x2d: torch.Tensor) -> None:
        if not cfg.sq_collect or not cfg.target_match(layer):
            return
        if torch.cuda.is_current_stream_capturing():
            return
        prefix = getattr(layer, "prefix", "")
        if not prefix:
            return
        act_absmax = x2d.detach().abs().amax(dim=0).float().cpu()
        prev = sq_stats_by_prefix.get(prefix)
        if prev is None or prev.numel() != act_absmax.numel():
            sq_stats_by_prefix[prefix] = act_absmax
        else:
            sq_stats_by_prefix[prefix] = torch.maximum(prev, act_absmax)
        if sq_stats_path is not None:
            _persist_sq_stats()

    def _load_sq_artifact_layers() -> dict[str, torch.Tensor] | None:
        nonlocal sq_artifact_layers, sq_artifact_attempted
        if sq_artifact_attempted:
            return sq_artifact_layers
        sq_artifact_attempted = True
        if not cfg.sq_artifact_path:
            _record({"event": "missing_sq_artifact_path"})
            return None
        artifact_path = Path(cfg.sq_artifact_path)
        if not artifact_path.exists():
            _record({"event": "missing_sq_artifact", "path": str(artifact_path)})
            return None
        rec = torch.load(artifact_path, map_location="cpu")
        if isinstance(rec, dict) and isinstance(rec.get("layers"), dict):
            sq_artifact_layers = rec["layers"]
        elif isinstance(rec, dict):
            sq_artifact_layers = rec
        else:
            sq_artifact_layers = None
        _record(
            {
                "event": "load_sq_artifact",
                "path": str(artifact_path),
                "num_layers": len(sq_artifact_layers or {}),
            }
        )
        return sq_artifact_layers

    def _apl_model_key() -> str:
        for env_name in (
            "APPROX_SGLANG_APL_MODEL_PATH",
            "APPROX_SGLANG_MODEL_PATH",
            "MODEL_PATH",
            "SGLANG_MODEL_PATH",
        ):
            value = os.environ.get(env_name, "")
            if value:
                return value
        return "unknown_model"

    def _apl_cache_path(prefix: str, weight_shape: tuple[int, int]) -> Path | None:
        if not cfg.apl_artifact_dir:
            return None
        key = {
            "model_path": _apl_model_key(),
            "layer_prefix": prefix,
            "weight_shape": list(weight_shape),
            "bits": cfg.apl_bits,
            "layout_version": cfg.apl_layout_version,
            "quantizer_version": cfg.apl_quantizer_version,
        }
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        safe_prefix = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in prefix)[:120] or "layer"
        return Path(cfg.apl_artifact_dir) / f"apl_lut_{safe_prefix}_{digest}.pt"

    def _load_or_build_apl_artifact(layer: torch.nn.Module, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict, bool] | None:
        prefix = getattr(layer, "prefix", "")
        cache_path = _apl_cache_path(prefix, tuple(weight.shape))
        artifact_hit = False
        if cache_path is not None and cache_path.exists():
            try:
                rec = torch.load(cache_path, map_location="cpu")
                meta = dict(rec.get("metadata", {}))
                expected = {
                    "backend": "triton_apl_lut",
                    "bits": cfg.apl_bits,
                    "K_orig": int(weight.shape[1]),
                    "N_orig": int(weight.shape[0]),
                    "layout_version": cfg.apl_layout_version,
                    "quantizer_version": cfg.apl_quantizer_version,
                }
                if all(meta.get(k) == v for k, v in expected.items()):
                    q_cpu = rec["qweight"].contiguous()
                    lut_cpu = rec["lut"].contiguous()
                    expected_q_shape = (
                        int(meta["bits"]),
                        int(meta["N_padded"]),
                        int(meta["K_padded"]) // 32,
                    )
                    expected_lut_shape = (int(meta["N_padded"]), 1 << int(meta["bits"]))
                    if (
                        q_cpu.dtype == torch.int32
                        and lut_cpu.dtype == torch.float16
                        and tuple(q_cpu.shape) == expected_q_shape
                        and tuple(lut_cpu.shape) == expected_lut_shape
                    ):
                        artifact_hit = True
                        return q_cpu, lut_cpu, meta, artifact_hit
                _record(
                    {
                        "event": "apl_artifact_reject",
                        "prefix": prefix,
                        "path": str(cache_path),
                        "reason": "metadata_or_dtype_mismatch",
                        "expected": expected,
                        "actual": meta,
                    }
                )
            except Exception as exc:
                _record({"event": "apl_artifact_reject", "prefix": prefix, "path": str(cache_path), "reason": repr(exc)})

        max_runtime_elems = int(os.environ.get("APPROX_SGLANG_APL_MAX_RUNTIME_QUANT_ELEMS", "67108864"))
        allow_runtime_quant = os.environ.get("APPROX_SGLANG_APL_ALLOW_RUNTIME_QUANT", "1") == "1"
        if cfg.apl_quantizer_version == APL_QUANTIZER_IMPORTED_ANYPRECISION:
            _record(
                {
                    "event": "skip_prepare_apl_weight",
                    "prefix": prefix,
                    "backend": "triton_apl_lut",
                    "artifact_hit": False,
                    "artifact_path": "" if cache_path is None else str(cache_path),
                    "reason": "imported_anyprecision_requires_artifact",
                    "weight_shape": list(weight.shape),
                    "bits": cfg.apl_bits,
                    "layout_version": cfg.apl_layout_version,
                    "quantizer_version": cfg.apl_quantizer_version,
                }
            )
            return None
        if cfg.apl_quantizer_version not in (APL_QUANTIZER_ROW_UNIFORM, APL_QUANTIZER_ROW_KMEANS):
            _record(
                {
                    "event": "skip_prepare_apl_weight",
                    "prefix": prefix,
                    "backend": "triton_apl_lut",
                    "artifact_hit": False,
                    "artifact_path": "" if cache_path is None else str(cache_path),
                    "reason": "unsupported_runtime_quantizer",
                    "weight_shape": list(weight.shape),
                    "bits": cfg.apl_bits,
                    "layout_version": cfg.apl_layout_version,
                    "quantizer_version": cfg.apl_quantizer_version,
                }
            )
            return None
        if not allow_runtime_quant or weight.numel() > max_runtime_elems:
            _record(
                {
                    "event": "skip_prepare_apl_weight",
                    "prefix": prefix,
                    "backend": "triton_apl_lut",
                    "artifact_hit": False,
                    "artifact_path": "" if cache_path is None else str(cache_path),
                    "reason": "missing_artifact_runtime_quant_disabled_or_too_large",
                    "weight_shape": list(weight.shape),
                    "bits": cfg.apl_bits,
                    "layout_version": cfg.apl_layout_version,
                    "quantizer_version": cfg.apl_quantizer_version,
                }
            )
            return None

        try:
            q_cpu, lut_cpu, meta = apl_lut_quantize_weight(
                weight.detach().cpu(),
                bits=cfg.apl_bits,
                layout_version=cfg.apl_layout_version,
                quantizer_version=cfg.apl_quantizer_version,
            )
        except Exception as exc:
            _record(
                {
                    "event": "skip_prepare_apl_weight",
                    "prefix": prefix,
                    "backend": "triton_apl_lut",
                    "artifact_hit": False,
                    "artifact_path": "" if cache_path is None else str(cache_path),
                    "reason": repr(exc),
                    "weight_shape": list(weight.shape),
                    "bits": cfg.apl_bits,
                    "layout_version": cfg.apl_layout_version,
                    "quantizer_version": cfg.apl_quantizer_version,
                }
            )
            return None
        meta = {
            **meta,
            "backend": "triton_apl_lut",
            "prefix": prefix,
            "model_path": _apl_model_key(),
            "layout_version": cfg.apl_layout_version,
            "quantizer_version": cfg.apl_quantizer_version,
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "qweight": q_cpu,
                    "lut": lut_cpu,
                    "metadata": meta,
                },
                cache_path,
            )
        return q_cpu, lut_cpu, meta, artifact_hit

    def _prepared_weight_name() -> str:
        if cfg.backend == "triton_apl_lut":
            return "_approx_apl_qweight"
        if cfg.backend in ("triton_sq_w4a16", "triton_awq_w4a16"):
            return "_approx_sq_qweight_i4_t_packed"
        return "_approx_qweight_i8_t"

    def prepare_weight(layer: torch.nn.Module, event: str = "prepare_weight") -> bool:
        weight = getattr(layer, "weight", None)
        if (
            weight is None
            or weight.ndim != 2
            or not weight.is_cuda
            or weight.dtype not in (torch.float16, torch.bfloat16)
        ):
            return False

        prefix = getattr(layer, "prefix", "")
        qweight_name = _prepared_weight_name()
        extra_record: dict[str, object] = {}
        if cfg.backend == "triton_apl_lut":
            artifact = _load_or_build_apl_artifact(layer, weight.data)
            if artifact is None:
                return False
            q_cpu, lut_cpu, apl_meta, artifact_hit = artifact
            layout_version = str(apl_meta.get("layout_version", ""))
            if cfg.apl_kernel_variant != "natural_tl_dot" or layout_version != APL_LAYOUT_NATURAL:
                _record(
                    {
                        "event": "skip_prepare_apl_weight",
                        "prefix": prefix,
                        "backend": "triton_apl_lut",
                        "artifact_hit": artifact_hit,
                        "artifact_path": str(_apl_cache_path(prefix, tuple(weight.shape)) or ""),
                        "reason": "unsupported_layout_kernel_combination",
                        "layout_version": layout_version,
                        "kernel_variant": cfg.apl_kernel_variant,
                    }
                )
                return False
            q = q_cpu.to(device=weight.device, non_blocking=True).contiguous()
            lut = lut_cpu.to(device=weight.device, non_blocking=True).contiguous()
            layer.register_buffer(qweight_name, q, persistent=False)
            layer.register_buffer("_approx_apl_lut", lut, persistent=False)
            layer._approx_apl_bits = int(apl_meta["bits"])
            layer._approx_apl_k_orig = int(apl_meta["K_orig"])
            layer._approx_apl_k_padded = int(apl_meta["K_padded"])
            layer._approx_apl_n_orig = int(apl_meta["N_orig"])
            layer._approx_apl_n_padded = int(apl_meta["N_padded"])
            layer._approx_apl_artifact_hit = bool(artifact_hit)
            layer._approx_apl_layout_version = layout_version
            layer._approx_apl_quantizer_version = str(apl_meta.get("quantizer_version", cfg.apl_quantizer_version))
            layer._approx_apl_kernel_variant = cfg.apl_kernel_variant
            layer._approx_apl_source_dtype = str(apl_meta.get("source_dtype", weight.dtype))
            qweight_bits = int(apl_meta["bits"])
            extra_record = {
                "lut_shape": list(lut.shape),
                "K_orig": int(apl_meta["K_orig"]),
                "K_padded": int(apl_meta["K_padded"]),
                "N_orig": int(apl_meta["N_orig"]),
                "N_padded": int(apl_meta["N_padded"]),
                "artifact_hit": artifact_hit,
                "artifact_path": str(_apl_cache_path(prefix, tuple(weight.shape)) or ""),
                "layout_version": layout_version,
                "quantizer_version": str(apl_meta.get("quantizer_version", cfg.apl_quantizer_version)),
                "kernel_variant": cfg.apl_kernel_variant,
                "source_dtype": str(apl_meta.get("source_dtype", weight.dtype)),
            }
        elif cfg.backend in ("triton_sq_w4a16", "triton_awq_w4a16"):
            sq_layers = _load_sq_artifact_layers()
            act_absmax = None if sq_layers is None else sq_layers.get(prefix)
            if act_absmax is None:
                _record(
                    {
                        "event": "skip_prepare_sq_weight",
                        "prefix": prefix,
                        "reason": "missing_layer_artifact",
                    }
                )
                return False
            try:
                if cfg.backend == "triton_sq_w4a16":
                    q, scale_g, smooth_inv = smoothquant_quantize_weight_i4_groupwise_packed(
                        weight.data,
                        torch.as_tensor(act_absmax),
                        alpha=cfg.sq_alpha,
                        group_size=cfg.sq_group_size,
                    )
                    extra_record = {
                        "quant_method": "smoothquant",
                        "scale_shape": list(scale_g.shape),
                        "group_size": cfg.sq_group_size,
                        "alpha": cfg.sq_alpha,
                    }
                else:
                    q, scale_g, smooth_inv, awq_ratio = awq_quantize_weight_i4_groupwise_packed(
                        weight.data,
                        torch.as_tensor(act_absmax),
                        group_size=cfg.sq_group_size,
                        grid_size=cfg.awq_grid_size,
                    )
                    extra_record = {
                        "quant_method": "awq",
                        "scale_shape": list(scale_g.shape),
                        "group_size": cfg.sq_group_size,
                        "awq_grid_size": cfg.awq_grid_size,
                        "awq_ratio": awq_ratio,
                    }
            except Exception as exc:
                _record(
                    {
                        "event": "skip_prepare_sq_weight",
                        "prefix": prefix,
                        "reason": repr(exc),
                    }
                )
                return False
            layer.register_buffer(qweight_name, q, persistent=False)
            layer.register_buffer(
                "_approx_sq_qweight_scale_g",
                scale_g.to(dtype=weight.dtype),
                persistent=False,
            )
            layer.register_buffer(
                "_approx_sq_act_smooth_inv",
                smooth_inv.to(dtype=weight.dtype),
                persistent=False,
            )
            decode_block_n = int(os.environ.get("APPROX_SGLANG_BLOCK_N", "128"))
            decode_block_k = int(
                os.environ.get(
                    "APPROX_SGLANG_SQ_BLOCK_K",
                    os.environ.get("APPROX_SGLANG_BLOCK_K", "64"),
                )
            )
            try:
                q_decode, scale_decode = repack_i4_packed_for_decode_tile(
                    q,
                    scale_g,
                    group_size=cfg.sq_group_size,
                    block_k=decode_block_k,
                    block_n=decode_block_n,
                )
                layer.register_buffer(
                    "_approx_sq_qweight_i4_t_decode_tiled",
                    q_decode,
                    persistent=False,
                )
                layer.register_buffer(
                    "_approx_sq_qweight_scale_g_decode_tiled",
                    scale_decode.to(dtype=weight.dtype),
                    persistent=False,
                )
                layer._approx_sq_decode_block_n = decode_block_n
                layer._approx_sq_decode_block_k = decode_block_k
                extra_record.update(
                    {
                        "decode_qweight_shape": list(q_decode.shape),
                        "decode_scale_shape": list(scale_decode.shape),
                        "decode_block_n": decode_block_n,
                        "decode_block_k": decode_block_k,
                    }
                )
            except Exception as exc:
                _record(
                    {
                        "event": "skip_prepare_sq_decode_layout",
                        "prefix": prefix,
                        "reason": repr(exc),
                    }
                )
            qweight_bits = 4
        else:
            q, scale = quantize_weight_i8_per_col(weight.data)
            layer.register_buffer(qweight_name, q, persistent=False)
            layer.register_buffer("_approx_qweight_scale", scale, persistent=False)
            qweight_bits = 8
            extra_record = {"scale_shape": list(scale.shape)}

        layer._approx_qweight_name = qweight_name
        if cfg.drop_weight:
            layer.weight = Parameter(
                torch.empty(0, device=weight.device, dtype=weight.dtype),
                requires_grad=False,
            )
        _record(
            {
                "event": event,
                "prefix": prefix,
                "weight_shape": list(weight.shape),
                "qweight_shape": list(q.shape),
                "qweight_stride": list(q.stride()),
                "qweight_bits": qweight_bits,
                "backend": cfg.backend,
                "dtype": str(weight.dtype),
                **extra_record,
            }
        )
        return True

    def process_weights_after_loading(self, layer):
        original_process(self, layer)
        if cfg.mode == "approx" and cfg.target_match(layer):
            prepare_weight(layer)

    def _apply_weight_only_triton(
        layer,
        x2d,
        x_shape,
        bias,
        *,
        backend_name: str,
        qweight_attr: str,
        primary,
        substitute,
        substitute_name: str,
    ):
        import triton

        bq = getattr(layer, qweight_attr)
        bs = layer._approx_qweight_scale
        M, K = x2d.shape[0], x2d.shape[1]
        N = bq.shape[1]
        block_n, block_k = _block_dims(128)
        grid = (M, triton.cdiv(N, block_n))
        out = torch.empty((M, N), device=x2d.device, dtype=x2d.dtype)
        args = (
            x2d,
            bq,
            bs,
            out,
            M,
            N,
            K,
            x2d.stride(0),
            x2d.stride(1),
            bq.stride(0),
            bq.stride(1),
            out.stride(0),
            out.stride(1),
        )
        kwargs = dict(BLOCK_N=block_n, BLOCK_K=block_k)
        shape_key = (
            backend_name,
            str(x2d.dtype),
            M,
            N,
            K,
            x2d.stride(0),
            x2d.stride(1),
            bq.stride(0),
            bq.stride(1),
            out.stride(0),
            out.stride(1),
            block_n,
            block_k,
        )
        _run_substituted_kernel(
            backend_name=backend_name,
            primary=primary,
            substitute=substitute,
            substitute_name=substitute_name,
            grid=grid,
            args=args,
            kwargs=kwargs,
            shape_key=shape_key,
            use_substitute=cfg.use_substitute,
            record_common={
                "prefix": getattr(layer, "prefix", ""),
                "M": M,
                "N": N,
                "K": K,
                "block_n": block_n,
                "block_k": block_k,
            },
            layer=layer,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*x_shape[:-1], N)

    def _apply_w8a16(layer, x2d, x_shape, bias):
        return _apply_weight_only_triton(
            layer,
            x2d,
            x_shape,
            bias,
            backend_name="triton_w8a16",
            qweight_attr="_approx_qweight_i8_t",
            primary=sglang_w8a16_linear_kernel,
            substitute=approx_sglang_w8a16_linear_kernel_1,
            substitute_name="approx_sglang_w8a16_linear_kernel_1",
        )

    def _apply_sq_w4a16(layer, x2d, x_shape, bias):
        import triton

        generic_bq = layer._approx_sq_qweight_i4_t_packed
        generic_bs = layer._approx_sq_qweight_scale_g
        smooth_inv = layer._approx_sq_act_smooth_inv
        M, K = x2d.shape[0], x2d.shape[1]
        N = generic_bs.shape[1]
        smooth_mode = os.environ.get("APPROX_SGLANG_SQ_SMOOTH_MODE", "kernel")
        block_n = int(os.environ.get("APPROX_SGLANG_BLOCK_N", "128"))
        block_k = int(
            os.environ.get(
                "APPROX_SGLANG_SQ_BLOCK_K",
                os.environ.get("APPROX_SGLANG_BLOCK_K", "64"),
            )
        )
        if block_k <= 0 or block_k > cfg.sq_group_size or (cfg.sq_group_size % block_k) != 0:
            raise ValueError(
                f"invalid SQ block/group combination: block_k={block_k}, group_k={cfg.sq_group_size}"
            )
        use_decode_tiled = (
            M == 1
            and hasattr(layer, "_approx_sq_qweight_i4_t_decode_tiled")
            and hasattr(layer, "_approx_sq_qweight_scale_g_decode_tiled")
            and getattr(layer, "_approx_sq_decode_block_n", -1) == block_n
            and getattr(layer, "_approx_sq_decode_block_k", -1) == block_k
        )
        if use_decode_tiled:
            bq = layer._approx_sq_qweight_i4_t_decode_tiled
            bs = layer._approx_sq_qweight_scale_g_decode_tiled
        else:
            bq = generic_bq
            bs = generic_bs
        grid = (M, triton.cdiv(N, block_n))
        out = torch.empty((M, N), device=x2d.device, dtype=x2d.dtype)
        if smooth_mode == "host":
            x_input = (x2d * smooth_inv).contiguous()
            smooth_input = torch.ones_like(smooth_inv)
        elif smooth_mode == "kernel":
            x_input = x2d
            smooth_input = smooth_inv
        else:
            raise ValueError(f"invalid SQ smooth mode: {smooth_mode}")
        args = (
            x_input,
            bq,
            bs,
            smooth_input,
            out,
            M,
            N,
            K,
            x_input.stride(0),
            x_input.stride(1),
            bq.stride(0),
            bq.stride(1),
            bs.stride(0),
            bs.stride(1),
            out.stride(0),
            out.stride(1),
        )
        kwargs = dict(
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_K=cfg.sq_group_size,
            DECODE_TILE_LAYOUT=1 if use_decode_tiled else 0,
        )
        backend_name = cfg.backend if cfg.backend in ("triton_sq_w4a16", "triton_awq_w4a16") else "triton_sq_w4a16"
        shape_key = (
            backend_name,
            str(x2d.dtype),
            str(bs.dtype),
            smooth_mode,
            use_decode_tiled,
            M,
            N,
            K,
            x_input.stride(0),
            x_input.stride(1),
            bq.stride(0),
            bq.stride(1),
            bs.stride(0),
            bs.stride(1),
            out.stride(0),
            out.stride(1),
            block_n,
            block_k,
            cfg.sq_group_size,
        )
        _run_substituted_kernel(
            backend_name=backend_name,
            primary=sglang_sq_w4a16_linear_kernel,
            substitute=approx_sglang_sq_w4a16_linear_kernel_1,
            substitute_name="approx_sglang_sq_w4a16_linear_kernel_1",
            grid=grid,
            args=args,
            kwargs=kwargs,
            shape_key=shape_key,
            use_substitute=cfg.use_substitute,
            record_common={
                "prefix": getattr(layer, "prefix", ""),
                "M": M,
                "N": N,
                "K": K,
                "block_n": block_n,
                "block_k": block_k,
                "group_k": cfg.sq_group_size,
                "smooth_mode": smooth_mode,
                "decode_tiled": use_decode_tiled,
            },
            layer=layer,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*x_shape[:-1], N)

    def _apply_apl_lut(layer, x2d, x_shape, bias):
        import triton

        qweight = layer._approx_apl_qweight
        lut = layer._approx_apl_lut
        M, K = x2d.shape[0], x2d.shape[1]
        K_orig = int(layer._approx_apl_k_orig)
        K_padded = int(layer._approx_apl_k_padded)
        N_orig = int(layer._approx_apl_n_orig)
        bits = int(layer._approx_apl_bits)
        layout_version = str(getattr(layer, "_approx_apl_layout_version", ""))
        quantizer_version = str(getattr(layer, "_approx_apl_quantizer_version", ""))
        kernel_variant = str(getattr(layer, "_approx_apl_kernel_variant", cfg.apl_kernel_variant))
        source_dtype = str(getattr(layer, "_approx_apl_source_dtype", ""))
        if K != K_orig:
            _record(
                {
                    "event": "skip_apply_apl",
                    "backend": "triton_apl_lut",
                    "prefix": getattr(layer, "prefix", ""),
                    "M": M,
                    "N": N_orig,
                    "K": K,
                    "K_orig": K_orig,
                    "reason": "activation_k_mismatch",
                    "bits": bits,
                    "layout_version": layout_version,
                    "quantizer_version": quantizer_version,
                    "kernel_variant": kernel_variant,
                    "source_dtype": source_dtype,
                }
            )
            return None
        block_n, block_k, num_warps = _apl_launch_dims(bits)
        if block_k % 32 != 0:
            raise ValueError(f"APL BLOCK_K must be divisible by 32, got {block_k}")
        grid = (M, triton.cdiv(N_orig, block_n))
        out = torch.empty((M, N_orig), device=x2d.device, dtype=x2d.dtype)
        args = (
            x2d,
            qweight,
            lut,
            out,
            M,
            N_orig,
            K_orig,
            K_padded,
            x2d.stride(0),
            x2d.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            qweight.stride(2),
            lut.stride(0),
            lut.stride(1),
            out.stride(0),
            out.stride(1),
        )
        kwargs = dict(BITS=bits, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps)
        shape_key = (
            "triton_apl_lut",
            str(x2d.dtype),
            str(lut.dtype),
            M,
            N_orig,
            K_orig,
            K_padded,
            bits,
            x2d.stride(0),
            x2d.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            qweight.stride(2),
            lut.stride(0),
            lut.stride(1),
            out.stride(0),
            out.stride(1),
            block_n,
            block_k,
            num_warps,
        )
        _run_substituted_kernel(
            backend_name="triton_apl_lut",
            primary=sglang_apl_lut_linear_kernel,
            substitute=approx_sglang_apl_lut_linear_kernel_1,
            substitute_name="approx_sglang_apl_lut_linear_kernel_1",
            grid=grid,
            args=args,
            kwargs=kwargs,
            shape_key=shape_key,
            use_substitute=cfg.use_substitute,
            record_common={
                "prefix": getattr(layer, "prefix", ""),
                "M": M,
                "N": N_orig,
                "K": K_orig,
                "K_padded": K_padded,
                "bits": bits,
                "layout_version": layout_version,
                "quantizer_version": quantizer_version,
                "kernel_variant": kernel_variant,
                "source_dtype": source_dtype,
                "block_n": block_n,
                "block_k": block_k,
                "num_warps": num_warps,
                "artifact_hit": getattr(layer, "_approx_apl_artifact_hit", False),
            },
            layer=layer,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*x_shape[:-1], N_orig)

    def _apply_prequant(layer, x2d, x_shape, bias):
        import triton

        bq = layer._approx_qweight_i8_t
        bs = layer._approx_qweight_scale
        aq, a_scale = quantize_activation_i8_per_row(x2d)
        M, K = aq.shape[0], aq.shape[1]
        N = bq.shape[1]
        block_n, block_k = _block_dims(128)
        grid = (M, triton.cdiv(N, block_n))
        out = torch.empty((M, N), device=x2d.device, dtype=x2d.dtype)
        args = (
            aq,
            a_scale,
            bq,
            bs,
            out,
            M,
            N,
            K,
            aq.stride(0),
            aq.stride(1),
            bq.stride(0),
            bq.stride(1),
            out.stride(0),
            out.stride(1),
        )
        kwargs = dict(BLOCK_N=block_n, BLOCK_K=block_k)
        shape_key = (
            "prequant",
            str(aq.dtype),
            str(out.dtype),
            M,
            N,
            K,
            aq.stride(0),
            aq.stride(1),
            bq.stride(0),
            bq.stride(1),
            out.stride(0),
            out.stride(1),
            block_n,
            block_k,
        )
        _run_substituted_kernel(
            backend_name="triton_prequant",
            primary=sglang_prequant_w8a8_linear_kernel,
            substitute=approx_sglang_prequant_w8a8_linear_kernel_1,
            substitute_name="approx_sglang_prequant_w8a8_linear_kernel_1",
            grid=grid,
            args=args,
            kwargs=kwargs,
            shape_key=shape_key,
            use_substitute=cfg.use_substitute,
            record_common={
                "prefix": getattr(layer, "prefix", ""),
                "M": M,
                "N": N,
                "K": K,
                "block_n": block_n,
                "block_k": block_k,
            },
            layer=layer,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*x_shape[:-1], N)

    def _apply_dynamic_w8a8(layer, x2d, x_shape, bias):
        import triton

        bq = layer._approx_qweight_i8_t
        bs = layer._approx_qweight_scale
        M, K = x2d.shape[0], x2d.shape[1]
        N = bq.shape[1]
        block_n, block_k = _block_dims(64)
        grid = (M, triton.cdiv(N, block_n))
        out = torch.empty((M, N), device=x2d.device, dtype=x2d.dtype)
        args = (
            x2d,
            bq,
            bs,
            out,
            M,
            N,
            K,
            x2d.stride(0),
            x2d.stride(1),
            bq.stride(0),
            bq.stride(1),
            out.stride(0),
            out.stride(1),
        )
        kwargs = dict(BLOCK_N=block_n, BLOCK_K=block_k)
        shape_key = (
            str(x2d.dtype),
            M,
            N,
            K,
            x2d.stride(0),
            x2d.stride(1),
            bq.stride(0),
            bq.stride(1),
            out.stride(0),
            out.stride(1),
            block_n,
            block_k,
        )
        _run_substituted_kernel(
            backend_name="triton",
            primary=sglang_dynamic_w8a8_linear_kernel,
            substitute=approx_sglang_dynamic_w8a8_linear_kernel_1,
            substitute_name="approx_sglang_dynamic_w8a8_linear_kernel_1",
            grid=grid,
            args=args,
            kwargs=kwargs,
            shape_key=shape_key,
            use_substitute=cfg.use_substitute,
            record_common={
                "prefix": getattr(layer, "prefix", ""),
                "M": M,
                "N": N,
                "K": K,
                "block_n": block_n,
                "block_k": block_k,
            },
            layer=layer,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*x_shape[:-1], N)

    def _apply_sgl_kernel(layer, x2d, x_shape, bias):
        if int8_scaled_mm is None or per_token_quant_int8 is None:
            return None
        bq = layer._approx_qweight_i8_t
        bs = layer._approx_qweight_scale
        M, K = x2d.shape[0], x2d.shape[1]
        N = bq.shape[1]
        xq, xs = per_token_quant_int8(x2d)
        out = int8_scaled_mm(
            xq,
            bq,
            xs.view(-1, xs.shape[-1]),
            bs.view(-1, 1),
            out_dtype=x2d.dtype,
            bias=bias,
        )
        _record(
            {
                "event": "apply_approx",
                "backend": "sgl_kernel",
                "prefix": getattr(layer, "prefix", ""),
                "M": M,
                "N": N,
                "K": K,
                "substituted": False,
            }
        )
        return out.reshape(*x_shape[:-1], N)

    _BACKENDS = {
        "triton_w8a16": _apply_w8a16,
        "triton_sq_w4a16": _apply_sq_w4a16,
        "triton_awq_w4a16": _apply_sq_w4a16,
        "triton_apl_lut": _apply_apl_lut,
        "triton_prequant": _apply_prequant,
        "triton": _apply_dynamic_w8a8,
        "sgl_kernel": _apply_sgl_kernel,
    }

    def apply(self, layer, x, bias=None):
        is_target = cfg.target_match(layer)
        x_shape = None
        x2d = None
        if cfg.sq_collect and is_target and x.dtype in (torch.float16, torch.bfloat16) and x.is_cuda:
            x_shape = tuple(x.shape)
            x2d = x.reshape(-1, x_shape[-1]).contiguous()
            _collect_sq_stats(layer, x2d)

        if cfg.mode != "approx" or not is_target:
            return original_apply(self, layer, x, bias)
        if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_cuda:
            return original_apply(self, layer, x, bias)
        if x2d is None:
            x_shape = tuple(x.shape)
            x2d = x.reshape(-1, x_shape[-1]).contiguous()

        qweight_name = getattr(layer, "_approx_qweight_name", _prepared_weight_name())
        if not hasattr(layer, qweight_name) and not prepare_weight(layer, "prepare_weight_lazy"):
            return original_apply(self, layer, x, bias)
        if cfg.decode_only and x2d.shape[0] != 1:
            return original_apply(self, layer, x, bias)

        backend_fn = _BACKENDS.get(cfg.backend, _apply_dynamic_w8a8)
        result = backend_fn(layer, x2d, x_shape, bias)
        if result is None:
            return original_apply(self, layer, x, bias)
        return result

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._approx_sglang_quant_patched = True
    print("[approx-sglang-quant] patched UnquantizedLinearMethod", file=sys.stderr)


_install()
