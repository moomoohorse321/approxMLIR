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
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.parameter import Parameter

from mixed_backend_config import (
    W4_BACKENDS as _W4_BACKENDS,
    backend_or_none as _backend_or_none,
    parse_mixed_backend_rules as _parse_mixed_backend_rules,
)


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
    mixed_backend_rules: tuple[tuple[str, str], ...]
    decode_only: bool
    use_substitute: bool
    drop_weight: bool
    sq_collect: bool
    sq_stats_dir: str
    sq_artifact_path: str
    sq_alpha: float
    sq_group_size: int
    awq_grid_size: int

    @classmethod
    def from_env(cls) -> "_Config":
        return cls(
            mode=os.environ.get("APPROX_SGLANG_MODE", "exact"),
            target=os.environ.get("APPROX_SGLANG_TARGET", "proj"),
            backend=os.environ.get("APPROX_SGLANG_BACKEND", "triton"),
            mixed_backend_rules=_parse_mixed_backend_rules(
                os.environ.get("APPROX_SGLANG_MIXED_BACKENDS", "")
            ),
            decode_only=os.environ.get("APPROX_SGLANG_DECODE_ONLY", "1") == "1",
            use_substitute=os.environ.get("APPROX_SGLANG_USE_SUBSTITUTE", "0") == "1",
            drop_weight=os.environ.get("APPROX_SGLANG_DROP_ORIGINAL_WEIGHT", "0") == "1",
            sq_collect=os.environ.get("APPROX_SGLANG_SQ_COLLECT", "0") == "1",
            sq_stats_dir=os.environ.get("APPROX_SGLANG_SQ_STATS_DIR", ""),
            sq_artifact_path=os.environ.get("APPROX_SGLANG_SQ_ARTIFACT_PATH", ""),
            sq_alpha=float(os.environ.get("APPROX_SGLANG_SQ_ALPHA", "0.85")),
            sq_group_size=int(os.environ.get("APPROX_SGLANG_SQ_GROUP_SIZE", "128")),
            awq_grid_size=int(os.environ.get("APPROX_SGLANG_AWQ_GRID_SIZE", "20")),
        )

    def backend_for_prefix(self, prefix: str) -> str | None:
        if self.mixed_backend_rules:
            for target, backend in self.mixed_backend_rules:
                if target == "all" or (target and target in prefix):
                    return _backend_or_none(backend)
            return None
        if self.target == "all" or any(
            tok and tok in prefix for tok in self.target.split(",")
        ):
            return _backend_or_none(self.backend)
        return None

    def backend_for_layer(self, layer: torch.nn.Module) -> str | None:
        return self.backend_for_prefix(getattr(layer, "prefix", ""))

    def target_match(self, layer: torch.nn.Module) -> bool:
        return self.backend_for_layer(layer) is not None

    def needs_sq_stats(self, layer: torch.nn.Module) -> bool:
        backend = self.backend_for_layer(layer)
        return backend in _W4_BACKENDS


def _block_dims(default_n: int, default_k: int = 64) -> tuple[int, int]:
    return (
        int(os.environ.get("APPROX_SGLANG_BLOCK_N", str(default_n))),
        int(os.environ.get("APPROX_SGLANG_BLOCK_K", str(default_k))),
    )


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

    from approx_kernels import (
        approx_sglang_dynamic_w8a8_linear_kernel_1,
        approx_sglang_prequant_w8a8_linear_kernel_1,
        approx_sglang_sq_w4a16_linear_kernel_1,
        awq_quantize_weight_i4_groupwise_packed,
        approx_sglang_w8a16_linear_kernel_1,
        quantize_activation_i8_per_row,
        quantize_weight_i8_per_col,
        repack_i4_packed_for_decode_tile,
        sglang_dynamic_w8a8_linear_kernel,
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
        if not cfg.sq_collect or not cfg.needs_sq_stats(layer):
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

    def _prepared_weight_name(backend: str) -> str:
        if backend in _W4_BACKENDS:
            return "_approx_sq_qweight_i4_t_packed"
        return "_approx_qweight_i8_t"

    def prepare_weight(layer: torch.nn.Module, event: str = "prepare_weight") -> bool:
        backend = cfg.backend_for_layer(layer)
        if backend is None:
            return False
        weight = getattr(layer, "weight", None)
        if (
            weight is None
            or weight.ndim != 2
            or not weight.is_cuda
            or weight.dtype not in (torch.float16, torch.bfloat16)
        ):
            return False

        prefix = getattr(layer, "prefix", "")
        qweight_name = _prepared_weight_name(backend)
        extra_record: dict[str, object] = {}
        if backend in _W4_BACKENDS:
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
                if backend == "triton_sq_w4a16":
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
                "backend": backend,
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

    def _apply_sq_w4a16(layer, x2d, x_shape, bias, backend_name: str):
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
        "triton_prequant": _apply_prequant,
        "triton": _apply_dynamic_w8a8,
        "sgl_kernel": _apply_sgl_kernel,
    }

    def apply(self, layer, x, bias=None):
        backend = cfg.backend_for_layer(layer)
        is_target = backend is not None
        x_shape = None
        x2d = None
        if (
            cfg.sq_collect
            and cfg.needs_sq_stats(layer)
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.is_cuda
        ):
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

        qweight_name = getattr(
            layer,
            "_approx_qweight_name",
            _prepared_weight_name(backend),
        )
        if not hasattr(layer, qweight_name) and not prepare_weight(
            layer,
            "prepare_weight_lazy",
        ):
            return original_apply(self, layer, x, bias)
        if cfg.decode_only and x2d.shape[0] != 1:
            return original_apply(self, layer, x, bias)

        if backend in _W4_BACKENDS:
            result = _apply_sq_w4a16(layer, x2d, x_shape, bias, backend)
        else:
            backend_fn = _BACKENDS.get(backend, _apply_dynamic_w8a8)
            result = backend_fn(layer, x2d, x_shape, bias)
        if result is None:
            return original_apply(self, layer, x, bias)
        return result

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._approx_sglang_quant_patched = True
    print("[approx-sglang-quant] patched UnquantizedLinearMethod", file=sys.stderr)


_install()
