from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def _enabled() -> bool:
    return os.environ.get("APPROX_SGLANG_QUANT", "0") == "1"


def _target(layer: torch.nn.Module) -> bool:
    include = os.environ.get("APPROX_SGLANG_TARGET", "proj")
    prefix = getattr(layer, "prefix", "")
    if include == "all":
        return True
    return any(tok and tok in prefix for tok in include.split(","))


def _decode_only() -> bool:
    return os.environ.get("APPROX_SGLANG_DECODE_ONLY", "1") == "1"


def _backend() -> str:
    return os.environ.get("APPROX_SGLANG_BACKEND", "triton")


def _record(event: dict) -> None:
    path = os.environ.get("APPROX_SGLANG_STATS_PATH", "")
    if not path:
        return
    event = {"ts": time.time(), "pid": os.getpid(), **event}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def _install() -> None:
    if not _enabled():
        return

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from approx_kernels import (
        approx_sglang_dynamic_w8a8_linear_kernel_1,
        approx_sglang_prequant_w8a8_linear_kernel_1,
        approx_sglang_w8a16_linear_kernel_1,
        quantize_activation_i8_per_row,
        quantize_weight_i8_per_col,
        sglang_dynamic_w8a8_linear_kernel,
        sglang_prequant_w8a8_linear_kernel,
        sglang_w8a16_linear_kernel,
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

    def prepare_weight(layer: torch.nn.Module, event: str = "prepare_weight") -> bool:
        weight = getattr(layer, "weight", None)
        if (
            weight is None
            or weight.ndim != 2
            or not weight.is_cuda
            or weight.dtype not in (torch.float16, torch.bfloat16)
        ):
            return False
        q, scale = quantize_weight_i8_per_col(weight.data)
        layer.register_buffer("_approx_qweight_i8_t", q, persistent=False)
        layer.register_buffer("_approx_qweight_scale", scale, persistent=False)
        if os.environ.get("APPROX_SGLANG_DROP_ORIGINAL_WEIGHT", "0") == "1":
            layer.weight = Parameter(torch.empty(0, device=weight.device, dtype=weight.dtype), requires_grad=False)
        _record(
            {
                "event": event,
                "prefix": getattr(layer, "prefix", ""),
                "weight_shape": list(weight.shape),
                "qweight_shape": list(q.shape),
                "qweight_stride": list(q.stride()),
                "dtype": str(weight.dtype),
            }
        )
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        original_process(self, layer)
        if not _target(layer):
            return
        prepare_weight(layer)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias=None) -> torch.Tensor:
        if os.environ.get("APPROX_SGLANG_MODE", "exact") != "approx":
            return original_apply(self, layer, x, bias)
        if not _target(layer):
            return original_apply(self, layer, x, bias)
        if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_cuda:
            return original_apply(self, layer, x, bias)
        if not hasattr(layer, "_approx_qweight_i8_t") and not prepare_weight(layer, "prepare_weight_lazy"):
            return original_apply(self, layer, x, bias)

        x_shape = tuple(x.shape)
        x2d = x.reshape(-1, x_shape[-1]).contiguous()
        if _decode_only() and x2d.shape[0] != 1:
            return original_apply(self, layer, x, bias)

        bq = layer._approx_qweight_i8_t
        bs = layer._approx_qweight_scale
        if _backend() == "triton_w8a16":
            import triton

            out = torch.empty((x2d.shape[0], bq.shape[1]), device=x.device, dtype=x.dtype)
            block_n = int(os.environ.get("APPROX_SGLANG_BLOCK_N", "128"))
            block_k = int(os.environ.get("APPROX_SGLANG_BLOCK_K", "64"))
            grid = (x2d.shape[0], triton.cdiv(bq.shape[1], block_n))
            if os.environ.get("APPROX_SGLANG_USE_SUBSTITUTE", "0") == "1":
                import approx_substitution_state as subst_state

                shape_key = (
                    "w8a16",
                    str(x2d.dtype),
                    int(x2d.shape[0]),
                    int(bq.shape[1]),
                    int(bq.shape[0]),
                    int(x2d.stride(0)),
                    int(x2d.stride(1)),
                    int(bq.stride(0)),
                    int(bq.stride(1)),
                    int(out.stride(0)),
                    int(out.stride(1)),
                    block_n,
                    block_k,
                )
                if shape_key not in subst_state.ttir_by_shape:
                    helper = approx_sglang_w8a16_linear_kernel_1[grid](
                        x2d,
                        bq,
                        bs,
                        out,
                        x2d.shape[0],
                        bq.shape[1],
                        bq.shape[0],
                        x2d.stride(0),
                        x2d.stride(1),
                        bq.stride(0),
                        bq.stride(1),
                        out.stride(0),
                        out.stride(1),
                        BLOCK_N=block_n,
                        BLOCK_K=block_k,
                    )
                    subst_state.ttir_by_shape[shape_key] = helper.asm["ttir"]
                    subst_state.seen_shapes.add(shape_key)
                    _record({"event": "prepare_substitute_ttir", "shape_key": list(shape_key)})
                subst_state.extra_ttir_texts[:] = [subst_state.ttir_by_shape[shape_key]]
            handle = sglang_w8a16_linear_kernel[grid](
                x2d,
                bq,
                bs,
                out,
                x2d.shape[0],
                bq.shape[1],
                bq.shape[0],
                x2d.stride(0),
                x2d.stride(1),
                bq.stride(0),
                bq.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_N=block_n,
                BLOCK_K=block_k,
            )
            substituted = "approx_sglang_w8a16_linear_kernel_1" in handle.asm.get("ttir", "")
            _record(
                {
                    "event": "apply_approx",
                    "backend": "triton_w8a16",
                    "prefix": getattr(layer, "prefix", ""),
                    "M": int(x2d.shape[0]),
                    "N": int(bq.shape[1]),
                    "K": int(bq.shape[0]),
                    "block_n": block_n,
                    "block_k": block_k,
                    "substituted": substituted,
                }
            )
            if bias is not None:
                out = out + bias
            if substituted:
                layer._approx_substitution_hits = getattr(layer, "_approx_substitution_hits", 0) + 1
            return out.reshape(*x_shape[:-1], bq.shape[1])

        if _backend() == "triton_prequant":
            import triton

            aq, a_scale = quantize_activation_i8_per_row(x2d)
            out = torch.empty((x2d.shape[0], bq.shape[1]), device=x.device, dtype=x.dtype)
            block_n = int(os.environ.get("APPROX_SGLANG_BLOCK_N", "128"))
            block_k = int(os.environ.get("APPROX_SGLANG_BLOCK_K", "64"))
            grid = (x2d.shape[0], triton.cdiv(bq.shape[1], block_n))
            if os.environ.get("APPROX_SGLANG_USE_SUBSTITUTE", "0") == "1":
                import approx_substitution_state as subst_state

                shape_key = (
                    "prequant",
                    str(aq.dtype),
                    str(out.dtype),
                    int(aq.shape[0]),
                    int(bq.shape[1]),
                    int(bq.shape[0]),
                    int(aq.stride(0)),
                    int(aq.stride(1)),
                    int(bq.stride(0)),
                    int(bq.stride(1)),
                    int(out.stride(0)),
                    int(out.stride(1)),
                    block_n,
                    block_k,
                )
                if shape_key not in subst_state.ttir_by_shape:
                    helper = approx_sglang_prequant_w8a8_linear_kernel_1[grid](
                        aq,
                        a_scale,
                        bq,
                        bs,
                        out,
                        aq.shape[0],
                        bq.shape[1],
                        bq.shape[0],
                        aq.stride(0),
                        aq.stride(1),
                        bq.stride(0),
                        bq.stride(1),
                        out.stride(0),
                        out.stride(1),
                        BLOCK_N=block_n,
                        BLOCK_K=block_k,
                    )
                    subst_state.ttir_by_shape[shape_key] = helper.asm["ttir"]
                    subst_state.seen_shapes.add(shape_key)
                    _record({"event": "prepare_substitute_ttir", "shape_key": list(shape_key)})
                subst_state.extra_ttir_texts[:] = [subst_state.ttir_by_shape[shape_key]]
            handle = sglang_prequant_w8a8_linear_kernel[grid](
                aq,
                a_scale,
                bq,
                bs,
                out,
                aq.shape[0],
                bq.shape[1],
                bq.shape[0],
                aq.stride(0),
                aq.stride(1),
                bq.stride(0),
                bq.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_N=block_n,
                BLOCK_K=block_k,
            )
            substituted = "approx_sglang_prequant_w8a8_linear_kernel_1" in handle.asm.get("ttir", "")
            _record(
                {
                    "event": "apply_approx",
                    "backend": "triton_prequant",
                    "prefix": getattr(layer, "prefix", ""),
                    "M": int(x2d.shape[0]),
                    "N": int(bq.shape[1]),
                    "K": int(bq.shape[0]),
                    "block_n": block_n,
                    "block_k": block_k,
                    "substituted": substituted,
                }
            )
            if bias is not None:
                out = out + bias
            if substituted:
                layer._approx_substitution_hits = getattr(layer, "_approx_substitution_hits", 0) + 1
            return out.reshape(*x_shape[:-1], bq.shape[1])

        if _backend() == "sgl_kernel":
            if int8_scaled_mm is None or per_token_quant_int8 is None:
                return original_apply(self, layer, x, bias)
            xq, xs = per_token_quant_int8(x2d)
            out = int8_scaled_mm(
                xq,
                bq,
                xs.view(-1, xs.shape[-1]),
                bs.view(-1, 1),
                out_dtype=x.dtype,
                bias=bias,
            )
            _record(
                {
                    "event": "apply_approx",
                    "backend": "sgl_kernel",
                    "prefix": getattr(layer, "prefix", ""),
                    "M": int(x2d.shape[0]),
                    "N": int(bq.shape[1]),
                    "K": int(bq.shape[0]),
                    "substituted": False,
                }
            )
            return out.reshape(*x_shape[:-1], bq.shape[1])

        out = torch.empty((x2d.shape[0], bq.shape[1]), device=x.device, dtype=x.dtype)
        import triton

        block_n = int(os.environ.get("APPROX_SGLANG_BLOCK_N", "64"))
        block_k = int(os.environ.get("APPROX_SGLANG_BLOCK_K", "64"))
        grid = (x2d.shape[0], triton.cdiv(bq.shape[1], block_n))
        if os.environ.get("APPROX_SGLANG_USE_SUBSTITUTE", "0") == "1":
            import approx_substitution_state as subst_state

            shape_key = (
                str(x2d.dtype),
                int(x2d.shape[0]),
                int(bq.shape[1]),
                int(bq.shape[0]),
                int(x2d.stride(0)),
                int(x2d.stride(1)),
                int(bq.stride(0)),
                int(bq.stride(1)),
                int(out.stride(0)),
                int(out.stride(1)),
                block_n,
                block_k,
            )
            if shape_key not in subst_state.ttir_by_shape:
                helper = approx_sglang_dynamic_w8a8_linear_kernel_1[grid](
                    x2d,
                    bq,
                    bs,
                    out,
                    x2d.shape[0],
                    bq.shape[1],
                    bq.shape[0],
                    x2d.stride(0),
                    x2d.stride(1),
                    bq.stride(0),
                    bq.stride(1),
                    out.stride(0),
                    out.stride(1),
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                )
                subst_state.ttir_by_shape[shape_key] = helper.asm["ttir"]
                subst_state.seen_shapes.add(shape_key)
                _record({"event": "prepare_substitute_ttir", "shape_key": list(shape_key)})
            subst_state.extra_ttir_texts[:] = [subst_state.ttir_by_shape[shape_key]]
        handle = sglang_dynamic_w8a8_linear_kernel[grid](
            x2d,
            bq,
            bs,
            out,
            x2d.shape[0],
            bq.shape[1],
            bq.shape[0],
            x2d.stride(0),
            x2d.stride(1),
            bq.stride(0),
            bq.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
        substituted = "approx_sglang_dynamic_w8a8_linear_kernel_1" in handle.asm.get("ttir", "")
        _record(
            {
                "event": "apply_approx",
                "prefix": getattr(layer, "prefix", ""),
                "M": int(x2d.shape[0]),
                "N": int(bq.shape[1]),
                "K": int(bq.shape[0]),
                "block_n": block_n,
                "block_k": block_k,
                "substituted": substituted,
            }
        )
        if bias is not None:
            out = out + bias
        if substituted:
            layer._approx_substitution_hits = getattr(layer, "_approx_substitution_hits", 0) + 1
        return out.reshape(*x_shape[:-1], bq.shape[1])

    UnquantizedLinearMethod.process_weights_after_loading = process_weights_after_loading
    UnquantizedLinearMethod.apply = apply
    UnquantizedLinearMethod._approx_sglang_quant_patched = True
    print("[approx-sglang-quant] patched UnquantizedLinearMethod", file=sys.stderr)


_install()
