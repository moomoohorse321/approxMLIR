# approx_quant_patch.py
#
# Runtime monkeypatch that injects approxMLIR quantization into SGLang's
# linear-layer dispatch path. When the APPROX_SGLANG_QUANT env flag is set
# and this module is imported (via sitecustomize.py inside SGLang worker
# subprocesses), it replaces `UnquantizedLinearMethod.apply` with a version
# that, on matching layers, routes through one of a handful of hand-written
# Triton int8/int4 quant backends. The module also optionally warms up an
# "approx_..._1" twin kernel so the approxMLIR Triton pass plugin can do
# function-substitution at TTIR time.
#
# Layout:
#   * env config + stats logger (top of file)
#   * `_run_substituted_kernel`  — shared warmup + launch + stats path
#   * `_install`                 — the monkeypatch installer; defines the
#                                  per-backend `_apply_*` closures and the
#                                  replacement `apply` that dispatches to them
#   * module-level `_install()` call at the bottom runs on import

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.parameter import Parameter


def _enabled() -> bool:
    # Single env gate for the whole patch; anything else is opt-in on top of this.
    return os.environ.get("APPROX_SGLANG_QUANT", "0") == "1"


# Append one JSON line per event to APPROX_SGLANG_STATS_PATH (if set). Used by
# the sweep driver to correlate apply-time events with per-case medians; opt-in
# so we don't pay file-IO when the stats path is unset.
def _record(event: dict) -> None:
    # path:  stats file path from env; empty string disables logging entirely.
    # event: already-typed fields from the caller; we prepend ts + pid here so
    #        call sites stay shape-free.
    path = os.environ.get("APPROX_SGLANG_STATS_PATH", "")
    if not path:
        return
    event = {"ts": time.time(), "pid": os.getpid(), **event}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


# Snapshot of every env var the patch reads. Built once in `_install()` so the
# four `_apply_*` backends share one consistent view instead of re-parsing env
# on each forward pass.
@dataclass(frozen=True)
class _Config:
    mode: str
    target: str
    backend: str
    decode_only: bool
    use_substitute: bool
    drop_weight: bool

    @classmethod
    def from_env(cls) -> "_Config":
        return cls(
            mode=os.environ.get("APPROX_SGLANG_MODE", "exact"),
            target=os.environ.get("APPROX_SGLANG_TARGET", "proj"),
            backend=os.environ.get("APPROX_SGLANG_BACKEND", "triton"),
            decode_only=os.environ.get("APPROX_SGLANG_DECODE_ONLY", "1") == "1",
            use_substitute=os.environ.get("APPROX_SGLANG_USE_SUBSTITUTE", "0") == "1",
            drop_weight=os.environ.get("APPROX_SGLANG_DROP_ORIGINAL_WEIGHT", "0") == "1",
        )

    # Does this layer's SGLang `prefix` (e.g. "model.layers.7.mlp.gate_up_proj")
    # match the comma-separated substring filter in APPROX_SGLANG_TARGET?
    # Called from both the weight-prep hook and the forward-time apply() so
    # targeting stays consistent across load-time and run-time.
    def target_match(self, layer: torch.nn.Module) -> bool:
        # prefix: SGLang-assigned path string, set by the LinearBase init patch
        #         below; "" for layers constructed before the patch landed.
        prefix = getattr(layer, "prefix", "")
        if self.target == "all":
            return True
        return any(tok and tok in prefix for tok in self.target.split(","))


# Read Triton block tile sizes from env, falling back to a per-backend default.
# Keeps autotune-style knob discovery easy without baking numbers into code.
def _block_dims(default_n: int, default_k: int = 64) -> tuple[int, int]:
    return (
        int(os.environ.get("APPROX_SGLANG_BLOCK_N", str(default_n))),
        int(os.environ.get("APPROX_SGLANG_BLOCK_K", str(default_k))),
    )


# Shared "run a quantized matmul with optional approxMLIR substitution" path.
# All four backends funnel through here so the substitute-warmup, primary
# launch, substitution-detection, and stats logic live in one place. Returns
# True if the approxMLIR pass actually swapped in the `_1` substitute kernel.
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
    # primary:         the @triton.jit kernel we actually want to run.
    # substitute:      its `_1` twin — launched only for its TTIR; the approxMLIR
    #                  pass plugin will pattern-match against that TTIR when the
    #                  primary kernel is compiled.
    # substitute_name: string name to grep for in the primary's compiled TTIR
    #                  to confirm the substitution fired.
    # shape_key:       cache key for the substitute TTIR; must uniquely identify
    #                  every constexpr/stride that would change the compiled IR.
    # record_common:   shared fields (prefix, M/N/K, block_n/block_k) that the
    #                  stats log wants but that the helper shouldn't own.
    import approx_substitution_state as state

    # 1) Warm up the substitute kernel so its TTIR is available to the plugin.
    #    Cached by shape_key so we only pay this once per (dtype, shape, stride,
    #    block) tuple.
    if use_substitute:
        if shape_key not in state.ttir_by_shape:
            helper = substitute[grid](*args, **kwargs)
            state.ttir_by_shape[shape_key] = helper.asm["ttir"]
            state.seen_shapes.add(shape_key)
            _record({"event": "prepare_substitute_ttir", "shape_key": list(shape_key)})
        # Publish the cached TTIR to the module-level list that the Triton
        # stage hook reads on the next compile.
        state.extra_ttir_texts[:] = [state.ttir_by_shape[shape_key]]

    # 2) Launch the primary kernel (its compile is what the plugin rewrites).
    handle = primary[grid](*args, **kwargs)

    # 3) Confirm substitution actually landed by searching the compiled TTIR
    #    for the substitute function name.
    substituted = substitute_name in handle.asm.get("ttir", "")
    _record({"event": "apply_approx", "backend": backend_name, "substituted": substituted, **record_common})
    if substituted:
        layer._approx_substitution_hits = getattr(layer, "_approx_substitution_hits", 0) + 1
    return substituted


# Top-level installer. Runs once on import (from module-level `_install()` at
# the bottom of the file) and mutates SGLang's `UnquantizedLinearMethod` in
# place. Reading order inside this function:
#   1) Import Triton kernels + sgl_kernel fallbacks.
#   2) Patch LinearBase.__init__ to remember each layer's `prefix` (so we can
#      target by name later).
#   3) Snapshot env into `cfg` and define `prepare_weight` + the
#      `process_weights_after_loading` override that calls it.
#   4) Define per-backend `_apply_*` closures and the replacement `apply`.
#   5) Swap the two methods onto the SGLang class.
def _install() -> None:
    if not _enabled():
        return

    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    # 1) Import our Triton kernels and (optionally) sgl_kernel's fast int8 mm.
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

    # Idempotent: multiple SGLang workers can import sitecustomize; only patch once.
    if getattr(UnquantizedLinearMethod, "_approx_sglang_quant_patched", False):
        return

    from sglang.srt.layers.linear import LinearBase

    # 2) Remember each linear layer's symbolic path on `self.prefix` so
    #    target_match() can filter by name (gate_up_proj, qkv_proj, ...).
    if not getattr(LinearBase, "_approx_prefix_patched", False):
        original_linear_init = LinearBase.__init__

        def linear_init_with_prefix(self, *args, **kwargs):
            prefix = kwargs.get("prefix", args[5] if len(args) > 5 else "")
            original_linear_init(self, *args, **kwargs)
            self.prefix = prefix

        LinearBase.__init__ = linear_init_with_prefix
        LinearBase._approx_prefix_patched = True

    # 3) Snapshot env, save originals, define weight prep.
    original_process = UnquantizedLinearMethod.process_weights_after_loading
    original_apply = UnquantizedLinearMethod.apply
    cfg = _Config.from_env()

    # Quantize the fp16/bf16 weight to int8 per-output-channel and park the
    # result on the layer as non-persistent buffers. Called both at load-time
    # (via process_weights_after_loading) and lazily in apply() if the
    # load-time hook missed the layer.
    def prepare_weight(layer: torch.nn.Module, event: str = "prepare_weight") -> bool:
        # weight:  the live fp16/bf16 Parameter on the layer — we read .data
        #          but don't mutate it (unless drop_weight is set below).
        # q:       int8 weight in [K, N] layout — quantize_weight_i8_per_col
        #          emits it transposed to match the kernel's K-major loads.
        # scale:   per-column fp32 scale vector of length N.
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
        # Optional memory saver for approx-only runs: toss the fp weight after
        # quantization so we don't carry both. Note: peak memory still spikes
        # during prepare_weight itself (both tensors alive briefly) — this is
        # why gate_up_qkv_drop OOMs on 8 GB cards.
        if cfg.drop_weight:
            layer.weight = Parameter(
                torch.empty(0, device=weight.device, dtype=weight.dtype),
                requires_grad=False,
            )
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

    # Load-time hook: run the original process (preserves whatever SGLang did)
    # and then prepare quantized buffers for layers that match the target
    # filter. Non-matching layers are left untouched so they keep the fp path.
    def process_weights_after_loading(self, layer):
        original_process(self, layer)
        if cfg.target_match(layer):
            prepare_weight(layer)

    # 4) Backend closures. Each one does the same four steps in order:
    #    a) fetch the cached int8 weight buffers from the layer
    #    b) (optionally) quantize the activation
    #    c) assemble the kernel args tuple + shape_key
    #    d) hand off to `_run_substituted_kernel` for the warmup + launch.
    # They differ only in which kernels they call and which side (weight-only
    # vs activation+weight) is quantized.

    # W8A16: int8 weights (quantized at load time), fp16/bf16 activations.
    # This is the primary decode-bandwidth-bound path — ~2x weight bandwidth
    # reduction vs fp16 with no activation quant cost.
    def _apply_w8a16(layer, x2d, x_shape, bias):
        # x2d: activation flattened to [M, K], same dtype as original (fp16/bf16).
        # bq:  int8 weight in [K, N] layout (already transposed at prep time).
        # bs:  per-column fp32 scale, length N.
        # out: fp output in x2d's dtype; kernel writes upcasted fp32 then stores.
        import triton
        bq = layer._approx_qweight_i8_t
        bs = layer._approx_qweight_scale
        M, K = x2d.shape[0], x2d.shape[1]
        N = bq.shape[1]
        block_n, block_k = _block_dims(128)
        grid = (M, triton.cdiv(N, block_n))
        out = torch.empty((M, N), device=x2d.device, dtype=x2d.dtype)
        args = (
            x2d, bq, bs, out, M, N, K,
            x2d.stride(0), x2d.stride(1),
            bq.stride(0), bq.stride(1),
            out.stride(0), out.stride(1),
        )
        kwargs = dict(BLOCK_N=block_n, BLOCK_K=block_k)
        shape_key = (
            "w8a16", str(x2d.dtype),
            M, N, K,
            x2d.stride(0), x2d.stride(1),
            bq.stride(0), bq.stride(1),
            out.stride(0), out.stride(1),
            block_n, block_k,
        )
        _run_substituted_kernel(
            backend_name="triton_w8a16",
            primary=sglang_w8a16_linear_kernel,
            substitute=approx_sglang_w8a16_linear_kernel_1,
            substitute_name="approx_sglang_w8a16_linear_kernel_1",
            grid=grid, args=args, kwargs=kwargs, shape_key=shape_key,
            use_substitute=cfg.use_substitute,
            record_common={
                "prefix": getattr(layer, "prefix", ""),
                "M": M, "N": N, "K": K,
                "block_n": block_n, "block_k": block_k,
            },
            layer=layer,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*x_shape[:-1], N)

    # W8A8 "prequant": activation is int8-quantized *outside* the matmul
    # kernel (in a dedicated per-row quantizer), then passed in. Useful when
    # the same activation feeds multiple matmuls (would only quantize once) —
    # currently not the default because dispatch doesn't exploit that reuse.
    def _apply_prequant(layer, x2d, x_shape, bias):
        # aq:      per-row int8 activation, shape [M, K].
        # a_scale: per-row fp32 activation scale, length M.
        # bq/bs:   same as W8A16 path.
        # The matmul accumulates int32 then multiplies by a_scale * b_scale at the end.
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
            aq, a_scale, bq, bs, out, M, N, K,
            aq.stride(0), aq.stride(1),
            bq.stride(0), bq.stride(1),
            out.stride(0), out.stride(1),
        )
        kwargs = dict(BLOCK_N=block_n, BLOCK_K=block_k)
        shape_key = (
            "prequant", str(aq.dtype), str(out.dtype),
            M, N, K,
            aq.stride(0), aq.stride(1),
            bq.stride(0), bq.stride(1),
            out.stride(0), out.stride(1),
            block_n, block_k,
        )
        _run_substituted_kernel(
            backend_name="triton_prequant",
            primary=sglang_prequant_w8a8_linear_kernel,
            substitute=approx_sglang_prequant_w8a8_linear_kernel_1,
            substitute_name="approx_sglang_prequant_w8a8_linear_kernel_1",
            grid=grid, args=args, kwargs=kwargs, shape_key=shape_key,
            use_substitute=cfg.use_substitute,
            record_common={
                "prefix": getattr(layer, "prefix", ""),
                "M": M, "N": N, "K": K,
                "block_n": block_n, "block_k": block_k,
            },
            layer=layer,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*x_shape[:-1], N)

    # W8A8 "dynamic": a single fused kernel that does per-row activation
    # absmax → quantize → int8 matmul → dequant, all in one pass. Default
    # backend when APPROX_SGLANG_BACKEND is unset or set to "triton".
    def _apply_dynamic_w8a8(layer, x2d, x_shape, bias):
        # Same inputs as W8A16 but the kernel itself does the activation quant
        # inline instead of caller-side. block_n defaults to 64 here (not 128)
        # because the fused version has more register pressure per program.
        import triton
        bq = layer._approx_qweight_i8_t
        bs = layer._approx_qweight_scale
        M, K = x2d.shape[0], x2d.shape[1]
        N = bq.shape[1]
        block_n, block_k = _block_dims(64)
        grid = (M, triton.cdiv(N, block_n))
        out = torch.empty((M, N), device=x2d.device, dtype=x2d.dtype)
        args = (
            x2d, bq, bs, out, M, N, K,
            x2d.stride(0), x2d.stride(1),
            bq.stride(0), bq.stride(1),
            out.stride(0), out.stride(1),
        )
        kwargs = dict(BLOCK_N=block_n, BLOCK_K=block_k)
        shape_key = (
            str(x2d.dtype),
            M, N, K,
            x2d.stride(0), x2d.stride(1),
            bq.stride(0), bq.stride(1),
            out.stride(0), out.stride(1),
            block_n, block_k,
        )
        _run_substituted_kernel(
            backend_name="triton",
            primary=sglang_dynamic_w8a8_linear_kernel,
            substitute=approx_sglang_dynamic_w8a8_linear_kernel_1,
            substitute_name="approx_sglang_dynamic_w8a8_linear_kernel_1",
            grid=grid, args=args, kwargs=kwargs, shape_key=shape_key,
            use_substitute=cfg.use_substitute,
            record_common={
                "prefix": getattr(layer, "prefix", ""),
                "M": M, "N": N, "K": K,
                "block_n": block_n, "block_k": block_k,
            },
            layer=layer,
        )
        if bias is not None:
            out = out + bias
        return out.reshape(*x_shape[:-1], N)

    # Escape hatch: route through sgl_kernel's `int8_scaled_mm` C++ path
    # instead of a Triton kernel. Used for baselining against a known-good
    # int8 implementation. No approxMLIR substitution happens here.
    # Returns None to signal the outer apply() to fall back to the original
    # fp path when the required imports are missing.
    def _apply_sgl_kernel(layer, x2d, x_shape, bias):
        if int8_scaled_mm is None or per_token_quant_int8 is None:
            return None
        bq = layer._approx_qweight_i8_t
        bs = layer._approx_qweight_scale
        M, K = x2d.shape[0], x2d.shape[1]
        N = bq.shape[1]
        xq, xs = per_token_quant_int8(x2d)
        out = int8_scaled_mm(
            xq, bq, xs.view(-1, xs.shape[-1]), bs.view(-1, 1),
            out_dtype=x2d.dtype, bias=bias,
        )
        _record(
            {
                "event": "apply_approx",
                "backend": "sgl_kernel",
                "prefix": getattr(layer, "prefix", ""),
                "M": M, "N": N, "K": K,
                "substituted": False,
            }
        )
        return out.reshape(*x_shape[:-1], N)

    # Backend registry. Order doesn't matter — selected by cfg.backend string.
    # Anything not in this dict silently falls through to _apply_dynamic_w8a8.
    _BACKENDS = {
        "triton_w8a16": _apply_w8a16,
        "triton_prequant": _apply_prequant,
        "triton": _apply_dynamic_w8a8,
        "sgl_kernel": _apply_sgl_kernel,
    }

    # Replacement for UnquantizedLinearMethod.apply. The bail-out checks at
    # the top come in a deliberate order — cheapest first — so that the hot
    # path (approx-disabled or non-target layer) exits before we touch x.
    # Reading order:
    #   1) check mode + target match    (cheap string ops)
    #   2) check dtype/device           (cheap tensor attr reads)
    #   3) lazily prepare int8 buffers  (one-time per layer)
    #   4) reshape to 2D + decode gate  (avoids prefill when decode_only)
    #   5) dispatch to backend closure
    def apply(self, layer, x, bias=None):
        # self:  the UnquantizedLinearMethod instance (unused here, kept for ABI).
        # layer: the SGLang linear layer — carries .prefix and the cached
        #        _approx_qweight_* buffers.
        # x:     [..., K] activation; reshaped to [M, K] for the kernel.
        # bias:  optional bias tensor added after the matmul inside each backend.
        if cfg.mode != "approx" or not cfg.target_match(layer):
            return original_apply(self, layer, x, bias)
        if x.dtype not in (torch.float16, torch.bfloat16) or not x.is_cuda:
            return original_apply(self, layer, x, bias)
        if not hasattr(layer, "_approx_qweight_i8_t") and not prepare_weight(layer, "prepare_weight_lazy"):
            return original_apply(self, layer, x, bias)

        x_shape = tuple(x.shape)
        x2d = x.reshape(-1, x_shape[-1]).contiguous()
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
