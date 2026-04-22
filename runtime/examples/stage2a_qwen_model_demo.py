#!/usr/bin/env python3
"""Stage 2(a): real-model decode demo with Triton-backed Qwen projections.

This runs a real Hugging Face Qwen model, but patches selected decode-time
projection layers with a Triton-backed linear wrapper. The wrapper keeps prefill
exact and only switches decode-time M=1 launches onto Triton. Exact and approx
modes share the same caller shell; only the weight data path changes.

Env:
- TRITON_PASS_PLUGIN_PATH: required
- QWEN_MODEL: default Qwen/Qwen2.5-0.5B-Instruct
- QWEN_PROMPT: default short prompt
- QWEN_DECODE_STEPS: default 8
- APPROX_GROUP_SIZE: default 128
- APPROX_BITS: 8 or 4
- QWEN_PATCH: comma-separated set from
  q_proj,k_proj,v_proj,o_proj,in_proj_qkv,out_proj,gate_proj,up_proj,down_proj
- QWEN_LAYERS: "all" or comma-separated layer indices
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_TRITON_PYTHON = _REPO_ROOT / "triton" / "python"
if str(_SOURCE_TRITON_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SOURCE_TRITON_PYTHON))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "fork")

import torch
import torch.nn as nn

from approx_quant_api import (
    ApproxAffineQuantizedTensor,
    ApproxInt8DynamicActivationInt8WeightConfig,
    ApproxLinearActivationQuantizedTensor,
    quantize_,
)


def _setup_dlopen_flags() -> None:
    try:
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    except AttributeError:
        pass


def _prepend_plugin_env(plugin: str) -> None:
    paths = [p for p in os.environ.get("TRITON_PLUGIN_PATHS", "").split(":") if p]
    if plugin not in paths:
        os.environ["TRITON_PLUGIN_PATHS"] = ":".join([plugin] + paths)


def _configure_inductor_env() -> None:
    if os.environ.get("QWEN_INDUCTOR_ALLOW_UNSPEC_INT", "1") == "1":
        torch._dynamo.config.allow_unspec_int_on_nn_module = True


def _compile_module(module: nn.Module) -> nn.Module:
    mode = os.environ.get("QWEN_INDUCTOR_MODE")
    fullgraph = os.environ.get("QWEN_INDUCTOR_FULLGRAPH", "0") == "1"
    kwargs = {"backend": "inductor", "fullgraph": fullgraph}
    if mode:
        kwargs["mode"] = mode
    return torch.compile(module, **kwargs)


def _maybe_compile_model(model: nn.Module, scope: str, wrappers: list[nn.Module]) -> nn.Module:
    if scope == "none":
        return model
    _configure_inductor_env()
    if scope == "patched":
        for wrapper in wrappers:
            compiled = _compile_module(wrapper)
            wrapper.forward = compiled.forward
        return model
    if scope == "model":
        model.forward = _compile_module(model.forward)
        return model
    raise RuntimeError("QWEN_INDUCTOR_SCOPE must be one of: none, patched, model")
    return model


def _force_stable_qwen3_5_linear_attn_mask(model: nn.Module) -> None:
    text_model = getattr(model, "model", None)
    updater = getattr(text_model, "_update_linear_attn_mask", None)
    if updater is None:
        return

    def _stable_update_linear_attn_mask(attention_mask, past_key_values):
        return attention_mask

    text_model._update_linear_attn_mask = _stable_update_linear_attn_mask


def _bench_decode(torch_mod, step_fn, inputs: list[torch.Tensor]) -> float:
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    torch_mod.cuda.synchronize()
    start.record()
    for token in inputs:
        step_fn(token)
    end.record()
    torch_mod.cuda.synchronize()
    return start.elapsed_time(end) / len(inputs)


@dataclass
class DecodeStats:
    avg_ms: float
    logits: list[torch.Tensor]
    next_tokens: list[int]
    substitution_hits: int


def _token_rank(logits: torch.Tensor, token_id: int) -> int:
    order = torch.argsort(logits, dim=-1, descending=True)
    hits = (order == token_id).nonzero(as_tuple=False)
    if hits.numel() == 0:
        return int(logits.shape[-1])
    return int(hits[0, 1].item()) + 1


def _topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> float:
    top_a = set(torch.topk(logits_a, k=min(k, logits_a.shape[-1]), dim=-1).indices[0].tolist())
    top_b = set(torch.topk(logits_b, k=min(k, logits_b.shape[-1]), dim=-1).indices[0].tolist())
    return len(top_a & top_b) / float(k)


class TritonDecodeLinear(nn.Module):
    # Cache key includes _step_id so stale entries from a prior decode step can never
    # be returned when torch's caching allocator reuses a data_ptr.
    _step_id: int = 0
    _quant_cache: dict[tuple[int, int, tuple[int, ...], torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
    _cache_hits: int = 0
    _cache_misses: int = 0
    _profile_enabled: bool = False
    _profile_stats: dict[str, float] = {}
    _profile_counts: dict[str, int] = {}

    @classmethod
    def clear_quant_cache(cls) -> None:
        cls._quant_cache.clear()
        cls._cache_hits = 0
        cls._cache_misses = 0

    @classmethod
    def clear_profile_stats(cls) -> None:
        cls._profile_stats = {}
        cls._profile_counts = {}

    @classmethod
    def set_profile_enabled(cls, enabled: bool) -> None:
        cls._profile_enabled = enabled

    @classmethod
    def new_step(cls) -> None:
        cls._step_id += 1
        cls._quant_cache.clear()

    @classmethod
    def _record_ms(cls, key: str, ms: float) -> None:
        if not cls._profile_enabled:
            return
        cls._profile_stats[key] = cls._profile_stats.get(key, 0.0) + ms
        cls._profile_counts[key] = cls._profile_counts.get(key, 0) + 1

    @classmethod
    def _measure_cuda_ms(cls, key: str, fn):
        if not cls._profile_enabled:
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        cls._record_ms(key, start.elapsed_time(end))
        return out

    def __init__(self, linear: nn.Linear, group_size_k: int, quant_bits: int, approx_impl: str):
        super().__init__()
        self.original = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.group_size_k = group_size_k
        self.quant_bits = quant_bits
        self.approx_impl = approx_impl
        self.last_input_dtype = linear.weight.dtype
        self.mode = "exact"
        self.decode_only = True
        self.substitution_hits = 0

        weight_t = linear.weight.detach().t().contiguous()
        self.register_buffer("weight_t_exact", weight_t)
        self.register_buffer("weight_t_packed", torch.empty(0, dtype=torch.float16))
        self.register_buffer("weight_t_i8", torch.empty(0, dtype=torch.int8))
        self.register_buffer("weight_t_i8_scale", torch.empty(0, dtype=torch.float32))
        self.weight_layout = ""
        self.weight_group_size_runtime = self.in_features
        self.quantized_weight: ApproxAffineQuantizedTensor | ApproxLinearActivationQuantizedTensor | None = None
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().contiguous().to(dtype=torch.float16))

    def finalize_approx(self, pack_fn, quant_b_fn) -> None:
        if self.approx_impl == "packed_dequant":
            self.weight_t_packed = pack_fn(self.weight_t_exact, self.group_size_k, self.quant_bits)
        elif self.approx_impl == "fast_int8":
            self.quantized_weight = getattr(self.original, "_approx_quantized_weight", None)
            if not isinstance(self.quantized_weight, ApproxLinearActivationQuantizedTensor):
                raise RuntimeError("fast_int8 expected activation-aware quantized weight")
            self.weight_t_i8 = self.quantized_weight.original_weight_tensor.qdata
            self.weight_t_i8_scale = self.quantized_weight.original_weight_tensor.scale
            self.weight_layout = self.quantized_weight.original_weight_tensor.layout
            self.weight_group_size_runtime = self.quantized_weight.original_weight_tensor.group_size
        else:
            raise RuntimeError(f"Unknown approx impl: {self.approx_impl}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input_dtype = x.dtype
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            return self.original(x)
        x_2d = x.reshape(-1, self.in_features).contiguous()
        if self.decode_only and x_2d.shape[0] != 1:
            return self.original(x)

        sys.path.insert(0, str(Path(__file__).parent))
        from kernels.llm_like import exact_matmul_kernel, matmul_kernel
        from kernels.quant_kernels import int8_dequant_matmul_kernel, int8_matmul_kernel

        import triton

        out = torch.empty((x_2d.shape[0], self.out_features), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(x_2d.shape[0], 64), triton.cdiv(self.out_features, 64))
        if self.mode == "exact":
            b = self.weight_t_exact
            stride_bk, stride_bn = b.stride(0), b.stride(1)
            self._measure_cuda_ms(
                "exact_mm_ms",
                lambda: exact_matmul_kernel[grid](
                    x_2d, b, out,
                    x_2d.shape[0], self.out_features, self.in_features,
                    x_2d.stride(0), x_2d.stride(1),
                    stride_bk, stride_bn,
                    out.stride(0), out.stride(1),
                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                ),
            )
        elif self.approx_impl == "packed_dequant":
            b = self.weight_t_packed
            stride_bk, stride_bn = self.group_size_k, self.quant_bits
            handle = self._measure_cuda_ms(
                "packed_dequant_mm_ms",
                lambda: matmul_kernel[grid](
                    x_2d, b, out,
                    x_2d.shape[0], self.out_features, self.in_features,
                    x_2d.stride(0), x_2d.stride(1),
                    stride_bk, stride_bn,
                    out.stride(0), out.stride(1),
                    BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                ),
            )
            if "approx_matmul_kernel_1" in handle.asm.get("ttir", ""):
                self.substitution_hits += 1
        else:
            cls = type(self)
            cache_key = (cls._step_id, x_2d.data_ptr(), tuple(x_2d.shape), x_2d.dtype)
            cached = cls._quant_cache.get(cache_key)
            if cached is None:
                if not isinstance(self.quantized_weight, ApproxLinearActivationQuantizedTensor):
                    raise RuntimeError("fast_int8 expected ApproxLinearActivationQuantizedTensor")
                cached = self._measure_cuda_ms(
                    "fast_int8_actq_ms",
                    lambda: self.quantized_weight.input_quant_func(x_2d),
                )
                cls._quant_cache[cache_key] = cached
                cls._cache_misses += 1
            else:
                cls._cache_hits += 1
            a_q, a_scale = cached
            acc = torch.zeros((x_2d.shape[0], self.out_features), device=x.device, dtype=torch.float32)
            if self.weight_layout == "per_group_int8":
                num_groups = self.weight_t_i8_scale.shape[0]
                for group_idx in range(num_groups):
                    k0 = group_idx * self.weight_group_size_runtime
                    k1 = min(k0 + self.weight_group_size_runtime, self.in_features)
                    a_q_g = a_q[:, k0:k1].contiguous()
                    b_q_g = self.weight_t_i8[k0:k1, :].contiguous()
                    c_i32 = torch.empty((x_2d.shape[0], self.out_features), device=x.device, dtype=torch.int32)
                    handle = self._measure_cuda_ms(
                        "fast_int8_mm_ms",
                        lambda: int8_matmul_kernel[grid](
                            a_q_g, b_q_g, c_i32,
                            x_2d.shape[0], self.out_features, k1 - k0,
                            a_q_g.stride(0), a_q_g.stride(1),
                            b_q_g.stride(0), b_q_g.stride(1),
                            c_i32.stride(0), c_i32.stride(1),
                            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                        ),
                    )
                    if "approx_int8_matmul_kernel_1" in handle.asm.get("ttir", ""):
                        self.substitution_hits += 1
                    acc += c_i32.float() * self.weight_t_i8_scale[group_idx][None, :]
                self._measure_cuda_ms(
                    "fast_int8_epilogue_ms",
                    lambda: out.copy_((acc * a_scale[:, None]).to(out.dtype)),
                )
            else:
                handle = self._measure_cuda_ms(
                    "fast_int8_mm_ms",
                    lambda: int8_dequant_matmul_kernel[grid](
                        a_q, self.weight_t_i8, a_scale, self.weight_t_i8_scale, out,
                        x_2d.shape[0], self.out_features, self.in_features,
                        a_q.stride(0), a_q.stride(1),
                        self.weight_t_i8.stride(0), self.weight_t_i8.stride(1),
                        out.stride(0), out.stride(1),
                        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                    ),
                )
                if (
                    "approx_int8_dequant_matmul_kernel_1" in handle.asm.get("ttir", "")
                    or "approx_int8_matmul_kernel_1" in handle.asm.get("ttir", "")
                ):
                    self.substitution_hits += 1
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], self.out_features)


class TritonFusedGateUpMLP(nn.Module):
    def __init__(self, mlp: nn.Module, group_size_k: int):
        super().__init__()
        self.original = mlp
        self.group_size_k = group_size_k
        self.act_fn = mlp.act_fn
        self.down_proj = mlp.down_proj
        self.mode = "exact"
        self.substitution_hits = 0
        self.decode_only = True

        self.gate = TritonDecodeLinear(mlp.gate_proj, group_size_k, 8, "fast_int8")
        self.up = TritonDecodeLinear(mlp.up_proj, group_size_k, 8, "fast_int8")
        self.down_exact = TritonDecodeLinear(mlp.down_proj, group_size_k, 8, "packed_dequant")
        self.gate.to("cuda")
        self.up.to("cuda")
        self.down_exact.to("cuda")
        self.gate.finalize_approx(None, None)
        self.up.finalize_approx(None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            return self.original(x)
        x_2d = x.reshape(-1, self.gate.in_features).contiguous()
        if self.decode_only and x_2d.shape[0] != 1:
            return self.original(x)
        if self.mode == "exact":
            return self.original(x)

        sys.path.insert(0, str(Path(__file__).parent))
        from kernels.quant_kernels import int8_dual_dequant_matmul_kernel
        import triton

        cls = TritonDecodeLinear
        cache_key = (cls._step_id, x_2d.data_ptr(), tuple(x_2d.shape), x_2d.dtype)
        cached = cls._quant_cache.get(cache_key)
        if cached is None:
            if not isinstance(self.gate.quantized_weight, ApproxLinearActivationQuantizedTensor):
                raise RuntimeError("fused gate/up expected activation-aware quantized weight")
            cached = cls._measure_cuda_ms(
                "fast_int8_actq_ms",
                lambda: self.gate.quantized_weight.input_quant_func(x_2d),
            )
            cls._quant_cache[cache_key] = cached
            cls._cache_misses += 1
        else:
            cls._cache_hits += 1
        a_q, a_scale = cached

        if self.gate.weight_layout != "per_col_int8" or self.up.weight_layout != "per_col_int8":
            gate = self.gate(x)
            up = self.up(x)
            self.down_exact.mode = "exact"
            out = self.down_exact(self.act_fn(gate) * up)
            self.substitution_hits = (
                self.gate.substitution_hits
                + self.up.substitution_hits
                + self.down_exact.substitution_hits
            )
            return out

        gate_out = torch.empty((x_2d.shape[0], self.gate.out_features), device=x.device, dtype=x.dtype)
        up_out = torch.empty_like(gate_out)
        grid = (triton.cdiv(x_2d.shape[0], 64), triton.cdiv(self.gate.out_features, 64))
        handle = cls._measure_cuda_ms(
            "fast_int8_mm_ms",
            lambda: int8_dual_dequant_matmul_kernel[grid](
                a_q,
                self.gate.weight_t_i8,
                self.up.weight_t_i8,
                a_scale,
                self.gate.weight_t_i8_scale,
                self.up.weight_t_i8_scale,
                gate_out,
                up_out,
                x_2d.shape[0],
                self.gate.out_features,
                self.gate.in_features,
                a_q.stride(0),
                a_q.stride(1),
                self.gate.weight_t_i8.stride(0),
                self.gate.weight_t_i8.stride(1),
                gate_out.stride(0),
                gate_out.stride(1),
                BLOCK_M=64,
                BLOCK_N=64,
                BLOCK_K=32,
            ),
        )
        if "approx_int8_dual_dequant_matmul_kernel_1" in handle.asm.get("ttir", ""):
            self.substitution_hits += 1
        hidden = self.act_fn(gate_out) * up_out
        self.down_exact.mode = "exact"
        out = self.down_exact(hidden.reshape(*x.shape[:-1], self.gate.out_features))
        self.substitution_hits += self.down_exact.substitution_hits
        return out


def _iter_target_modules(
    model, patch_set: set[str], layer_filter: set[int] | None
) -> Iterable[tuple[int, str, nn.Module, str, nn.Linear]]:
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_filter is not None and layer_idx not in layer_filter:
            continue
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if name in patch_set:
                    mod = getattr(attn, name, None)
                    if isinstance(mod, nn.Linear):
                        yield layer_idx, f"model.layers.{layer_idx}.self_attn.{name}", attn, name, mod
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is not None:
            for name in ("in_proj_qkv", "out_proj"):
                if name in patch_set:
                    mod = getattr(linear_attn, name, None)
                    if isinstance(mod, nn.Linear):
                        yield layer_idx, f"model.layers.{layer_idx}.linear_attn.{name}", linear_attn, name, mod
        mlp = layer.mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            if name in patch_set:
                mod = getattr(mlp, name)
                if isinstance(mod, nn.Linear):
                    yield layer_idx, f"model.layers.{layer_idx}.mlp.{name}", mlp, name, mod


def _resolve_layer_filter(layers_spec: str, num_layers: int) -> set[int] | None:
    spec = layers_spec.strip()
    if spec == "all":
        return None
    if spec.startswith("last"):
        count = int(spec[4:])
        if count <= 0:
            raise RuntimeError("QWEN_LAYERS=lastN requires N > 0")
        start = max(0, num_layers - count)
        return set(range(start, num_layers))
    return {int(v) for v in spec.split(",") if v.strip()}


def _build_hook(
    plugin: str,
    activation_dtype: torch.dtype,
    approx_impl: str,
    *,
    fused_gate_up: bool = False,
):
    import approx_runtime as ar
    sys.path.insert(0, str(Path(__file__).parent))
    from kernels.llm_like import matmul_kernel
    from kernels.quant_kernels import (
        approx_int8_dual_dequant_matmul_kernel_1,
        approx_int8_dequant_matmul_kernel_1,
        approx_int8_matmul_kernel_1,
        approx_matmul_kernel_1,
    )

    if approx_impl == "packed_dequant":
        func_name = "matmul_kernel"
        approx_kernel = approx_matmul_kernel_1
        a = torch.randn((1, 896), device="cuda", dtype=activation_dtype)
        b = torch.randn((896, 896), device="cuda", dtype=torch.float16)
        c = torch.empty((1, 896), device="cuda", dtype=activation_dtype)
        approx_handle = approx_kernel[(1, 14)](
            a, b, c, 1, 896, 896,
            a.stride(0), a.stride(1),
            128, 0,
            c.stride(0), c.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )
    else:
        a = torch.randint(-127, 128, (1, 2048), device="cuda", dtype=torch.int8)
        b = torch.randint(-127, 128, (2048, 6144), device="cuda", dtype=torch.int8)
        a_scale = torch.ones((1,), device="cuda", dtype=torch.float32)
        b_scale = torch.ones((6144,), device="cuda", dtype=torch.float32)
        c = torch.empty((1, 6144), device="cuda", dtype=activation_dtype)
        if fused_gate_up:
            func_name = "int8_dual_dequant_matmul_kernel"
            approx_kernel = approx_int8_dual_dequant_matmul_kernel_1
            c2 = torch.empty_like(c)
            approx_handle = approx_kernel[(1, 96)](
                a, b, b, a_scale, b_scale, b_scale, c, c2, 1, 6144, 2048,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            )
        else:
            func_name = "int8_dequant_matmul_kernel"
            approx_kernel = approx_int8_dequant_matmul_kernel_1
            approx_handle = approx_kernel[(1, 96)](
                a, b, a_scale, b_scale, c, 1, 6144, 2048,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            )
    extra_ttir = [approx_handle.asm["ttir"]] if "ttir" in approx_handle.asm else []
    if not extra_ttir:
        raise RuntimeError("approx kernel did not expose ttir asm")
    config = {
        "decision_tree": None,
        "safety_contract": None,
        "static_transform": ar.StaticTransform(
            transform_type="func_substitute",
            knob_val=1,
            approx_kernel=approx_kernel,
        ),
    }
    pipeline = ar.get_pipeline_for_config(config, workload=ar.WorkloadType.TRITON)
    return ar.make_triton_stages_hook(
        passes=pipeline,
        plugin_path=plugin,
        stage_name="make_ttir_approx",
        func_name=func_name,
        config=config,
        extra_ttir_texts=extra_ttir,
        verbose=False,
    )


def _make_decode_cache(model, input_ids: torch.Tensor, decode_steps: int):
    from transformers import StaticCache

    max_cache_len = int(input_ids.shape[1]) + int(decode_steps) + 1
    return StaticCache(config=model.config, max_cache_len=max_cache_len)


def _mark_compile_step() -> None:
    marker = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def _make_fixed_attention_mask(input_ids: torch.Tensor, max_cache_len: int | None) -> Optional[torch.Tensor]:
    if max_cache_len is None:
        return None
    mask = torch.zeros((input_ids.shape[0], max_cache_len), device=input_ids.device, dtype=torch.bool)
    mask[:, : input_ids.shape[1]] = True
    return mask


def _cache_max_len(cache) -> Optional[int]:
    if cache is None:
        return None
    getter = getattr(cache, "get_max_cache_shape", None)
    if getter is None:
        return None
    try:
        return int(getter())
    except Exception:
        layers = getattr(cache, "layers", None)
        if layers is None:
            return None
        for idx, layer in enumerate(layers):
            if hasattr(layer, "get_max_cache_shape"):
                return int(getter(idx))
        return None


def _decode_trace(model, input_ids: torch.Tensor, steps: int, base_cache=None) -> list[int]:
    tokens: list[int] = []
    with torch.inference_mode():
        TritonDecodeLinear.new_step()
        _mark_compile_step()
        cache = base_cache
        if cache is not None:
            cache.reset()
        attention_mask = _make_fixed_attention_mask(input_ids, _cache_max_len(cache))
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(steps):
            tokens.append(int(next_token.item()))
            TritonDecodeLinear.new_step()
            _mark_compile_step()
            if attention_mask is not None:
                cur_len = int(cache.get_seq_length())
                attention_mask[:, cur_len : cur_len + next_token.shape[1]] = 1
            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return tokens


def _run_decode(model, input_ids: torch.Tensor, teacher_tokens: list[int], base_cache=None) -> DecodeStats:
    teacher_inputs = [torch.tensor([[tok]], device=input_ids.device, dtype=input_ids.dtype) for tok in teacher_tokens]

    with torch.inference_mode():
        # Warm decode caches and any Triton compilation before timing.
        TritonDecodeLinear.new_step()
        _mark_compile_step()
        warm_cache = base_cache
        if warm_cache is not None:
            warm_cache.reset()
        warm_attention_mask = _make_fixed_attention_mask(input_ids, _cache_max_len(warm_cache))
        warm = model(
            input_ids=input_ids,
            attention_mask=warm_attention_mask,
            past_key_values=warm_cache,
            use_cache=True,
            return_dict=True,
        )
        warm_cache = warm.past_key_values
        for token in teacher_inputs:
            TritonDecodeLinear.new_step()
            _mark_compile_step()
            if warm_attention_mask is not None:
                cur_len = int(warm_cache.get_seq_length())
                warm_attention_mask[:, cur_len : cur_len + token.shape[1]] = 1
            warm = model(
                input_ids=token,
                attention_mask=warm_attention_mask,
                past_key_values=warm_cache,
                use_cache=True,
                return_dict=True,
            )
            warm_cache = warm.past_key_values

        TritonDecodeLinear.new_step()
        _mark_compile_step()
        cache = base_cache
        if cache is not None:
            cache.reset()
        attention_mask = _make_fixed_attention_mask(input_ids, _cache_max_len(cache))
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logits: list[torch.Tensor] = []
        next_tokens: list[int] = []

        def timed_step(token: torch.Tensor) -> None:
            nonlocal cache, logits, next_tokens
            TritonDecodeLinear.new_step()
            _mark_compile_step()
            if attention_mask is not None:
                cur_len = int(cache.get_seq_length())
                attention_mask[:, cur_len : cur_len + token.shape[1]] = 1
            step_out = model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = step_out.past_key_values
            step_logits = step_out.logits[:, -1, :].detach().float().cpu()
            logits.append(step_logits)
            next_tokens.append(int(step_logits.argmax(dim=-1).item()))

        avg_ms = _bench_decode(torch, timed_step, teacher_inputs)
    substitution_hits = 0
    for module in model.modules():
        if isinstance(module, (TritonDecodeLinear, TritonFusedGateUpMLP)):
            substitution_hits += module.substitution_hits
    return DecodeStats(
        avg_ms=avg_ms,
        logits=logits,
        next_tokens=next_tokens,
        substitution_hits=substitution_hits,
    )


def main() -> int:
    _setup_dlopen_flags()
    plugin = os.environ.get("TRITON_PASS_PLUGIN_PATH", "")
    if not plugin:
        raise RuntimeError("Set TRITON_PASS_PLUGIN_PATH to libApproxTritonPlugin.so before running")
    _prepend_plugin_env(plugin)

    import approx_runtime as ar
    import triton  # noqa: F401
    from triton import knobs
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from kernels.quant_kernels import pack_b_i8_per_group_fp16_storage, quantize_b_i8_per_col

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model_id = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    prompt = os.environ.get("QWEN_PROMPT", "Write one short sentence about Paris.")
    decode_steps = int(os.environ.get("QWEN_DECODE_STEPS", "8"))
    group_size = int(os.environ.get("APPROX_GROUP_SIZE", "128"))
    quant_bits = int(os.environ.get("APPROX_BITS", "8"))
    if quant_bits not in (4, 8):
        raise RuntimeError("APPROX_BITS must be 4 or 8")
    approx_impl = os.environ.get("APPROX_IMPL", "packed_dequant")
    if approx_impl not in ("packed_dequant", "fast_int8"):
        raise RuntimeError("APPROX_IMPL must be packed_dequant or fast_int8")
    if approx_impl == "fast_int8" and quant_bits != 8:
        raise RuntimeError("fast_int8 currently supports APPROX_BITS=8 only")
    use_substitute = os.environ.get("APPROX_USE_SUBSTITUTE", "1") == "1"
    use_inductor = os.environ.get("QWEN_USE_INDUCTOR", "0") == "1"
    inductor_scope = os.environ.get("QWEN_INDUCTOR_SCOPE", "model" if use_inductor else "none")
    use_static_cache = os.environ.get("QWEN_USE_STATIC_CACHE", "1" if use_inductor else "0") == "1"
    profile_components = os.environ.get("APPROX_PROFILE_COMPONENTS", "0") == "1"
    fuse_gate_up = os.environ.get("APPROX_FUSE_GATE_UP", "1") == "1"
    layers_spec = os.environ.get("QWEN_LAYERS", "all")
    patch_set = {
        s.strip() for s in os.environ.get("QWEN_PATCH", "q_proj,o_proj").split(",") if s.strip()
    }

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to("cuda")
    model.eval()
    if use_inductor:
        model.generation_config.cache_implementation = "static"
        if os.environ.get("QWEN_FORCE_STABLE_LINEAR_ATTN_MASK", "1") == "1":
            _force_stable_qwen3_5_linear_attn_mask(model)
    layer_filter = _resolve_layer_filter(layers_spec, len(model.model.layers))

    targets = list(_iter_target_modules(model, patch_set, layer_filter))
    if not targets:
        raise RuntimeError("No target modules were patched")

    if approx_impl == "fast_int8":
        target_names = {fqn for _, fqn, _, _, _ in targets}

        def _filter_fn(mod: nn.Module, fqn: str) -> bool:
            return isinstance(mod, nn.Linear) and fqn in target_names

        quantize_(
            model,
            ApproxInt8DynamicActivationInt8WeightConfig(
                group_size=group_size,
                weight_only_decode=False,
            ),
            _filter_fn,
        )

    wrappers: list[nn.Module] = []
    patched_names: list[str] = []
    consumed_layers: set[int] = set()
    if (
        approx_impl == "fast_int8"
        and fuse_gate_up
        and patch_set == {"gate_proj", "up_proj"}
    ):
        for layer_idx in sorted({layer_idx for layer_idx, *_ in targets}):
            fused = TritonFusedGateUpMLP(model.model.layers[layer_idx].mlp, group_size)
            fused.to("cuda")
            model.model.layers[layer_idx].mlp = fused
            wrappers.append(fused)
            patched_names.append(f"{layer_idx}.mlp(gate_up_fused)")
            consumed_layers.add(layer_idx)

    for layer_idx, _fqn, parent, name, linear in targets:
        if layer_idx in consumed_layers:
            continue
        wrapper = TritonDecodeLinear(linear, group_size, quant_bits, approx_impl)
        wrapper.to("cuda")
        wrapper.finalize_approx(pack_b_i8_per_group_fp16_storage, quantize_b_i8_per_col)
        setattr(parent, name, wrapper)
        wrappers.append(wrapper)
        patched_names.append(f"{layer_idx}.{name}")
    TritonDecodeLinear.set_profile_enabled(profile_components)
    model = _maybe_compile_model(model, inductor_scope if use_inductor else "none", wrappers)

    pad_multiple = int(os.environ.get("QWEN_PAD_TO_MULTIPLE_OF", "8"))
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=pad_multiple,
    ).to("cuda")
    base_cache = _make_decode_cache(model, inputs["input_ids"], decode_steps) if use_static_cache else None

    for wrapper in wrappers:
        wrapper.mode = "exact"
    knobs.runtime.add_stages_inspection_hook = None
    TritonDecodeLinear.clear_quant_cache()
    TritonDecodeLinear.clear_profile_stats()
    teacher_tokens = _decode_trace(model, inputs["input_ids"], decode_steps, base_cache=base_cache)
    exact_free_tokens = list(teacher_tokens)
    TritonDecodeLinear.clear_quant_cache()
    TritonDecodeLinear.clear_profile_stats()
    exact = _run_decode(model, inputs["input_ids"], teacher_tokens, base_cache=base_cache)
    exact_profile = dict(TritonDecodeLinear._profile_stats)
    exact_profile_counts = dict(TritonDecodeLinear._profile_counts)

    hook_activation_dtype = wrappers[0].last_input_dtype if wrappers else torch.float16
    hook = _build_hook(
        plugin,
        activation_dtype=hook_activation_dtype,
        approx_impl=approx_impl,
        fused_gate_up=fuse_gate_up and patch_set == {"gate_proj", "up_proj"},
    ) if use_substitute else None
    for wrapper in wrappers:
        wrapper.mode = "approx"
    if hook is not None:
        knobs.runtime.add_stages_inspection_hook = hook
    TritonDecodeLinear.clear_quant_cache()
    TritonDecodeLinear.clear_profile_stats()
    approx_free_tokens = _decode_trace(model, inputs["input_ids"], decode_steps, base_cache=base_cache)
    TritonDecodeLinear.clear_quant_cache()
    TritonDecodeLinear.clear_profile_stats()
    approx = _run_decode(model, inputs["input_ids"], teacher_tokens, base_cache=base_cache)
    approx_profile = dict(TritonDecodeLinear._profile_stats)
    approx_profile_counts = dict(TritonDecodeLinear._profile_counts)
    actq_cache_hits = TritonDecodeLinear._cache_hits
    actq_cache_misses = TritonDecodeLinear._cache_misses
    knobs.runtime.add_stages_inspection_hook = None

    max_logit_err = 0.0
    mean_logit_err = 0.0
    teacher_argmax_match = 0
    exact_token_rank_sum = 0.0
    top5_overlap_sum = 0.0
    for exact_logits, approx_logits, exact_next, approx_next in zip(
        exact.logits, approx.logits, exact.next_tokens, approx.next_tokens
    ):
        abs_diff = (exact_logits - approx_logits).abs()
        max_logit_err = max(max_logit_err, float(abs_diff.max().item()))
        mean_logit_err += float(abs_diff.mean().item())
        teacher_argmax_match += int(exact_next == approx_next)
        exact_token_rank_sum += _token_rank(approx_logits, exact_next)
        top5_overlap_sum += _topk_overlap(exact_logits, approx_logits, 5)
    if exact.logits:
        mean_logit_err /= len(exact.logits)
        exact_token_rank_sum /= len(exact.logits)
        top5_overlap_sum /= len(exact.logits)
    free_run_match = sum(int(a == b) for a, b in zip(exact_free_tokens, approx_free_tokens))
    substitution_hits = sum(wrapper.substitution_hits for wrapper in wrappers)

    first = wrappers[0]
    if isinstance(first, TritonFusedGateUpMLP):
        exact_bytes = (
            first.gate.weight_t_exact.numel() * first.gate.weight_t_exact.element_size()
            + first.up.weight_t_exact.numel() * first.up.weight_t_exact.element_size()
        )
        packed_bytes = (
            first.gate.weight_t_i8.numel() * first.gate.weight_t_i8.element_size()
            + first.gate.weight_t_i8_scale.numel() * first.gate.weight_t_i8_scale.element_size()
            + first.up.weight_t_i8.numel() * first.up.weight_t_i8.element_size()
            + first.up.weight_t_i8_scale.numel() * first.up.weight_t_i8_scale.element_size()
        )
    else:
        exact_bytes = first.weight_t_exact.numel() * first.weight_t_exact.element_size()
        if approx_impl == "packed_dequant":
            packed_bytes = first.weight_t_packed.numel() * first.weight_t_packed.element_size()
        else:
            packed_bytes = (
                first.weight_t_i8.numel() * first.weight_t_i8.element_size()
                + first.weight_t_i8_scale.numel() * first.weight_t_i8_scale.element_size()
            )
    print(f"model:                     {model_id}")
    print(f"patched modules:           {len(wrappers)} ({','.join(sorted(patch_set))})")
    print(f"patched layer names:       {','.join(patched_names[:8])}{'...' if len(patched_names) > 8 else ''}")
    print(f"decode steps:              {decode_steps}")
    print(f"approx impl:               {approx_impl}")
    print(f"use inductor:              {use_inductor}")
    print(f"inductor scope:            {inductor_scope if use_inductor else 'none'}")
    print(f"use static cache:          {use_static_cache}")
    print(f"quant bits:                {quant_bits}")
    print(f"group_size_k:              {group_size}")
    print(f"substitution hits:         {substitution_hits}")
    print(f"actq cache hits/misses:    {actq_cache_hits}/{actq_cache_misses}")
    print(f"per-module exact bytes:    {exact_bytes}")
    print(f"per-module packed bytes:   {packed_bytes}")
    print(f"exact decode avg (ms):     {exact.avg_ms:.5f}")
    print(f"approx decode avg (ms):    {approx.avg_ms:.5f}")
    print(f"speedup vs exact:          {exact.avg_ms / approx.avg_ms:.3f}x")
    print(f"max logit abs diff:        {max_logit_err:.9e}")
    print(f"mean logit abs diff:       {mean_logit_err:.9e}")
    print(f"teacher argmax agreement:  {teacher_argmax_match}/{len(exact.next_tokens)}")
    print(f"avg exact-token rank:      {exact_token_rank_sum:.3f}")
    print(f"avg top5 overlap:          {top5_overlap_sum:.3f}")
    print(f"free-run token agreement:  {free_run_match}/{len(exact_free_tokens)}")
    if profile_components:
        for key in sorted(exact_profile):
            count = max(exact_profile_counts.get(key, 1), 1)
            print(f"profile exact {key}:       {exact_profile[key] / count:.5f} ms/call ({count} calls)")
        for key in sorted(approx_profile):
            count = max(approx_profile_counts.get(key, 1), 1)
            print(f"profile approx {key}:      {approx_profile[key] / count:.5f} ms/call ({count} calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
