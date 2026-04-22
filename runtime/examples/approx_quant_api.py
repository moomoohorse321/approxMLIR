from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class ApproxInt8WeightOnlyConfig:
    group_size: Optional[int] = None


@dataclass
class ApproxInt8DynamicActivationInt8WeightConfig:
    group_size: Optional[int] = None
    weight_only_decode: bool = False


@dataclass
class ApproxAffineQuantizedTensor:
    qdata: torch.Tensor
    scale: torch.Tensor
    group_size: int
    layout: str


@dataclass
class ApproxLinearActivationQuantizedTensor:
    original_weight_tensor: ApproxAffineQuantizedTensor
    input_quant_func: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    weight_only_decode: bool = False


def _quantize_weight_per_col(weight_t: torch.Tensor) -> ApproxAffineQuantizedTensor:
    from kernels.quant_kernels import quantize_b_i8_per_col

    qdata, scale = quantize_b_i8_per_col(weight_t)
    return ApproxAffineQuantizedTensor(
        qdata=qdata,
        scale=scale.float(),
        group_size=weight_t.shape[0],
        layout="per_col_int8",
    )


def _quantize_weight_per_group(
    weight_t: torch.Tensor, group_size: int
) -> ApproxAffineQuantizedTensor:
    from kernels.quant_kernels import quantize_b_i8_per_group

    qdata, scale = quantize_b_i8_per_group(weight_t, group_size)
    return ApproxAffineQuantizedTensor(
        qdata=qdata,
        scale=scale.float(),
        group_size=group_size,
        layout="per_group_int8",
    )


def to_affine_quantized_intx(
    weight_t: torch.Tensor,
    config: ApproxInt8WeightOnlyConfig | ApproxInt8DynamicActivationInt8WeightConfig,
) -> ApproxAffineQuantizedTensor:
    if weight_t.dtype not in (torch.float16, torch.bfloat16) or not weight_t.is_cuda:
        raise ValueError("expected CUDA fp16/bf16 weight tensor")
    group_size = config.group_size
    if group_size is None or group_size >= weight_t.shape[0]:
        return _quantize_weight_per_col(weight_t)
    return _quantize_weight_per_group(weight_t, group_size)


def to_linear_activation_quantized(
    weight: ApproxAffineQuantizedTensor,
    input_quant_func: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    *,
    weight_only_decode: bool = False,
) -> ApproxLinearActivationQuantizedTensor:
    return ApproxLinearActivationQuantizedTensor(
        original_weight_tensor=weight,
        input_quant_func=input_quant_func,
        weight_only_decode=weight_only_decode,
    )


def quantize_linear_weight(
    weight_t: torch.Tensor,
    config: ApproxInt8WeightOnlyConfig | ApproxInt8DynamicActivationInt8WeightConfig,
) -> ApproxAffineQuantizedTensor | ApproxLinearActivationQuantizedTensor:
    from kernels.quant_kernels import quantize_a_i8_per_row

    weight = to_affine_quantized_intx(weight_t, config)
    if isinstance(config, ApproxInt8DynamicActivationInt8WeightConfig):
        return to_linear_activation_quantized(
            weight,
            quantize_a_i8_per_row,
            weight_only_decode=config.weight_only_decode,
        )
    return weight


def _is_linear_module(module: nn.Module, _fqn: str) -> bool:
    return isinstance(module, nn.Linear)


def quantize_(
    module: nn.Module,
    config: ApproxInt8WeightOnlyConfig | ApproxInt8DynamicActivationInt8WeightConfig,
    filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
    *,
    cur_fqn: str = "",
) -> None:
    if filter_fn is None:
        filter_fn = _is_linear_module

    for name, child in list(module.named_children()):
        fqn = f"{cur_fqn}{name}"
        if filter_fn(child, fqn):
            if not isinstance(child, nn.Linear):
                raise TypeError("approx quantize_ currently supports nn.Linear only")
            weight_t = child.weight.detach().t().contiguous()
            child._approx_quantized_weight = quantize_linear_weight(weight_t, config)
            child._approx_quant_config = config
        quantize_(child, config, filter_fn, cur_fqn=f"{fqn}.")
