#!/usr/bin/env python3
"""Triton kernels for diffusion quantization experiments."""

from __future__ import annotations

import os

_plugin_path = os.environ.get("TRITON_PASS_PLUGIN_PATH")
if _plugin_path:
    _existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
    _paths = _existing.split(":") if _existing else []
    if _plugin_path not in _paths:
        _paths.insert(0, _plugin_path)
        os.environ["TRITON_PLUGIN_PATHS"] = ":".join(path for path in _paths if path)

import torch
import triton
import triton.language as tl


def quantize_weight_i8_per_col(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("expected a 2D floating-point weight")
    weight_f = weight.detach().float()
    scale = weight_f.abs().amax(dim=1).clamp_min(1.0e-8) / 127.0
    q_out_in = torch.clamp(torch.round(weight_f / scale[:, None]), -127, 127).to(torch.int8).contiguous()
    return q_out_in.t().contiguous(), scale.contiguous()


def _pack_i4_codes_along_k(q_codes_t: torch.Tensor) -> torch.Tensor:
    if q_codes_t.ndim != 2 or q_codes_t.dtype != torch.uint8:
        raise ValueError("expected a 2D uint8 tensor")
    k, n = q_codes_t.shape
    k_padded = triton.cdiv(k, 2) * 2
    if k_padded != k:
        padded = torch.full((k_padded, n), 7, device=q_codes_t.device, dtype=torch.uint8)
        padded[:k] = q_codes_t
        q_codes_t = padded
    lo = q_codes_t[0::2].to(torch.int32)
    hi = q_codes_t[1::2].to(torch.int32)
    return (lo | (hi << 4)).to(torch.uint8).contiguous()


def quantize_weight_i4_groupwise_packed(
    weight: torch.Tensor,
    *,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("expected a 2D floating-point weight")
    if group_size <= 0:
        raise ValueError("expected a positive group_size")

    weight_f = weight.detach().float()
    n, k = weight_f.shape
    num_groups = triton.cdiv(k, group_size)
    k_padded = num_groups * group_size
    if k_padded != k:
        padded = torch.zeros((n, k_padded), device=weight_f.device, dtype=weight_f.dtype)
        padded[:, :k] = weight_f
        weight_f = padded

    groups = weight_f.view(n, num_groups, group_size)
    scale = groups.abs().amax(dim=2).clamp_min(1.0e-8) / 7.0
    q = torch.clamp(torch.round(groups / scale[:, :, None]), -7, 7).to(torch.int16)
    q_codes_t = (q + 7).to(torch.uint8).reshape(n, k_padded).t().contiguous()
    packed = _pack_i4_codes_along_k(q_codes_t)
    act_scale_inv = torch.ones(k, device=weight.device, dtype=weight.dtype)
    return packed, scale.t().to(dtype=weight.dtype).contiguous(), act_scale_inv.contiguous()


@triton.jit
def diffusion_w8a16_linear_kernel(
    a_ptr,
    b_q_ptr,
    b_scale_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        bq = tl.load(
            b_q_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        ).to(tl.float32)
        b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
        b = (bq * b_scale[None, :]).to(tl.float16)
        acc += tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def diffusion_w4a16_linear_kernel(
    a_ptr,
    b_q_ptr,
    b_scale_ptr,
    a_scale_inv_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        smooth = tl.load(a_scale_inv_ptr + k, mask=k < K, other=1.0)
        a = a * smooth[None, :]
        k_packed = k // 2
        b_packed = tl.load(
            b_q_ptr + k_packed[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int32)
        b_lo = b_packed & 0xF
        b_hi = (b_packed >> 4) & 0xF
        b_q = tl.where((offs_k[:, None] % 2) == 0, b_lo, b_hi).to(tl.float16) - 7.0
        group_id = k0 // GROUP_K
        b_scale = tl.load(
            b_scale_ptr + group_id * stride_bsg + offs_n * stride_bsn,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float16)
        b = b_q * b_scale[None, :]
        acc += tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def approx_diffusion_w4a16_linear_kernel_1(
    a_ptr,
    b_q_ptr,
    b_scale_ptr,
    a_scale_inv_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bsg: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        smooth = tl.load(a_scale_inv_ptr + k, mask=k < K, other=1.0)
        a = a * smooth[None, :]
        k_packed = k // 2
        b_packed = tl.load(
            b_q_ptr + k_packed[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int32)
        b_lo = b_packed & 0xF
        b_hi = (b_packed >> 4) & 0xF
        b_q = tl.where((offs_k[:, None] % 2) == 0, b_lo, b_hi).to(tl.float16) - 7.0
        group_id = k0 // GROUP_K
        b_scale = tl.load(
            b_scale_ptr + group_id * stride_bsg + offs_n * stride_bsn,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float16)
        b = b_q * b_scale[None, :]
        acc += tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def approx_diffusion_w8a16_linear_kernel_1(
    a_ptr,
    b_q_ptr,
    b_scale_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        bq = tl.load(
            b_q_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        ).to(tl.float32)
        b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
        b = (bq * b_scale[None, :]).to(tl.float16)
        acc += tl.dot(a, b, out_dtype=tl.float32)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )
