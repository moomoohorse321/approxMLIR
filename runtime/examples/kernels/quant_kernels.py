"""Approximate Triton kernels for Stage 2(a) func_substitute demo."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_a_i8_per_row_kernel(
    a_ptr, q_ptr, scale_ptr,
    M, K,
    stride_am, stride_ak,
    stride_qm, stride_qk,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    row_max = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        idx = k * BLOCK_K + offs_k
        mask = idx < K
        v = tl.load(a_ptr + row * stride_am + idx * stride_ak, mask=mask, other=0.0).to(tl.float32)
        row_max = tl.maximum(row_max, tl.abs(v))
    amax = tl.max(row_max, axis=0)
    scale = tl.maximum(amax / 127.0, 1.0e-8)
    tl.store(scale_ptr + row, scale)
    inv_scale = 1.0 / scale
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        idx = k * BLOCK_K + offs_k
        mask = idx < K
        v = tl.load(a_ptr + row * stride_am + idx * stride_ak, mask=mask, other=0.0).to(tl.float32)
        q = v * inv_scale
        q = tl.extra.cuda.libdevice.round(q)
        q = tl.maximum(tl.minimum(q, 127.0), -127.0).to(tl.int8)
        tl.store(q_ptr + row * stride_qm + idx * stride_qk, q, mask=mask)


def quantize_a_i8_per_row(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to int8 with one scale per row in a single kernel launch."""
    if a.dtype not in (torch.float16, torch.bfloat16) or a.ndim != 2 or not a.is_cuda:
        raise ValueError("expected a CUDA fp16/bf16 matrix")
    M, K = a.shape
    q = torch.empty((M, K), device=a.device, dtype=torch.int8)
    scale = torch.empty((M,), device=a.device, dtype=torch.float32)
    block_k = min(1024, triton.next_power_of_2(K))
    _quantize_a_i8_per_row_kernel[(M,)](
        a, q, scale,
        M, K,
        a.stride(0), a.stride(1),
        q.stride(0), q.stride(1),
        BLOCK_K=block_k,
    )
    return q, scale


def quantize_b_i8_per_col(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to int8 with one scale per output column."""
    if b.dtype not in (torch.float16, torch.bfloat16) or b.ndim != 2 or not b.is_cuda:
        raise ValueError("expected a CUDA fp16/bf16 matrix")
    scale = b.float().abs().amax(dim=0).clamp_min(1.0e-8) / 127.0
    q = torch.clamp(torch.round(b.float() / scale[None, :]), -127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def quantize_b_i8_per_group(
    b: torch.Tensor, group_size_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to int8 with one scale per K-group and output column."""
    if b.dtype not in (torch.float16, torch.bfloat16) or b.ndim != 2 or not b.is_cuda:
        raise ValueError("expected a CUDA fp16/bf16 matrix")
    if group_size_k <= 0:
        raise ValueError("group_size_k must be positive")
    k, n = b.shape
    num_groups = (k + group_size_k - 1) // group_size_k
    q = torch.empty((k, n), device=b.device, dtype=torch.int8)
    scales = torch.empty((num_groups, n), device=b.device, dtype=torch.float32)
    for g in range(num_groups):
        start = g * group_size_k
        end = min(start + group_size_k, k)
        block = b[start:end, :].float()
        scale = block.abs().amax(dim=0).clamp_min(1.0e-8) / 127.0
        q[start:end, :] = torch.clamp(torch.round(block / scale[None, :]), -127, 127).to(torch.int8)
        scales[g, :] = scale
    return q.contiguous(), scales.contiguous()


def pack_b_i8_per_group_fp16_storage(
    b: torch.Tensor, group_size_k: int, quant_bits: int = 8
) -> torch.Tensor:
    """Pack a fp16 weight matrix into fp16 storage plus fp16 group scales.

    Layout:
    - first packed words: signed low-bit values packed into uint16 lanes,
      then reinterpreted as fp16 storage
    - final ``ceil(K / group_size_k) * N`` fp16 values: one dequant scale per
      K-group and output column
    """
    if b.dtype not in (torch.float16, torch.bfloat16) or b.ndim != 2 or not b.is_cuda:
        raise ValueError("expected a CUDA fp16/bf16 matrix")
    if group_size_k <= 0:
        raise ValueError("group_size_k must be positive")
    if quant_bits not in (4, 8):
        raise ValueError("quant_bits must be 4 or 8")
    k, n = b.shape
    qmax = 127.0 if quant_bits == 8 else 7.0
    pack_factor = 2 if quant_bits == 8 else 4
    num_groups = (k + group_size_k - 1) // group_size_k
    q = torch.empty_like(b, dtype=torch.int8)
    scales = torch.empty((num_groups, n), device=b.device, dtype=torch.float16)
    for g in range(num_groups):
        start = g * group_size_k
        end = min(start + group_size_k, k)
        block = b[start:end, :].float()
        scale = block.abs().amax(dim=0).clamp_min(1.0e-8) / qmax
        q[start:end, :] = torch.clamp(
            torch.round(block / scale.unsqueeze(0)), -qmax, qmax
        ).to(torch.int8)
        scales[g, :] = scale
    q_i16 = q.to(torch.int16)
    remainder = k % pack_factor
    if remainder:
        q_i16 = torch.cat(
            [
                q_i16,
                torch.zeros((pack_factor - remainder, n), device=q.device, dtype=torch.int16),
            ],
            dim=0,
        )
        k += pack_factor - remainder
    if quant_bits == 8:
        low = (q_i16[0::2, :] & 0xFF).to(torch.int16)
        high = ((q_i16[1::2, :] & 0xFF) << 8).to(torch.int16)
        packed_u16 = (low | high).to(torch.uint16)
    else:
        q_u16 = (q_i16 & 0x0F).to(torch.int16)
        packed_u16 = (
            q_u16[0::4, :]
            | (q_u16[1::4, :] << 4)
            | (q_u16[2::4, :] << 8)
            | (q_u16[3::4, :] << 12)
        ).to(torch.uint16)
    packed_u16 = packed_u16.contiguous()
    packed_storage = packed_u16.view(torch.float16).reshape(-1)
    return torch.cat([packed_storage, scales.contiguous().reshape(-1)], dim=0)


@triton.jit
def int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


@triton.jit
def approx_int8_matmul_kernel_1(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


@triton.jit
def int8_dequant_matmul_kernel(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=mask_c)


@triton.jit
def approx_int8_dequant_matmul_kernel_1(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=mask_c)


@triton.jit
def int8_dual_dequant_matmul_kernel(
    a_ptr, b1_ptr, b2_ptr, a_scale_ptr, b1_scale_ptr, b2_scale_ptr, c1_ptr, c2_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b1_ptrs = b1_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b2_ptrs = b2_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b1 = tl.load(b1_ptrs, mask=mask_b, other=0)
        b2 = tl.load(b2_ptrs, mask=mask_b, other=0)
        acc1 = tl.dot(a, b1, acc=acc1, out_dtype=tl.int32)
        acc2 = tl.dot(a, b2, acc=acc2, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b1_ptrs += BLOCK_K * stride_bk
        b2_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    b1_scale = tl.load(b1_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    b2_scale = tl.load(b2_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out1 = acc1.to(tl.float32) * a_scale[:, None] * b1_scale[None, :]
    out2 = acc2.to(tl.float32) * a_scale[:, None] * b2_scale[None, :]
    c1_ptrs = c1_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c2_ptrs = c2_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c1_ptrs, out1.to(c1_ptr.dtype.element_ty), mask=mask_c)
    tl.store(c2_ptrs, out2.to(c2_ptr.dtype.element_ty), mask=mask_c)


@triton.jit
def approx_int8_dual_dequant_matmul_kernel_1(
    a_ptr, b1_ptr, b2_ptr, a_scale_ptr, b1_scale_ptr, b2_scale_ptr, c1_ptr, c2_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b1_ptrs = b1_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b2_ptrs = b2_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b1 = tl.load(b1_ptrs, mask=mask_b, other=0)
        b2 = tl.load(b2_ptrs, mask=mask_b, other=0)
        acc1 = tl.dot(a, b1, acc=acc1, out_dtype=tl.int32)
        acc2 = tl.dot(a, b2, acc=acc2, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b1_ptrs += BLOCK_K * stride_bk
        b2_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    a_scale = tl.load(a_scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    b1_scale = tl.load(b1_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    b2_scale = tl.load(b2_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out1 = acc1.to(tl.float32) * a_scale[:, None] * b1_scale[None, :]
    out2 = acc2.to(tl.float32) * a_scale[:, None] * b2_scale[None, :]
    c1_ptrs = c1_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c2_ptrs = c2_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c1_ptrs, out1.to(c1_ptr.dtype.element_ty), mask=mask_c)
    tl.store(c2_ptrs, out2.to(c2_ptr.dtype.element_ty), mask=mask_c)


@triton.jit
def approx_matmul_kernel_1(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # b_ptr points to host-prepacked storage: packed low-bit weights followed by fp16 scales.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    group_size_k = stride_bk
    quant_bits = stride_bn
    pack_factor = 2 if quant_bits == 8 else 4
    packed_k = (K + pack_factor - 1) // pack_factor
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < (K - k * BLOCK_K)
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)
        k_idx = k * BLOCK_K + offs_k
        packed_rows = k_idx[:, None] // pack_factor
        packed_ptrs = b_ptr + packed_rows * N + offs_n[None, :]
        valid_b = k_idx[:, None] < K
        packed_f16 = tl.load(packed_ptrs, mask=valid_b, other=0.0)
        packed_u16 = packed_f16.to(tl.uint16, bitcast=True)
        if quant_bits == 8:
            low = (packed_u16 & 0x00FF).to(tl.int16)
            high = ((packed_u16 >> 8) & 0x00FF).to(tl.int16)
            q_i16 = tl.where((k_idx[:, None] & 1) == 0, low, high)
            q_i16 = tl.where(q_i16 >= 128, q_i16 - 256, q_i16)
        else:
            nib0 = (packed_u16 & 0x000F).to(tl.int16)
            nib1 = ((packed_u16 >> 4) & 0x000F).to(tl.int16)
            nib2 = ((packed_u16 >> 8) & 0x000F).to(tl.int16)
            nib3 = ((packed_u16 >> 12) & 0x000F).to(tl.int16)
            sel = k_idx[:, None] & 3
            q_i16 = tl.where(
                sel == 0,
                nib0,
                tl.where(sel == 1, nib1, tl.where(sel == 2, nib2, nib3)),
            )
            q_i16 = tl.where(q_i16 >= 8, q_i16 - 16, q_i16)
        group_idx = k_idx[:, None] // group_size_k
        scale_ptrs = b_ptr + packed_k * N + group_idx * N + offs_n[None, :]
        scale = tl.load(scale_ptrs, mask=valid_b & (offs_n[None, :] < N), other=0.0)
        b_dq = q_i16.to(tl.float32) * scale.to(tl.float32)
        acc += tl.dot(a, b_dq.to(a.dtype))
        a_ptrs += BLOCK_K * stride_ak
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
