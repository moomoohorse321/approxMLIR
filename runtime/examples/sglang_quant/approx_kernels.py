from __future__ import annotations

import torch
import triton
import triton.language as tl


def quantize_weight_i8_per_col(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expected a 2D fp16/bf16 weight")
    weight_f = weight.detach().float()
    scale = weight_f.abs().amax(dim=1).clamp_min(1.0e-8) / 127.0
    q_out_in = torch.clamp(torch.round(weight_f / scale[:, None]), -127, 127).to(torch.int8).contiguous()
    return q_out_in.t(), scale.contiguous()


@triton.jit
def quantize_activation_i8_per_row_kernel(
    a_ptr,
    q_ptr,
    scale_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    amax_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        mask = k < K
        a = tl.load(a_ptr + row * stride_am + k * stride_ak, mask=mask, other=0.0).to(tl.float32)
        amax_vec = tl.maximum(amax_vec, tl.abs(a))
    amax = tl.max(amax_vec, axis=0)
    scale = tl.maximum(amax / 127.0, 1.0e-8)
    inv_scale = 1.0 / scale
    tl.store(scale_ptr + row, scale)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        mask = k < K
        a = tl.load(a_ptr + row * stride_am + k * stride_ak, mask=mask, other=0.0).to(tl.float32)
        q = tl.extra.cuda.libdevice.round(a * inv_scale)
        q = tl.maximum(tl.minimum(q, 127.0), -127.0).to(tl.int8)
        tl.store(q_ptr + row * stride_qm + k * stride_qk, q, mask=mask)


def quantize_activation_i8_per_row(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.ndim != 2 or not a.is_cuda or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expected a 2D CUDA fp16/bf16 activation")
    m, k = a.shape
    q = torch.empty((m, k), device=a.device, dtype=torch.int8)
    scale = torch.empty((m,), device=a.device, dtype=torch.float32)
    block_k = min(1024, triton.next_power_of_2(k))
    quantize_activation_i8_per_row_kernel[(m,)](
        a,
        q,
        scale,
        m,
        k,
        a.stride(0),
        a.stride(1),
        q.stride(0),
        q.stride(1),
        BLOCK_K=block_k,
    )
    return q, scale


@triton.jit
def sglang_prequant_w8a8_linear_kernel(
    a_q_ptr,
    a_scale_ptr,
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((1, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        aq = tl.load(
            a_q_ptr + row * stride_am + k * stride_ak,
            mask=k < K,
            other=0,
        )
        bq = tl.load(
            b_q_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc = tl.dot(aq[None, :], bq, acc=acc, out_dtype=tl.int32)

    a_scale = tl.load(a_scale_ptr + row)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * a_scale * b_scale[None, :]
    tl.store(
        c_ptr + row * stride_cm + offs_n[None, :] * stride_cn,
        out,
        mask=offs_n[None, :] < N,
    )


@triton.jit
def approx_sglang_prequant_w8a8_linear_kernel_1(
    a_q_ptr,
    a_scale_ptr,
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((1, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        aq = tl.load(
            a_q_ptr + row * stride_am + k * stride_ak,
            mask=k < K,
            other=0,
        )
        bq = tl.load(
            b_q_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc = tl.dot(aq[None, :], bq, acc=acc, out_dtype=tl.int32)

    a_scale = tl.load(a_scale_ptr + row)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * a_scale * b_scale[None, :]
    tl.store(
        c_ptr + row * stride_cm + offs_n[None, :] * stride_cn,
        out,
        mask=offs_n[None, :] < N,
    )


@triton.jit
def sglang_w8a16_linear_kernel(
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + row * stride_am + k * stride_ak,
            mask=k < K,
            other=0.0,
        ).to(tl.float32)
        bq = tl.load(
            b_q_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        ).to(tl.float32)
        b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
        b = bq * b_scale[None, :]
        acc += tl.sum(a[:, None] * b, axis=0)[None, :]

    tl.store(
        c_ptr + row * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=offs_n[None, :] < N,
    )


@triton.jit
def approx_sglang_w8a16_linear_kernel_1(
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + row * stride_am + k * stride_ak,
            mask=k < K,
            other=0.0,
        ).to(tl.float32)
        bq = tl.load(
            b_q_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        ).to(tl.float32)
        b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
        b = bq * b_scale[None, :]
        acc += tl.sum(a[:, None] * b, axis=0)[None, :]

    tl.store(
        c_ptr + row * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=offs_n[None, :] < N,
    )


@triton.jit
def sglang_dynamic_w8a8_linear_kernel(
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    amax = tl.full((), 0.0, tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        mask = k < K
        a = tl.load(a_ptr + row * stride_am + k * stride_ak, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(a), axis=0))

    a_scale = tl.maximum(amax / 127.0, 1.0e-8)
    inv_a_scale = 1.0 / a_scale
    acc = tl.zeros((1, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(a_ptr + row * stride_am + k * stride_ak, mask=k < K, other=0.0).to(tl.float32)
        aq = tl.extra.cuda.libdevice.round(a * inv_a_scale)
        aq = tl.maximum(tl.minimum(aq, 127.0), -127.0).to(tl.int8)
        bq = tl.load(
            b_q_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc = tl.dot(aq[None, :], bq, acc=acc, out_dtype=tl.int32)

    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * a_scale * b_scale[None, :]
    tl.store(
        c_ptr + row * stride_cm + offs_n[None, :] * stride_cn,
        out,
        mask=offs_n[None, :] < N,
    )


@triton.jit
def approx_sglang_dynamic_w8a8_linear_kernel_1(
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    amax = tl.full((), 0.0, tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        mask = k < K
        a = tl.load(a_ptr + row * stride_am + k * stride_ak, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(a), axis=0))

    a_scale = tl.maximum(amax / 127.0, 1.0e-8)
    inv_a_scale = 1.0 / a_scale
    acc = tl.zeros((1, BLOCK_N), dtype=tl.int32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(a_ptr + row * stride_am + k * stride_ak, mask=k < K, other=0.0).to(tl.float32)
        aq = tl.extra.cuda.libdevice.round(a * inv_a_scale)
        aq = tl.maximum(tl.minimum(aq, 127.0), -127.0).to(tl.int8)
        bq = tl.load(
            b_q_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )
        acc = tl.dot(aq[None, :], bq, acc=acc, out_dtype=tl.int32)

    b_scale = tl.load(b_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    out = acc.to(tl.float32) * a_scale * b_scale[None, :]
    tl.store(
        c_ptr + row * stride_cm + offs_n[None, :] * stride_cn,
        out,
        mask=offs_n[None, :] < N,
    )
