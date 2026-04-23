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


def smoothquant_quantize_weight_i4_groupwise_packed(
    weight: torch.Tensor,
    act_absmax: torch.Tensor,
    *,
    alpha: float = 0.85,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expected a 2D fp16/bf16 weight")
    if act_absmax.ndim != 1 or act_absmax.numel() != weight.shape[1]:
        raise ValueError("expected act_absmax to match the weight input dimension")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("expected alpha in [0, 1]")
    if group_size <= 0:
        raise ValueError("expected a positive group_size")

    weight_f = weight.detach().float()
    act_f = act_absmax.detach().float().to(weight_f.device).clamp_min(1.0e-5)
    weight_max = weight_f.abs().amax(dim=0).clamp_min(1.0e-5)
    smooth = (act_f.pow(alpha) / weight_max.pow(1.0 - alpha)).clamp(1.0e-4, 1.0e4)
    smooth_weight = weight_f * smooth[None, :]

    n, k = smooth_weight.shape
    num_groups = triton.cdiv(k, group_size)
    k_padded = num_groups * group_size
    if k_padded != k:
        padded = torch.zeros((n, k_padded), device=smooth_weight.device, dtype=smooth_weight.dtype)
        padded[:, :k] = smooth_weight
        smooth_weight = padded

    groups = smooth_weight.view(n, num_groups, group_size)
    scale = groups.abs().amax(dim=2).clamp_min(1.0e-8) / 7.0
    q = torch.clamp(torch.round(groups / scale[:, :, None]), -7, 7).to(torch.int16)
    q_codes_t = (q + 7).to(torch.uint8).reshape(n, k_padded).t().contiguous()
    packed = _pack_i4_codes_along_k(q_codes_t)
    return packed, scale.t().contiguous(), smooth.reciprocal().contiguous()


def awq_quantize_weight_i4_groupwise_packed(
    weight: torch.Tensor,
    act_absmax: torch.Tensor,
    *,
    group_size: int = 64,
    grid_size: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expected a 2D fp16/bf16 weight")
    if act_absmax.ndim != 1 or act_absmax.numel() != weight.shape[1]:
        raise ValueError("expected act_absmax to match the weight input dimension")
    if group_size <= 0:
        raise ValueError("expected a positive group_size")
    if grid_size <= 0:
        raise ValueError("expected a positive grid_size")

    weight_f = weight.detach().float()
    weight_t = weight_f.t().contiguous()
    act_f = act_absmax.detach().float().to(weight_f.device).clamp_min(1.0e-5)
    act_norm = act_f / act_f.mean().clamp_min(1.0e-5)
    weight_mean = weight_t.abs().mean(dim=1).clamp_min(1.0e-5)

    def _quantize_scaled_weight(
        scaled_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, k = scaled_weight.shape
        num_groups = triton.cdiv(k, group_size)
        k_padded = num_groups * group_size
        if k_padded != k:
            padded = torch.zeros((n, k_padded), device=scaled_weight.device, dtype=scaled_weight.dtype)
            padded[:, :k] = scaled_weight
            scaled_weight = padded
        groups = scaled_weight.view(n, num_groups, group_size)
        scale = groups.abs().amax(dim=2).clamp_min(1.0e-8) / 7.0
        q = torch.clamp(torch.round(groups / scale[:, :, None]), -7, 7).to(torch.int16)
        q_t = q.reshape(n, k_padded).t().contiguous()
        packed = _pack_i4_codes_along_k((q_t + 7).to(torch.uint8))
        return packed, scale.t().contiguous(), q_t[:k]

    best = None
    for step in range(grid_size + 1):
        ratio = float(step) / float(grid_size)
        scales = (act_norm.pow(ratio) / weight_mean.pow(1.0 - ratio)).clamp(1.0e-4, 1.0e4)
        scales = scales / (scales.max() * scales.min()).sqrt().clamp_min(1.0e-5)
        scaled_weight = weight_f * scales[None, :]
        packed, scale_g, q_t = _quantize_scaled_weight(scaled_weight)
        expanded_scale = scale_g.t().repeat_interleave(group_size, dim=1)[:, : weight.shape[1]]
        dequant_weight = (q_t.t().contiguous().float() * expanded_scale) / scales[None, :]
        loss = (((dequant_weight - weight_f) * act_norm[None, :]) ** 2).mean()
        candidate = (
            float(loss.item()),
            ratio,
            packed,
            scale_g,
            scales.reciprocal().contiguous(),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
        del scaled_weight, packed, scale_g, q_t, expanded_scale, dequant_weight, loss
        if weight.is_cuda and (((step + 1) % 4) == 0 or step == grid_size):
            torch.cuda.empty_cache()

    assert best is not None
    _, best_ratio, packed, scale_g, act_scale_inv = best
    scale_g = scale_g.to(weight.dtype).contiguous()
    act_scale_inv = act_scale_inv.to(weight.dtype).contiguous()
    del weight_f, weight_t, act_f, act_norm, weight_mean
    if weight.is_cuda:
        torch.cuda.empty_cache()
    return packed, scale_g, act_scale_inv, best_ratio


def repack_i4_packed_for_decode_tile(
    q_packed_t: torch.Tensor,
    scale_g: torch.Tensor,
    *,
    group_size: int,
    block_k: int,
    block_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q_packed_t.ndim != 2 or q_packed_t.dtype != torch.uint8:
        raise ValueError("expected packed int4 weights as a 2D uint8 tensor")
    if scale_g.ndim != 2:
        raise ValueError("expected group scales as a 2D tensor")
    if group_size <= 0 or block_k <= 0 or block_n <= 0:
        raise ValueError("expected positive group/block sizes")
    if block_k % 2 != 0:
        raise ValueError("expected an even block_k for packed int4 weights")
    if block_k > group_size or (group_size % block_k) != 0:
        raise ValueError("expected block_k to divide group_size")

    k_padded = q_packed_t.shape[0] * 2
    if (k_padded % block_k) != 0:
        raise ValueError("expected packed K dimension to be divisible by block_k")

    num_groups = triton.cdiv(k_padded, group_size)
    if scale_g.shape[0] != num_groups:
        raise ValueError("expected scale_g leading dimension to match the number of K groups")

    n = q_packed_t.shape[1]
    num_n_tiles = triton.cdiv(n, block_n)
    n_padded = num_n_tiles * block_n

    q_padded = q_packed_t
    if n_padded != n:
        padded = torch.full((q_packed_t.shape[0], n_padded), 0x77, device=q_packed_t.device, dtype=torch.uint8)
        padded[:, :n] = q_packed_t
        q_padded = padded

    num_k_blocks = k_padded // block_k
    q_tiled = (
        q_padded.view(num_k_blocks, block_k // 2, n_padded)
        .view(num_k_blocks, block_k // 2, num_n_tiles, block_n)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(num_k_blocks * num_n_tiles * (block_k // 2), block_n)
        .contiguous()
    )

    scale_padded = scale_g
    if n_padded != n:
        padded = torch.zeros((scale_g.shape[0], n_padded), device=scale_g.device, dtype=scale_g.dtype)
        padded[:, :n] = scale_g
        scale_padded = padded

    scale_tiled = (
        scale_padded.view(scale_g.shape[0], num_n_tiles, block_n)
        .contiguous()
        .view(scale_g.shape[0] * num_n_tiles, block_n)
        .contiguous()
    )
    return q_tiled, scale_tiled


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
def sglang_sq_w4a16_linear_kernel(
    a_ptr,
    b_q_ptr,
    b_scale_ptr,
    a_smooth_inv_ptr,
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    DECODE_TILE_LAYOUT: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_k_pair = tl.arange(0, BLOCK_K // 2)
    offs_n = pid_n * BLOCK_N + offs_bn
    acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    if DECODE_TILE_LAYOUT:
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        for k0 in range(0, K, BLOCK_K):
            k_lo = k0 + 2 * offs_k_pair
            k_hi = k_lo + 1
            a_lo = tl.load(
                a_ptr + row * stride_am + k_lo * stride_ak,
                mask=k_lo < K,
                other=0.0,
            ).to(tl.float16)
            a_hi = tl.load(
                a_ptr + row * stride_am + k_hi * stride_ak,
                mask=k_hi < K,
                other=0.0,
            ).to(tl.float16)
            smooth_lo = tl.load(
                a_smooth_inv_ptr + k_lo,
                mask=k_lo < K,
                other=1.0,
            ).to(tl.float16)
            smooth_hi = tl.load(
                a_smooth_inv_ptr + k_hi,
                mask=k_hi < K,
                other=1.0,
            ).to(tl.float16)
            a_lo = a_lo * smooth_lo
            a_hi = a_hi * smooth_hi
            k_block = k0 // BLOCK_K
            packed_row = (k_block * num_n_tiles + pid_n) * (BLOCK_K // 2) + offs_k_pair
            b_packed = tl.load(
                b_q_ptr + packed_row[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                mask=offs_n[None, :] < N,
                other=0,
            ).to(tl.int32)
            b_lo = (b_packed & 0xF).to(tl.float16) - 7.0
            b_hi = ((b_packed >> 4) & 0xF).to(tl.float16) - 7.0
            group_id = k0 // GROUP_K
            scale_row = group_id * num_n_tiles + pid_n
            b_scale = tl.load(
                b_scale_ptr + scale_row * stride_bsg + offs_bn * stride_bsn,
                mask=offs_n < N,
                other=0.0,
            ).to(tl.float16)
            b_lo = b_lo * b_scale[None, :]
            b_hi = b_hi * b_scale[None, :]
            acc = tl.dot(a_lo[None, :], b_lo, acc=acc, out_dtype=tl.float32)
            acc = tl.dot(a_hi[None, :], b_hi, acc=acc, out_dtype=tl.float32)
    else:
        for k0 in range(0, K, BLOCK_K):
            k_lo = k0 + 2 * offs_k_pair
            k_hi = k_lo + 1
            a_lo = tl.load(
                a_ptr + row * stride_am + k_lo * stride_ak,
                mask=k_lo < K,
                other=0.0,
            ).to(tl.float16)
            a_hi = tl.load(
                a_ptr + row * stride_am + k_hi * stride_ak,
                mask=k_hi < K,
                other=0.0,
            ).to(tl.float16)
            smooth_lo = tl.load(
                a_smooth_inv_ptr + k_lo,
                mask=k_lo < K,
                other=1.0,
            ).to(tl.float16)
            smooth_hi = tl.load(
                a_smooth_inv_ptr + k_hi,
                mask=k_hi < K,
                other=1.0,
            ).to(tl.float16)
            a_lo = a_lo * smooth_lo
            a_hi = a_hi * smooth_hi
            k_packed = (k0 // 2) + offs_k_pair
            b_packed = tl.load(
                b_q_ptr + k_packed[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(k_lo[:, None] < K) & (offs_n[None, :] < N),
                other=0,
            ).to(tl.int32)
            b_lo = (b_packed & 0xF).to(tl.float16) - 7.0
            b_hi = ((b_packed >> 4) & 0xF).to(tl.float16) - 7.0
            group_id = k0 // GROUP_K
            b_scale = tl.load(
                b_scale_ptr + group_id * stride_bsg + offs_n * stride_bsn,
                mask=offs_n < N,
                other=0.0,
            ).to(tl.float16)
            b_lo = b_lo * b_scale[None, :]
            b_hi = b_hi * b_scale[None, :]
            acc = tl.dot(a_lo[None, :], b_lo, acc=acc, out_dtype=tl.float32)
            acc = tl.dot(a_hi[None, :], b_hi, acc=acc, out_dtype=tl.float32)

    tl.store(
        c_ptr + row * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=offs_n[None, :] < N,
    )


@triton.jit
def approx_sglang_sq_w4a16_linear_kernel_1(
    a_ptr,
    b_q_ptr,
    b_scale_ptr,
    a_smooth_inv_ptr,
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    DECODE_TILE_LAYOUT: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_bn = tl.arange(0, BLOCK_N)
    offs_k_pair = tl.arange(0, BLOCK_K // 2)
    offs_n = pid_n * BLOCK_N + offs_bn
    acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    if DECODE_TILE_LAYOUT:
        num_n_tiles = tl.cdiv(N, BLOCK_N)
        for k0 in range(0, K, BLOCK_K):
            k_lo = k0 + 2 * offs_k_pair
            k_hi = k_lo + 1
            a_lo = tl.load(
                a_ptr + row * stride_am + k_lo * stride_ak,
                mask=k_lo < K,
                other=0.0,
            ).to(tl.float16)
            a_hi = tl.load(
                a_ptr + row * stride_am + k_hi * stride_ak,
                mask=k_hi < K,
                other=0.0,
            ).to(tl.float16)
            smooth_lo = tl.load(
                a_smooth_inv_ptr + k_lo,
                mask=k_lo < K,
                other=1.0,
            ).to(tl.float16)
            smooth_hi = tl.load(
                a_smooth_inv_ptr + k_hi,
                mask=k_hi < K,
                other=1.0,
            ).to(tl.float16)
            a_lo = a_lo * smooth_lo
            a_hi = a_hi * smooth_hi
            k_block = k0 // BLOCK_K
            packed_row = (k_block * num_n_tiles + pid_n) * (BLOCK_K // 2) + offs_k_pair
            b_packed = tl.load(
                b_q_ptr + packed_row[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                mask=offs_n[None, :] < N,
                other=0,
            ).to(tl.int32)
            b_lo = (b_packed & 0xF).to(tl.float16) - 7.0
            b_hi = ((b_packed >> 4) & 0xF).to(tl.float16) - 7.0
            group_id = k0 // GROUP_K
            scale_row = group_id * num_n_tiles + pid_n
            b_scale = tl.load(
                b_scale_ptr + scale_row * stride_bsg + offs_bn * stride_bsn,
                mask=offs_n < N,
                other=0.0,
            ).to(tl.float16)
            b_lo = b_lo * b_scale[None, :]
            b_hi = b_hi * b_scale[None, :]
            acc = tl.dot(a_lo[None, :], b_lo, acc=acc, out_dtype=tl.float32)
            acc = tl.dot(a_hi[None, :], b_hi, acc=acc, out_dtype=tl.float32)
    else:
        for k0 in range(0, K, BLOCK_K):
            k_lo = k0 + 2 * offs_k_pair
            k_hi = k_lo + 1
            a_lo = tl.load(
                a_ptr + row * stride_am + k_lo * stride_ak,
                mask=k_lo < K,
                other=0.0,
            ).to(tl.float16)
            a_hi = tl.load(
                a_ptr + row * stride_am + k_hi * stride_ak,
                mask=k_hi < K,
                other=0.0,
            ).to(tl.float16)
            smooth_lo = tl.load(
                a_smooth_inv_ptr + k_lo,
                mask=k_lo < K,
                other=1.0,
            ).to(tl.float16)
            smooth_hi = tl.load(
                a_smooth_inv_ptr + k_hi,
                mask=k_hi < K,
                other=1.0,
            ).to(tl.float16)
            a_lo = a_lo * smooth_lo
            a_hi = a_hi * smooth_hi
            k_packed = (k0 // 2) + offs_k_pair
            b_packed = tl.load(
                b_q_ptr + k_packed[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(k_lo[:, None] < K) & (offs_n[None, :] < N),
                other=0,
            ).to(tl.int32)
            b_lo = (b_packed & 0xF).to(tl.float16) - 7.0
            b_hi = ((b_packed >> 4) & 0xF).to(tl.float16) - 7.0
            group_id = k0 // GROUP_K
            b_scale = tl.load(
                b_scale_ptr + group_id * stride_bsg + offs_n * stride_bsn,
                mask=offs_n < N,
                other=0.0,
            ).to(tl.float16)
            b_lo = b_lo * b_scale[None, :]
            b_hi = b_hi * b_scale[None, :]
            acc = tl.dot(a_lo[None, :], b_lo, acc=acc, out_dtype=tl.float32)
            acc = tl.dot(a_hi[None, :], b_hi, acc=acc, out_dtype=tl.float32)

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
