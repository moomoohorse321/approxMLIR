"""LLM-shaped Triton kernels for Stage 1 observation.

Sample-only. Not a runtime API. Shapes are taken from Qwen2.5-0.5B
(hidden=896, intermediate=4864, num_heads=14, num_kv_heads=2, head_dim=64)
so the dumped TTIR structurally resembles real serving kernels rather than
being cherry-picked to flatter the later demo.

Each kernel entry is ``(name, launcher, shape_iter)`` where ``shape_iter``
yields dicts describing concrete launch shapes. The launcher takes that dict
and actually fires the kernel on CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def matmul_kernel(
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_K))
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        mask_b = (offs_k[:, None] < (K - k * BLOCK_K)) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b.to(a.dtype))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def exact_matmul_kernel(
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < (K - k * BLOCK_K))
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        mask_b = (offs_k[:, None] < (K - k * BLOCK_K)) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b.to(a.dtype))
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    stride_xm, stride_ym,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + row * stride_ym + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def silu_gate_kernel(
    gate_ptr, up_ptr, out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    silu = g * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(out_ptr + offs, (silu * u).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,
    M, N,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Single-batch single-head FlashAttention-style forward. Iterates over KV
    # blocks, accumulates with online softmax. Stage 1 observation only —
    # no causal mask, no GQA broadcast, no paged KV.
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < M

    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], -1.0e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        k_ptrs = K_ptr + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
        v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(mask_n[None, :], qk, -1.0e9)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


def _launch_matmul(shape: dict, recorder=None) -> None:
    M, N, K = shape["M"], shape["N"], shape["K"]
    dtype = shape.get("dtype", torch.float16)
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    c = torch.empty((M, N), device="cuda", dtype=dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = shape.get("BLOCK_M", 64), shape.get("BLOCK_N", 64), shape.get("BLOCK_K", 32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    if recorder is not None:
        recorder.record(
            "matmul_kernel",
            shape.get("tag"),
            {"M": M, "N": N, "K": K, "dtype": str(dtype),
             "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
        )


def _launch_rmsnorm(shape: dict, recorder=None) -> None:
    M, N = shape["M"], shape["N"]
    dtype = shape.get("dtype", torch.float16)
    x = torch.randn((M, N), device="cuda", dtype=dtype)
    w = torch.randn((N,), device="cuda", dtype=dtype)
    y = torch.empty_like(x)
    BLOCK_N = shape.get("BLOCK_N", triton.next_power_of_2(N))
    grid = (M,)
    rmsnorm_kernel[grid](
        x, w, y,
        x.stride(0), y.stride(0),
        N, 1e-6,
        BLOCK_N=BLOCK_N,
    )
    if recorder is not None:
        recorder.record(
            "rmsnorm_kernel",
            shape.get("tag"),
            {"M": M, "N": N, "dtype": str(dtype), "BLOCK_N": BLOCK_N},
        )


def _launch_silu_gate(shape: dict, recorder=None) -> None:
    M, D = shape["M"], shape["D"]
    dtype = shape.get("dtype", torch.float16)
    gate = torch.randn((M, D), device="cuda", dtype=dtype)
    up = torch.randn((M, D), device="cuda", dtype=dtype)
    out = torch.empty_like(gate)
    n = M * D
    BLOCK = shape.get("BLOCK", 1024)
    grid = (triton.cdiv(n, BLOCK),)
    silu_gate_kernel[grid](gate, up, out, n, BLOCK=BLOCK)
    if recorder is not None:
        recorder.record(
            "silu_gate_kernel",
            shape.get("tag"),
            {"M": M, "D": D, "dtype": str(dtype), "BLOCK": BLOCK},
        )


def _launch_attention(shape: dict, recorder=None) -> None:
    M, N = shape["M"], shape["N"]
    head_dim = shape.get("HEAD_DIM", 64)
    dtype = shape.get("dtype", torch.float16)
    q = torch.randn((M, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((N, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((N, head_dim), device="cuda", dtype=dtype)
    o = torch.empty_like(q)
    BLOCK_M = shape.get("BLOCK_M", 16 if M < 16 else 64)
    BLOCK_N = shape.get("BLOCK_N", 64)
    sm_scale = head_dim ** -0.5
    grid = (triton.cdiv(M, BLOCK_M),)
    attn_fwd_kernel[grid](
        q, k, v, o, sm_scale,
        M, N,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
    )
    if recorder is not None:
        recorder.record(
            "attn_fwd_kernel",
            shape.get("tag"),
            {"M": M, "N": N, "HEAD_DIM": head_dim, "dtype": str(dtype),
             "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
        )


# ---------------------------------------------------------------------------
# Qwen2.5-0.5B-derived shape menu
# ---------------------------------------------------------------------------

QWEN_HIDDEN = 896
QWEN_INTER = 4864
QWEN_KV = 128  # num_kv_heads * head_dim = 2 * 64

# Two representative sequence lengths: prefill chunk and decode.
_PREFILL = 128
_DECODE = 1


def _matmul_shapes() -> Iterable[dict]:
    for seq in (_PREFILL, _DECODE):
        # attention projections
        yield {"tag": "attn_qo_proj",   "M": seq, "N": QWEN_HIDDEN, "K": QWEN_HIDDEN}
        yield {"tag": "attn_kv_proj",   "M": seq, "N": QWEN_KV,     "K": QWEN_HIDDEN}
        # MLP projections
        yield {"tag": "mlp_gate_up",    "M": seq, "N": QWEN_INTER,  "K": QWEN_HIDDEN}
        yield {"tag": "mlp_down",       "M": seq, "N": QWEN_HIDDEN, "K": QWEN_INTER}


def _rmsnorm_shapes() -> Iterable[dict]:
    for seq in (_PREFILL, _DECODE):
        yield {"tag": "rmsnorm_hidden", "M": seq, "N": QWEN_HIDDEN}


def _silu_gate_shapes() -> Iterable[dict]:
    for seq in (_PREFILL, _DECODE):
        yield {"tag": "mlp_silu_glu", "M": seq, "D": QWEN_INTER}


def _attention_shapes() -> Iterable[dict]:
    # Self-attention prefill: seq_q == seq_kv == 128
    yield {"tag": "attn_prefill",      "M": _PREFILL, "N": _PREFILL, "HEAD_DIM": 64}
    # Decode at short context: 1 query token, small KV history
    yield {"tag": "attn_decode_short", "M": _DECODE,  "N": 128,      "HEAD_DIM": 64}
    # Decode at long context: 1 query token, large KV history — KV bandwidth dominates
    yield {"tag": "attn_decode_long",  "M": _DECODE,  "N": 1024,     "HEAD_DIM": 64}


@dataclass
class KernelEntry:
    name: str
    launcher: Callable[[dict], None]
    shapes: Callable[[], Iterable[dict]]


MENU: list[KernelEntry] = [
    KernelEntry("matmul", _launch_matmul, _matmul_shapes),
    KernelEntry("rmsnorm", _launch_rmsnorm, _rmsnorm_shapes),
    KernelEntry("silu_gate", _launch_silu_gate, _silu_gate_shapes),
    KernelEntry("attention", _launch_attention, _attention_shapes),
]
