from __future__ import annotations

import argparse
from typing import NamedTuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on hosts without Triton.
    triton = None
    tl = None


DEFAULT_BLOCK_K = 64
DEFAULT_BLOCK_N = 128
DEFAULT_BLOCK_M = 16


class CompactKPrunePlan(NamedTuple):
    block_mask: torch.Tensor
    kept_k_blocks: torch.Tensor
    kept_counts: torch.Tensor


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _validate_prune_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    sparsity: float,
    block_k: int,
    block_n: int,
) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected 2D lhs and rhs tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("expected lhs.shape[1] to equal rhs.shape[0]")
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError("expected sparsity in [0, 1]")
    if block_k <= 0 or block_n <= 0:
        raise ValueError("expected positive block sizes")
    return int(a.shape[0]), int(b.shape[0]), int(b.shape[1])


def block_magnitude_prune_mask(
    b: torch.Tensor,
    sparsity: float,
    *,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
) -> torch.Tensor:
    """Return a [ceil(K/BLOCK_K), ceil(N/BLOCK_N)] bool keep mask for B."""
    if b.ndim != 2:
        raise ValueError("expected a 2D rhs tensor")
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError("expected sparsity in [0, 1]")
    if block_k <= 0 or block_n <= 0:
        raise ValueError("expected positive block sizes")

    k, n = int(b.shape[0]), int(b.shape[1])
    num_k_blocks = _cdiv(k, block_k)
    num_n_blocks = _cdiv(n, block_n)
    num_blocks = num_k_blocks * num_n_blocks
    keep = torch.ones((num_k_blocks, num_n_blocks), device=b.device, dtype=torch.bool)
    if sparsity == 0.0 or num_blocks == 0:
        return keep

    prune_blocks = int(round(num_blocks * float(sparsity)))
    if sparsity > 0.0:
        prune_blocks = max(1, prune_blocks)
    prune_blocks = min(num_blocks, prune_blocks)
    if prune_blocks == 0:
        return keep

    scores = torch.empty((num_k_blocks, num_n_blocks), device=b.device, dtype=torch.float32)
    b_abs = b.detach().float().abs()
    for kb in range(num_k_blocks):
        k0 = kb * block_k
        k1 = min(k0 + block_k, k)
        for nb in range(num_n_blocks):
            n0 = nb * block_n
            n1 = min(n0 + block_n, n)
            scores[kb, nb] = b_abs[k0:k1, n0:n1].sum()

    flat_order = torch.argsort(scores.reshape(-1), stable=True)
    keep_flat = keep.reshape(-1)
    keep_flat[flat_order[:prune_blocks]] = False
    return keep


def expand_block_mask(
    block_mask: torch.Tensor,
    k: int,
    n: int,
    *,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
) -> torch.Tensor:
    if block_mask.ndim != 2 or block_mask.dtype != torch.bool:
        raise ValueError("expected a 2D bool block mask")
    return block_mask.repeat_interleave(block_k, dim=0).repeat_interleave(block_n, dim=1)[:k, :n]


def torch_block_prune_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    sparsity: float,
    *,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    block_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch same-mask reference for block-magnitude-pruned matmul."""
    _validate_prune_inputs(a, b, sparsity, block_k, block_n)
    if block_mask is None:
        block_mask = block_magnitude_prune_mask(b, sparsity, block_k=block_k, block_n=block_n)
    dense_mask = expand_block_mask(block_mask, int(b.shape[0]), int(b.shape[1]), block_k=block_k, block_n=block_n)
    return a @ (b * dense_mask.to(dtype=b.dtype)), block_mask


def build_compact_k_prune_plan(
    b: torch.Tensor,
    sparsity: float,
    *,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    block_mask: torch.Tensor | None = None,
) -> CompactKPrunePlan:
    """Build per-N-block kept K-block lists for the compact-K Triton backend."""
    if b.ndim != 2:
        raise ValueError("expected a 2D rhs tensor")
    if block_mask is None:
        block_mask = block_magnitude_prune_mask(b, sparsity, block_k=block_k, block_n=block_n)
    if block_mask.ndim != 2 or block_mask.dtype != torch.bool:
        raise ValueError("expected a 2D bool block mask")

    num_k_blocks, num_n_blocks = int(block_mask.shape[0]), int(block_mask.shape[1])
    kept_counts = block_mask.sum(dim=0).to(torch.int32).contiguous()
    max_keep = max(1, int(kept_counts.max().item()) if kept_counts.numel() else 1)
    kept_k_blocks = torch.zeros((num_n_blocks, max_keep), device=b.device, dtype=torch.int32)
    for nb in range(num_n_blocks):
        ids = torch.nonzero(block_mask[:, nb], as_tuple=False).flatten().to(torch.int32)
        if ids.numel():
            kept_k_blocks[nb, : ids.numel()] = ids
    return CompactKPrunePlan(block_mask.contiguous(), kept_k_blocks.contiguous(), kept_counts)


if triton is not None:

    @triton.jit
    def sglang_block_prune_linear_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        mask_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        NUM_N_BLOCKS: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            k_block = k0 // BLOCK_K
            block_live = tl.load(mask_ptr + k_block * NUM_N_BLOCKS + pid_n)
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            b = tl.where(block_live, b, 0.0)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def approx_sglang_block_prune_linear_kernel_1(
        a_ptr,
        b_ptr,
        c_ptr,
        mask_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        NUM_N_BLOCKS: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            k_block = k0 // BLOCK_K
            block_live = tl.load(mask_ptr + k_block * NUM_N_BLOCKS + pid_n)
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                mask=block_live & (offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=block_live & (k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def sglang_compact_k_prune_linear_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        kept_k_ptr,
        counts_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        MAX_KEEP: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        stride_kept_nb: tl.constexpr,
        stride_kept_i: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        keep_count = tl.load(counts_ptr + pid_n)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for keep_i in range(0, MAX_KEEP):
            valid_block = keep_i < keep_count
            k_block = tl.load(
                kept_k_ptr + pid_n * stride_kept_nb + keep_i * stride_kept_i,
                mask=valid_block,
                other=0,
            )
            k = k_block * BLOCK_K + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                mask=valid_block & (offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=valid_block & (k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

    @triton.jit
    def approx_sglang_compact_k_prune_linear_kernel_1(
        a_ptr,
        b_ptr,
        c_ptr,
        kept_k_ptr,
        counts_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        MAX_KEEP: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        stride_kept_nb: tl.constexpr,
        stride_kept_i: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        keep_count = tl.load(counts_ptr + pid_n)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for keep_i in range(0, MAX_KEEP):
            valid_block = keep_i < keep_count
            k_block = tl.load(
                kept_k_ptr + pid_n * stride_kept_nb + keep_i * stride_kept_i,
                mask=valid_block,
                other=0,
            )
            k = k_block * BLOCK_K + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
                mask=valid_block & (offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=valid_block & (k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def _require_triton_cuda(a: torch.Tensor, b: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("Triton is not importable")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Triton pruning backends require CUDA tensors")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise ValueError("expected matching fp16/bf16 CUDA tensors")


def triton_block_prune(
    a: torch.Tensor,
    b: torch.Tensor,
    sparsity: float,
    *,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    block_m: int = DEFAULT_BLOCK_M,
    block_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-magnitude-pruned A @ B using a dense K loop and a block keep mask."""
    m, k, n = _validate_prune_inputs(a, b, sparsity, block_k, block_n)
    _require_triton_cuda(a, b)
    if block_mask is None:
        block_mask = block_magnitude_prune_mask(b, sparsity, block_k=block_k, block_n=block_n)
    block_mask = block_mask.to(device=b.device, dtype=torch.bool).contiguous()

    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (_cdiv(m, block_m), _cdiv(n, block_n))
    sglang_block_prune_linear_kernel[grid](
        a,
        b,
        c,
        block_mask,
        m,
        n,
        k,
        int(block_mask.shape[1]),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return c, block_mask


def triton_compact_k_prune(
    a: torch.Tensor,
    b: torch.Tensor,
    sparsity: float,
    *,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    block_m: int = DEFAULT_BLOCK_M,
    plan: CompactKPrunePlan | None = None,
    block_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, CompactKPrunePlan]:
    """Block-magnitude-pruned A @ B using compact per-output-tile K block lists."""
    m, k, n = _validate_prune_inputs(a, b, sparsity, block_k, block_n)
    _require_triton_cuda(a, b)
    if plan is None:
        plan = build_compact_k_prune_plan(b, sparsity, block_k=block_k, block_n=block_n, block_mask=block_mask)
    kept_k_blocks = plan.kept_k_blocks.to(device=b.device, dtype=torch.int32).contiguous()
    kept_counts = plan.kept_counts.to(device=b.device, dtype=torch.int32).contiguous()

    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (_cdiv(m, block_m), _cdiv(n, block_n))
    sglang_compact_k_prune_linear_kernel[grid](
        a,
        b,
        c,
        kept_k_blocks,
        kept_counts,
        m,
        n,
        k,
        int(kept_k_blocks.shape[1]),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        kept_k_blocks.stride(0),
        kept_k_blocks.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return c, CompactKPrunePlan(plan.block_mask, kept_k_blocks, kept_counts)


BACKENDS = {
    "triton_block_prune": triton_block_prune,
    "triton_compact_k_prune": triton_compact_k_prune,
}

__all__ = [
    "BACKENDS",
    "DEFAULT_BLOCK_K",
    "DEFAULT_BLOCK_M",
    "DEFAULT_BLOCK_N",
    "CompactKPrunePlan",
    "block_magnitude_prune_mask",
    "build_compact_k_prune_plan",
    "expand_block_mask",
    "approx_sglang_block_prune_linear_kernel_1",
    "approx_sglang_compact_k_prune_linear_kernel_1",
    "sglang_block_prune_linear_kernel",
    "sglang_compact_k_prune_linear_kernel",
    "torch_block_prune_reference",
    "triton_block_prune",
    "triton_compact_k_prune",
]


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-1, msg=f"{name} mismatch")


def _run_cpu_reference_microtests() -> None:
    for k, n in ((64, 128), (65, 129), (130, 257)):
        a = torch.randn((7, k), dtype=torch.float32)
        b = torch.randn((k, n), dtype=torch.float32)
        dense_ref, dense_mask = torch_block_prune_reference(a, b, 0.0)
        torch.testing.assert_close(dense_ref, a @ b)
        if not bool(dense_mask.all().item()):
            raise AssertionError("sparsity 0 should keep every block")

        mask = block_magnitude_prune_mask(b, 0.25)
        pruned_ref, ref_mask = torch_block_prune_reference(a, b, 0.25, block_mask=mask)
        torch.testing.assert_close(pruned_ref, a @ (b * expand_block_mask(mask, k, n).to(dtype=b.dtype)))
        if not torch.equal(ref_mask, mask):
            raise AssertionError("reference returned a different mask")

        dense = expand_block_mask(mask, k, n)
        if dense.shape != b.shape:
            raise AssertionError(f"expanded mask shape mismatch: {dense.shape} != {b.shape}")
        plan = build_compact_k_prune_plan(b, 0.25, block_mask=mask)
        if plan.block_mask.shape != mask.shape:
            raise AssertionError("compact plan mask shape mismatch")


def _run_cuda_microtests() -> None:
    if triton is None or not torch.cuda.is_available():
        print("CUDA/Triton unavailable; ran CPU reference microtests only")
        return

    torch.manual_seed(0)
    device = torch.device("cuda")
    cases = ((16, 64, 128), (17, 65, 129), (33, 130, 257))
    for m, k, n in cases:
        a = torch.randn((m, k), device=device, dtype=torch.float16)
        b = torch.randn((k, n), device=device, dtype=torch.float16)
        for sparsity in (0.0, 0.1, 0.25, 0.5):
            ref = a @ b if sparsity == 0.0 else None
            pruned_ref, mask = torch_block_prune_reference(a, b, sparsity)
            expected = ref if ref is not None else pruned_ref

            out_block, out_mask = triton_block_prune(a, b, sparsity, block_mask=mask)
            _assert_close(f"triton_block_prune shape={(m, k, n)} sparsity={sparsity}", out_block, expected)
            if not torch.equal(out_mask, mask):
                raise AssertionError("triton_block_prune returned a different mask")

            plan = build_compact_k_prune_plan(b, sparsity, block_mask=mask)
            out_compact, out_plan = triton_compact_k_prune(a, b, sparsity, plan=plan)
            _assert_close(f"triton_compact_k_prune shape={(m, k, n)} sparsity={sparsity}", out_compact, expected)
            if not torch.equal(out_plan.block_mask, mask):
                raise AssertionError("triton_compact_k_prune returned a different mask")


def main() -> None:
    parser = argparse.ArgumentParser(description="Microtests for SGLang block-pruning Triton kernels")
    parser.add_argument("--cpu-only", action="store_true", help="skip CUDA/Triton kernel microtests")
    args = parser.parse_args()
    _run_cpu_reference_microtests()
    if not args.cpu_only:
        _run_cuda_microtests()
    print("sglang_pruning approx_kernels microtests passed")


if __name__ == "__main__":
    main()
