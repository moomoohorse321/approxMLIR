from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "sglang_quant"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from approx_apl_lut import (  # noqa: E402
    APL_LAYOUT_CUDA_PERMUTED,
    APL_LAYOUT_NATURAL,
    APL_QUANTIZER_ROW_KMEANS,
    APL_QUANTIZER_ROW_UNIFORM,
    apl_lut_convert_layout,
    apl_lut_dequantize_weight,
    apl_lut_quantize_weight,
)

try:
    from approx_kernels import sglang_apl_lut_linear_kernel  # noqa: E402
except ModuleNotFoundError:
    sglang_apl_lut_linear_kernel = None


def test_apl_lut_pack_shape_padding_and_msb_order() -> None:
    codes_weight = torch.arange(64, dtype=torch.float32).reshape(2, 32).to(torch.float16)
    qweight, lut, meta = apl_lut_quantize_weight(codes_weight, bits=3, pad_n_to=4, pad_k_to=64)

    assert qweight.shape == (3, 4, 2)
    assert qweight.dtype == torch.int32
    assert lut.shape == (4, 8)
    assert lut.dtype == torch.float16
    assert meta["K_orig"] == 32
    assert meta["K_padded"] == 64
    assert meta["N_orig"] == 2
    assert meta["N_padded"] == 4
    assert meta["layout_version"] == APL_LAYOUT_NATURAL
    assert meta["quantizer_version"] == APL_QUANTIZER_ROW_UNIFORM

    dequant = apl_lut_dequantize_weight(qweight, lut, 3, k_orig=32, n_orig=2)
    assert dequant.shape == codes_weight.shape
    assert torch.allclose(dequant, codes_weight, atol=5.0, rtol=0.0)

    first_code = int(torch.bucketize(codes_weight[0, 0].float(), ((lut[0, :-1] + lut[0, 1:]) * 0.5).float()))
    for plane in range(3):
        expected = (first_code >> (2 - plane)) & 1
        actual = (int(qweight[plane, 0, 0]) >> 0) & 1
        assert actual == expected


def test_apl_lut_dequant_matches_reference_lookup() -> None:
    weight = torch.randn(5, 70, dtype=torch.float16)
    qweight, lut, meta = apl_lut_quantize_weight(weight, bits=4, pad_n_to=8, pad_k_to=96)
    dequant = apl_lut_dequantize_weight(qweight, lut, 4, k_orig=meta["K_orig"], n_orig=meta["N_orig"])

    planes = qweight[:4].to(torch.int64)
    bit_positions = torch.arange(32, dtype=torch.int64)
    unpacked = ((planes.unsqueeze(-1) >> bit_positions) & 1).to(torch.long)
    codes = torch.zeros((planes.shape[1], planes.shape[2], 32), dtype=torch.long)
    for plane_idx in range(4):
        codes |= unpacked[plane_idx] << (3 - plane_idx)
    codes = codes.reshape(planes.shape[1], planes.shape[2] * 32)
    expected = torch.gather(lut, 1, codes)[:5, :70].contiguous()

    assert torch.equal(dequant, expected)


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_apl_lut_cuda_permuted_layout_dequant_matches_natural(bits: int) -> None:
    weight = torch.randn(6, 65, dtype=torch.float16)
    q_nat, lut_nat, meta_nat = apl_lut_quantize_weight(
        weight,
        bits=bits,
        pad_n_to=4,
        pad_k_to=96,
        layout_version=APL_LAYOUT_NATURAL,
    )
    q_cuda, lut_cuda = apl_lut_convert_layout(
        q_nat,
        lut_nat,
        src_layout=APL_LAYOUT_NATURAL,
        dst_layout=APL_LAYOUT_CUDA_PERMUTED,
    )
    q_round, lut_round = apl_lut_convert_layout(
        q_cuda,
        lut_cuda,
        src_layout=APL_LAYOUT_CUDA_PERMUTED,
        dst_layout=APL_LAYOUT_NATURAL,
    )

    assert torch.equal(q_round, q_nat)
    assert torch.equal(lut_round, lut_nat)
    assert q_cuda.shape == (bits, 8, 3)
    assert lut_cuda.shape == (8, 1 << bits)

    dequant_nat = apl_lut_dequantize_weight(
        q_nat,
        lut_nat,
        bits,
        k_orig=meta_nat["K_orig"],
        n_orig=meta_nat["N_orig"],
    )
    dequant_cuda = apl_lut_dequantize_weight(
        q_cuda,
        lut_cuda,
        bits,
        k_orig=meta_nat["K_orig"],
        n_orig=meta_nat["N_orig"],
        layout_version=APL_LAYOUT_CUDA_PERMUTED,
    )
    assert torch.equal(dequant_cuda, dequant_nat)


def test_apl_lut_quantizer_versions_are_explicit() -> None:
    weight = torch.randn(4, 48, dtype=torch.float32)
    q_uniform, lut_uniform, meta_uniform = apl_lut_quantize_weight(
        weight,
        bits=4,
        quantizer_version=APL_QUANTIZER_ROW_UNIFORM,
    )
    q_kmeans, lut_kmeans, meta_kmeans = apl_lut_quantize_weight(
        weight,
        bits=4,
        quantizer_version=APL_QUANTIZER_ROW_KMEANS,
    )

    assert q_uniform.shape == q_kmeans.shape == (4, 4, 2)
    assert lut_uniform.shape == lut_kmeans.shape == (4, 16)
    assert meta_uniform["quantizer_version"] == APL_QUANTIZER_ROW_UNIFORM
    assert meta_kmeans["quantizer_version"] == APL_QUANTIZER_ROW_KMEANS
    assert meta_uniform["layout_version"] == APL_LAYOUT_NATURAL
    assert meta_kmeans["layout_version"] == APL_LAYOUT_NATURAL


def test_apl_lut_rejects_invalid_artifact_padding() -> None:
    qweight = torch.zeros((4, 5, 2), dtype=torch.int32)
    lut = torch.zeros((5, 16), dtype=torch.float16)
    with pytest.raises(ValueError, match="N_padded"):
        apl_lut_dequantize_weight(qweight, lut, 4)


@pytest.mark.skipif(sglang_apl_lut_linear_kernel is None or not torch.cuda.is_available(), reason="requires Triton and CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_apl_lut_triton_kernel_matches_dequant_matmul(dtype: torch.dtype) -> None:
    import triton

    torch.manual_seed(0)
    weight = torch.randn(96, 128, dtype=torch.float16)
    qweight, lut, meta = apl_lut_quantize_weight(weight, bits=4, pad_n_to=128, pad_k_to=128)
    qweight = qweight.cuda()
    lut = lut.cuda()
    x = torch.randn(1, 128, device="cuda", dtype=dtype)
    out = torch.empty((1, 96), device="cuda", dtype=dtype)

    block_n = 64
    block_k = 64
    grid = (1, triton.cdiv(96, block_n))
    sglang_apl_lut_linear_kernel[grid](
        x,
        qweight,
        lut,
        out,
        1,
        96,
        128,
        meta["K_padded"],
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        qweight.stride(2),
        lut.stride(0),
        lut.stride(1),
        out.stride(0),
        out.stride(1),
        BITS=4,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    dequant = apl_lut_dequantize_weight(qweight.cpu(), lut.cpu(), 4, k_orig=128, n_orig=96).cuda()
    expected = x.float() @ dequant.t().float()
    assert torch.allclose(out.float(), expected, atol=0.5, rtol=0.02)
