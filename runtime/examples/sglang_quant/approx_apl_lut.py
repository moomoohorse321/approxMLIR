from __future__ import annotations

import torch


def _ceil_div_int(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def apl_lut_dequantize_weight(
    qweight: torch.Tensor,
    lut: torch.Tensor,
    bits: int,
    *,
    k_orig: int | None = None,
    n_orig: int | None = None,
) -> torch.Tensor:
    if qweight.ndim != 3 or qweight.dtype != torch.int32:
        raise ValueError("expected qweight with shape (bits, N, K/32) and dtype int32")
    if lut.ndim != 2 or lut.shape[1] != (1 << int(bits)):
        raise ValueError("expected lut with shape (N, 2**bits)")
    if int(bits) <= 0 or int(bits) > qweight.shape[0]:
        raise ValueError("invalid APL bit width")
    n_padded = qweight.shape[1]
    k_padded = qweight.shape[2] * 32
    n = n_padded if n_orig is None else int(n_orig)
    k = k_padded if k_orig is None else int(k_orig)
    if n > n_padded or k > k_padded:
        raise ValueError("requested original shape exceeds padded qweight shape")

    planes = qweight[: int(bits)].to(torch.int64)
    bit_positions = torch.arange(32, device=planes.device, dtype=torch.int64)
    unpacked = ((planes.unsqueeze(-1) >> bit_positions) & 1).to(torch.long)
    codes = torch.zeros(
        (planes.shape[1], planes.shape[2], 32),
        dtype=torch.long,
        device=planes.device,
    )
    for plane_idx in range(int(bits)):
        codes |= unpacked[plane_idx] << (int(bits) - 1 - plane_idx)
    codes = codes.reshape(planes.shape[1], planes.shape[2] * 32)
    weight = torch.gather(lut, 1, codes)
    return weight[:n, :k].contiguous()


def apl_lut_quantize_weight(
    weight: torch.Tensor,
    *,
    bits: int,
    pad_n_to: int = 1,
    pad_k_to: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("expected a 2D floating-point weight")
    if bits < 1 or bits > 8:
        raise ValueError("expected APL bits in [1, 8]")
    if pad_n_to <= 0 or pad_k_to <= 0 or pad_k_to % 32 != 0:
        raise ValueError("expected positive padding, with K padding divisible by 32")

    n_orig, k_orig = weight.shape
    n_padded = _ceil_div_int(n_orig, pad_n_to) * pad_n_to
    k_padded = _ceil_div_int(k_orig, pad_k_to) * pad_k_to
    levels = 1 << int(bits)
    weight_f = weight.detach().float().cpu()

    lut = torch.empty((n_padded, levels), dtype=torch.float16)
    codes = torch.zeros((n_padded, k_padded), dtype=torch.int64)
    for row in range(n_orig):
        row_w = weight_f[row]
        if levels == 1:
            centroids = row_w.mean().reshape(1)
            row_codes = torch.zeros_like(row_w, dtype=torch.int64)
        else:
            w_min = row_w.min()
            w_max = row_w.max()
            if torch.isclose(w_min, w_max):
                centroids = torch.full((levels,), float(w_min), dtype=torch.float32)
                row_codes = torch.zeros_like(row_w, dtype=torch.int64)
            else:
                centroids = torch.linspace(float(w_min), float(w_max), levels, dtype=torch.float32)
                boundaries = (centroids[:-1] + centroids[1:]) * 0.5
                row_codes = torch.bucketize(row_w, boundaries)
        lut[row] = centroids.to(torch.float16)
        codes[row, :k_orig] = row_codes
    if n_padded > n_orig:
        lut[n_orig:] = 0

    qweight = torch.zeros((int(bits), n_padded, k_padded // 32), dtype=torch.int32)
    for plane_idx in range(int(bits)):
        bit = ((codes >> (int(bits) - 1 - plane_idx)) & 1).to(torch.int32)
        words = torch.zeros((n_padded, k_padded // 32), dtype=torch.int32)
        for bit_pos in range(32):
            words |= bit[:, bit_pos::32] << bit_pos
        qweight[plane_idx] = words

    meta = {
        "bits": int(bits),
        "K_orig": int(k_orig),
        "K_padded": int(k_padded),
        "N_orig": int(n_orig),
        "N_padded": int(n_padded),
        "source_dtype": str(weight.dtype),
        "quantizer": "row_uniform_lut_v1",
    }
    return qweight.contiguous(), lut.contiguous(), meta
