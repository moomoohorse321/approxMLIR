from __future__ import annotations

import torch


APL_LAYOUT_NATURAL = "natural_bitplane_v1"
APL_LAYOUT_CUDA_PERMUTED = "cuda_permuted_bitplane_v1"
APL_LAYOUTS = (APL_LAYOUT_NATURAL, APL_LAYOUT_CUDA_PERMUTED)

APL_QUANTIZER_ROW_UNIFORM = "row_uniform_lut_v1"
APL_QUANTIZER_ROW_KMEANS = "row_kmeans_lut_v1"
APL_QUANTIZER_IMPORTED_ANYPRECISION = "imported_anyprecision_v1"
APL_QUANTIZERS = (
    APL_QUANTIZER_ROW_UNIFORM,
    APL_QUANTIZER_ROW_KMEANS,
    APL_QUANTIZER_IMPORTED_ANYPRECISION,
)


def _ceil_div_int(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _validate_apl_shapes(qweight: torch.Tensor, lut: torch.Tensor, bits: int) -> tuple[int, int]:
    if qweight.ndim != 3 or qweight.dtype != torch.int32:
        raise ValueError("expected qweight with shape (bits, N_padded, K_padded/32) and dtype int32")
    if lut.ndim != 2 or lut.shape[1] != (1 << int(bits)):
        raise ValueError("expected lut with shape (N_padded, 2**bits)")
    if int(bits) <= 0 or int(bits) > qweight.shape[0] or int(bits) > 8:
        raise ValueError("invalid APL bit width")
    if qweight.shape[1] != lut.shape[0]:
        raise ValueError("qweight and lut disagree on N_padded")
    n_padded = int(qweight.shape[1])
    k_padded = int(qweight.shape[2]) * 32
    if n_padded % 4 != 0:
        raise ValueError(f"APL artifact requires N_padded % 4 == 0, got {n_padded}")
    if k_padded % 32 != 0:
        raise ValueError(f"APL artifact requires K_padded % 32 == 0, got {k_padded}")
    return n_padded, k_padded


def _cuda_row_permutation(n_padded: int) -> torch.Tensor:
    if int(n_padded) % 4 != 0:
        raise ValueError(f"cuda_permuted_bitplane_v1 requires N_padded % 4 == 0, got {n_padded}")
    base = torch.arange(int(n_padded), dtype=torch.long).reshape(-1, 4)
    return base[:, [0, 2, 1, 3]].reshape(-1)


def apl_lut_convert_layout(
    qweight: torch.Tensor,
    lut: torch.Tensor,
    *,
    src_layout: str,
    dst_layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if src_layout not in APL_LAYOUTS:
        raise ValueError(f"unknown APL source layout: {src_layout}")
    if dst_layout not in APL_LAYOUTS:
        raise ValueError(f"unknown APL destination layout: {dst_layout}")
    if src_layout == dst_layout:
        return qweight.contiguous(), lut.contiguous()
    n_padded = int(qweight.shape[1])
    perm = _cuda_row_permutation(n_padded).to(qweight.device)
    if src_layout == APL_LAYOUT_NATURAL and dst_layout == APL_LAYOUT_CUDA_PERMUTED:
        lut_perm = perm.to(lut.device)
        return qweight[:, perm, :].contiguous(), lut[lut_perm].contiguous()
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n_padded, device=perm.device)
    lut_inv = inv.to(lut.device)
    return qweight[:, inv, :].contiguous(), lut[lut_inv].contiguous()


def apl_lut_dequantize_weight(
    qweight: torch.Tensor,
    lut: torch.Tensor,
    bits: int,
    *,
    k_orig: int | None = None,
    n_orig: int | None = None,
    layout_version: str = APL_LAYOUT_NATURAL,
) -> torch.Tensor:
    if layout_version not in APL_LAYOUTS:
        raise ValueError(f"unknown APL layout: {layout_version}")
    n_padded, k_padded = _validate_apl_shapes(qweight, lut, bits)
    if layout_version == APL_LAYOUT_CUDA_PERMUTED:
        qweight, lut = apl_lut_convert_layout(
            qweight,
            lut,
            src_layout=APL_LAYOUT_CUDA_PERMUTED,
            dst_layout=APL_LAYOUT_NATURAL,
        )
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


def _row_uniform_codes(row_w: torch.Tensor, levels: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    return centroids, row_codes


def _row_kmeans_codes(row_w: torch.Tensor, levels: int, *, iters: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    if levels == 1 or row_w.numel() == 0:
        return row_w.mean().reshape(1), torch.zeros_like(row_w, dtype=torch.int64)
    w_min = row_w.min()
    w_max = row_w.max()
    if torch.isclose(w_min, w_max):
        return torch.full((levels,), float(w_min), dtype=torch.float32), torch.zeros_like(row_w, dtype=torch.int64)

    centroids = torch.quantile(row_w, torch.linspace(0.0, 1.0, levels, dtype=torch.float32))
    row_codes = torch.zeros_like(row_w, dtype=torch.int64)
    for _ in range(iters):
        dist = torch.abs(row_w[:, None] - centroids[None, :])
        row_codes = torch.argmin(dist, dim=1).to(torch.int64)
        for code in range(levels):
            mask = row_codes == code
            if bool(mask.any()):
                centroids[code] = row_w[mask].mean()
    order = torch.argsort(centroids)
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(levels, dtype=torch.long)
    return centroids[order].to(torch.float32), inv_order[row_codes].to(torch.int64)


def apl_lut_quantize_weight(
    weight: torch.Tensor,
    *,
    bits: int,
    pad_n_to: int = 4,
    pad_k_to: int = 32,
    layout_version: str = APL_LAYOUT_NATURAL,
    quantizer_version: str = APL_QUANTIZER_ROW_UNIFORM,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("expected a 2D floating-point weight")
    if bits < 1 or bits > 8:
        raise ValueError("expected APL bits in [1, 8]")
    if layout_version not in APL_LAYOUTS:
        raise ValueError(f"unknown APL layout: {layout_version}")
    if quantizer_version not in APL_QUANTIZERS:
        raise ValueError(f"unknown APL quantizer version: {quantizer_version}")
    if quantizer_version == APL_QUANTIZER_IMPORTED_ANYPRECISION:
        raise ValueError("imported_anyprecision_v1 artifacts must be loaded, not built from runtime weights")
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
        if quantizer_version == APL_QUANTIZER_ROW_KMEANS:
            centroids, row_codes = _row_kmeans_codes(row_w, levels)
        else:
            centroids, row_codes = _row_uniform_codes(row_w, levels)
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
        "backend": "triton_apl_lut",
        "bits": int(bits),
        "K_orig": int(k_orig),
        "K_padded": int(k_padded),
        "N_orig": int(n_orig),
        "N_padded": int(n_padded),
        "source_dtype": str(weight.dtype),
        "layout_version": APL_LAYOUT_NATURAL,
        "quantizer_version": quantizer_version,
    }
    qweight = qweight.contiguous()
    lut = lut.contiguous()
    if layout_version != APL_LAYOUT_NATURAL:
        qweight, lut = apl_lut_convert_layout(
            qweight,
            lut,
            src_layout=APL_LAYOUT_NATURAL,
            dst_layout=layout_version,
        )
        meta["layout_version"] = layout_version
    return qweight, lut, meta
