from __future__ import annotations

EXACT_BACKENDS = frozenset({"exact", "none", "original", "fp", "fp16", "bf16"})
W4_BACKENDS = frozenset({"triton_sq_w4a16", "triton_awq_w4a16"})
QUANT_BACKENDS = frozenset(
    {
        "triton",
        "triton_prequant",
        "triton_w8a16",
        "sgl_kernel",
        *W4_BACKENDS,
    }
)
NO_SUBSTITUTE_BACKENDS = EXACT_BACKENDS | {"sgl_kernel"}


def backend_or_none(backend: str) -> str | None:
    if backend in EXACT_BACKENDS:
        return None
    if backend not in QUANT_BACKENDS:
        raise ValueError(f"unsupported APPROX_SGLANG backend: {backend}")
    return backend


def parse_mixed_backend_rules(raw: str) -> tuple[tuple[str, str], ...]:
    rules: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        sep = "=" if "=" in item else ":"
        if sep not in item:
            raise ValueError(
                "APPROX_SGLANG_MIXED_BACKENDS entries must be target=backend"
            )
        target, backend = (part.strip() for part in item.split(sep, 1))
        if not target or not backend:
            raise ValueError(
                "APPROX_SGLANG_MIXED_BACKENDS entries must be target=backend"
            )
        backend_or_none(backend)
        rules.append((target, backend))
    return tuple(rules)


def unique_in_order(values) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
