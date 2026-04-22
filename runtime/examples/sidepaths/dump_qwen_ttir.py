#!/usr/bin/env python3
"""Stage 1 dump harness: HF transformers + torch.compile -> Triton TTIR.

Boots a HuggingFace causal-LM (default: a small Qwen variant) through
``torch.compile`` so the forward pass emits Triton kernels via TorchInductor,
and routes every kernel through :func:`approx_runtime.make_triton_dump_hook`
so the TTIR + correlation metadata land in ``OUT_DIR``.

CAVEAT: these are TorchInductor-lowered kernels, not LLM-serving kernels (no
vLLM / FlashAttention / hand-written fused ops). Fine for early observation,
not sufficient for locking in quantization targets — rerun on a serving-like
input path before doing target selection.

Environment variables
---------------------
OUT_DIR         Directory to write dumps into (default: /tmp/qwen_ttir_dump)
MODEL_ID        HF model id                   (default: Qwen/Qwen2.5-0.5B)
PROMPT          Prompt text                   (default: a short sentence)
MAX_NEW_TOKENS  Tokens to generate            (default: 4)
DUMP_VERBOSE    "1" to log each dumped kernel (default: "0")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


_EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_ROOT))


def _setup_dlopen_flags() -> None:
    # Per plugin-architecture notes: libtriton.so's MLIR symbols must be
    # globally visible when the pass plugin loads. Load this even for the
    # observe-only path so the same process can later switch to the rewriting
    # hook without reimporting triton.
    try:
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    except AttributeError:
        pass


def main() -> int:
    _setup_dlopen_flags()

    import torch
    import triton
    from triton import knobs

    import approx_runtime as ar

    out_dir = Path(os.environ.get("OUT_DIR", "/tmp/qwen_ttir_dump"))
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B")
    prompt = os.environ.get("PROMPT", "The quick brown fox jumps over")
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "4"))
    verbose = os.environ.get("DUMP_VERBOSE", "0") == "1"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this dump harness")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required: `pip install transformers accelerate`"
        ) from exc

    print(f"[dump-qwen] out_dir={out_dir}")
    print(f"[dump-qwen] model_id={model_id}")
    print(f"[dump-qwen] prompt={prompt!r}")
    print(f"[dump-qwen] max_new_tokens={max_new_tokens}")

    hook = ar.make_triton_dump_hook(
        out_dir=out_dir,
        source="hf_torch_compile",
        verbose=verbose,
    )
    knobs.runtime.add_stages_inspection_hook = hook

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    compiled = torch.compile(model, backend="inductor", mode="reduce-overhead")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = compiled.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"[dump-qwen] generated: {decoded!r}")

    knobs.runtime.add_stages_inspection_hook = None

    jsons = sorted(out_dir.glob("*.json"))
    print(f"[dump-qwen] dumped {len(jsons)} distinct kernels into {out_dir}")
    by_name: dict[str, int] = {}
    for p in jsons:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        by_name[rec["func_name"]] = by_name.get(rec["func_name"], 0) + 1
    for name, count in sorted(by_name.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d} shape variants  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
