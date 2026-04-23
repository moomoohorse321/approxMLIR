# SGLang Triton Quantization Mainline

This is the current mainline for testing approxMLIR quantization inside a
serving-style SGLang runtime.

Goals:

- verify that `sglang` imports in the selected Python environment
- verify which `triton` package SGLang sees
- verify whether `triton.knobs.runtime.add_stages_inspection_hook` exists
- optionally run one tiny offline SGLang generation with `attention_backend=triton`
- dump any Triton TTIR into a probe-only output directory
- install the dump hook inside SGLang worker subprocesses through a local
  `sitecustomize.py`
- monkeypatch SGLang linear layers with torchao-style weight prequantization
- optionally run approxMLIR function substitution on the Triton approx kernel

Non-goals:

- no package installation
- no mutation of the current venv
- no direct changes to installed SGLang files
- no compiler ABI-changing rewrite yet

Example:

```bash
source .venv/bin/activate
OUT_DIR=/tmp/approx_sglang_probe \
MODEL_PATH=Qwen/Qwen3.5-2B \
BATCH_SIZE=4 \
MAX_NEW_TOKENS=8 \
WARMUP_RUNS=2 \
MEASURE_RUNS=5 \
SGLANG_MEM_FRACTION_STATIC=0.75 \
SGLANG_DISABLE_CUDA_GRAPH=0 \
APPROX_SGLANG_QUANT=1 \
APPROX_SGLANG_MODE=approx \
APPROX_SGLANG_TARGET=gate_up_proj \
APPROX_SGLANG_BACKEND=triton_w8a16 \
APPROX_SGLANG_USE_SUBSTITUTE=1 \
TRITON_PASS_PLUGIN_PATH=$PWD/approxMLIR/external-tools/approx-triton-plugin/build/lib/libApproxTritonPlugin.so \
python3 approxMLIR/runtime/examples/sglang_quant/probe_sglang_triton_dump.py
```

Useful environment variables:

- `OUT_DIR`: output directory, default `/tmp/approx_sglang_probe`
- `MODEL_PATH`: SGLang model path, default `Qwen/Qwen2.5-0.5B-Instruct`
- `PROMPT`: prompt, default `The capital of France is`
- `BATCH_SIZE`: repeated prompt batch size, default `1`
- `MAX_NEW_TOKENS`: generation length, default `2`
- `WARMUP_RUNS`: warmup generate calls, default `0`
- `MEASURE_RUNS`: measured generate calls, default `1`
- `ATTENTION_BACKEND`: default `triton`
- `SAMPLING_BACKEND`: default `pytorch`
- `SGLANG_DISABLE_CUDA_GRAPH`: set `0` for serving-like timing, default `1`
- `SGLANG_FORCE_SOURCE_TRITON`: prepend repo `triton/python`, default `1`
- `RUN_GENERATE`: set `0` for import/hook checks only, default `1`
- `APPROX_SGLANG_QUANT`: set `1` to monkeypatch SGLang linear layers
- `APPROX_SGLANG_MODE`: `exact` or `approx`, default `exact`
- `APPROX_SGLANG_TARGET`: comma-separated substring filter, default `proj`
- `APPROX_SGLANG_BACKEND`: `triton_w8a16`, `triton_prequant`, `triton`, or
  `sgl_kernel`; current useful path is `triton_w8a16`
- `APPROX_SGLANG_DECODE_ONLY`: only patch M=1 decode calls, default `1`
- `APPROX_SGLANG_USE_SUBSTITUTE`: set `1` to run approxMLIR function
  substitution on the selected Triton kernel
- `APPROX_SGLANG_DROP_ORIGINAL_WEIGHT`: drop original fp weights after
  prequantization for wide coverage; only safe in approx mode

Current Qwen3.5-2B result on the RTX 4060 Laptop, batch=4, 8 generated tokens,
CUDA graph enabled:

- exact SGLang/Triton median: `0.212s`
- `gate_up_proj` W8A16 substitute median: `0.152s` (`1.40x`)
- `qkv_proj` W8A16 substitute median: `0.201s` (`1.06x`)
- `gate_up_proj,qkv_proj` median: `0.206s`
- `gate_up_proj,down_proj` median: `0.211s`

The practical frontier is selective, not maximum coverage. `gate_up_proj` is
the current useful target; `down_proj` regresses because the W8A16 kernel is not
a fast path for that shape. CUDA graph must be enabled for serving-like numbers,
otherwise launch overhead dominates and the same replacement can look flat or
negative.
