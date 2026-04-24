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
- `APPROX_SGLANG_BACKEND`: `triton_w8a16`, `triton_sq_w4a16`,
  `triton_awq_w4a16`,
  `triton_prequant`, `triton`, or `sgl_kernel`
- `APPROX_SGLANG_MIXED_BACKENDS`: comma-separated `target=backend` rules for
  mixed precision, for example
  `gate_up_proj=triton_w8a16,qkv_proj=triton_awq_w4a16`; when set, this
  supersedes `APPROX_SGLANG_TARGET` and `APPROX_SGLANG_BACKEND`
- `APPROX_SGLANG_DECODE_ONLY`: only patch M=1 decode calls, default `1`
- `APPROX_SGLANG_USE_SUBSTITUTE`: set `1` to run approxMLIR function
  substitution on the selected Triton kernel
- `APPROX_SGLANG_DROP_ORIGINAL_WEIGHT`: drop original fp weights after
  prequantization for wide coverage; only safe in approx mode
- `APPROX_SGLANG_SQ_COLLECT`: collect per-layer activation absmax for
  SmoothQuant-style calibration
- `APPROX_SGLANG_SQ_STATS_DIR`: directory for calibration shards
- `APPROX_SGLANG_SQ_ARTIFACT_PATH`: merged SmoothQuant artifact path
- `APPROX_SGLANG_SQ_ALPHA`: smoothing exponent, default `0.85`
- `APPROX_SGLANG_SQ_GROUP_SIZE`: W4 group size, default `128`
- `APPROX_SGLANG_SQ_BLOCK_K`: SQ-W4 K tile size, default `64`
- `APPROX_SGLANG_AWQ_GRID_SIZE`: AWQ-style ratio search grid size, default `20`

SmoothQuant-style calibration flow:

```bash
source .venv/bin/activate

OUT_DIR=/tmp/approx_sq_cal \
MODEL_PATH=Qwen/Qwen3.5-2B \
BATCH_SIZE=4 \
MAX_NEW_TOKENS=8 \
WARMUP_RUNS=1 \
MEASURE_RUNS=1 \
APPROX_SGLANG_QUANT=1 \
APPROX_SGLANG_MODE=exact \
APPROX_SGLANG_TARGET=gate_up_proj \
APPROX_SGLANG_BACKEND=triton_sq_w4a16 \
APPROX_SGLANG_SQ_COLLECT=1 \
TRITON_PASS_PLUGIN_PATH=$PWD/approxMLIR/external-tools/approx-triton-plugin/build/lib/libApproxTritonPlugin.so \
python3 approxMLIR/runtime/examples/sglang_quant/probe_sglang_triton_dump.py

APPROX_SGLANG_SQ_STATS_DIR=/tmp/approx_sq_cal/sq_stats \
APPROX_SGLANG_SQ_ARTIFACT_PATH=/tmp/approx_sq_cal/sq_artifact.pt \
python3 approxMLIR/runtime/examples/sglang_quant/build_smoothquant_artifact.py
```

AWQ-style W4 uses the same calibration artifact path, but swaps the load-time
quantizer:

```bash
APPROX_SGLANG_QUANT=1 \
APPROX_SGLANG_MODE=approx \
APPROX_SGLANG_TARGET=gate_up_proj \
APPROX_SGLANG_BACKEND=triton_awq_w4a16 \
APPROX_SGLANG_SQ_ARTIFACT_PATH=/tmp/approx_sq_cal/sq_artifact.pt \
APPROX_SGLANG_SQ_GROUP_SIZE=128 \
APPROX_SGLANG_SQ_BLOCK_K=64 \
APPROX_SGLANG_AWQ_GRID_SIZE=20 \
python3 approxMLIR/runtime/examples/sglang_quant/probe_sglang_triton_dump.py
```

Mixed precision is expressed as a runtime binding plan, not as a new kernel
family. The current implementation reuses the existing W8A16 and W4A16 kernels
and installs one approxMLIR function-substitution hook per backend kernel that
appears in `APPROX_SGLANG_MIXED_BACKENDS`.

Example W8/W4 mix:

```bash
APPROX_SGLANG_QUANT=1 \
APPROX_SGLANG_MODE=approx \
APPROX_SGLANG_MIXED_BACKENDS=gate_up_proj=triton_w8a16,qkv_proj=triton_awq_w4a16 \
APPROX_SGLANG_USE_SUBSTITUTE=1 \
APPROX_SGLANG_SQ_ARTIFACT_PATH=/tmp/approx_sq_cal_gq/sq_artifact.pt \
APPROX_SGLANG_SQ_GROUP_SIZE=128 \
APPROX_SGLANG_SQ_BLOCK_K=64 \
APPROX_SGLANG_AWQ_GRID_SIZE=20 \
python3 approxMLIR/runtime/examples/sglang_quant/probe_sglang_triton_dump.py
```

The best tuned selective W4 point found so far moves both `gate_up_proj` and
`qkv_proj` to AWQ-W4 and uses a smaller W4 K tile:

```bash
APPROX_SGLANG_QUANT=1 \
APPROX_SGLANG_MODE=approx \
APPROX_SGLANG_MIXED_BACKENDS=gate_up_proj=triton_awq_w4a16,qkv_proj=triton_awq_w4a16 \
APPROX_SGLANG_USE_SUBSTITUTE=1 \
APPROX_SGLANG_SQ_ARTIFACT_PATH=/tmp/approx_sq_cal_gq/sq_artifact.pt \
APPROX_SGLANG_SQ_GROUP_SIZE=128 \
APPROX_SGLANG_SQ_BLOCK_K=32 \
APPROX_SGLANG_AWQ_GRID_SIZE=20 \
python3 approxMLIR/runtime/examples/sglang_quant/probe_sglang_triton_dump.py
```

Rules are matched in order against the SGLang layer prefix; `all=...` is
accepted. Backends `exact`, `none`, `original`, `fp`, `fp16`, and `bf16` leave a
matching site on the original SGLang linear path. W4 rules require the SQ/AWQ
artifact to contain the matched layer prefix; otherwise that site falls back to
the original path and records a skip event.

Current Qwen3.5-2B result on the RTX 4060 Laptop, batch=4, 8 generated tokens,
CUDA graph enabled:

- exact SGLang/Triton median: about `0.204s`
- `gate_up_proj` W8A16 substitute median: about `0.156s` (`1.31x`)
- `qkv_proj` W8A16 substitute median: about `0.199s` (`1.02x`)
- `gate_up_proj` SmoothQuant-style W4A16 full coverage median: about `0.226s`
  with checked-output divergence
- `gate_up_proj` SmoothQuant-style W4A16 decode-only, group size `128`,
  block size `64`: about `0.194s` (`1.05x`) with checked-output match on
  the probe prompt
- mixed `gate_up_proj=triton_w8a16,qkv_proj=triton_awq_w4a16`, using a
  gate/qkv calibration artifact: about `0.151s` in a two-measurement smoke
  probe, with `90/90` substitution hits
- tuned selective AWQ-W4 on `gate_up_proj,qkv_proj` with `SQ_BLOCK_K=32`:
  about `0.137s` in a five-measurement probe, with `90/90` substitution hits

The practical frontier is selective, not maximum coverage. `gate_up_proj` is
the current useful target; `down_proj` regresses because the W8A16 kernel is not
a fast path for that shape. On the tested Qwen3.5-2B setup, the new
`triton_sq_w4a16` path is calibrated and functional, but its current kernel only
becomes a useful secondary point when tuned for decode-only with `group=128`
and `block_k=64`. `triton_awq_w4a16` now runs in parallel to that path with the
same online kernel and decode-tiled layout, but a different load-time
activation-aware group-wise W4 quantizer. CUDA graph must be enabled for
serving-like numbers, otherwise launch overhead dominates and the same
replacement can look flat or negative.

For the mixed W8/AWQ-W4 smoke above, the decode-step logprob probe over 32
scored tokens reported `top1_agreement_rate=1.0`,
`teacher_forced_perplexity_ratio=0.9854`, and `topk_js_mean=0.000882`.
For the tuned selective AWQ-W4 policy, the same probe shape reported
`top1_agreement_rate=1.0`, `teacher_forced_perplexity_ratio=0.9791`, and
`topk_js_mean=0.00366`.
