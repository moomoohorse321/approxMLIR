# Quantization In approxMLIR

This document describes the current SGLang-based quantization mainline in
`approxMLIR/runtime/examples`.

## Current Goal

The active goal is a real serving-style accuracy/performance frontier:

- run Qwen through SGLang with a Triton backend
- use approxMLIR's Triton hook inside SGLang worker subprocesses
- precompute quantized weight state outside per-token kernel launches
- use function substitution for the Triton kernel body
- measure e2e generation latency, not only microbenchmarks

The old TorchInductor demo is no longer the mainline.

## Mainline Files

- `runtime/examples/sglang_quant/probe_sglang_triton_dump.py`
  - main single-run driver
  - exact/approx generation
  - e2e latency collection
  - TTIR dump hook setup
  - SGLang child-process bootstrap setup

- `runtime/examples/sglang_quant/sweep_sglang_quant.py`
  - sequential sweep over exact and approximate configs
  - records median latency, output text, and substitution hit counts

- `runtime/examples/sglang_quant/approx_quant_patch.py`
  - runtime patch for SGLang `UnquantizedLinearMethod`
  - prepares quantized weights, scales, and SmoothQuant calibration state
  - redirects selected linear call sites to quantized Triton kernels

- `runtime/examples/sglang_quant/approx_kernels.py`
  - W8A16, W8A8, and SmoothQuant-style W4A16 Triton kernels
  - same-signature substitute kernels used by approxMLIR function substitution

- `runtime/examples/sglang_quant/compare_sglang_logprobs.py`
  - exact-vs-approx accuracy probe using SGLang logprob APIs
  - reports teacher-forced token-logprob drift and top-k distribution drift
  - uses stepwise decode when `APPROX_SGLANG_DECODE_ONLY=1`

- `runtime/examples/sglang_quant/bootstrap/sitecustomize.py`
  - installs hooks inside SGLang worker subprocesses

## Current Quantization Strategy

The useful path today is still `APPROX_SGLANG_BACKEND=triton_w8a16`.

The new lower-bit path is:

- `APPROX_SGLANG_BACKEND=triton_sq_w4a16`

This is not a naive int4 path. It is a SmoothQuant-style calibrated `W4A16`
flow with:

- offline activation-stat collection on targeted linear layers
- smoothing-factor computation from activation and weight statistics
- one-time load-time packing into group-wise int4 weights
- online activation-side inverse smoothing inside the Triton kernel

It is weight-only at runtime:

- activation input stays fp16/bf16
- weight is prequantized outside the kernel to low-bit values plus fp32 scale
- the Triton kernel loads quantized weight and scale, dequantizes weight values,
  and multiplies by fp16/bf16 activation
- no activation quantization is used in the current best result

This is intentionally different from the older W8A8 attempts. The W8A8 paths
are still present for comparison, but they are not the current useful frontier
because activation quantization cost can dominate at this granularity.

## SmoothQuant-Style Split: Offline, Load Time, Online

The `triton_sq_w4a16` path is split across three phases.

Offline calibration:

- run the exact model with `APPROX_SGLANG_SQ_COLLECT=1`
- collect per-layer activation absmax statistics
- merge worker shards into one artifact with
  `runtime/examples/sglang_quant/build_smoothquant_artifact.py`

Load time:

- read the artifact for each targeted layer
- fold smoothing into the weight tensor
- quantize to group-wise int4
- pack the weight tensor once and cache it on the layer

Online serving:

- route the targeted linear call site to `sglang_sq_w4a16_linear_kernel`
- apply the activation-side inverse smoothing in-kernel
- unpack int4 weights, load group-wise scales, and run the matmul
- if `M==1` and tile metadata matches, use a decode-only tiled W4 layout that is
  repacked at load time; otherwise fall back to the generic packed W4 path

## Function Substitution Boundary

Current function substitution is same-ABI:

```text
sglang_w8a16_linear_kernel(...)
  -> approx_sglang_w8a16_linear_kernel_1(...)

sglang_sq_w4a16_linear_kernel(...)
  -> approx_sglang_sq_w4a16_linear_kernel_1(...)
```

The ABI change from original SGLang fp weight to `qweight + scale` is currently
done by the SGLang runtime patch, not by the compiler pass.

For `triton_sq_w4a16`, that ABI is:

- packed int4 weight
- group-wise scale tensor
- activation smoothing inverse vector

What this proves:

- approxMLIR can hook Triton compilation inside SGLang workers
- approxMLIR can substitute the selected Triton kernel body
- the serving path can produce a real positive e2e speedup

What remains future work:

- compiler-level call-site / ABI rewrite from original fp weights to quantized
  weight metadata
- more optimized W4 kernels; current decode-tiled SQ-W4 is now ahead on speed
  but still behind on accuracy
- richer calibration schemes such as AWQ/GPTQ-style protection on especially
  sensitive layers

## Current Best Configuration

The repeated PoC configuration is:

```bash
MODEL_PATH=Qwen/Qwen3.5-2B
BATCH_SIZE=4
MAX_NEW_TOKENS=8
WARMUP_RUNS=2
MEASURE_RUNS=5
ATTENTION_BACKEND=triton
SAMPLING_BACKEND=pytorch
SGLANG_MEM_FRACTION_STATIC=0.75
SGLANG_DISABLE_CUDA_GRAPH=0
APPROX_SGLANG_QUANT=1
APPROX_SGLANG_MODE=approx
APPROX_SGLANG_TARGET=gate_up_proj
APPROX_SGLANG_BACKEND=triton_w8a16
APPROX_SGLANG_USE_SUBSTITUTE=1
APPROX_SGLANG_DECODE_ONLY=0
APPROX_SGLANG_BLOCK_N=128
APPROX_SGLANG_BLOCK_K=64
```

Observed repeated result on the RTX 4060 Laptop:

- exact SGLang/Triton batch=4, 8-token e2e median: about `0.204s`
- `gate_up_proj` W8A16 median: about `0.156s`
- speedup: about `1.31x`
- substitution hits: `100%`
- output text matched the exact run in the checked prompt

Use medians for reporting. SGLang e2e latency has occasional outliers.

## Current Accuracy Definition

For decode-oriented points, the useful accuracy definition is no longer
"output text matched on one prompt".

The current accuracy probe is:

- exact model generates a reference continuation
- exact and approx are then compared on that same continuation
- when `APPROX_SGLANG_DECODE_ONLY=1`, the comparison is done as stepwise decode:
  one token at a time, conditioned on the exact prefix
- metrics are computed from SGLang logprob APIs

Current reported metrics:

- `teacher_forced_perplexity_ratio`
  - ratio of approx NLL to exact NLL on the exact continuation
  - `1.0` is exact; higher is worse
- `teacher_forced_mean_logprob_delta`
  - mean `(approx_logprob - exact_logprob)` on the reference token
- `top1_agreement_rate`
  - fraction of decode steps where approx and exact agree on top-1 token
- `topk_js_mean`
  - Jensen-Shannon divergence on returned top-k token distributions
  - `0.0` is exact; lower is better

This is still not full-vocab KL, because the public SGLang interface returns
top-k / selected-token logprobs rather than full logits. But it is much more
meaningful than a single prompt-level text match.

## Current Pareto Frontier

Measured on:

- `MODEL_PATH=Qwen/Qwen3.5-2B`
- `BATCH_SIZE=4`
- `MAX_NEW_TOKENS=8`
- target: `gate_up_proj`
- decode-heavy serving with CUDA graph enabled
- accuracy probe on 4 prompts x 8 generated tokens = 32 scored decode steps

Latency points:

- `exact`: median about `0.202s`
- `gate_up_proj` `triton_w8a16`: median about `0.176s` (`~1.15x`)
- decode-tiled `gate_up_proj` `triton_sq_w4a16`: median about `0.136s`
  (`~1.48x` vs exact, `~1.29x` vs W8)

Decode-step logprob accuracy:

- `exact`
  - `teacher_forced_perplexity_ratio = 1.0000`
  - `top1_agreement_rate = 1.000`
  - `topk_js_mean = 0.000000`
- `W8A16`
  - `teacher_forced_perplexity_ratio = 1.0048`
  - `teacher_forced_mean_logprob_delta = -0.00483`
  - `top1_agreement_rate = 1.000`
  - `topk_js_mean = 0.000643`
- decode-tiled `W4A16`
  - `teacher_forced_perplexity_ratio = 1.1569`
  - `teacher_forced_mean_logprob_delta = -0.14578`
  - `top1_agreement_rate = 0.875`
  - `topk_js_mean = 0.009500`

Interpretation:

- `W8A16` is the current safe Pareto point
  - clearly faster than exact
  - logits distribution remains very close to exact
- decode-tiled `W4A16` is a strong speed point but not yet on the safe
  accuracy/performance frontier
  - it is much faster than both exact and W8
  - but the decode-step distribution drift is now clearly visible under the new
    logprob definition

## Frontier Interpretation

The current frontier is selective coverage:

- `gate_up_proj` W8A16 is the best target so far
- `qkv_proj` W8A16 gives a smaller positive point
- `down_proj` is not a Pareto point with the current W8A16 kernel
- `triton_sq_w4a16` full coverage is not a Pareto point with the current W4
  kernel
- decode-tiled `triton_sq_w4a16` now gives a major speedup, but under the
  decode-step logprob definition it is still too inaccurate to count as a safe
  frontier point
- blindly increasing coverage can erase the speedup

CUDA graph must be enabled for serving-like measurements. With CUDA graph
disabled, launch overhead dominates and the same replacement can look flat or
negative.

## Accuracy Probe Command

Decode-step exact-vs-approx logprob comparison:

```bash
source .venv/bin/activate
MODEL_PATH=Qwen/Qwen3.5-2B \
BATCH_SIZE=4 \
MAX_NEW_TOKENS=8 \
TOP_LOGPROBS_NUM=20 \
SGLANG_MEM_FRACTION_STATIC=0.75 \
SGLANG_DISABLE_CUDA_GRAPH=0 \
PROMPTS_JSON='["The capital of France is","The capital of France is","The capital of France is","The capital of France is"]' \
APPROX_SGLANG_QUANT=1 \
APPROX_SGLANG_MODE=approx \
APPROX_SGLANG_TARGET=gate_up_proj \
APPROX_SGLANG_BACKEND=triton_w8a16 \
APPROX_SGLANG_DECODE_ONLY=1 \
python3 approxMLIR/runtime/examples/sglang_quant/compare_sglang_logprobs.py
```

Swap `APPROX_SGLANG_BACKEND` and related W4 envs to compare other points.

## Run Commands

Single best-config run:

```bash
source .venv/bin/activate
OUT_DIR=/tmp/approx_sglang_qwen35_2b_gate \
MODEL_PATH=Qwen/Qwen3.5-2B \
BATCH_SIZE=4 \
MAX_NEW_TOKENS=8 \
WARMUP_RUNS=2 \
MEASURE_RUNS=5 \
ATTENTION_BACKEND=triton \
SAMPLING_BACKEND=pytorch \
SGLANG_MEM_FRACTION_STATIC=0.75 \
SGLANG_DISABLE_CUDA_GRAPH=0 \
APPROX_SGLANG_QUANT=1 \
APPROX_SGLANG_MODE=approx \
APPROX_SGLANG_TARGET=gate_up_proj \
APPROX_SGLANG_BACKEND=triton_w8a16 \
APPROX_SGLANG_USE_SUBSTITUTE=1 \
APPROX_SGLANG_DECODE_ONLY=0 \
TRITON_PASS_PLUGIN_PATH=$PWD/approxMLIR/external-tools/approx-triton-plugin/build/lib/libApproxTritonPlugin.so \
python3 approxMLIR/runtime/examples/sglang_quant/probe_sglang_triton_dump.py
```

Sequential sweep:

```bash
source .venv/bin/activate
OUT_DIR=/tmp/approx_sglang_quant_sweep \
MODEL_PATH=Qwen/Qwen3.5-2B \
BATCH_SIZE=4 \
MAX_NEW_TOKENS=8 \
WARMUP_RUNS=2 \
MEASURE_RUNS=5 \
SGLANG_MEM_FRACTION_STATIC=0.75 \
python3 approxMLIR/runtime/examples/sglang_quant/sweep_sglang_quant.py
```
