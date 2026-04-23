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
  - prepares quantized weights and scales
  - redirects selected linear call sites to quantized Triton kernels

- `runtime/examples/sglang_quant/approx_kernels.py`
  - W8A16 and W8A8 Triton kernels
  - same-signature substitute kernels used by approxMLIR function substitution

- `runtime/examples/sglang_quant/bootstrap/sitecustomize.py`
  - installs hooks inside SGLang worker subprocesses

## Current Quantization Strategy

The useful path today is `APPROX_SGLANG_BACKEND=triton_w8a16`.

It is weight-only at runtime:

- activation input stays fp16/bf16
- weight is prequantized outside the kernel to int8 plus fp32 scale
- the Triton kernel loads int8 weight and scale, dequantizes weight values, and
  multiplies by fp16/bf16 activation
- no activation quantization is used in the current best result

This is intentionally different from the older W8A8 attempts. The W8A8 paths
are still present for comparison, but they are not the current useful frontier
because activation quantization cost can dominate at this granularity.

## Function Substitution Boundary

Current function substitution is same-ABI:

```text
sglang_w8a16_linear_kernel(...)
  -> approx_sglang_w8a16_linear_kernel_1(...)
```

The ABI change from original SGLang fp weight to `qweight + scale` is currently
done by the SGLang runtime patch, not by the compiler pass.

What this proves:

- approxMLIR can hook Triton compilation inside SGLang workers
- approxMLIR can substitute the selected Triton kernel body
- the serving path can produce a real positive e2e speedup

What remains future work:

- compiler-level call-site / ABI rewrite from original fp weights to quantized
  weight metadata
- passing offline information such as AWQ scales through explicit metadata
  rather than Python monkeypatch state

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

- exact SGLang/Triton batch=4, 8-token e2e median: about `0.212s`
- approximate repeated medians: `0.166s`, `0.150s`, `0.151s`
- speedup range: about `1.27x` to `1.41x`
- substitution hits: `100%`
- output text matched the exact run in the checked prompt

Use medians for reporting. SGLang e2e latency has occasional outliers.

## Frontier Interpretation

The current frontier is selective coverage:

- `gate_up_proj` W8A16 is the best target so far
- `qkv_proj` W8A16 gives a smaller positive point
- `down_proj` is not a Pareto point with the current W8A16 kernel
- blindly increasing coverage can erase the speedup

CUDA graph must be enabled for serving-like measurements. With CUDA graph
disabled, launch overhead dominates and the same replacement can look flat or
negative.

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
