# Quantization In approxMLIR

This file documents the current quantization implementation in
`approxMLIR/runtime/examples`, the current main entrypoint, and the current
engineering direction.

## Current Goal

The active goal is not "insert some quant/dequant ops into TTIR". The active
goal is:

- run real Qwen decode through our framework
- use our Triton hook and function substitution path
- get a real accuracy/performance frontier on a memory-bound decode workload
- do it on top of `torch.compile` / `Inductor`, not only eager mode

## Current Main Entry

The canonical real-model driver is:

- `runtime/examples/stage2a_qwen_model_demo.py`

This is the script to use for current experiments. Older one-off frontier sweep
scripts were removed once their logic was folded back into this driver.

For stage-labelled usage, the same driver is also exposed as:

- `runtime/examples/stage2c_qwen_inductor_demo.py`

See:

- `runtime/examples/docs/STAGE_MAP.md`

## Mainline vs Side Paths

### Mainline

The smallest current code path for "quantization through approxMLIR on a real
model" is:

- `runtime/examples/stage2c_qwen_inductor_demo.py`
- `runtime/examples/stage2a_qwen_model_demo.py`
- `runtime/examples/approx_quant_api.py`
- `runtime/examples/kernels/quant_kernels.py`

This is the mainline. If someone wants the minimum set of files to understand
or modify the current implementation, this is the set.

### Side Paths

The following files are support or investigation tools:

- `runtime/examples/sidepaths/stage2a_regime_characterizer.py`
- `runtime/examples/sidepaths/stage2a_fastpath_probe.py`
- `runtime/examples/sidepaths/stage2a_func_substitute_demo.py`
- `runtime/examples/sidepaths/dump_qwen_ttir.py`
- `runtime/examples/sidepaths/dump_triton_kernels.py`

These are useful for regime analysis, fast-path verification, and Stage 1
observation, but they are not the main implementation path.

## Current Supporting Files

The minimal current file set for quantization work is:

- `runtime/examples/stage2a_qwen_model_demo.py`
  - real Qwen model driver
  - exact/approx comparison
  - `torch.compile` / `Inductor` path
  - static-cache decode path
  - Triton function substitution hook wiring
- `runtime/examples/approx_quant_api.py`
  - torchao-like quantized-weight representation layer
  - module-level `quantize_(...)` entry for `nn.Linear`
- `runtime/examples/kernels/quant_kernels.py`
  - activation quant kernel
  - weight quant helpers
  - INT8 matmul kernels
  - substitute kernels used by the Triton hook
- `runtime/examples/sidepaths/stage2a_regime_characterizer.py`
  - decode-regime characterization for real target shapes
- `runtime/examples/sidepaths/stage2a_fastpath_probe.py`
  - microbenchmark to confirm whether a real bad-case shape is on a true INT8
    fast path
- `runtime/examples/sidepaths/stage2a_func_substitute_demo.py`
  - minimal function-substitution demo path
- `runtime/examples/sidepaths/dump_triton_kernels.py`
  - direct Triton kernel dump harness
- `runtime/examples/sidepaths/dump_qwen_ttir.py`
  - Qwen TTIR dump harness

## How Quantization Is Implemented Today

There are two layers: representation and execution.

### 1. Quantized Representation Layer

`runtime/examples/approx_quant_api.py` defines a small torchao-like interface.

Main types:

- `ApproxAffineQuantizedTensor`
  - holds quantized weight storage (`qdata`)
  - holds scale tensor (`scale`)
  - records `group_size`
  - records weight layout string
- `ApproxLinearActivationQuantizedTensor`
  - wraps a quantized weight representation
  - also stores `input_quant_func`
  - this is the current representation for dynamic-activation + INT8-weight
    linear paths

Main APIs:

- `quantize_linear_weight(...)`
- `quantize_(module, config, filter_fn)`

Current quantization configs:

- `ApproxInt8WeightOnlyConfig`
- `ApproxInt8DynamicActivationInt8WeightConfig`

This layer is the first step toward a torchao-like boundary: quantization is
described as a module/weight representation, not only as handwritten kernel
logic inside one wrapper.

### 2. Execution Layer

`runtime/examples/stage2a_qwen_model_demo.py` replaces selected Qwen linear
modules with Triton-backed wrappers and then runs exact/approx decode through
the same model shell.

Current wrapper types:

- `TritonDecodeLinear`
- `TritonFusedGateUpMLP`

Execution modes:

- `exact`
  - keep original floating-point path
- `packed_dequant`
  - pre-pack weights offline
  - unpack and dequantize inside the substitute kernel
  - useful as a weight-compression experiment
  - not a true low-bit compute path
- `fast_int8`
  - pre-quantize weights to INT8
  - quantize activations online with a fused Triton activation quant kernel
  - run INT8 matmul / INT8+dequant matmul kernels
  - this is the current main path

### 3. Function Substitution

The actual Triton rewrite path still goes through our existing hook/runtime
system.

Current structure:

- `stage2a_qwen_model_demo.py` builds a hook with `_build_hook(...)`
- the hook injects extra TTIR containing the approximate kernel
- `runtime/approx_runtime/triton_hook.py` merges the extra TTIR and runs the
  `TransformApproxPass`
- the pass applies `func_substitute`

Current substitute kernels in `quant_kernels.py` include:

- `approx_int8_matmul_kernel_1`
- `approx_int8_dequant_matmul_kernel_1`
- `approx_int8_dual_dequant_matmul_kernel_1`
- legacy `approx_matmul_kernel_1`

## What Is Actually On The Fast Path

Current important distinction:

- `packed_dequant`
  - saves weight storage / bandwidth
  - but the heavy op falls back to floating-point MMA
  - this is not the main path for speed
- `fast_int8`
  - the heavy op is intentionally driven toward true INT8 MMA
  - `stage2a_fastpath_probe.py` is the probe that confirms this on real bad
    shapes

So the current practical direction is:

- use `fast_int8` for real speed experiments
- treat `packed_dequant` as a reference weight-compression path, not the main
  performance path

## Current Inductor Status

The current real-model path has moved to `torch.compile` / `Inductor`.

Current settings in the demo:

- `TORCHINDUCTOR_COMPILE_THREADS=1`
- `TORCHINDUCTOR_WORKER_START=fork`

We intentionally removed the multi-thread compile workaround path after
debugging it.

Current debugging conclusion:

- `compile_threads > 1` with this source-Triton environment is not yet a stable
  mainline path
- the reproducible failure was in Inductor async compile workers, not in the
  model logic itself
- until that is solved properly, the canonical path is single-threaded compile

## Current Compile-Boundary Fixes

Recent work showed that a large fraction of "2B compile is too slow" was
actually caused by bad compile-boundary instability, not by model size alone.

Two concrete issues already identified in `Qwen3.5-2B`:

- `attention_mask` was switching between tensor and `None`
- `attention_mask` was also drifting across incompatible shapes/dtypes

The current demo already forces a more stable path, including:

- static cache
- fixed mask construction
- stable linear-attention mask override for Qwen3.5

This is still under active refinement.

## Current Practical Result

The current real positive result on the 2B line is:

- `Qwen3.5-2B`
- `out_proj`
- `last4`
- `fast_int8`
- `substitution=1`
- `Inductor=model`
- `StaticCache=1`

This path has already produced a real positive speedup in our framework.

The exact number is not treated as final because compile-boundary cleanup is
still in progress, but the line is now real: it is no longer blocked on eager
mode and it is no longer failing before function substitution is exercised.

## Current Engineering Direction

The current next step is:

1. keep the experiment inside `stage2a_qwen_model_demo.py`
2. continue stabilizing the `Inductor` compile boundary on `Qwen3.5-2B`
3. only then expand coverage to build a wider real frontier

The current order of priorities is:

- stable `Inductor` path
- real function substitution hits
- real speedup on 2B
- then broader coverage / frontier sweep

Not the other way around.
