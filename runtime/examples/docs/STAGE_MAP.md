# Stage Map

This directory now distinguishes between a small mainline and several side-path
tools.

## Mainline

These files are the minimal path for "run quantization through approxMLIR on a
real model".

- `stage2c_qwen_inductor_demo.py`
  - preferred stage-labelled entrypoint
  - current real-model quantization demo
  - runs Qwen decode through `Inductor`, Triton hook, and function
    substitution

- `stage2a_qwen_model_demo.py`
  - implementation file behind the stage-2(c) alias above
  - keep this if you are editing the real driver

- `approx_quant_api.py`
  - quantized module/weight representation layer
  - current torchao-like entry boundary

- `kernels/quant_kernels.py`
  - Triton kernels for activation quant, INT8 matmul, dequant matmul, and
    substitute kernels

If someone wants the shortest path to understand the current quantization
implementation, start with exactly these four files.

## Side Paths

These are still useful, but they are support or investigation tools rather
than the mainline implementation.

- `sidepaths/stage2a_regime_characterizer.py`
  - decode-regime characterization tool
  - used to confirm whether a target decode path is actually memory-bound

- `sidepaths/stage2a_fastpath_probe.py`
  - fast-path probe
  - microbenchmark for real bad-case shapes
  - used to verify whether a kernel reaches true INT8 compute

- `sidepaths/dump_qwen_ttir.py`
  - Stage 1 observation tool for Qwen TTIR

- `sidepaths/dump_triton_kernels.py`
  - Stage 1 observation tool for direct Triton kernels

## Why The Alias Exists

The real implementation converged into one maintained Qwen driver instead of
multiple drifting frontier scripts. To preserve stage semantics without
duplicating logic, the active Inductor path is exposed as
`stage2c_qwen_inductor_demo.py`.

For the side-path overview, see:

- `SIDEPATHS.md`
