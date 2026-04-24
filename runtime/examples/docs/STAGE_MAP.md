# Stage Map

This directory now has one mainline quantization implementation: the SGLang
serving path.

## Mainline

Use these files for the current "run Qwen quantization through approxMLIR"
path:

- `sglang_quant/probe_sglang_triton_dump.py`
  - single-run SGLang driver
  - installs the Triton hook in SGLang worker subprocesses
  - can run exact or approximate generation
  - reports e2e latency, TTIR dumps, and substitution hit counts

- `sglang_quant/sweep_sglang_quant.py`
  - sequential frontier sweep over exact and approximate SGLang configs
  - keeps runs serialized to avoid overloading the GPU

- `sglang_quant/approx_quant_patch.py`
  - monkeypatches SGLang linear layers at runtime
  - prequantizes selected weights outside the kernel
  - redirects selected call sites to the Triton quantized kernels

- `sglang_quant/approx_kernels.py`
  - Triton W8A16/W8A8 kernels and same-ABI substitute kernels

- `sglang_quant/bootstrap/sitecustomize.py`
  - child-process bootstrap used because SGLang runs model workers out of
    process

## Current Result

The useful PoC target is Qwen3.5-2B with SGLang CUDA graph enabled:

- exact SGLang/Triton batch=4, 8-token e2e median: about `0.212s`
- `gate_up_proj` W8A16 function-substitute median: `0.150s` to `0.166s`
- repeated runs show about `1.27x` to `1.41x` e2e speedup
- substitution hit rate was `100%` in the repeated best-config runs

This is a selective-coverage frontier. `gate_up_proj` is the useful target;
adding `down_proj` is not a Pareto point with the current kernel.

## Retired Material

The old direct-Triton and TorchInductor side-path probes were removed from the
current runtime examples. They were useful during early target discovery, but
they are not part of the SGLang quantization mainline or its verification path.

For more detail, see:

- `QUANTIZATION_SPEC.md`
