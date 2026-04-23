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

## Side Paths

These files are support tools, not the active implementation:

- `sidepaths/stage2a_regime_characterizer.py`
  - decode-regime characterization for direct Triton shapes

- `sidepaths/stage2a_fastpath_probe.py`
  - microbenchmark for checking whether a candidate kernel reaches a fast path

- `sidepaths/dump_qwen_ttir.py`
  - old Stage 1 Qwen TTIR observation tool

- `sidepaths/dump_triton_kernels.py`
  - direct Triton kernel dump harness

For more detail, see:

- `QUANTIZATION_SPEC.md`
- `SIDEPATHS.md`
