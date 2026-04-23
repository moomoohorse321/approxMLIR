# Side Paths

This folder contains support scripts for the quantization work, not the main
implementation path.

Use the mainline first:

- `../sglang_quant/probe_sglang_triton_dump.py`
- `../sglang_quant/sweep_sglang_quant.py`

Side-path scripts are for:

- regime characterization
- fast-path probing
- TTIR dumping / observation
- historical comparison against earlier direct-Triton or TorchInductor probes

Current side paths:

- `stage2a_regime_characterizer.py`
- `stage2a_fastpath_probe.py`
- `dump_qwen_ttir.py`
- `dump_triton_kernels.py`
- `stage2c_validate_mlir_quant_path.py`

These are not the active Qwen quantization implementation.
