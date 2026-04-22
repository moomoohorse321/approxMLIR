# Side Paths

This folder contains support scripts for the quantization work, not the main
implementation path.

Use these when you need:

- regime characterization
- fast-path probing
- TTIR dumping / observation

If you want the minimal real-model quantization path, do not start here.
Start with:

- `../stage2c_qwen_inductor_demo.py`
- `../stage2a_qwen_model_demo.py`
- `../approx_quant_api.py`
- `../kernels/quant_kernels.py`

See also:

- `STAGE_MAP.md`
- `QUANTIZATION_SPEC.md`
