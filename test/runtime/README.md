`approx-runtime` converts JAX program to MLIR, optimizing it using `approx-opt` and deploying it using IREE.

`test_kernels.py` contains unit test for the frontend (JAX to MLIR conversion) and backend (IREE deployment).

`test_opt.py` contains unit test for the `approx-opt` optimization passes.