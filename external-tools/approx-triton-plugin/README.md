# approx-triton-plugin

An out-of-tree Triton pass plugin that brings [ApproxMLIR](https://github.com/moomoohorse321/approxMLIR) function substitution to Triton GPU kernels.

## What this does

Loads ApproxMLIR's approximate computing passes into Triton's compilation pipeline via `TRITON_PASS_PLUGIN_PATH`. This lets you replace expensive math operations (e.g., `exp`, `sin`) in Triton kernels with cheaper approximations — without forking Triton.

## Prerequisites

- C++17 compiler
- CMake 3.20+
- LLVM/MLIR built as shared libraries
- Triton built from source (shared libs)

## Build

```bash
# 1. Build LLVM with shared libs (if you don't have it already)
# See triton-ext/.github/actions/build-llvm/action.yml for reference

# 2. Build Triton from source
cd /path/to/triton
pip install -e python  # or build with cmake directly

# 3. Build this plugin
cd approx-triton-plugin
mkdir build && cd build
cmake .. \
  -G Ninja \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DTRITON_DIR=/path/to/triton \
  -DTRITON_BUILD_DIR=/path/to/triton/python/build \
  -DAPPROX_MLIR_DIR=/path/to/approxMLIR
cmake --build .
```

The plugin shared library will be at `build/lib/libApproxTritonPlugin.so`.

## Usage

```bash
export TRITON_PASS_PLUGIN_PATH=/path/to/build/lib/libApproxTritonPlugin.so
python my_triton_kernel.py
```

## Project layout

```
approx-triton-plugin/
├── CMakeLists.txt              # Top-level build (pulls in approxMLIR + Triton)
├── pass/
│   ├── CMakeLists.txt          # Builds libApproxTritonPlugin.so
│   └── ApproxTritonPlugin.cpp  # Plugin entry point: registers approx passes with Triton
├── test/
│   └── func_substitute.mlir    # Skeleton test for function substitution
└── README.md
```

## How it works

See [ONBOARDING.md](./ONBOARDING.md) for the full technical background.

**Short version:** ApproxMLIR's passes lower to SCF control flow, which Triton already uses. The approx dialect operations are dialect-agnostic. The only work needed is adding pattern matching for `tt.extern_elementwise` (Triton's op for libdevice math calls) in the `transform-approx` pass.
