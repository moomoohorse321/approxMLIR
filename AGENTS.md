# AGENTS.md

## Purpose

This repository is ApproxMLIR, an accuracy-aware compilation framework built on MLIR.

ApproxMLIR is not just a plugin or deployment artifact. Its main role is to provide a reusable substrate for expressing, lowering, transforming, and tuning approximations across different software stacks and abstraction levels.

Agents should treat the repository as supporting several equally legitimate kinds of work:

- core compiler research and development
- runtime and autotuning infrastructure
- frontend and backend integrations
- workload-specific projects built on top of ApproxMLIR

Examples of the last category include systems such as Triton-based kernel approximation or projects like ApproxLM that use ApproxMLIR as the approximation substrate for large-model workloads.

## Core Framing

ApproxMLIR exists to support unified approximation management across heterogeneous systems.

The important design ideas are:

- approximation should be represented explicitly, not buried in backend-specific ad hoc logic
- approximation metadata and control should be non-invasive with respect to existing dialects
- the same approximation interface should be reusable across ML and non-ML compilation flows
- backend-specific lowering should not redefine the meaning of the shared approximation abstractions

When in doubt, preserve these properties.

## What ApproxMLIR Provides

ApproxMLIR consists of:

- the `approx` dialect, which represents approximation scopes and decisions
- passes that lower approximation semantics and apply transformations
- `approx-opt`, a reusable optimizer for ApproxMLIR pipelines
- `approx-runtime`, which supports runtime control and autotuning flows
- frontends and integrations for different host ecosystems

Core dialect concepts include:

- `approx.knob`: marks an approximation region
- `approx.decide`: computes runtime approximation state
- `approx.try`: expresses try-check-recover safety behavior
- `approx.transform`: selects a transformation strategy and knob value
- `approx.yield`: region terminator
- `approx.util.annotation.*`: frontend annotations consumed by early passes

Approximation strategies currently include:

- `func_substitute`
- `loop_perforate`
- `task_skipping`

## Canonical Pipeline

The standard ApproxMLIR pass story is:

1. `emit-approx`
2. `emit-management`
3. `config-approx`
4. `transform-approx`
5. `finalize-approx`

Some flows also use `pre-emit-transform` before `transform-approx`.

The high-level expectation is:

- frontends emit annotations or approximation structure
- passes materialize approximation semantics
- transformations rewrite the underlying program
- approx ops are eliminated by the end of the pipeline
- downstream IR remains legal for the host compiler stack

## Main Work Modes

Before acting, identify which layer the task belongs to. Do not let one layer's constraints dominate the others by accident.

### 1. Core compiler work

This includes:

- dialect design
- pass implementation
- legality and lowering
- transformation correctness
- IR invariants
- regression tests

Optimize for semantic clarity and reusable abstractions. Prefer backend-neutral designs unless the task is explicitly integration-specific.

### 2. Runtime and tuning work

This includes:

- Python APIs
- configuration objects
- export and compilation glue
- runtime semantics
- OpenTuner integration
- workload-specific pipeline selection

Optimize for stable interfaces and clean separation between shared runtime logic and backend-specific execution paths.

### 3. Integration work

This includes integrating ApproxMLIR into systems such as:

- IREE / StableHLO / JAX
- Triton
- C/C++ flows through Polygeist or related toolchains

Optimize for preserving ApproxMLIR semantics while adapting to the host compiler's IR and pass APIs.

Do not assume the host compiler's quirks should leak back into the core ApproxMLIR model unless that change is deliberate and broadly beneficial.

### 4. Research extension work

This includes projects that build on ApproxMLIR rather than merely deploying it.

Examples:

- identifying approximation opportunities in kernels or whole workloads
- designing new approximation families
- generating or inferring annotations automatically
- constructing approximation search spaces
- evaluating quality-performance tradeoffs
- building systems like ApproxLM on top of ApproxMLIR

For this kind of work, treat ApproxMLIR as the reusable approximation substrate. Triton, JAX, C++, or other ecosystems are frontend and evaluation paths, not the definition of the project itself.

## Guidance For ApproxLM-Style Work

If a task is about approximating kernels for large models, identifying opportunities automatically, or generating new approximate variants, prefer the following framing:

- ask first whether the logic belongs in shared ApproxMLIR analysis or transformation infrastructure
- only make it Triton-specific when the problem is truly tied to Triton IR or Triton runtime behavior
- keep search-space construction and approximation semantics as general as possible
- treat workload evaluation as separate from the core representation of approximation

Good places for ApproxLM-style contributions may include:

- generic analysis passes
- annotation generation utilities
- new transform abstractions
- runtime support for evaluating candidate approximations
- thin frontend hooks for Triton or other kernel generators

Avoid prematurely hard-coding research ideas into one backend integration if they could live at a shared layer.

## How To Avoid Misleading Future Work

Keep these boundaries in mind:

- ApproxMLIR is not "the Triton plugin project"
- Triton is one important integration target, not the sole target
- JAX/IREE, C/C++, and future frontends should remain conceptually first-class
- deployment goals are local priorities, not repository-wide definitions
- backend-specific compatibility code should stay scoped to backend-specific layers where possible

If a task is ambiguous, classify it explicitly as one of:

- core compiler work
- runtime/tuning work
- integration work
- research extension work

Then optimize for that layer.

## Current Local Priority

On this machine, there is an active Triton deployment effort. The immediate operational target is:

- `runtime/examples/example_triton_wo_tuning.py` runs successfully on the Jetson machine

This is a local deployment milestone. It should guide debugging for Jetson setup work, but it should not override the broader ApproxMLIR framing above for unrelated tasks.

## Triton Integration Summary

Triton's rough pipeline is:

- Python AST -> TTIR -> TTGIR -> LLVM IR -> PTX/AMDGCN

ApproxMLIR integrates at the TTIR stage through an out-of-tree pass plugin.

Relevant repo locations:

- Triton plugin: `external-tools/approx-triton-plugin`
- Triton example target: `runtime/examples/example_triton_wo_tuning.py`
- Triton runtime hook code: `runtime/approx_runtime/triton_hook.py`
- Triton compiler bridge: `runtime/approx_runtime/triton_compiler.py`
- Pipeline selection logic: `runtime/approx_runtime/compiler.py`

The plugin is loaded with:

- `TRITON_PASS_PLUGIN_PATH=/path/to/libApproxTritonPlugin.so`

The Triton path does not use `approx-opt`; it drives passes through Triton's pass manager.

## Triton Pass Pipeline

The current Triton plugin registers:

1. `emit-approx`
2. `emit-management`
3. `config-approx`
4. `pre-emit-transform`
5. `transform-approx`
6. `finalize-approx`

`legalize-to-stablehlo` is not part of the Triton path.

For `func_substitute`, `pre-emit-transform` may be needed because Triton uses Triton-specific function/call ops and compatibility handling may be required before the generic transformation pass runs.

## Recommended Debug Order For Triton Deployment

If `runtime/examples/example_triton_wo_tuning.py` fails, check in this order:

1. Python imports:
   - `approx_runtime`
   - `triton`
   - `torch`
2. GPU availability:
   - `torch.cuda.is_available()`
3. Plugin build artifact exists:
   - `libApproxTritonPlugin.so`
4. Plugin path exported:
   - `TRITON_PASS_PLUGIN_PATH`
5. Triton can load the plugin and enumerate passes
6. TTIR hook is being installed
7. Approx annotations are injected into TTIR
8. The approx helper TTIR is available for `func_substitute`
9. The transformed TTIR still lowers through Triton successfully

Prefer confirming the failure stage with the example script before making broad changes.

## Setup Guide: ApproxMLIR with Triton Support

This section walks through building ApproxMLIR's Triton plugin from scratch. The end goal is running `runtime/examples/example_triton_wo_tuning.py`.

### Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | >= 3.9 | 3.10+ recommended |
| CMake | >= 3.20, < 4.0 | |
| Ninja | >= 1.11.1 | |
| C++17 compiler | GCC 11+ or Clang 14+ | |
| PyTorch | >= 2.0 | Must match your CUDA version. On Jetson, use NVIDIA's pip index |
| CUDA toolkit | >= 12.0 | For GPU execution |

### Step 1: Clone Triton

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
git checkout 3f0a0085a  # tested commit; newer may work but API may drift
git submodule update --init --recursive
```

### Step 2: Create a Python virtual environment

```bash
python3 -m venv /path/to/envs/triton-dev
source /path/to/envs/triton-dev/bin/activate
pip install --upgrade pip setuptools wheel
pip install pybind11>=2.13.1 ninja cmake
```

Install PyTorch into the venv:

```bash
# x86_64 Linux / Mac:
pip install torch

# Jetson (aarch64, JetPack 6+):
pip install torch --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
```

### Step 3: Build Triton from source with plugin support

Triton must be built with `TRITON_EXT_ENABLED=ON` so that `libtriton.so` exports symbols needed by external plugins. Without this flag, the plugin will fail to load at runtime.

```bash
# Triton downloads its own LLVM during build. Set a cache directory:
export TRITON_CACHE_PATH=/path/to/triton-cache

# Option A: pip install (simpler, handles LLVM download automatically)
cd triton/python
TRITON_BUILD_WITH_CCACHE=true pip install -e . \
  --config-settings=cmake.define.TRITON_EXT_ENABLED=ON

# Option B: cmake build (more control, useful for debugging)
mkdir -p /path/to/build/triton-cmake
cd /path/to/build/triton-cmake

# Triton bundles a specific LLVM. Download it first:
python triton/python/setup.py --llvm-download  # or let cmake find it
# The downloaded LLVM lands at: triton/.triton-home/.triton/llvm/llvm-<hash>-<platform>/

LLVM_SYSPATH=/path/to/triton/.triton-home/.triton/llvm/llvm-7f77ca0d-<platform>

cmake /path/to/triton \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_EXTERNAL_LIT=$(which lit || echo "") \
  -DTRITON_EXT_ENABLED=ON \
  -DCMAKE_CXX_FLAGS="-Wno-error=attributes"
ninja -j$(nproc) triton
```

> **Jetson note:** Use `ninja -j2` instead of `-j$(nproc)` — the Jetson Orin Nano has limited RAM and will OOM with more parallel jobs.

If you used Option B (cmake), make `triton` importable in Python:

```bash
# Create a .pth file so Python finds the Triton package
echo "/path/to/triton/python" > \
  $(python -c "import site; print(site.getsitepackages()[0])")/triton-dev.pth
```

Verify Triton is importable:

```bash
python -c "import triton; print(triton.__version__)"
```

### Step 4: Clone and build the ApproxMLIR plugin

```bash
git clone https://github.com/moomoohorse321/approxMLIR.git
cd approxMLIR
git submodule update --init --recursive
```

Triton downloads a pre-built LLVM during its build. The plugin must be built against that same LLVM — not a separately installed one. Find it at:

```bash
# The hash comes from triton/cmake/llvm-hash.txt
LLVM_DIR=/path/to/triton/.triton-home/.triton/llvm/llvm-7f77ca0d-<platform>/lib/cmake/llvm
MLIR_DIR=/path/to/triton/.triton-home/.triton/llvm/llvm-7f77ca0d-<platform>/lib/cmake/mlir
```

Replace `<platform>` with your system:
- `ubuntu-x64` for x86_64 Linux
- `ubuntu-arm64` for aarch64 / Jetson
- `macos-arm64` for Apple Silicon Mac

Build the plugin:

```bash
mkdir -p /path/to/build/approx-triton-plugin
cd /path/to/build/approx-triton-plugin

cmake /path/to/approxMLIR/external-tools/approx-triton-plugin \
  -G Ninja \
  -DMLIR_DIR=$MLIR_DIR \
  -DLLVM_DIR=$LLVM_DIR \
  -DTRITON_DIR=/path/to/triton \
  -DTRITON_BUILD_DIR=/path/to/build/triton-cmake \
  -DAPPROX_MLIR_DIR=/path/to/approxMLIR

ninja -j$(nproc)  # or ninja -j2 on Jetson
```

Output: `lib/libApproxTritonPlugin.so` (or `.dylib` on Mac).

### Step 5: Install the approx_runtime Python package

```bash
cd /path/to/approxMLIR/runtime
pip install -e .
```

### Step 6: Set environment variables and run

```bash
# Required: tells Triton where to find the plugin
export TRITON_PLUGIN_PATHS=/path/to/build/approx-triton-plugin/lib/libApproxTritonPlugin.so
export TRITON_PASS_PLUGIN_PATH=$TRITON_PLUGIN_PATHS

# Required if Triton was built with cmake (Option B), not pip:
export TRITON_BACKENDS_IN_TREE=1

# Run the example
python runtime/examples/example_triton_wo_tuning.py
```

Expected output:

```
kernel launch ok
ttir length: 6270
substitution active: True
max|hooked - exact|: ~0.02
max|hooked - approx_ref|: 0.0
```

> **Tip:** Add these exports to your venv's `bin/activate` script so you don't need to set them every time.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to load plugin ... undefined symbol` | Plugin built against wrong LLVM | Rebuild plugin using LLVM from `triton/.triton-home/.triton/llvm/` |
| `operation being parsed with an unregistered dialect` | `TRITON_EXT_ENABLED` was OFF when building Triton | Rebuild Triton with `-DTRITON_EXT_ENABLED=ON` |
| `0 active drivers` | cmake-built Triton can't find backends | `export TRITON_BACKENDS_IN_TREE=1` |
| `No module named 'triton'` | Triton not on Python path | Add a `.pth` file or `pip install -e python` |
| OOM during build on Jetson | Too many parallel compile jobs | Use `ninja -j2` |
| `libcudss.so.0: cannot open` (Jetson) | PyTorch needs libcudss at non-standard path | `export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/libcudss/12:$LD_LIBRARY_PATH` |

## Files Worth Reading First

For general ApproxMLIR understanding:

- `README.md`
- `include/`
- `lib/`
- `runtime/README.md`

For Triton-specific work:

- `runtime/examples/example_triton_wo_tuning.py`
- `runtime/approx_runtime/triton_hook.py`
- `runtime/approx_runtime/triton_compiler.py`
- `runtime/approx_runtime/compiler.py`
- `external-tools/approx-triton-plugin/README.md`
- `external-tools/approx-triton-plugin/pass/ApproxTritonPlugin.cpp`