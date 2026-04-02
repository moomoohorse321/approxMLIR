# Onboarding Task: Port ApproxMLIR's C++ Flow from Polygeist to Reactant C++

**Author:** Hao | **Assignee:** Kevin | **Date:** 2026-04-02

---

## What This Is

ApproxMLIR is an MLIR-based framework for approximate computing. It has an `approx` dialect and 6 passes that transform annotated code. Read `approxMLIR/AGENTS.md` and `approxMLIR/runtime/README.md` for full context.

The C++ compilation flow currently depends on **Polygeist** (an older C-to-MLIR frontend, no longer maintained). Your job: **replace Polygeist with Reactant C++**, its successor from the same group (EnzymeAD).

---

## Current Flow (Polygeist)

Read `approxMLIR/runtime/approx_runtime/cpp_pipeline.py` -- this is the actual pipeline, not `run_exp.sh`.

```
C source (with // @approx comments)
  |
  v
cgeist -O0 -S  -->  base MLIR
  |
  v
Python parses @approx comments, injects annotation ops into MLIR (in-memory)
  |
  v
approx-opt-cpp (= polygeist-opt) runs passes:
  emit-approx -> emit-management -> config-approx -> transform-approx -> finalize-approx
  |
  v
Clean MLIR (no approx ops remaining)
```

**Where Polygeist is wired in:**

- `CMakeLists.txt:133` -- `add_subdirectory(approxMLIR)` into the Polygeist CMake tree
- `tools/polygeist-opt/polygeist-opt.cpp` -- registers `approxDialect` + 6 passes
- `tools/polygeist-opt/CMakeLists.txt` -- links `MLIRapprox`, `MLIRApproxTransforms`
- `approxMLIR/runtime/approx_runtime/toolchain.py` -- `APPROX_OPT_CPP` env var points to the opt binary; `cpp_pipeline.py` shells out to `cgeist` via `CgeistConfig`

The approx dialect and passes themselves (`approxMLIR/include/`, `approxMLIR/lib/`) are MLIR-standard -- not Polygeist-specific.

---

## New Target (Reactant C++)

Reactant C++ replaces cgeist/polygeist-opt with **stock clang + runtime-loaded plugins**:

```bash
clang -fplugin=ClangEnzyme-22.so -mllvm -raising-plugin-path=libRaise.so \
  -fno-exceptions -O1 input.c -o output
```

| Component | Repo | Build | Artifact |
|-----------|------|-------|----------|
| LLVM/Clang | `llvm/llvm-project` | CMake | `clang` |
| Enzyme plugin | `EnzymeAD/Reactant` (enzyme/) | CMake | `ClangEnzyme-XX.so` |
| Raising plugin | `EnzymeAD/Enzyme-JaX` | Bazel | `libRaise.so` |

`ClangEnzyme` intercepts compilation; `libRaise.so` performs the "raising" (LLVM IR -> high-level MLIR) that cgeist used to do. See the Reactant C++ install doc (provided separately) for build instructions.

---

## What You Need to Figure Out

1. **Where do approx passes run?** Options: as another `.so` plugin loaded by ClangEnzyme, compiled into an existing plugin, or kept as a standalone opt tool in a two-step flow. Look at `approxMLIR/external-tools/approx-triton-plugin/` -- we already package approxMLIR as a standalone `.so` for Triton's pass manager.

2. **How do annotations get into the IR?** Currently Python injects them into MLIR text between cgeist and approx-opt. If Reactant runs end-to-end inside clang, there may be no intermediate MLIR text. Options: dump MLIR then re-ingest, or inject annotations within the plugin pipeline.

3. **How to build?** ApproxMLIR must build standalone against the same LLVM that Reactant uses. The standalone mode in `approxMLIR/CMakeLists.txt` already exists but may need updates.

4. **How to update the runtime?** `cpp_pipeline.py` and `toolchain.py` shell out to cgeist and approx-opt-cpp. These need new equivalents.

---

## Approach

1. **Get Reactant C++ working** -- build LLVM, ClangEnzyme, libRaise. Compile a simple C file. Then try an approxMLIR benchmark C file (without approximation) to confirm it works as a cgeist replacement.

2. **Understand the plugin mechanism** -- how does ClangEnzyme load libRaise.so? Can you load additional pass plugins? Study the Reactant source and the Triton plugin reference.

3. **Integrate approxMLIR** -- build it standalone, wire the passes into the new pipeline, get one end-to-end test working (C source with `@approx` annotations -> approximated output).

4. **Validate** -- run the existing benchmarks (`bm25`, `lavaMD`, etc.) through the new flow and verify correctness.

---

## Key Files

| File | Why |
|------|-----|
| `approxMLIR/runtime/approx_runtime/cpp_pipeline.py` | **The current C++ flow** |
| `approxMLIR/runtime/approx_runtime/toolchain.py` | Toolchain config (opt paths, pipelines) |
| `tools/polygeist-opt/polygeist-opt.cpp` | How approx dialect/passes are registered now |
| `approxMLIR/CMakeLists.txt` | Standalone vs. in-tree build |
| `approxMLIR/external-tools/approx-triton-plugin/` | Reference: approxMLIR as a standalone `.so` |
| `approxMLIR/AGENTS.md` | Project context |
| `approxMLIR/runtime/README.md` | Runtime API and pass pipeline docs |

## Success Criteria

1. The benchmark scripts in `approxMLIR/runtime/examples/benchmark/` (`example_bm25_tuning.py`, `example_lavamd_tuning.py`, `example_kmeans_tuning.py`, etc.) run successfully using Reactant C++ instead of Polygeist
2. `approx_runtime` Python package works with the new toolchain -- `ar.compile_cpp_source()` produces correct transformed MLIR
3. Clear build/usage instructions documented

## Resources

- Reactant C++ install doc (provided separately)
- Reactant repo: https://github.com/EnzymeAD/Reactant
- Enzyme-JaX repo: https://github.com/EnzymeAD/Enzyme-JaX
- Example usage: https://github.com/wsmoses/Enzyme-GPU-Tests/pull/2

---
---

# Follow-Up Task: Apply ApproxMLIR to ILLIXR

After Reactant C++ integration is working, the next goal is to apply ApproxMLIR to **ILLIXR** (https://github.com/ILLIXR/ILLIXR) -- an open-source XR runtime with ~31 C++ plugins connected via a message-passing switchboard. Each plugin compiles independently via CMake, so we can approximate individual plugins and measure end-to-end quality/performance tradeoffs.

## XR System Subsystems (General Context)

An XR pipeline has these standard subsystems. Understanding them helps you evaluate what's worth approximating and how to measure quality.

| Subsystem | Compute Profile | Latency Sensitivity | Quality Metric |
|-----------|----------------|--------------------|--------------------|
| **VIO / SLAM** (visual-inertial odometry) | CPU-heavy: feature detection, Kalman filter / graph optimization | Critical (<100ms) | Pose drift (mm), tracking loss rate |
| **IMU integration** (sensor fusion) | CPU, sub-ms per sample, 200-1000Hz | Critical (<5ms) | Orientation accuracy vs. ground truth |
| **Scene reconstruction** | GPU-heavy: depth estimation, voxel fusion | Low-medium | Mesh completeness, depth RMSE |
| **Hand/eye/body tracking** | GPU: CNN inference + CPU post-processing | Medium (50-100ms ok) | Keypoint accuracy, jitter |
| **Rendering** | GPU-bound (rasterization/shading) | Medium (frame drops if missed) | FPS, visual fidelity |
| **Timewarp / reprojection** | GPU, <5ms, runs just before vsync | Critical (<2ms) | Motion-to-photon latency |
| **Audio spatialization** | CPU: HRTF convolution, FIR filtering | Medium (50ms ok) | Spectral distortion, localization error |
| **Network / offloading** | I/O-bound | Critical (30-50ms RTT) | Codec artifacts, latency |

## ILLIXR Plugin Reality Check

**Most ILLIXR plugins are thin wrappers around third-party libraries.** Source-level approximation with ApproxMLIR requires C++ code we can actually compile and annotate. Here's what the source code actually shows:

### Plugins With No ILLIXR Source Code (pure library wrappers -- cannot approximate)

| Plugin | Wraps | Plugin LOC |
|--------|-------|------------|
| `openvins` | OpenVINS (external git, full VIO system) | 0 |
| `orb_slam3` | ORB-SLAM3 + g2o + Sophus + DBoW2 (external git) | 0 |
| `hand_tracking` | MediaPipe hand tracking (external git); plugin is viewer UI only | 0 (525 for viewer) |
| `hand_tracking_gpu` | Same + CUDA | 0 |
| `lighthouse` | LibSurvive (external git) | 125 (coordinate transform only) |
| `ada/infinitam` | InfiniTAM (external git, 3D reconstruction) | 0 |
| `audio_pipeline` | libspatialaudio (external git) | 0 (cmake include only) |

### Plugins With ILLIXR's Own C++ (approximation candidates)

| Plugin | What It Does | LOC | Compute Type | Hot Loops |
|--------|-------------|-----|-------------|-----------|
| `rk4_integrator` | RK4 numerical integration of IMU (quaternion dynamics, position, velocity) | 208 | CPU, pure math (Eigen) | 4-stage RK4 kernel per IMU sample; skew-symmetric matrix construction; quaternion propagation |
| `ada/scene_management` | Spatial hash table for incremental mesh updates (voxel block indexing, face/vertex management) | 897 | CPU | Hash table ops, mesh face iteration and deletion, vertex append |
| `timewarp_gl` | Async reprojection (OpenGL) -- CPU computes transform matrices, GPU does distortion | 967 | GPU-bound, CPU matrix math | Per-frame timewarp transform calculation, distortion mesh generation |
| `timewarp_vk` | Same as above, Vulkan | 952 | GPU-bound | Same pattern |
| `openwarp_vk` | 6-DOF reprojection with dual Vulkan render passes | 1652 | GPU-bound | Mesh generation, inverse projection matrices, dual-pass command buffers |
| `gtsam_integrator` | Thin wrapper around GTSAM's PreintegratedCombinedMeasurements | 343 | CPU | Loop over IMU samples calling `pim_obj_->integrateMeasurement()` (library call); custom One-Euro filter + IMU interpolation |

### Plugins That Are Pure I/O (nothing to approximate)

`offline_cam`, `offline_imu`, `record_imu_cam`, `record_rgb_depth`, `webcam`, `realsense`, `zed`, `depthai`, `openni` -- camera drivers or data replay. `offload_rendering_server/client` and `offload_vio` wrap FFmpeg/GStreamer codecs. `tcp_network_backend` is socket I/O. `debugview` is ImGui UI.

## Approximation Opportunities, Ranked

Given the source code reality, here are the actual candidates:

### Tier 1: Best Targets

**`rk4_integrator`** (208 LOC, all custom C++)
- The RK4 kernel in `runge-kutta.hpp` (~60 lines) computes 4 stages of quaternion/position/velocity integration per IMU sample. This is textbook approximable numerical computation.
- **Loop perforation:** Skip IMU samples in the propagation loop (lines 94-113 iterate over every sample pair).
- **Function substitution:** Replace 4-stage RK4 with Euler integration (1 stage).
- **Quality metric:** Pose drift vs. ground truth (ILLIXR has `ground_truth_slam` plugin for comparison, plus `gtsam_integrator` and `passthrough_integrator` as baselines).
- **Why first:** Small, self-contained, pure C++ with Eigen only, clear quality metrics, multiple comparison baselines already in ILLIXR.

**`ada/scene_management`** (897 LOC, custom spatial hash)
- `spatial_hash.cpp` (570 LOC) implements voxel block hashing, mesh face iteration/deletion, and vertex appending.
- **Loop perforation:** Skip voxel blocks during mesh cleaning (`delete_mesh` iterates over block ranges) or reduce frequency of mesh rebuild.
- **Task skipping:** Skip mesh update cycles entirely when scene change is below threshold.
- **Quality metric:** Mesh completeness, reconstruction error vs. ground truth (ScanNet datasets available via `ada/offline_scannet`).

### Tier 2: Possible but Harder

**`gtsam_integrator`** (343 LOC, mostly library calls)
- The IMU integration loop calls GTSAM's `integrateMeasurement()` -- that's a library call we can't touch. But the surrounding code (IMU sample selection/interpolation at lines 219-265, One-Euro filtering) is custom ILLIXR code.
- **Function substitution:** Replace the GTSAM integration with a simpler custom integrator.
- **Task skipping:** Skip the One-Euro filter or reduce IMU interpolation precision.
- Harder because the expensive part is inside GTSAM.

**`timewarp_gl` / `timewarp_vk`** (~950 LOC each, GPU-bound)
- The CPU-side computation (transform matrix calculation, distortion mesh setup) is custom ILLIXR code, but it's a small fraction of the per-frame cost -- the GPU does the real work.
- ApproxMLIR operates at the C++ source level, so GPU shader code is out of scope.
- Could approximate the CPU-side matrix math (reduced precision, skip interpolation), but impact would be small.

### Not Viable for Source-Level Approximation

Everything in the "pure library wrapper" and "pure I/O" categories above. For `openvins`, `orb_slam3`, `hand_tracking`, and `ada/infinitam`, approximation would require modifying the upstream libraries themselves -- a different (larger) project.

## Where to Start

1. **`rk4_integrator`** -- 208 LOC of pure numerical C++, the ideal first target.
2. **`ada/scene_management`** -- larger but still self-contained custom code with clear quality metrics.
3. If the pipeline is validated on those two, consider `gtsam_integrator`'s custom code portions.

## What Success Looks Like

1. `rk4_integrator` compiled through Reactant C++ + ApproxMLIR with `@approx` annotations on the RK4 propagation loop
2. End-to-end ILLIXR runs with the approximated integrator, producing pose drift measurements against `ground_truth_slam`
3. Reproducible workflow documented: annotate plugin C++ -> compile with ApproxMLIR -> swap `.so` back into ILLIXR build

## Resources

- ILLIXR docs: https://illixr.github.io/ILLIXR/latest/
- ILLIXR repo: https://github.com/ILLIXR/ILLIXR
- ILLIXR plugin docs: https://illixr.github.io/ILLIXR/latest/illixr_plugins/
- ILLIXR paper: https://rsim.cs.illinois.edu/Pubs/IISWC_2021_ILLIXR.pdf
