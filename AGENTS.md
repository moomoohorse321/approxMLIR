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
