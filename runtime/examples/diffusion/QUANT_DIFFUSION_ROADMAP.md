# Quant Diffusion Roadmap

This directory is currently an exact diffusion application benchmark scaffold.
The target is a `quant_diffusion` counterpart to `sglang_quant`: an
application-level image generation benchmark that discovers quantization
opportunities, binds them through approxMLIR-controlled function substitution,
and accepts candidates only when the end-to-end image generation QoS contract
passes.

## Current Status

Implemented and tested:

- `app.py`: CLI runner for inspect and exact application runs.
- `dataset.py`: JSONL request schema, validation, checksums, and stable request
  IDs.
- `exact_backend.py`: lazy Diffusers Stable Diffusion exact backend.
- `qos.py`: output records, image validity checks, latency/memory summary, and
  hard QoS guards.
- `data/smoke_requests.jsonl`: four real PartiPrompts text-to-image requests.
- `tests/test_diffusion_exact.py`: dependency-light unit tests using a fake
  backend.
- `.gitignore`: ignores diffusion outputs, staged models, goldens,
  calibration artifacts, and Python cache files.
- `site_inventory.py`: deterministic linear/conv candidate inventory.
- `approx_manifest.py`: legal W8A16/W4A16 binding manifest and validation.
- `approx_kernels.py`: W8A16 and groupwise W4A16 Triton linear kernels plus
  same-ABI approx twins.
- `memory_bound.py`: deterministic large prompt-conditioning tower for a
  latency-memory-bound serving profile.
- `substitute.py`: runtime Diffusers linear binding and approxMLIR hook setup.
- `approx_backend.py`: approximate backend that wraps the exact Diffusers path.
- `compare_outputs.py`: exact-vs-approx image comparison.
- `sweep_diffusion.py`: exact-vs-approx frontier sweep.

Verified:

- Unit tests pass under `/home/hao/triton-trails/.venv/bin/python`.
- `app.py --mode inspect` sees Torch, Diffusers, Pillow, CUDA, and the RTX 4060
  Laptop GPU in that venv.
- A previous exact laptop smoke run completed all four SD 1.5 requests with
  accepted QoS, p50 latency about 3.38 seconds, p95 latency about 4.01 seconds,
  and peak CUDA memory about 2.81 GB.
- The current latency-memory-bound profile uses 8 low-resolution diffusion
  requests plus a deterministic prompt-conditioning tower. Each request scores
  32 candidates with a `4096 -> 32768` projection over 128 tower steps, so the
  profile uses small-batch parallel inference instead of one input repeated
  thousands of times. The sweep binds `memory_bound_conditioner.proj`, drops
  the original fp weight, and uses approxMLIR function substitution for W8A16
  and W4A16 Triton kernels. The W4A16 kernel uses a single-dot unpack path
  tuned at `BLOCK_M=32`, `BLOCK_N=64`, `BLOCK_K=32`, and `GROUP_K=128`. On the
  RTX 4060 Laptop probe, exact p50 was about 259.61 ms, W8A16 p50 was about
  202.67 ms (`1.28x`), and W4A16 p50 was about 195.39 ms (`1.33x`). W4A16
  passed QoS with conditioner-output relative error below 0.73% and reduced
  peak CUDA memory by about 8.0%; W4A16 is now the selected latency frontier.

Not implemented yet:

- Persisted calibration collection for W4/AWQ beyond the current load-time
  groupwise W4 artifact materialization.
- Broader W8/W4 kernels for native UNet/text-encoder diffusion shapes.
- Exact-vs-approx comparison against persisted golden output directories.
- Pruning path.

## Target Directory Shape

The complete directory should grow toward:

```text
runtime/examples/quant_diffusion/
  README.md
  app.py
  dataset.py
  exact_backend.py
  qos.py
  site_inventory.py
  approx_manifest.py
  calibration.py
  approx_kernels.py
  substitute.py
  compare_outputs.py
  sweep_diffusion.py
  data/
    smoke_requests.jsonl
    latency_memory_bound_requests.jsonl
    eval_requests.jsonl
  tests/
    test_diffusion_exact.py
    test_site_inventory.py
    test_manifest.py
    test_substitute.py
    test_compare_outputs.py
    test_sweep_acceptance.py
```

It can either be renamed from `runtime/examples/diffusion` once the
substitution path lands, or a new `quant_diffusion` directory can be created
with the exact benchmark moved in. The important boundary is conceptual:
`diffusion` is currently the exact app benchmark; `quant_diffusion` should mean
the exact benchmark plus approxMLIR quantization and policy search.

## Implementation Plan

### 1. Freeze the exact baseline contract

Promote the current exact scaffold from draft to baseline:

- Commit or otherwise preserve the current exact benchmark files.
- Keep `local_files_only=True` as the benchmark default.
- Regenerate exact goldens for the pinned model, dataset, profile, precision,
  scheduler, step count, and seed policy.
- Store only manifests, checksums, and small fixtures in git; keep images and
  model snapshots in ignored staging directories.
- Extend the runner to report stage timing for text encoding, denoising, VAE
  decode, postprocess, and scoring instead of only total latency.

Acceptance for this phase:

- Exact unit tests pass.
- Exact laptop smoke runs from staged local weights with no network.
- Exact goldens can be reproduced or rejected with a clear reproducibility
  error.

### 2. Add site inventory

Implement `site_inventory.py` to inspect the loaded exact Diffusers pipeline and
emit stable candidate sites:

- Prompt-conditioning tower `Linear` modules for the latency-memory-bound
  serving profile.
- Text encoder `Linear` modules.
- UNet cross-attention and self-attention projections.
- UNet feed-forward `Linear` modules.
- UNet convolution blocks.
- VAE decoder convolutions.

Each site record should include a stable `site_id`, host module path, operation
kind, input/output channel or feature shape, dtype, weight checksum, and
matchers. This should follow the quantization integration spec rather than
using host names as permanent identity.

Acceptance for this phase:

- Inventory is deterministic across two loads of the same model.
- Tests catch duplicate IDs, stale shapes, dtype mismatch, and checksum
  mismatch.
- Inventory output is written into the run manifest.

### 3. Define the diffusion quantization manifest

Implement `approx_manifest.py` using the ApproxMLIR quantization integration
contract:

- `SiteInventory`
- `QuantPlan`
- `ArtifactRecord`
- `QuantManifest`
- selected bindings
- tuning evidence
- negative evidence

Start with legal plans for:

- `exact`
- `linear_w8a16`
- `linear_w4a16_awq`
- `conv_w8a16` or defer convolution with explicit unsupported status

Acceptance for this phase:

- Strict manifest validation rejects malformed bindings.
- Non-strict mode falls back to exact with explicit telemetry.
- Serving-time code never performs tuning or artifact construction.

### 4. Build the first same-ABI W8A16 linear substitution

Mirror the successful `sglang_quant` split:

- Runtime adapter identifies selected Diffusers `Linear` modules after weights
  load.
- Load-time materialization quantizes weights to int8 and registers qweight and
  scale buffers.
- Online dispatch calls a quantized wrapper for the selected site.
- approxMLIR function substitution swaps the online Triton kernel body with a
  same-signature substitute.

Initial target order:

1. UNet attention projections, because denoising dominates latency.
2. UNet feed-forward linear layers.
3. Text encoder linear layers only if they show material latency.

Acceptance for this phase:

- Exact mode remains byte-for-byte unchanged at the output schema level.
- W8A16 selected sites report expected hit/miss counts.
- Missing artifact or unsupported shape produces explicit exact fallback.
- Unit tests cover adapter binding with fake modules.

### 5. Add calibration and W4A16 candidates

Port the SGLang W4 lesson, but only for sites that survive W8 profiling:

- Collect activation absmax or representative activation statistics on a
  calibration split disjoint from evaluation.
- Materialize AWQ-style W4 artifacts at load time or prebuild time.
- Keep activations fp16/bf16 at runtime.
- Start with linear layers; add convolution W4 only after W8 or pruning proves
  convolution is worth targeting.

Acceptance for this phase:

- Calibration artifacts have checksums, source split fingerprints, and build
  parameters.
- W4 bindings are rejected when calibration coverage is missing.
- W4 is evaluated only against exact goldens and QoS thresholds.

### 6. Add exact-vs-approx image quality comparison

Implement `compare_outputs.py` and extend `qos.py`:

- Compare approximate outputs to exact goldens by request ID.
- Add SSIM and either LPIPS or another local perceptual metric.
- Add optional CLIP prompt-image alignment when a local scorer is staged.
- Treat missing required metrics as failure, not pass.

Acceptance for this phase:

- Approximate runs cannot be accepted with only nonblank-image heuristics.
- Quality thresholds are profile-specific and recorded in the manifest.
- The output record clearly separates exact-only metrics, optional metrics, and
  required acceptance metrics.

### 7. Add the sweep and acceptance gate

Implement `sweep_diffusion.py` as the diffusion counterpart of
`sweep_sglang_quant.py`:

- Run exact baseline first.
- Generate or consume candidate manifests.
- Run candidates sequentially to avoid GPU memory interference.
- Parse run summaries, QoS summaries, substitution telemetry, and image
  comparison metrics.
- Write `results.json` with speedup, memory deltas, QoS verdict, selected
  bindings, and negative evidence.

Default acceptance:

- all hard guards pass;
- required exact-vs-approx quality metrics pass;
- p50 or p95 latency improves by at least 5 percent;
- no silent site misses.

### 8. Add pruning after quantization is measurable

Pruning should come after the quantization/evaluation loop is credible:

- Start with structured channel or attention-head pruning only when output ABI
  remains stable through a wrapper.
- Require exact fallback and per-site telemetry.
- Keep pruning plans in the same manifest/evidence system as quantization.

Acceptance for this phase:

- Pruned candidates preserve output schema and pass exact-vs-approx QoS.
- Rejected pruning attempts write negative evidence with site, plan, and
  failure reason.

## Immediate Next Patch

The next concrete patch should broaden the latency frontier beyond the
prompt-conditioning tower: persist calibration artifacts for W4/AWQ, then
sweep native UNet/text-encoder sites with the tiled W8A16/W4A16 kernels and
negative evidence for shapes that remain compute-bound.
