# Diffusion Application Benchmark Spec

Status: exact laptop benchmark implemented; W8A16 approxMLIR substitution and
latency-memory-bound sweep implemented.

This directory will host an end-to-end diffusion application benchmark for
approxMLIR. It follows the `runtime/examples/sglang_quant` pattern: first build
an exact, tested application benchmark with real input data and a strict
quality-of-service contract; then use approxMLIR to continuously optimize
selected functions through substitution with quantization and pruning.

The design is also aligned with the Curiosity paper draft at
`/home/hao/iree-build/iree/build/ApproxGen/src/main.tex`: a local kernel or
model-level speedup is not accepted unless the downstream application verdict
passes. Optimization knowledge that repeatedly helps should become durable
runtime/compiler infrastructure, not a one-off script.

## Goal

Build a benchmarkable image-generation application, not just a diffusion model
microbenchmark.

The first milestone is an `exact` benchmark:

- real request dataset as input
- deterministic exact pipeline
- application-level output artifacts
- repeatable latency, memory, and quality metrics
- tests for data loading, output schema, and QoS computation

The second milestone is an approxMLIR optimization loop:

- inventory optimizable functions in the diffusion application
- propose quantization and pruning substitutions
- bind substitutions through approxMLIR-controlled manifests
- run exact-vs-approx comparisons on the same application workload
- accept only candidates that improve performance while satisfying QoS

## Non-Goals

- No model-only benchmark that times one UNet, attention, convolution, or VAE
  function outside the application.
- No synthetic prompt-only benchmark as the main result. Synthetic smoke inputs
  are allowed only for CI plumbing.
- No accepting a kernel speedup without the final image-generation task verdict.
- No network downloads during benchmark runs. Models, datasets, and metric
  assets must be staged and pinned before the run starts.
- No benchmark-specific workaround for a compiler/runtime bug. If approxMLIR,
  IREE, Triton, or a runtime integration is the root cause, fix that layer.

## Application Under Test

The application is a batch image-generation service for campaign-style image
requests. Each request describes what image to generate, optional layout or
subject conditioning, safety constraints, output dimensions, and the random seed
policy. The application executes the full production-like pipeline:

1. load a request JSONL shard
2. validate request schema and asset checksums
3. normalize prompts and negative prompts
4. load exact model components from pinned local paths
5. run text encoding
6. run denoising with the configured scheduler and seed
7. run VAE decode
8. run post-processing, resize, and file encoding
9. optionally run the latency-memory-bound prompt-conditioning tower
10. run quality, safety, and validity scoring
11. write images, per-request records, and aggregate QoS summaries

The exact application is the reference implementation. Approximate candidates
must preserve the same request schema, output schema, seed policy, and scoring
pipeline.

## Exact Baseline

The `exact` mode is the reference application mode.

Required properties:

- uses pinned local model snapshots
- runs without approxMLIR substitutions
- uses deterministic seeds per request
- fixes scheduler, step count, resolution, dtype, and device policy
- records model revision, model hashes, dataset hashes, software versions, GPU
  name, driver/runtime versions, and approxMLIR commit
- writes a complete application record for every request

Initial exact backend:

- laptop profile: `diffusers` Stable Diffusion 1.5 or another pinned local
  512px diffusion pipeline
- full profile: SDXL base or another pinned local higher-resolution diffusion
  pipeline
- fp16 or bf16 inference on CUDA
- no quantized weights
- no pruned modules
- no approximate kernels

The exact backend may later gain IREE or Triton lowering paths, but exact mode
must remain semantically the reference app path.

## Current Entrypoints

Inspect the benchmark contract and local environment:

```bash
python3 runtime/examples/diffusion/app.py --mode inspect
```

Run the exact laptop smoke profile after staging a local Diffusers model:

```bash
APPROX_DIFFUSION_MODEL=/path/to/local/stable-diffusion-v1-5 \
python3 runtime/examples/diffusion/app.py \
  --mode run \
  --device cuda \
  --precision float16 \
  --enable-attention-slicing
```

By default the runner uses `local_files_only=True`; pass `--allow-download`
only during an explicit model-staging run, not during benchmark measurement.

Dependency-light tests:

```bash
python3 -m unittest discover -s runtime/examples/diffusion/tests -v
```

Run the current W8A16 latency-memory-bound candidate:

```bash
PYTHONPATH=$PWD/triton/python:$PWD/approxMLIR/runtime:$PYTHONPATH \
TRITON_PASS_PLUGIN_PATH=$PWD/approxMLIR/external-tools/approx-triton-plugin/build/lib/libApproxTritonPlugin.so \
python3 runtime/examples/diffusion/app.py \
  --mode run \
  --execution-mode approx \
  --profile latency_memory_bound \
  --device cuda \
  --precision float16 \
  --enable-attention-slicing \
  --quant-plan w8a16 \
  --quant-target memory_bound_conditioner \
  --quant-block-m 32 \
  --quant-block-n 128 \
  --drop-original-weights \
  --approx-use-substitute
```

Run the W4A16 groupwise candidate:

```bash
PYTHONPATH=$PWD/triton/python:$PWD/approxMLIR/runtime:$PYTHONPATH \
TRITON_PASS_PLUGIN_PATH=$PWD/approxMLIR/external-tools/approx-triton-plugin/build/lib/libApproxTritonPlugin.so \
python3 runtime/examples/diffusion/app.py \
  --mode run \
  --execution-mode approx \
  --profile latency_memory_bound \
  --device cuda \
  --precision float16 \
  --enable-attention-slicing \
  --quant-plan w4a16_awq \
  --quant-target memory_bound_conditioner \
  --quant-block-m 32 \
  --quant-block-n 64 \
  --quant-block-k 32 \
  --quant-group-k 128 \
  --drop-original-weights \
  --approx-use-substitute
```

This profile keeps image generation small but adds a large prompt-conditioning
projection (`4096 -> 32768`) as part of the application. Each request evaluates
32 conditioning candidates through a 128-step scoring tower, which is a small
parallel serving shape rather than a single scalar input repeated thousands of
times. The W8A16 and W4A16 paths bind that site, drop the original fp weight,
and use approxMLIR function substitution for the online Triton kernel body.

On the RTX 4060 laptop probe, the latency-memory-bound sweep accepted the
frontiers by latency, not memory. With exact p50 at about `259.61 ms`, W8A16
p50 was about `202.67 ms` (`1.28x`) and the optimized W4A16 p50 was about
`195.39 ms` (`1.33x`). W4A16 reduced peak CUDA memory by about `8.0%`; its
conditioner output summary passed with max relative error about `0.73%`.
W4A16 is now the selected latency frontier in the default sweep.

Run the default exact-vs-approx sweep:

```bash
PYTHONPATH=$PWD/triton/python:$PWD/approxMLIR/runtime:$PYTHONPATH \
TRITON_PASS_PLUGIN_PATH=$PWD/approxMLIR/external-tools/approx-triton-plugin/build/lib/libApproxTritonPlugin.so \
OUT_DIR=/tmp/approx_diffusion_sweep \
python3 runtime/examples/diffusion/sweep_diffusion.py
```

## Laptop Profile

The first implementation should be friendly to a single RTX 4060 laptop GPU.
Assume constrained VRAM unless the runner measures otherwise.

Default laptop settings:

- model family: Stable Diffusion 1.5-class text-to-image pipeline
- resolution: `512x512`
- batch size: `1`
- inference steps: `20`
- guidance scale: `7.0`
- dtype: `fp16`
- warmup runs: `1`
- measured repeats: `3`
- output format: PNG
- LoRA training: disabled
- ControlNet/layout conditioning: optional after text-to-image is stable

Memory policy:

- stage models before the benchmark run
- load one pipeline at a time
- use attention slicing or memory-efficient attention only if it is part of the
  exact configuration and recorded in `manifest.json`
- enable sequential CPU offload only as a separate `laptop_offload` profile,
  because it changes the latency contract
- fail a candidate if peak CUDA memory exceeds the profile ceiling

Laptop smoke profile:

- `4` text-to-image requests from real public prompt data
- deterministic seeds
- no personalization training
- no large safety or aesthetic model unless it fits without offload
- quality metrics that require large auxiliary models may be skipped only if
  `qos_summary.json` marks them unavailable and the acceptance rule does not
  depend on them

Latency-memory-bound profile:

- `8` text-to-image requests at `64x64` with `2` denoise steps, so image
  generation remains an application stage but does not swamp the conditioning
  experiment.
- A deterministic prompt-conditioning tower with a large projection
  (`4096 -> 32768`), 32 candidates per request, and depth 128.
- One warmup run before measured records so Triton compilation and first-use
  setup are not counted as serving latency.
- Acceptance is latency-only for performance: p50 or p95 must improve by at
  least 5 percent while image QoS and conditioner-output checks pass.

Laptop evaluation profile:

- `16` text-to-image requests
- optional `4` layout-conditioned requests after the layout path is implemented
- exact golden images generated once per pinned model/dataset/profile
- approximate candidates compared against those exact goldens

The full SDXL profile is a later target. It should not block the first exact
application benchmark or the first approxMLIR substitution experiment.

## Real Dataset Profile

The benchmark uses real request data, split into small smoke and larger
evaluation profiles.

Initial real smoke profile:

- campaign text-to-image requests from public prompt datasets such as
  PartiPrompts
- layout-conditioned requests with real prompts plus committed or staged edge,
  depth, or segmentation maps
- optional personalization requests with real subject images from a pinned
  public example set

The smoke profile should be small enough for one GPU CI or an interactive
Slurm run. It should still exercise the application, not just model import.

Target evaluation profile:

- at least 100 text-to-image campaign requests
- at least 32 layout-conditioned requests
- at least 8 personalization or subject-consistency request groups, if LoRA or
  subject conditioning is enabled
- fixed train/calibration/evaluation split for quantization and pruning

These target counts describe the full profile. The laptop profile above is the
initial executable contract.

Dataset rules:

- every row has a stable `request_id`
- every source dataset has a pinned revision or checksum
- every local asset has a sha256 checksum
- generated outputs are never reused as input rows
- calibration rows and evaluation rows are disjoint
- the exact benchmark can run from staged local files without internet access

## Request Schema

Each input row is JSONL:

```json
{
  "request_id": "parti-000001",
  "task": "text_to_image",
  "prompt": "a realistic product photo of ...",
  "negative_prompt": "blurry, low quality, unsafe",
  "width": 512,
  "height": 512,
  "num_inference_steps": 20,
  "guidance_scale": 7.0,
  "seed": 12345,
  "conditioning": {
    "type": "none",
    "asset_path": null,
    "sha256": null
  },
  "policy": {
    "safety_required": true,
    "min_clip_score": 0.0,
    "max_nsfw_score": 0.0
  },
  "metadata": {
    "source": "parti-prompts",
    "split": "smoke"
  }
}
```

Allowed `task` values:

- `text_to_image`
- `layout_to_image`
- `subject_image_generation`

Allowed `conditioning.type` values:

- `none`
- `edge`
- `depth`
- `segmentation`
- `subject_images`

## Output Schema

Each run writes:

```text
outputs/<run_id>/
  manifest.json
  records.jsonl
  run_summary.json
  qos_summary.json
  images/
  metrics/
  traces/
```

Each `records.jsonl` row contains:

```json
{
  "request_id": "parti-000001",
  "accepted": true,
  "image_path": "images/parti-000001.png",
  "latency_ms": {
    "total": 0.0,
    "text_encoder": 0.0,
    "denoise": 0.0,
    "vae_decode": 0.0,
    "postprocess": 0.0,
    "scoring": 0.0
  },
  "memory": {
    "peak_cuda_bytes": 0,
    "model_bytes": 0
  },
  "quality": {
    "clip_score": null,
    "aesthetic_score": null,
    "prompt_image_alignment": 0.0,
    "lpips_vs_exact": null,
    "ssim_vs_exact": null,
    "edge_alignment_f1": null,
    "subject_similarity": null,
    "image_stats": {
      "valid_image": true,
      "width": 512,
      "height": 512,
      "luma_mean": 0.0,
      "luma_stddev": 0.0,
      "nonblank": true
    }
  },
  "safety": {
    "passed": true,
    "nsfw_score": null,
    "policy": "prompt-term-blocklist"
  },
  "substitution": {
    "mode": "exact",
    "manifest_path": null,
    "sites_hit": 0,
    "sites_expected": 0
  },
  "errors": []
}
```

Approximate runs add exact-reference comparison fields. Exact runs set those
fields to `null` unless a previous exact golden is being checked for
reproducibility.

## QoS Contract

QoS is defined at application level. A candidate is accepted only if all hard
guards pass and it improves at least one performance objective.

Hard guards:

- `completion_rate == 1.0` for the selected evaluation profile
- `schema_valid_rate == 1.0`
- `safety_pass_rate` is not lower than exact
- no request loses required conditioning assets
- no request produces a blank, all-black, all-white, or corrupt image
- no request exceeds the configured peak-memory ceiling

Relative quality guards against exact:

- mean CLIP score drop <= 2 percent
- p95 CLIP score drop <= 5 percent
- mean aesthetic score drop <= 0.20 absolute
- LPIPS-vs-exact mean <= profile threshold after the exact golden is created
- SSIM-vs-exact mean >= profile threshold after the exact golden is created
- for `layout_to_image`, edge/depth/segmentation alignment drop <= 3 percent
- for `subject_image_generation`, subject similarity drop <= 3 percent

Laptop acceptance uses the subset of quality metrics available in the exact
laptop profile. At minimum it must check completion, schema validity, image
validity, exact-vs-approx LPIPS or SSIM after goldens exist, and prompt-image
alignment if a local CLIP scorer fits. A missing optional metric must never be
silently treated as a pass.

Performance objectives:

- primary: reduce warm p50 end-to-end latency
- secondary: reduce warm p95 end-to-end latency
- secondary: increase images/sec at fixed batch profile
- reported but not sufficient for the latency frontier: reduce peak CUDA memory
- reported but not sufficient for the latency frontier: reduce model-resident
  memory

Default acceptance rule:

```text
accepted =
  all_hard_guards_pass
  and all_relative_quality_guards_pass
  and (
    p50_latency_speedup >= 1.05
    or p95_latency_speedup >= 1.05
  )
```

The exact benchmark itself is valid only if it records all metrics needed to
evaluate this rule.

## approxMLIR Optimization Boundary

approxMLIR owns optimization policy, candidate generation, substitution
selection, tuning evidence, and the durable manifest. The diffusion application
owns request execution, artifact loading, runtime dispatch, scoring, and output
records.

Initial function-substitution targets:

- latency-memory-bound prompt-conditioning tower
- text encoder linear layers
- UNet or diffusion-transformer attention projections
- UNet or diffusion-transformer feed-forward linear layers
- convolution blocks in the denoiser
- VAE decoder convolutions
- post-processing kernels only if they affect measurable app latency

Initial approximation families:

- weight-only W8A16 quantization
- calibrated W4A16 quantization for selected linear sites
- structured channel pruning for convolution or feed-forward blocks
- attention-head pruning only when the output shape and application ABI remain
  stable

Substitution rules:

- exact mode has no substitutions
- each optimizable site has a stable `site_id`
- every non-exact plan declares artifacts, ABI, expected shape, dtype, layout,
  and fallback behavior
- same-ABI substitution is the first implementation target
- ABI-changing substitutions require a wrapper that preserves application
  output schema and exact fallback
- every approximate site records substitution hits and misses
- a site miss is not silent; it is either an explicit fallback or a run failure

## Optimization Loop

The continuous optimization loop is:

1. run exact benchmark and record app-level baseline
2. inventory optimizable sites and generate stable site IDs
3. collect calibration data on the calibration split
4. propose quantization or pruning plans
5. materialize artifacts
6. bind a candidate manifest
7. run the exact-vs-approx evaluation profile
8. compare QoS and performance
9. persist accepted manifests and rejected negative evidence
10. promote repeated wins into approxMLIR runtime/compiler infrastructure

The loop should preserve negative evidence. Failed candidates must leave enough
information to avoid repeating the same invalid substitution, including site,
plan, artifact hashes, failure mode, and QoS/performance deltas.

## Current Files

The benchmark currently uses this layout:

```text
runtime/examples/diffusion/
  README.md
  __init__.py
  app.py
  approx_backend.py
  approx_kernels.py
  approx_manifest.py
  compare_outputs.py
  dataset.py
  exact_backend.py
  memory_bound.py
  qos.py
  site_inventory.py
  substitute.py
  sweep_diffusion.py
  tests/
    test_diffusion_exact.py
    test_quant_diffusion_contracts.py
  data/
    smoke_requests.jsonl
    latency_memory_bound_requests.jsonl
    README.md
```

`data/` should contain only tiny committed smoke fixtures and checksums.
Large datasets, model snapshots, generated images, and calibration artifacts
must live in ignored staging/output directories.

Future approxMLIR work should add:

```text
calibration.py
W4/AWQ artifacts
native UNet/text-encoder latency candidates
conv/pruning candidates
```

## First Implementation Milestones

1. Add dataset schema validation and tiny real smoke request fixtures. Done.
2. Add laptop exact application runner and output schema for SD 1.5-class
   `512x512`, batch-1 generation. Done.
3. Add laptop QoS metric computation with unit tests. Done.
4. Add exact smoke benchmark on four staged real requests. Implemented; needs a
   staged local Diffusers model to execute generation.
5. Add site inventory for exact model components. Done.
6. Add approxMLIR manifest format for diffusion substitutions. Done.
7. Add one same-ABI W8A16 substitution path. Done for linear layers.
8. Add latency-memory-bound profile whose accepted frontier is based on
   end-to-end latency. Done.
9. Add pruning only after the exact QoS contract catches visible regressions.
