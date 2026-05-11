# approxMLIR Quantization Integration Spec

Status: implementation-informed target contract.

This document defines the compiler/runtime contract for quantized serving
approximations in approxMLIR. It reflects the current SGLang quantization work
in `runtime/examples/sglang_quant`, the diffusion counterpart in
`runtime/examples/diffusion`, and the tuning lessons from W8A16 and W4A16
experiments.

The central design constraint is that quantization is an application-level
serving optimization, not only a kernel rewrite. The compiler may generate and
substitute Triton implementations, but a useful result requires stable
application sites, load-time quantized artifacts, end-to-end latency evidence,
quality checks, and explicit fallback accounting.

## Goal and Non-Goals

approxMLIR owns approximation policy, candidate generation, tuning output, and
the durable binding manifest. The host runtime owns artifact validation,
artifact loading or materialization, load-time binding, and serving-time
dispatch. Backend function substitution is an implementation mechanism, not the
quantization contract.

The target contract supports exact, W8A16, SQ-W4A16, AWQ-W4A16, and lightweight
groupwise W4A16 plans across site-specific, regime-specific, and workload
profile-specific bindings.

Non-goals:

- Calibration collection is not performed inside MLIR passes.
- Model weights and packed weight tensors are not encoded directly in IR.
- SGLang prefixes, diffusion module names, and environment variables are not
  permanent policy identity.
- Runtime policy does not override manifest policy.
- Bit width alone is not an optimization objective. W4 is not automatically
  better than W8 unless end-to-end evidence shows a Pareto improvement.

## Document Status

Normative requirements use `must`, `must not`, `required`, and `invalid`.
Sections labeled "Current evidence" are non-normative regression guidance from
the present examples. They must not be treated as portable policy for other
models, hardware, or workloads.

Glossary:

- `site`: a stable application operation that may legally bind an approximate
  implementation.
- `plan`: a complete executable quantization choice, including quantizer,
  artifact layout, kernel ABI, parameters, and fallback behavior.
- `artifact`: a packed weight, scale, calibration tensor, or file consumed by a
  plan.
- `regime`: the runtime shape/phase class used to choose among legal plans.
- `strict`: manifest mode that converts binding validation failures into hard
  failures instead of exact fallback.
- `BoundQuantState`: runtime-local binding state created after manifest and
  artifact validation.

## Non-Normative Lessons Folded Into the Contract

The SGLang and diffusion experiments changed the required contract in concrete
ways:

- Measure full request latency. Kernel-only speed is insufficient because CUDA
  graph capture, launch overhead, scheduling, artifact preparation, and fallback
  behavior can dominate the final result.
- Keep quantized artifact construction out of the serving path. W8/W4 weights,
  scales, smoothing vectors, and packed layouts must be prepared at load time or
  read from prebuilt artifacts.
- Treat kernel shape as part of the plan. The best W4 result changed when
  `BLOCK_K` moved to 32 and the unpack path became a single-dot path. A W4 plan
  with different tile or group parameters is a different executable plan.
- Preserve negative evidence. Naive W4/W2 implementations were slower because
  unpack/dequant instructions, register pressure, and shared-memory behavior
  outweighed bandwidth savings. The tuner must remember these rejected points.
- Bind selectively. `gate_up_proj` and `qkv_proj` were useful SGLang targets;
  `down_proj` and full W4 coverage were not Pareto points with the current
  kernels. The diffusion memory-bound conditioner produced a W4 frontier only
  after the workload exposed weight bandwidth and the W4 kernel was tuned.
- Count substitutions and fallbacks. A run with silent exact fallback is not an
  approximate run, even if latency looks good.
- Use workload-specific quality contracts. LLM decode uses logprob agreement;
  diffusion uses image and conditioner-output comparisons. These are smoke
  tests unless the manifest says the dataset is large enough for a stronger
  claim.

## Final Architecture

The system has six contract surfaces:

- `SiteInventory`: stable quantizable sites and host aliases.
- `WorkloadProfile`: the benchmark or serving profile whose latency and quality
  define the optimization target.
- `QuantPlan`: complete executable plans, including quantizer, layout, kernel
  ABI, artifact requirements, regimes, and fallback.
- `ArtifactRecord`: concrete tensors or files consumed by plans.
- `QuantManifest`: the single runtime source of truth joining sites, profiles,
  plans, artifacts, selected bindings, and tuning evidence.
- `Telemetry`: per-run records proving what was bound, substituted, measured,
  and rejected.

Load-time flow:

```text
manifest -> site match -> plan validation -> artifact validation/load/materialize -> BoundQuantState
```

Serve-time flow:

```text
BoundQuantState + exec_ctx -> profile/regime -> selected plan -> kernel wrapper -> output + telemetry
```

The serving path must not perform manifest lookup, tuning, calibration,
packing, or artifact construction.

## Site Identity

Each site has a canonical `site_id` and a derived `stable_key`.

Canonical `site_id` format:

```text
<model_fingerprint>:<component>.<module_path>:<op>:<shard_type>:<rank>/<world_size>
```

Examples:

```text
qwen35_2b_sha256_abcd:layers.0.mlp.gate_up_proj:linear:replica:0/1
sd15_sha256_abcd:memory_bound_conditioner.proj:linear:replica:0/1
```

`stable_key` is the hash of:

```text
model_fingerprint | component | module_path | op | shape | dtype | shard | weight_fingerprint
```

Site fields:

- `site_id`: canonical identifier.
- `stable_key`: deterministic hash used for stale binding detection.
- `op`: operation kind, initially `linear`; future extensions may include
  `conv2d` and attention-specific fused ops.
- `component`: application component such as `model`, `text_encoder`, `unet`,
  `transformer`, or `memory_bound_conditioner`.
- `module_path`: model-structure path independent of host wrapper names.
- `module_type`: host module class name for diagnostics only.
- `shape`: op-specific shape, such as `in_features` and `out_features`.
- `dtype`: accepted activation and weight dtypes.
- `weight_fingerprint`: hash or structured checksum of the exact weight.
- `shard`: object with `type`, `rank`, `world_size`, and `group_id`.
- `matchers`: host-specific match rules.

Matcher rule:

```json
{"type": "exact|prefix|contains|regex|callable_name", "value": "...", "priority": 100}
```

Matching is deterministic:

- The highest-priority matching rule wins.
- Equal-priority multiple matches are an error.
- A site match is valid only if shape, dtype, shard, and weight fingerprint also
  match.
- Host aliases are never durable policy identity; they only locate a site in a
  concrete runtime.

## Workload Profiles

`WorkloadProfile` defines the objective under which a plan was tuned and
selected. A plan selected for one profile is not automatically valid for
another.

Required profile fields:

- `profile_id`: unique string.
- `application`: `sglang`, `diffusion`, or another host adapter.
- `dataset_fingerprint`: hash of prompts, seeds, image settings, or request
  records.
- `hardware_fingerprint`: GPU, driver, CUDA, Triton, torch, and relevant runtime
  versions.
- `request_shape`: profile-specific shape summary, such as batch size, generated
  tokens, image resolution, denoising steps, number of candidates, or repeated
  tower steps.
- `measurement_scope`: `kernel`, `module`, or `end_to_end`; selected frontier
  points must include `end_to_end`.
- `latency_metric`: usually `p50_ms` and `p95_ms`.
- `memory_metric`: peak CUDA memory when available.
- `quality_contract_id`: reference to the quality thresholds used for selection.

Example profile summaries:

- SGLang decode profile: Qwen3.5-2B, batch 4, 8 generated tokens, CUDA graph
  enabled, logprob comparison over teacher-forced decode steps.
- Diffusion memory-bound profile: 8 requests, 64-by-64 images, 2 denoising
  steps, 32 candidates per request, 128 prompt-conditioning tower applications,
  and end-to-end latency as the performance metric.

## QuantPlan Contract

A `QuantPlan` is a complete executable option. Plans are global records in the
manifest and are referenced by bindings.

Required fields:

- `plan_id`: unique string.
- `kind`: `exact`, `w8a16`, `sq_w4a16`, `awq_w4a16`, or `w4a16_groupwise`.
- `op`: operation kind the plan can replace.
- `quantizer`: `none`, `int8_per_output_channel`, `smoothquant`,
  `int4_groupwise_awq`, or another registered quantizer.
- `layout`: `fp`, `int8_col`, `w4_generic`, `w4_decode_tiled`, or
  `uint4_packed_col`.
- `regimes`: nonempty list from the supported regime set.
- `params`: plan parameters such as `block_m`, `block_n`, `block_k`, `group_k`,
  SQ alpha, AWQ grid size, `use_substitute`, and whether exact weights may be
  dropped after binding.
- `artifact_types`: required artifact kinds.
- `kernel_id`: runtime wrapper or backend kernel family.
- `kernel_abi_id`: stable ABI version.
- `kernel_signature`: ordered arguments, expected dtypes, layouts, and scratch
  requirements.
- `input_layout`: activation layout expected by the kernel.
- `output_layout`: output layout produced by the kernel.
- `substitution`: object describing whether this plan uses same-ABI
  `func_substitute`, a wrapper ABI, no substitution, or a host-specific hook.
- `fallback`: `exact`, another `plan_id`, or `fail`.

Fallback references must not form cycles.

Plan identity rules:

- Any change to quantizer, layout, packed shape, group size, tile size, ABI, or
  substitution mechanism requires a new `plan_id`.
- W4 plans must include enough parameters to explain their online cost. A plan
  named only by bit width is invalid.
- A plan may be legal but unselected if tuning evidence shows it is dominated
  by exact, W8A16, or another W4 variant.

## Artifact Contract

Every non-exact plan declares required artifact types. Every selected binding
must either reference matching prebuilt artifacts or declare deterministic
load-time materialization rules.

Artifact record fields:

- `artifact_id`: unique string.
- `site_id`: site that owns the artifact.
- `plan_id`: plan that consumes the artifact.
- `artifact_type`: one of the plan's declared artifact types.
- `source`: `prebuilt` or `load_time`.
- `uri_scheme`: `file`, `memory`, or another registered scheme.
- `uri`: storage path or runtime-local identifier.
- `dtype`: tensor dtype.
- `shape`: tensor shape.
- `layout`: physical layout name.
- `byte_size`: expected byte size when known.
- `sha256`: required for `prebuilt`, null only for deterministic `load_time`.
- `created_utc`: timestamp for prebuilt artifacts.
- `build_params`: quantizer and packing parameters used to build it.

Artifact kind table:

| Plan kind | Artifact type | Dtype | Shape convention | Layout |
|---|---|---|---|---|
| `w8a16` | `qweight_i8_t` | `int8` | `[K, N]` | transposed int8 columns |
| `w8a16` | `qweight_scale` | `fp16/bf16/fp32` | `[N]` | per-output-column scale |
| `sq_w4a16` | `qweight_i4_t_packed` | packed int4 in `uint8` | ABI-defined, usually `[(K + 1) // 2, N]` | groupwise packed |
| `sq_w4a16` | `scale_g` | `fp16/bf16/fp32` | `[ceil(K/group_k), N]` | group scale |
| `sq_w4a16` | `act_smooth_inv` | `fp16/bf16/fp32` | `[K]` | activation inverse smoothing |
| `sq_w4a16` | `qweight_i4_decode_tiled` | packed int4 in `uint8` | ABI-defined tiled rows | decode-tiled |
| `sq_w4a16` | `scale_g_decode_tiled` | `fp16/bf16/fp32` | ABI-defined tiled rows | decode-tiled scale |
| `awq_w4a16` | W4 artifact types | same as W4 | same as W4 | AWQ-produced values |
| `w4a16_groupwise` | `qweight_i4_t_packed` | `uint8` | `[(K + 1) // 2, N]` or ABI-defined | packed columns |
| `w4a16_groupwise` | `qweight_scale_g` | `fp16/bf16/fp32` | `[ceil(K/group_k), N]` | groupwise per output column |
| `w4a16_groupwise` | `act_scale_inv` | `fp16/bf16/fp32` | `[K]` | activation scale inverse |

The exact ABI-defined packed shapes must be part of the `kernel_abi_id`
documentation and must be validated before binding.

ABI requirements for packed artifacts:

- `int8_col` stores transposed weights as `[K, N]`, contiguous, with
  per-output-column scales indexed by `N`.
- `uint4_packed_col.v1` stores two K-adjacent signed int4 codes per byte. The low
  nibble is even `K`, the high nibble is odd `K`, and the decoded signed value is
  `nibble - 7`. Padding nibbles must use code `7`, which decodes to zero.
- Group scales for untiled W4 are indexed as `[ceil(K / group_k), N]`; a group
  scale applies to all K values in that group for one output column.
- Artifact tensors must be contiguous on the target device before binding unless
  `kernel_abi_id` explicitly declares a strided layout.
- `kernel_abi_id` must name nibble order, padding code, scale broadcast
  semantics, alignment requirements, device placement, and tensor ownership.

Load-time artifacts are allowed only if:

- construction happens before any measured serving request;
- construction is deterministic from exact weights and declared calibration
  records;
- the manifest records the build parameters;
- telemetry records the artifact preparation event.

Dropping the original fp weight is a plan parameter. It can improve peak memory
but disables exact fallback unless the host keeps a separate exact path.

## Manifest Schema

The runtime consumes one `quant_manifest.json`. It is valid only if all required
fields and cross references are satisfied.

Schema validation rules:

- Unknown top-level fields are invalid for a given `schema_version`.
- Unknown nested fields are invalid unless the parent object has an explicit
  `extensions` map.
- Enum strings must match the enum set in this document exactly.
- Producers may emit an older schema only if the consumer advertises support for
  that schema version.
- Consumers must reject manifests with a newer unknown `schema_version`.
- `sha256:` values must be lowercase hexadecimal SHA-256 digests.
- Arrays that participate in identity or reference checks must not contain
  duplicate IDs.

Core enum sets:

| Field | Legal values |
|---|---|
| `application` | `sglang`, `diffusion`, registered adapter name |
| `op` | `linear`, `conv2d`, registered op |
| `kind` | `exact`, `w8a16`, `sq_w4a16`, `awq_w4a16`, `w4a16_groupwise` |
| `quantizer` | `none`, `int8_per_output_channel`, `smoothquant`, `int4_groupwise_awq`, registered quantizer |
| `layout` | `fp`, `int8_col`, `w4_generic`, `w4_decode_tiled`, `uint4_packed_col`, registered layout |
| `source` | `prebuilt`, `load_time` |
| `uri_scheme` | `file`, `memory`, registered scheme |
| `fallback` | `exact`, `fail`, or a valid `plan_id` |
| `state` | `unbound`, `bound_exact`, `bound_quant`, `degraded_exact_fallback`, `hard_fail` |

Top-level required fields:

- `schema_version`: integer.
- `regime_set_version`: string.
- `application`: string.
- `host_adapter`: string.
- `model`: string.
- `model_fingerprint`: string.
- `strict`: boolean.
- `sites`: array of site records.
- `workload_profiles`: array of workload profile records.
- `quality_contracts`: array of quality threshold records.
- `plans`: array of plan records.
- `artifacts`: array of artifact records.
- `bindings`: array of selected bindings.
- `tuning_evidence`: array of accepted or measured evidence records.
- `negative_evidence`: array of rejected candidate evidence records.

Uniqueness and reference rules:

- `sites[].site_id` is unique.
- `sites[].stable_key` is unique.
- `workload_profiles[].profile_id` is unique.
- `quality_contracts[].quality_contract_id` is unique.
- `plans[].plan_id` is unique.
- `artifacts[].artifact_id` is unique.
- `bindings[].site_id` references an existing site.
- `bindings[].default_plan` references an existing plan.
- `bindings[].regime_plans` values reference existing plans.
- `bindings[].profile_plans` values reference existing plans.
- `artifacts[].site_id` and `artifacts[].plan_id` reference existing records.
- Every selected non-exact plan has all required artifacts or load-time
  materialization rules.
- Every selected non-exact plan has at least one evidence record for the
  workload profile it claims to optimize.

Example:

```json
{
  "schema_version": 2,
  "regime_set_version": "decode-prefill.v1",
  "application": "sglang",
  "host_adapter": "sglang_quant.v1",
  "model": "Qwen/Qwen3.5-2B",
  "model_fingerprint": "qwen35_2b_sha256_abcd",
  "strict": false,
  "sites": [
    {
      "site_id": "qwen35_2b_sha256_abcd:layers.0.mlp.gate_up_proj:linear:replica:0/1",
      "stable_key": "sha256:sitekey",
      "op": "linear",
      "component": "model",
      "module_path": "layers.0.mlp.gate_up_proj",
      "module_type": "Linear",
      "shape": {"in_features": 2048, "out_features": 11008},
      "dtype": {"weight": ["float16", "bfloat16"], "activation": ["float16", "bfloat16"]},
      "weight_fingerprint": "sha256:weight",
      "shard": {"type": "replica", "rank": 0, "world_size": 1, "group_id": "default"},
      "matchers": [{"type": "contains", "value": "layers.0.mlp.gate_up_proj", "priority": 100}]
    }
  ],
  "workload_profiles": [
    {
      "profile_id": "qwen35_decode_batch4_tokens8_cuda_graph",
      "application": "sglang",
      "dataset_fingerprint": "sha256:prompts",
      "hardware_fingerprint": "rtx4060_laptop_cuda_triton",
      "request_shape": {"batch": 4, "generated_tokens": 8},
      "measurement_scope": "end_to_end",
      "latency_metric": ["p50_ms", "p95_ms"],
      "memory_metric": "peak_cuda_bytes",
      "quality_contract_id": "decode_logprob_smoke.v1"
    }
  ],
  "quality_contracts": [
    {
      "quality_contract_id": "decode_logprob_smoke.v1",
      "metrics": {"top1_agreement_min": 1.0, "topk_js_mean_max": 0.005}
    }
  ],
  "plans": [
    {
      "plan_id": "w8a16_int8_col_triton_bk64",
      "kind": "w8a16",
      "op": "linear",
      "quantizer": "int8_per_output_channel",
      "layout": "int8_col",
      "regimes": ["all"],
      "params": {"block_n": 128, "block_k": 64, "use_substitute": true},
      "artifact_types": ["qweight_i8_t", "qweight_scale"],
      "kernel_id": "sglang_w8a16_linear",
      "kernel_abi_id": "triton_w8a16.v1",
      "kernel_signature": ["x", "qweight", "scale", "out", "M", "N", "K", "strides"],
      "input_layout": "row_major",
      "output_layout": "row_major",
      "substitution": {"kind": "func_substitute", "same_abi": true},
      "fallback": "exact"
    }
  ],
  "artifacts": [
    {
      "artifact_id": "layers0_gate_up_qweight",
      "site_id": "qwen35_2b_sha256_abcd:layers.0.mlp.gate_up_proj:linear:replica:0/1",
      "plan_id": "w8a16_int8_col_triton_bk64",
      "artifact_type": "qweight_i8_t",
      "source": "load_time",
      "uri_scheme": "memory",
      "uri": "layer_buffer:_approx_qweight_i8_t",
      "dtype": "int8",
      "shape": [2048, 11008],
      "layout": "int8_col",
      "byte_size": 22544384,
      "sha256": null,
      "created_utc": null,
      "build_params": {"quantizer": "int8_per_output_channel"}
    },
    {
      "artifact_id": "layers0_gate_up_scale",
      "site_id": "qwen35_2b_sha256_abcd:layers.0.mlp.gate_up_proj:linear:replica:0/1",
      "plan_id": "w8a16_int8_col_triton_bk64",
      "artifact_type": "qweight_scale",
      "source": "load_time",
      "uri_scheme": "memory",
      "uri": "layer_buffer:_approx_qweight_scale",
      "dtype": "float32",
      "shape": [11008],
      "layout": "per_output_channel",
      "byte_size": 44032,
      "sha256": null,
      "created_utc": null,
      "build_params": {"quantizer": "int8_per_output_channel"}
    }
  ],
  "bindings": [
    {
      "site_id": "qwen35_2b_sha256_abcd:layers.0.mlp.gate_up_proj:linear:replica:0/1",
      "default_plan": "w8a16_int8_col_triton_bk64",
      "regime_plans": {},
      "profile_plans": {}
    }
  ],
  "tuning_evidence": [
    {
      "tuning_id": "example-w8a16-gate-smoke",
      "site_id": "qwen35_2b_sha256_abcd:layers.0.mlp.gate_up_proj:linear:replica:0/1",
      "profile_id": "qwen35_decode_batch4_tokens8_cuda_graph",
      "regime": "all",
      "candidate_plan": "w8a16_int8_col_triton_bk64",
      "baseline_plan": "exact",
      "hardware_fingerprint": "rtx4060_laptop_cuda_triton",
      "dataset_fingerprint": "sha256:prompts",
      "seed": 0,
      "latency_p50_ms": 156.0,
      "latency_p95_ms": null,
      "speedup_vs_exact": 1.31,
      "speedup_vs_best_lower_risk_plan": 1.0,
      "peak_cuda_bytes": null,
      "quality_metrics": {"scored_tokens": 32, "top1_agreement": 1.0},
      "quality_passed": true,
      "substitution_hits": 90,
      "fallbacks": 0,
      "selected": true,
      "dominated_by": null
    }
  ],
  "negative_evidence": []
}
```

## Load-Time Binding

For each host candidate op:

1. Match the host op to exactly one site.
2. Validate shape, dtype, shard, and weight fingerprint.
3. Resolve the binding and plan set.
4. Validate kernel ABI and artifact requirements.
5. Load or materialize artifacts outside the serving measurement window.
6. Install required function-substitution hooks or host wrappers.
7. Store `BoundQuantState`.

`BoundQuantState` contains site identity, selected plans, loaded artifact
tensors, fallback policy, hook metadata, and kernel dispatch metadata. It is
runtime-internal, but its state transitions are part of the contract.

Binding states:

- `unbound`
- `bound_exact`
- `bound_quant`
- `degraded_exact_fallback`
- `hard_fail`

Host adapters must declare hook granularity. SGLang can install a composite
substitution hook for multiple kernel families. The current diffusion adapter
installs one same-ABI hook family per run; mixed W8/W4 diffusion substitution
therefore requires a composite hook extension, separate runs, or strict failure
instead of silent partial substitution.

Multi-worker runtimes bind independently from the same immutable manifest. A
worker must not mutate manifest policy or artifact metadata in shared storage
during binding; any worker-local materialized buffers are runtime state and must
be reported through telemetry. If two workers observe different site
fingerprints for the same `site_id`, both must reject that binding rather than
choosing a winner.

## Serve-Time Dispatch

The generic host adapter dispatch path is:

1. If the bound state is `bound_exact` or `degraded_exact_fallback`, run exact.
2. If approximation mode is disabled, run exact.
3. Classify the call regime and workload profile.
4. Select the manifest-approved plan for the site, profile, and regime.
5. Launch the wrapper identified by `kernel_id` and `kernel_abi_id`.
6. Increment hit, substituted-hit, miss, or fallback counters.
7. Apply deterministic fallback on failure.

The SGLang adapter implements this path through its linear-layer apply hook. The
diffusion adapter implements it by replacing selected module forwards with
manifest-bound quantized wrappers. The contract is not tied to either host
method name.

## Regimes

Base supported regimes:

- `all`
- `prefill`
- `decode_m1`
- `decode_msmall`
- `decode_other`
- `batched_conditioning`
- `memory_bound_conditioner`

Regime classification uses:

- `M`, `N`, `K`
- batch size or candidate count
- decode vs prefill phase when available
- image resolution and denoising step profile when available
- dtype
- device
- CUDA graph capture status

Deterministic base classifiers:

- `prefill`: host reports prefill phase.
- `decode_m1`: host reports decode phase and `M == 1`.
- `decode_msmall`: host reports decode phase and `1 < M <= 16`.
- `decode_other`: host reports decode phase and `M > 16`.
- `batched_conditioning`: diffusion conditioning call with candidate count or
  effective batch greater than one.
- `memory_bound_conditioner`: profile explicitly declares the memory-bound
  conditioner component and the site belongs to that component.
- `all`: plan is legal for every regime accepted by the host adapter.

If the classifier returns a concrete regime with no selected override, runtime
uses the binding's `default_plan`. `unsupported_regime` is a validation failure
only when the host cannot classify the call or the selected plan does not list
the classified regime and does not list `all`.

Regime extensions require a new `regime_set_version`.

## Fallback and Strictness

Fallback is deterministic and per bound site.

Fallback triggers:

- `NO_SITE_MATCH`: host op does not match any manifest site.
- `AMBIGUOUS_SITE_MATCH`: host op matches multiple equal-priority sites.
- `SHAPE_MISMATCH`: host shape differs from site shape.
- `DTYPE_MISMATCH`: host dtype is not accepted by the site or plan.
- `WEIGHT_FINGERPRINT_MISMATCH`: exact weight fingerprint differs.
- `MISSING_ARTIFACT`: selected non-exact plan lacks a required artifact.
- `ARTIFACT_CHECKSUM_MISMATCH`: prebuilt artifact checksum differs.
- `KERNEL_ABI_MISMATCH`: kernel ABI does not match plan or artifacts.
- `UNSUPPORTED_REGIME`: call cannot be classified or selected plan is illegal.
- `MISSING_SUBSTITUTION_HOOK`: required plugin or hook is unavailable.
- `EXACT_FALLBACK_UNAVAILABLE`: plan dropped fp weights and has no exact path.
- `KERNEL_LAUNCH_FAILURE`: selected approximate kernel fails at serving time.

Rules:

- `strict=true` turns binding-time validation failures into `hard_fail`.
- `strict=false` turns binding-time validation failures into exact fallback.
- Serving-time kernel failure follows the selected plan's fallback.
- Fallback chains may contain at most one non-exact plan before exact.
- Fallback cycles are invalid manifest errors.
- Once a site enters `degraded_exact_fallback`, it remains there for the worker
  lifetime unless the host explicitly rebinds it.
- Final frontier claims must report zero unexpected fallbacks. Expected exact
  sites are not fallbacks; selected approximate sites that run exact are.
- Plans that set `drop_original_weights=true` must set `fallback` to `fail` or
  to another non-exact plan with its own artifacts. They must not name `exact`
  unless the host adapter declares a separate exact path.

State transitions:

| From | Event | To |
|---|---|---|
| `unbound` | exact binding selected | `bound_exact` |
| `unbound` | quant binding valid | `bound_quant` |
| `unbound` | binding validation fails and `strict=false` | `degraded_exact_fallback` |
| `unbound` | binding validation fails and `strict=true` | `hard_fail` |
| `bound_quant` | serving kernel failure with exact fallback | `degraded_exact_fallback` |
| `bound_quant` | serving kernel failure with `fallback=fail` | `hard_fail` |
| `degraded_exact_fallback` | explicit host rebind succeeds | `bound_exact` or `bound_quant` |

## Tuning Output Contract

Tuning writes selected bindings, positive evidence, and negative evidence into
the manifest.

Each evidence record contains:

- `tuning_id`
- `site_id`
- `profile_id`
- `regime`
- `candidate_plan`
- `baseline_plan`
- `hardware_fingerprint`
- `dataset_fingerprint`
- `seed`
- `latency_p50_ms`
- `latency_p95_ms`
- `speedup_vs_exact`
- `speedup_vs_best_lower_risk_plan`
- `peak_cuda_bytes`
- `quality_metrics`
- `quality_passed`: boolean
- `substitution_hits`
- `fallbacks`
- `selected`: boolean
- `dominated_by`: null or `plan_id`

Negative evidence records use the same identifiers and add:

- `rejection_reason`: for example `slower_than_w8`, `quality_regression`,
  `full_coverage_dominated`, `kernel_overhead`, or `missing_calibration`.
- `diagnosis`: short explanation, such as unpack/dequant overhead, register
  pressure, non-memory-bound workload, or fallback contamination.
- `do_not_retry_without`: required change before the tuner should revisit the
  point, such as a new kernel ABI, smaller `block_k`, calibration artifact, or
  larger memory-bound profile.

The tuner only evaluates complete legal plans. It does not expose independent
strategy, artifact, and kernel knobs unless they have already been assembled
into a legal plan.

## Telemetry

Required event types:

- `load_quant_manifest`
- `bind_quant_site`
- `skip_bind_quant_site`
- `prepare_quant_artifact`
- `install_substitution_hook`
- `disable_substitution_hook`
- `apply_approx`
- `substitution_hit`
- `substitution_miss`
- `fallback_exact`
- `fallback_plan`
- `reject_quant_binding`
- `record_quality_probe`
- `hard_fail`

Each event includes `site_id` when available, host alias, plan ID when
available, profile ID when available, state transition, reason, and process ID.

Every measured run reports aggregate counters:

- `sites_expected`
- `sites_bound`
- `sites_hit`
- `substituted_hits`
- `fallbacks`
- `plan` or list of plans
- exact/approx mode
- manifest path or manifest fingerprint

## Host Adapters

### SGLang Host Adapter

The SGLang adapter maps SGLang linear layers to manifest sites using matchers.
It prepares artifacts after weights load and dispatches through the generic
serve-time path.

SGLang-specific aliases are limited to `matchers`. They do not define site
identity.

The adapter must preserve the current exact path and must not mutate installed
SGLang files.

### Diffusion Host Adapter

The diffusion adapter maps pipeline modules to inventory sites using component
and module path matchers. It supports exact, W8A16, and W4A16 linear bindings
for the memory-bound prompt-conditioning tower and records substitution deltas
per request.

Diffusion-specific rules:

- Image outputs, run summaries, goldens, model caches, and calibration outputs
  are artifacts of experiments, not source files.
- The default latency frontier is selected by end-to-end request latency, not
  peak-memory reduction alone.
- W4A16 is a valid selected point only when the workload profile evidence meets
  its latency and quality thresholds and fallback count is zero.
- Native UNet, text-encoder, and adapter-layer substitutions should reuse the
  same manifest contract instead of adding ad hoc environment policies.

## Validation Requirements

Validation uses the existing serving probes:

- `runtime/examples/sglang_quant/probe_sglang_triton_dump.py` for SGLang
  latency and substitution counts.
- `runtime/examples/sglang_quant/compare_sglang_logprobs.py` for stepwise LLM
  decode quality.
- `runtime/examples/diffusion/app.py` for exact and approximate diffusion runs.
- `runtime/examples/diffusion/sweep_diffusion.py` for exact, W8A16, and W4A16
  diffusion frontier sweeps.
- `runtime/examples/diffusion/compare_outputs.py` for diffusion QoS checks.
- `runtime/examples/diffusion/tests` for manifest, inventory, and exact-path
  contracts.

Conformance validation cases:

- exact path unchanged
- diffusion exact/W8/W4 sweep reports p50, p95, peak memory, quality metrics,
  substitution hits, and fallbacks
- missing artifact fallback
- malformed manifest strict failure
- stale shape, dtype, or weight fingerprint rejection
- missing substitution plugin records either a strict failure or an exact
  fallback event
- no serving-path manifest lookup, calibration, packing, or artifact
  construction

Each conformance test must declare command line, fixture manifest, expected
state transitions, expected counters, and pass/fail thresholds. A test that only
names a script is not sufficient for a new host adapter.

## Current Evidence Profiles

This section is non-normative. It records the useful regression profiles from
the current examples so future changes do not accidentally lose known behavior.

SGLang Qwen3.5-2B decode on the RTX 4060 Laptop profile:

- W8A16 on `gate_up_proj` is a safe speed point in the current smoke benchmark.
- W8A16 on `qkv_proj` is a smaller positive point.
- tuned AWQ-W4A16 on `gate_up_proj` and `qkv_proj` with `BLOCK_K=32` is the
  fastest measured point in the current smoke benchmark.
- full W4 coverage and `down_proj` replacement are negative evidence for the
  current kernels.

Diffusion memory-bound conditioner profile:

- W8A16 and W4A16 must be compared by end-to-end request p50/p95, not only peak
  memory.
- The current selected W4 profile uses `BLOCK_K=32` and `GROUP_K=128`, records
  why it beats W8 only under the memory-bound profile, and requires zero
  fallbacks.

## ApproxMLIR Role

approxMLIR represents approximation sites, candidate plans, tuning evidence,
negative evidence, selected bindings, and durable manifests. It emits or
consumes `quant_manifest.json` and lowers selected plans to backend mechanisms
such as Triton wrapper calls or function substitution.

Quantization is represented at the plan, artifact, binding, profile, and
evidence level. `func_substitute` remains a backend rewrite mechanism. The
compiler should not infer that a lower bit width is better unless the manifest
contains end-to-end evidence showing a valid latency/quality frontier point.
