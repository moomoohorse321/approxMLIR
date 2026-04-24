# approxMLIR Quantization Integration Spec

Status: proposal for the complete target contract.

This document defines the target compiler/runtime contract for quantized linear
approximations in ApproxMLIR. The current SGLang implementation is described in
`QUANTIZATION_SPEC.md`.

## Goal and Non-Goals

ApproxMLIR owns approximation policy, candidate generation, tuning output, and
the binding manifest. The runtime owns artifact validation, artifact loading,
load-time binding, and serving-time dispatch. Backend function substitution is
one implementation mechanism, not the quantization contract.

The target contract supports exact, W8A16, SQ-W4A16, and AWQ-W4A16 plans across
site-specific and regime-specific bindings.

Non-goals:

- Calibration collection is not performed inside MLIR passes.
- Model weights and packed weight tensors are not encoded directly in IR.
- SGLang prefixes are not permanent site identity.
- Runtime policy does not override manifest policy.

## Final Architecture

The system has four contract surfaces:

- `SiteInventory`: stable quantizable sites and host aliases.
- `QuantPlan`: complete executable plans, including quantizer, layout, kernel
  ABI, artifact requirements, regimes, and fallback.
- `ArtifactRecord`: concrete tensors or files consumed by plans.
- `QuantManifest`: the single runtime source of truth joining sites, plans,
  artifacts, selected bindings, and tuning evidence.

Load-time flow:

```text
manifest -> site match -> plan validation -> artifact validation/load -> BoundQuantState
```

Serve-time flow:

```text
BoundQuantState + exec_ctx -> regime -> selected plan -> kernel wrapper -> output
```

The serving path must not perform manifest lookup, tuning, or artifact
construction.

## Site Identity

Each site has a canonical `site_id` and a derived `stable_key`.

Canonical `site_id` format:

```text
<model_fingerprint>:<module_path>:<op>:<shard_type>:<rank>/<world_size>
```

Example:

```text
qwen35_2b_sha256_abcd:layers.0.mlp.gate_up_proj:linear:replica:0/1
```

`stable_key` is the hash of:

```text
model_fingerprint | module_path | op | shape | dtype | shard
```

Site fields:

- `site_id`: canonical identifier.
- `stable_key`: deterministic hash used for stale binding detection.
- `op`: `linear`.
- `module_path`: model-structure path independent of host wrapper names.
- `shape`: `in_features`, `out_features`.
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

## QuantPlan Contract

A `QuantPlan` is a complete executable option. Plans are global records in the
manifest and are referenced by bindings.

Required fields:

- `plan_id`: unique string.
- `kind`: `exact`, `w8a16`, `sq_w4a16`, or `awq_w4a16`.
- `quantizer`: `none`, `int8_per_col`, `smoothquant`, or `awq`.
- `layout`: `fp`, `int8_col`, `w4_generic`, or `w4_decode_tiled`.
- `regimes`: nonempty list from the supported regime set.
- `params`: plan parameters such as block sizes, group size, SQ alpha, and AWQ
  grid size.
- `artifact_types`: required artifact kinds.
- `kernel_id`: runtime wrapper or backend kernel family.
- `kernel_abi_id`: stable ABI version.
- `kernel_signature`: ordered arguments, expected dtypes, layouts, and scratch
  requirements.
- `input_layout`: activation layout expected by the kernel.
- `output_layout`: output layout produced by the kernel.
- `fallback`: `exact`, another `plan_id`, or `fail`.

Fallback references must not form cycles.

## Artifact Contract

Every non-exact plan declares required artifact types. Every selected binding
must either reference matching prebuilt artifacts or declare load-time
materialization rules.

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
- `byte_size`: expected byte size.
- `sha256`: required for `prebuilt`, null only for `load_time`.
- `created_utc`: timestamp for prebuilt artifacts.
- `build_params`: quantizer and packing parameters used to build it.

Artifact kind table:

| Plan kind | Artifact type | Dtype | Shape convention | Layout |
|---|---|---|---|---|
| `w8a16` | `qweight_i8_t` | `int8` | `[K, N]` | transposed int8 columns |
| `w8a16` | `qweight_scale` | `fp16/bf16/fp32` | `[N]` | per-output-column scale |
| `sq_w4a16` | `qweight_i4_t_packed` | packed int4 | backend-defined | groupwise packed |
| `sq_w4a16` | `scale_g` | `fp16/bf16/fp32` | `[ceil(K/group), N]` | group scale |
| `sq_w4a16` | `act_smooth_inv` | `fp16/bf16/fp32` | `[K]` | activation inverse smoothing |
| `sq_w4a16` | `qweight_i4_decode_tiled` | packed int4 | backend-defined | decode-tiled |
| `sq_w4a16` | `scale_g_decode_tiled` | `fp16/bf16/fp32` | backend-defined | decode-tiled scale |
| `awq_w4a16` | W4 artifact types | same as W4 | same as W4 | AWQ-produced values |

The exact backend-defined packed shapes must be part of the `kernel_abi_id`
documentation and must be validated before binding.

## Manifest Schema

The runtime consumes one `quant_manifest.json`. It is valid only if all required
fields and cross references are satisfied.

Top-level required fields:

- `schema_version`: integer.
- `regime_set_version`: string.
- `model`: string.
- `model_fingerprint`: string.
- `strict`: boolean.
- `sites`: array of site records.
- `plans`: array of plan records.
- `artifacts`: array of artifact records.
- `bindings`: array of selected bindings.
- `tuning_evidence`: array of tuning evidence records.

Uniqueness and reference rules:

- `sites[].site_id` is unique.
- `sites[].stable_key` is unique.
- `plans[].plan_id` is unique.
- `artifacts[].artifact_id` is unique.
- `bindings[].site_id` references an existing site.
- `bindings[].default_plan` references an existing plan.
- `bindings[].regime_plans` values reference existing plans.
- `artifacts[].site_id` and `artifacts[].plan_id` reference existing records.
- Every selected non-exact plan has all required artifacts or load-time
  materialization rules.

Example:

```json
{
  "schema_version": 1,
  "regime_set_version": "decode-prefill.v1",
  "model": "Qwen/Qwen3.5-2B",
  "model_fingerprint": "qwen35_2b_sha256_abcd",
  "strict": false,
  "sites": [
    {
      "site_id": "qwen35_2b_sha256_abcd:layers.0.mlp.gate_up_proj:linear:replica:0/1",
      "stable_key": "sha256:sitekey",
      "op": "linear",
      "module_path": "layers.0.mlp.gate_up_proj",
      "shape": {"in_features": 2048, "out_features": 11008},
      "dtype": {"weight": ["float16", "bfloat16"], "activation": ["float16", "bfloat16"]},
      "weight_fingerprint": "sha256:weight",
      "shard": {"type": "replica", "rank": 0, "world_size": 1, "group_id": "default"},
      "matchers": [{"type": "contains", "value": "layers.0.mlp.gate_up_proj", "priority": 100}]
    }
  ],
  "plans": [
    {
      "plan_id": "w8a16_int8_col_triton",
      "kind": "w8a16",
      "quantizer": "int8_per_col",
      "layout": "int8_col",
      "regimes": ["all"],
      "params": {"block_n": 128, "block_k": 64, "use_substitute": true},
      "artifact_types": ["qweight_i8_t", "qweight_scale"],
      "kernel_id": "sglang_w8a16_linear",
      "kernel_abi_id": "triton_w8a16.v1",
      "kernel_signature": ["x", "qweight", "scale", "out", "M", "N", "K", "strides"],
      "input_layout": "row_major",
      "output_layout": "row_major",
      "fallback": "exact"
    }
  ],
  "artifacts": [
    {
      "artifact_id": "layers0_gate_up_qweight",
      "site_id": "qwen35_2b_sha256_abcd:layers.0.mlp.gate_up_proj:linear:replica:0/1",
      "plan_id": "w8a16_int8_col_triton",
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
      "build_params": {"quantizer": "int8_per_col"}
    }
  ],
  "bindings": [
    {
      "site_id": "qwen35_2b_sha256_abcd:layers.0.mlp.gate_up_proj:linear:replica:0/1",
      "default_plan": "w8a16_int8_col_triton",
      "regime_plans": {}
    }
  ],
  "tuning_evidence": []
}
```

## Load-Time Binding

For each host candidate op:

1. Match the host op to exactly one site.
2. Validate shape, dtype, shard, and weight fingerprint.
3. Resolve the binding and plan set.
4. Validate kernel ABI and artifact requirements.
5. Load or materialize artifacts.
6. Store `BoundQuantState`.

`BoundQuantState` contains site identity, selected plans, loaded artifact tensors,
fallback policy, and kernel dispatch metadata. It is runtime-internal, but its
state transitions are part of the contract.

Binding states:

- `unbound`
- `bound_exact`
- `bound_quant`
- `degraded_exact_fallback`
- `hard_fail`

## Serve-Time Dispatch

The generic host adapter dispatch path is:

1. If the bound state is `bound_exact` or `degraded_exact_fallback`, run exact.
2. If approximation mode is disabled, run exact.
3. Classify the call regime.
4. Select the manifest-approved plan for the site and regime.
5. Launch the wrapper identified by `kernel_id` and `kernel_abi_id`.
6. Apply deterministic fallback on failure.

The SGLang adapter implements this path through its linear-layer apply hook. The
contract is not tied to the SGLang method name.

## Regimes

Supported regimes:

- `all`
- `prefill`
- `decode_m1`
- `decode_msmall`
- `decode_other`

Regime classification uses:

- `M`, `N`, `K`
- batch size
- decode vs prefill phase when available
- dtype
- device
- CUDA graph capture status

If a concrete regime has no selected plan, runtime uses the binding's
`default_plan`.

Regime extensions require a new `regime_set_version`.

## Fallback and Strictness

Fallback is deterministic and per bound site.

Fallback triggers:

- no site match
- ambiguous site match
- shape mismatch
- dtype mismatch
- weight fingerprint mismatch
- missing artifact
- artifact checksum mismatch
- kernel ABI mismatch
- unsupported regime
- kernel launch failure

Rules:

- `strict=true` turns binding-time validation failures into `hard_fail`.
- `strict=false` turns binding-time validation failures into exact fallback.
- Serving-time kernel failure follows the selected plan's fallback.
- Fallback chains may contain at most one non-exact plan before exact.
- Fallback cycles are invalid manifest errors.
- Once a site enters `degraded_exact_fallback`, it remains there for the worker
  lifetime unless the host explicitly rebinds it.

## Tuning Output Contract

Tuning writes selected bindings and evidence into the manifest.

Each evidence record contains:

- `tuning_id`
- `site_id`
- `regime`
- `candidate_plan`
- `hardware_fingerprint`
- `dataset_fingerprint`
- `seed`
- `latency_p50`
- `latency_p95`
- `memory_bytes`
- `quality_metric`
- `quality_value`
- `selected`: boolean
- `dominated_by`: null or `plan_id`

The tuner only evaluates complete legal plans. It does not expose independent
strategy, artifact, and kernel knobs unless they have already been assembled
into a legal plan.

## Telemetry

Required event types:

- `load_quant_manifest`
- `bind_quant_site`
- `skip_bind_quant_site`
- `prepare_quant_artifact`
- `apply_approx`
- `fallback_exact`
- `fallback_plan`
- `reject_quant_binding`
- `hard_fail`

Each event includes `site_id` when available, host alias, plan ID when
available, state transition, reason, and process ID.

## SGLang Host Adapter

The SGLang adapter maps SGLang linear layers to manifest sites using matchers.
It prepares artifacts after weights load and dispatches through the generic
serve-time path.

SGLang-specific aliases are limited to `matchers`. They do not define site
identity.

The adapter must preserve the current exact path and must not mutate installed
SGLang files.

## Validation Requirements

Validation uses the existing serving probes:

- `probe_sglang_triton_dump.py` for latency and substitution counts.
- `compare_sglang_logprobs.py` for stepwise decode quality.

Required validation cases:

- exact path unchanged
- W8 `gate_up_proj` remains the safe speed point
- SQ-W4 decode-tiled remains the fast lower-quality point
- AWQ-W4 remains a valid calibrated W4 point
- missing artifact fallback
- malformed manifest strict failure
- stale shape, dtype, or weight fingerprint rejection
- no serving-path manifest lookup or artifact construction

## ApproxMLIR Role

ApproxMLIR represents approximation sites, candidate plans, tuning evidence, and
selected bindings. It emits or consumes `quant_manifest.json` and lowers selected
plans to backend mechanisms such as Triton wrapper calls or function
substitution.

Quantization is represented at the plan, artifact, and binding level.
`func_substitute` remains a backend rewrite mechanism.

