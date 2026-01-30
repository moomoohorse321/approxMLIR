# approx-runtime

Python interface for the ApproxMLIR compiler toolchain.

## Installation

From `approxMLIR/runtime`:

```bash
pip install -e .                    # Basic
pip install -e ".[jax]"             # With JAX export
pip install -e ".[iree]"            # With IREE deployment
pip install -e ".[tuning]"          # With OpenTuner auto-tuning
pip install -e ".[all]"             # Everything
```

## Quick Start (JAX)

```python
import approx_runtime as ar
import jax.numpy as jnp

def get_state(n):
    return n

# Original function stays untouched
def my_kernel(n, data):
    return jnp.sum(data[:n])

config = {
    "decision_tree": ar.DecisionTree(
        state_function=get_state,
        state_indices=[0],
        thresholds=[100],
        decisions=[0, 2],
        transform_type="loop_perforate",
    )
}

example_args = (100, jnp.zeros(1000))
mlir = ar.export_with_config(my_kernel, example_args, config)
mlir = ar.compile(mlir, workload=ar.WorkloadType.ML)
vmfb = ar.compile_to_iree(mlir, backend="cuda")
modules, device = ar.load_module(vmfb, backend="cuda")
result = modules.module["my_kernel"](n, data)
```

## Quick Start (C++)

```python
import approx_runtime as ar

cpp_source = open("approx_lavaMD.c").read()
annotated_mlir = ar.compile_cpp_source(cpp_source, emit="annotated")
transformed_mlir = ar.compile_cpp_source(cpp_source, emit="transformed")
```

Use `CgeistConfig` when you need include dirs or a resource dir:

```python
import approx_runtime as ar

config = ar.CgeistConfig(
    cgeist_path="/path/to/cgeist",
    resource_dir="/path/to/clang/resource",
    include_dirs=["/path/to/includes"],
)
mlir = ar.compile_cpp_source(cpp_source, cgeist_config=config, emit="annotated")
```

## Toolchains

ApproxMLIR supports separate toolchains for ML (StableHLO/IREE) and C++ (Polygeist).

Environment variables:
- `APPROX_OPT_ML` (default: `approx-opt`)
- `APPROX_OPT_CPP` (default: `approx-opt-cpp`, set to `polygeist-opt` for Polygeist)
- `CGEIST_PATH` (default: `cgeist`)

Example:
```python
import approx_runtime as ar
compiled = ar.compile(mlir_text, workload=ar.WorkloadType.CPP)
```

## Non-Invasive JAX API

Apply configurations at call-time without decorators:

```python
import approx_runtime as ar
import jax.numpy as jnp

def get_state(n):
    return n

def my_kernel(n, data):
    return jnp.sum(data[:n])

config = {
    "decision_tree": ar.DecisionTree(
        state_function=get_state,
        state_indices=[0],
        thresholds=[100],
        decisions=[0, 1],
        transform_type="loop_perforate",
    )
}

example_args = (100, jnp.zeros(1000))
mlir = ar.export_with_config(my_kernel, example_args, config)
```

Use `export_module_with_configs` to export helpers (state functions, approximate kernels):

```python
mlir = ar.export_module_with_configs({
    "my_kernel": (my_kernel, example_args, config),
    "get_state": (get_state, (100,), None),
})
```

## C++ Comment-Driven Annotations

Add structured comments to C/C++ source and generate MLIR annotations:

```c
// @approx:decision_tree {
//   transform_type: func_substitute
//   state_indices: [-1]         // optional, default: last arg
//   state_function: getState    // optional, default: identity helper
//   thresholds: [5, 10]         // required
//   thresholds_lower: [0, 0]    // optional (default: 0s)
//   thresholds_upper: [100, 100]// optional (default: 100s)
//   decisions: [0, 1, 2]        // required, len = thresholds + 1
//   decision_values: [0, 1, 2]  // optional tuning bounds
// }
void my_kernel(float* data, int n, int state) { ... }
```

```python
import approx_runtime as ar

annotations = ar.parse_and_generate(cpp_source)
annotated_mlir = ar.inject_annotations_text(base_mlir, annotations)
```

If `state_function` is omitted (or `identity`), a helper `__identity_<func>` is emitted.

## Approximation Types

### Decision Tree (Dynamic)
```python
@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=get_state,      # (state_args) -> i32
    state_indices=[0],             # which args to pass
    thresholds=[100, 500],         # decision boundaries
    decisions=[0, 1, 2],           # knob values per region
    transform_type="loop_perforate",
    # Optional: bounds for auto-tuning
    thresholds_lower=[0, 100],
    thresholds_upper=[500, 1000],
    decision_values=[0, 1, 2],
))
```

### Safety Contract (Try-Check-Recover)
```python
@ar.Knob(safety_contract=ar.SafetyContract(
    checker=my_checker,   # (results, inputs) -> bool
    recover=my_recover,   # (inputs) -> results
))
```

### Static Transform
```python
@ar.Knob(static_transform=ar.StaticTransform(
    transform_type="loop_perforate",
    knob_val=2
))
```

## Transform Types

- `loop_perforate`: Skip loop iterations (knob_val = stride multiplier)
- `func_substitute`: Replace function with approximate version
- `task_skipping`: Skip branches in if-else (knob_val = branch index)

## Auto-Tuning

The runtime integrates with [OpenTuner](https://github.com/jansel/opentuner) for automatic parameter optimization. The tuner searches the configuration space defined by `decision_tree` annotations to minimize execution time while maintaining accuracy above a threshold.

### Quick Start

```python
import approx_runtime as ar
import time

# Load MLIR with decision_tree annotations
mlir_source = open("my_kernel.mlir").read()
manager = ar.MLIRConfigManager()

def evaluate(config: dict) -> tuple[float, float]:
    """User-provided evaluation function.
    
    Args:
        config: Dict mapping parameter names to values
                e.g., {"my_kernel_threshold_0": 75, "my_kernel_decision_1": 2}
    
    Returns:
        (execution_time_ms, accuracy_score)
    """
    # Apply config to MLIR
    modified_mlir = manager.apply_config(mlir_source, config)
    
    # Compile and deploy
    compiled = ar.compile(modified_mlir, passes=ar.PASS_PIPELINE)
    vmfb = ar.compile_to_iree(compiled, backend="cuda")
    modules, _ = ar.load_module(vmfb, backend="cuda")
    
    # Measure execution time
    start = time.perf_counter()
    result = modules.module["my_kernel"](test_input)
    time_ms = (time.perf_counter() - start) * 1000
    
    # Compute accuracy (user-defined metric)
    accuracy = 1.0 / (1.0 + compute_error(result, ground_truth))
    
    return time_ms, accuracy

# Run tuning
result = ar.tune(
    mlir_source=mlir_source,
    evaluate_fn=evaluate,
    accuracy_threshold=0.95,
    time_budget=1800,  # 30 minutes
)

print(f"Best config: {result['best_config']}")
print(f"Best time: {result['best_time']:.2f}ms")
print(f"Best accuracy: {result['best_accuracy']:.4f}")

# Use the tuned MLIR
tuned_mlir = result['best_mlir']
```

### JAX and C++ Helpers

```python
result = ar.tune_jax(my_kernel, example_args, config, evaluate_fn)
result = ar.tune_cpp(cpp_source, base_mlir, evaluate_fn)
```

### Tuning API Reference

#### `ar.tune()` - High-level entry point

```python
result = ar.tune(
    mlir_source: str,          # MLIR with decision_tree annotations
    evaluate_fn: Callable,     # (config) -> (time_ms, accuracy)
    accuracy_threshold=0.9,    # Minimum acceptable accuracy [0, 1]
    time_budget=3600,          # Tuning duration in seconds
    database="opentuner.db",   # Results database path
    parallelism=1,             # Parallel evaluations
    result_callback=None,      # Optional: (config, time, accuracy) -> None
)
```

Returns:
```python
{
    "best_config": {"param_name": value, ...},
    "best_mlir": "...",        # MLIR with best config applied
    "best_time": 12.5,         # Best execution time in ms
    "best_accuracy": 0.97,     # Accuracy achieved
}
```

#### `ar.MLIRConfigManager` - Parse and modify annotations

```python
manager = ar.MLIRConfigManager()

# Extract tunable parameters from MLIR
params = manager.parse_annotations(mlir_source)
# Returns: {"func_threshold_0": TunableParam(...), "func_decision_0": ...}

# Apply a configuration to MLIR
modified = manager.apply_config(mlir_source, {"func_threshold_0": 150})

# Validate configuration (bounds + ascending thresholds)
is_valid = manager.validate_config(config, params)
```

#### `ar.TunableParam` - Parameter metadata

```python
@dataclass
class TunableParam:
    name: str          # e.g., "my_func_threshold_0"
    lower: int         # Minimum value (from thresholds_lowers)
    upper: int         # Maximum value (from thresholds_uppers)
    current: int       # Current value in MLIR
    param_type: str    # "threshold" or "decision"
    func_name: str     # Target function name
    index: int         # Parameter index
```

### Defining Tuning Bounds

Add bounds to your `DecisionTree` for auto-tuning:

```python
@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=get_state,
    state_indices=[0],
    thresholds=[100, 500],          # Initial values
    decisions=[0, 1, 2],
    transform_type="func_substitute",
    # Tuning bounds
    thresholds_lower=[0, 100],      # Min for each threshold
    thresholds_upper=[500, 1000],   # Max for each threshold
))
def my_kernel(...):
    ...
```

The tuner will:
1. Parse `thresholds_lowers` and `thresholds_uppers` to define search ranges
2. Use `decision_values` (if specified) to bound decision parameters
3. Enforce ascending order constraint on thresholds

### Advanced: Using ApproxTunerInterface

For full control over OpenTuner arguments:

```python
import approx_runtime as ar

# Create argument parser with OpenTuner options
parser = ar.create_tuner_arg_parser()
parser.add_argument("--my-custom-arg", type=int)
args = parser.parse_args()

# Create tuner interface
tuner = ar.ApproxTunerInterface(
    args=args,
    mlir_source=mlir_source,
    evaluate_fn=my_evaluate,
    accuracy_threshold=0.95,
    result_callback=track_results,
)

# Run tuning
result = tuner.run_tuning()
```

Command-line options include:
```
--stop-after SECONDS     Tuning time limit
--parallelism N          Parallel evaluations  
--no-dups                Skip duplicate configurations
--accuracy-threshold F   Minimum accuracy (added by approx-runtime)
--output-config FILE     Save best config JSON
--output-mlir FILE       Save tuned MLIR
```

### Examples

- `examples/example_w_tuning.py` (JAX tuning)
- `examples/benchmark/example_lavamd_tuning.py` (C++ tuning via cgeist)
- `examples/benchmark/example_bm25_tuning.py` (C++ tuning via cgeist)
- `examples/benchmark/example_choose_tuning.py` (C++ tuning via cgeist)
- `examples/benchmark/example_kmeans_tuning.py` (C++ tuning via cgeist)
- `examples/benchmark/example_pagerank_tuning.py` (C++ tuning via cgeist)
- `examples/benchmark/example_kb_tuning.py` (C++ tuning via cgeist)

The C/C++ benchmark scripts read annotated sources from `examples/benchmark/src/` (copies of the
unannotated inputs in `tools/cgeist/Test/approxMLIR/` with `@approx` comments added).

## Pass Pipeline

ML:
```
emit-approx → emit-management → config-approx → transform-approx → finalize-approx → legalize-to-stablehlo
```

CPP:
```
emit-approx → emit-management → config-approx → transform-approx → finalize-approx
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APPROX_OPT_PATH` | (deprecated) | Legacy approx-opt path (ML) |
| `APPROX_OPT_ML` | `approx-opt` | Path to ML toolchain |
| `APPROX_OPT_CPP` | `approx-opt-cpp` | Path to C++ toolchain (use `polygeist-opt`) |
| `CGEIST_PATH` | `cgeist` | Path to cgeist |
| `APPROX_BACKEND` | `cuda` | IREE backend (`cuda` or `llvm-cpu`) |
| `TUNING_TIME_BUDGET` | `300` | Default tuning time in seconds |
| `ACCURACY_THRESHOLD` | `0.9` | Default accuracy threshold |
