# approx-runtime

Minimal Python interface for the ApproxMLIR compiler toolchain.

## Installation

```bash
pip install -e .                    # Basic
pip install -e ".[jax]"             # With JAX export
pip install -e ".[iree]"            # With IREE deployment  
pip install -e ".[all]"             # Everything
```

## Usage

```python
import approx_runtime as ar

# 1. Define approximation with decorator
def get_state(n):
    return n

@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=get_state,
    state_indices=[0],
    thresholds=[100],
    decisions=[0, 2],
    transform_type="loop_perforate"
))
def my_kernel(n, data):
    return jnp.sum(data[:n])

# 2. Export to MLIR
mlir = ar.export_to_mlir(my_kernel, 100, jnp.zeros(1000))

# 3. Inject annotations
mlir = ar.inject_annotations(mlir, "my_kernel", ar.get_config(my_kernel))

# 4. Compile with approx-opt
mlir = ar.compile(mlir)

# 5. Deploy with IREE
vmfb = ar.compile_to_iree(mlir, backend="cuda")
module = ar.load_module(vmfb, backend="cuda")
result = module.my_kernel(n, data)
```

## Approximation Types

### Decision Tree (Dynamic)
```python
@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=get_state,      # (state_args) -> i32
    state_indices=[0],             # which args to pass
    thresholds=[100, 500],         # decision boundaries
    decisions=[0, 1, 2],           # knob values per region
    transform_type="loop_perforate"
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

## Pass Pipeline

```
emit-approx → emit-management → config-approx → transform-approx → finalize-approx
```
