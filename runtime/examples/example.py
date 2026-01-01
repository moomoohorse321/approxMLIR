#!/usr/bin/env python3
"""End-to-end function substitution example using ApproxMLIR.

This demonstrates the complete workflow:
1. Define kernels (exact + approximate variants)
2. Apply @Knob decorator with DecisionTree
3. Export all functions to single MLIR module via export_module()
4. Inject annotations via inject_annotations()
5. Compile with approx-opt
6. Deploy via IREE
"""

import os
import jax
import jax.numpy as jnp
import approx_runtime as ar

# Check for approx-opt binary
APPROX_OPT = os.environ.get(
    "APPROX_OPT_PATH",
    os.path.expanduser("~/iree-build/approxMLIR/build/bin/approx-opt")
)

print("=" * 70)
print("ApproxMLIR Function Substitution Example")
print("=" * 70)
print(f"\nUsing approx-opt: {APPROX_OPT}")

def get_state(n: int) -> int:
    """Runtime state function - returns n to compare against thresholds."""
    return n


def approx_my_kernel_1(n: int, data: jax.Array) -> jax.Array:
    """Approximate kernel v1 - skip sin() for speed."""
    dummy = (n * 0).astype(data.dtype)
    return data * 2.0 + dummy


def approx_my_kernel_2(n: int, data: jax.Array) -> jax.Array:
    """Approximate kernel v2 - even faster."""
    dummy = (n * 0).astype(data.dtype)
    return data * 1.9 + dummy


@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=get_state,
    state_indices=[0],  # First arg is state
    thresholds=[100, 500],
    decisions=[0, 1, 2],  # 0=exact, 1=approx_v1, 2=approx_v2
    transform_type="func_substitute",
    approx_kernels={
        1: approx_my_kernel_1,
        2: approx_my_kernel_2,
    },
))
def my_kernel(n: int, data: jax.Array) -> jax.Array:
    """Main kernel - exact implementation."""
    dummy = (n * 0).astype(data.dtype)
    
    # 2. Add it to the result.
    #    Mathematically: result + 0
    #    Computationally: result depends on n
    return (data * 2.0 + jnp.sin(data)) + dummy

config = ar.get_config(my_kernel)
helpers = ar.collect_helper_functions("my_kernel", config)

print(f"\n--- Step 1: Collected Functions ---")
print(f"Main kernel: my_kernel")
print(f"Helpers: {[name for name, _ in helpers]}")

# Define shapes for all functions
DATA_SHAPE = jax.ShapeDtypeStruct((1024,), jnp.float32)
N_SHAPE = jax.ShapeDtypeStruct((), jnp.int32)

functions = {
    # Main kernel: (n, data) -> result
    "my_kernel": (my_kernel, (N_SHAPE, DATA_SHAPE)),
    # get_state: (n,) -> n
    "get_state": (get_state, (N_SHAPE,)),
    # approx kernels: (data,) -> result  (no n needed at runtime)
    "approx_my_kernel_1": (approx_my_kernel_1, (N_SHAPE, DATA_SHAPE)),
    "approx_my_kernel_2": (approx_my_kernel_2, (N_SHAPE, DATA_SHAPE)),
}

print(f"\n--- Step 2: Exporting to MLIR ---")
mlir = ar.export_module(functions)
print(f"Exported {len(functions)} functions to single module")

print(f"\n--- Step 3: Injecting Annotations ---")
mlir_annotated = ar.inject_annotations(mlir, "my_kernel", config)

# Show generated annotations
annotations = ar.generate_annotations("my_kernel", config)
print("Generated annotations:")
print(annotations)

# Save input MLIR
with open("func_substitute_input.mlir", "w") as f:
    f.write(mlir_annotated)
print(f"\nSaved: func_substitute_input.mlir")

# Get the right pipeline for func_substitute
pipeline = ar.get_pipeline_for_config(config)
print(f"\n--- Step 4: Compiling with approx-opt ---")
print(f"Pipeline: {' -> '.join(pipeline)}")

try:
    transformed = ar.compile(mlir_annotated, passes=pipeline, approx_opt_path=APPROX_OPT)
    
    with open("func_substitute_output.mlir", "w") as f:
        f.write(transformed)
    print(f"Saved: func_substitute_output.mlir")
    
    # Verify passes worked
    if "approx.util.annotation" in transformed:
        print("\n⚠️  Warning: Some annotations were not lowered!")
    else:
        print("\n✓ All annotations lowered successfully")

except ar.CompilationError as e:
    print(f"Compilation failed: {e}")
    transformed = None


if transformed:
    print(f"\n--- Step 5: Deploying via IREE ---")
    
    # Compile to IREE VM flatbuffer
    vmfb = ar.compile_to_iree(transformed, backend='llvm-cpu', input_type=None)
    
    # Load module - returns (modules, device) tuple
    modules, device = ar.load_module(vmfb, backend='llvm-cpu')
    print("✓ IREE deployment successful!")
    
    # Test the module with different state values
    test_data = jnp.ones(1024, dtype=jnp.float32)
    n_tensor = jnp.array(100, dtype=jnp.int32)
    
    # Call the kernel
    result = modules.module["my_kernel"](n_tensor, test_data)
    print(f"\n--- Step 6: Testing ---")
    print(f"Input: n=100, data=ones(1024)")
    print(f"Result shape: {result.shape}")
    print(f"Result[0:5]: {result.to_host()[:5]}")
    
    # Test with different n values to trigger different approximation levels
    for n_val in [50, 200, 600]:
        n_tensor = jnp.array(n_val, dtype=jnp.int32)
        result = modules.module["my_kernel"](n_tensor, test_data)
        print(f"n={n_val}: result[0]={result.to_host()[0]:.4f}")
      
else:
    if not ar.IREE_AVAILABLE:
        print("\n--- IREE not available ---")


print("\n" + "=" * 70)
print("Done!")