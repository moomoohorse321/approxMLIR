#!/usr/bin/env python3
"""End-to-end function substitution example using ApproxMLIR with Auto-tuning.

This demonstrates the complete workflow:
1. Define kernels (exact + approximate variants)
2. Apply @Knob decorator with DecisionTree
3. Export all functions to single MLIR module via export_module()
4. Inject annotations via inject_annotations()
5. Compile with approx-opt
6. Deploy via IREE
7. Auto-tune the decision tree parameters

The tuning phase searches for optimal threshold and decision values
that minimize execution time while maintaining accuracy above a threshold.
"""

import os
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import approx_runtime as ar

# Check for approx-opt binary
APPROX_OPT = os.environ.get(
    "APPROX_OPT_PATH",
    os.path.expanduser("~/iree-build/approxMLIR/build/bin/approx-opt")
)

# Configuration
BACKEND = os.environ.get("APPROX_BACKEND", "cuda")  # "cuda" or "llvm-cpu"
TUNING_TIME_BUDGET = int(os.environ.get("TUNING_TIME_BUDGET", "300"))  # seconds
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.9"))


def get_state(n: int) -> int:
    """Runtime state function - returns n to compare against thresholds."""
    return n


def approx_my_kernel_1(n: int, data: jax.Array) -> jax.Array:
    """Approximate kernel v1 - skip sin() for speed."""
    dummy = (n * 0).astype(data.dtype)
    return data * 2.0 + dummy


def approx_my_kernel_2(n: int, data: jax.Array) -> jax.Array:
    """Approximate kernel v2 - even faster, less accurate."""
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
    # Bounds for auto-tuning
    thresholds_lower=[0, 100],
    thresholds_upper=[500, 1000],
))
def my_kernel(n: int, data: jax.Array) -> jax.Array:
    """Main kernel - exact implementation."""
    dummy = (n * 0).astype(data.dtype)
    return (data * 2.0 + jnp.sin(data)) + dummy


def compute_ground_truth(data: np.ndarray) -> np.ndarray:
    """Compute the exact (ground truth) result for accuracy comparison."""
    return data * 2.0 + np.sin(data)


def run_compilation_demo():
    """Run the basic compilation and deployment demo (Steps 1-5)."""
    print("=" * 70)
    print("ApproxMLIR Function Substitution Example")
    print("=" * 70)
    print(f"\nUsing approx-opt: {APPROX_OPT}")
    print(f"Backend: {BACKEND}")

    config = ar.get_config(my_kernel)
    helpers = ar.collect_helper_functions("my_kernel", config)

    print(f"\n--- Step 1: Collected Functions ---")
    print(f"Main kernel: my_kernel")
    print(f"Helpers: {[name for name, _ in helpers]}")

    # Define shapes for all functions
    DATA_SHAPE = jax.ShapeDtypeStruct((1024,), jnp.float32)
    N_SHAPE = jax.ShapeDtypeStruct((), jnp.int32)

    functions = {
        "my_kernel": (my_kernel, (N_SHAPE, DATA_SHAPE)),
        "get_state": (get_state, (N_SHAPE,)),
        "approx_my_kernel_1": (approx_my_kernel_1, (N_SHAPE, DATA_SHAPE)),
        "approx_my_kernel_2": (approx_my_kernel_2, (N_SHAPE, DATA_SHAPE)),
    }

    print(f"\n--- Step 2: Exporting to MLIR ---")
    mlir = ar.export_module(functions)
    print(f"Exported {len(functions)} functions to single module")

    print(f"\n--- Step 3: Injecting Annotations ---")
    mlir_annotated = ar.inject_annotations(mlir, "my_kernel", config)

    annotations = ar.generate_annotations("my_kernel", config)
    print("Generated annotations:")
    print(annotations)

    with open("func_substitute_input.mlir", "w") as f:
        f.write(mlir_annotated)
    print(f"\nSaved: func_substitute_input.mlir")

    pipeline = ar.get_pipeline_for_config(config)
    print(f"\n--- Step 4: Compiling with approx-opt ---")
    print(f"Pipeline: {' -> '.join(pipeline)}")

    try:
        transformed = ar.compile(mlir_annotated, passes=pipeline, approx_opt_path=APPROX_OPT)
        
        with open("func_substitute_output.mlir", "w") as f:
            f.write(transformed)
        print(f"Saved: func_substitute_output.mlir")
        
        if "approx.util.annotation" in transformed:
            print("\n⚠️  Warning: Some annotations were not lowered!")
        else:
            print("\n✓ All annotations lowered successfully")

    except ar.CompilationError as e:
        print(f"Compilation failed: {e}")
        return None, None, None

    print(f"\n--- Step 5: Deploying via IREE ---")
    
    vmfb = ar.compile_to_iree(transformed, backend=BACKEND)
    modules, device = ar.load_module(vmfb, backend=BACKEND)
    print("✓ IREE deployment successful!")

    # Quick functional test
    test_data = jnp.ones(1024, dtype=jnp.float32)
    n_tensor = jnp.array(100, dtype=jnp.int32)
    result = modules.module["my_kernel"](n_tensor, test_data)
    print(f"\nQuick test: n=100, result[0]={result.to_host()[0]:.4f}")

    return mlir_annotated, pipeline, config


def run_tuning(mlir_annotated: str, pipeline: list, config: dict):
    """Run the auto-tuning phase (Step 6).
    
    This creates an evaluation function that:
    1. Applies the config to MLIR
    2. Compiles with approx-opt
    3. Deploys via IREE
    4. Runs the kernel with test inputs
    5. Measures execution time and accuracy
    """
    print("\n" + "=" * 70)
    print("Auto-Tuning Phase")
    print("=" * 70)
    
    # Create MLIRConfigManager for applying configs
    manager = ar.MLIRConfigManager()
    
    # Prepare test data for evaluation
    # Use multiple n values to test different approximation paths
    test_n_values = [50, 150, 300, 600, 800]
    test_data = np.random.randn(1024).astype(np.float32)
    
    # Compute ground truth for each n value
    ground_truth = compute_ground_truth(test_data)
    
    # Convert to JAX arrays for IREE
    test_data_jax = jnp.array(test_data)
    
    # Compilation cache to avoid recompiling identical configs
    compilation_cache = {}
    
    def evaluate_fn(config: dict) -> tuple:
        """Evaluation function for the tuner.
        
        Args:
            config: Dict mapping parameter names to values
                    e.g., {"my_kernel_threshold_0": 75, "my_kernel_decision_0": 1}
            
        Returns:
            (execution_time_ms, accuracy_score)
        """
        # Create a hashable key from config
        config_key = tuple(sorted(config.items()))
        
        if config_key in compilation_cache:
            modules, device = compilation_cache[config_key]
        else:
            try:
                # Apply config to MLIR
                modified_mlir = manager.apply_config(mlir_annotated, config)
                
                # Compile with approx-opt
                compiled = ar.compile(modified_mlir, passes=pipeline, approx_opt_path=APPROX_OPT)
                
                # Deploy via IREE
                vmfb = ar.compile_to_iree(compiled, backend=BACKEND)
                modules, device = ar.load_module(vmfb, backend=BACKEND)
                
                # Cache the compiled module
                compilation_cache[config_key] = (modules, device)
                
            except Exception as e:
                print(f"Compilation error: {e}")
                return float('inf'), 0.0
        
        # Warmup runs
        n_warmup = jnp.array(test_n_values[0], dtype=jnp.int32)
        for _ in range(3):
            _ = modules.module["my_kernel"](n_warmup, test_data_jax)
        
        # Timing runs - test across different n values
        total_time_ms = 0.0
        total_error = 0.0
        num_runs = 10
        
        for n_val in test_n_values:
            n_tensor = jnp.array(n_val, dtype=jnp.int32)
            
            # Time the execution
            start = time.perf_counter()
            for _ in range(num_runs):
                result = modules.module["my_kernel"](n_tensor, test_data_jax)
            elapsed_ms = (time.perf_counter() - start) / num_runs * 1000
            total_time_ms += elapsed_ms
            
            # Compute accuracy (MSE-based)
            result_np = np.array(result.to_host())
            mse = np.mean((result_np - ground_truth) ** 2)
            total_error += mse
        
        # Average time across all n values
        avg_time_ms = total_time_ms / len(test_n_values)
        
        # Convert total error to accuracy score [0, 1]
        # Using: accuracy = 1 / (1 + normalized_error)
        avg_error = total_error / len(test_n_values)
        accuracy = 1.0 / (1.0 + avg_error)
        
        return avg_time_ms, accuracy
    
    # Track results for analysis
    all_results = []
    
    def result_callback(config_dict, time_ms, accuracy):
        """Callback to track all explored configurations."""
        all_results.append({
            'config': config_dict.copy(),
            'time_ms': time_ms,
            'accuracy': accuracy,
        })
        if len(all_results) % 10 == 0:
            print(f"  Explored {len(all_results)} configs, "
                  f"best so far: {min(r['time_ms'] for r in all_results if r['accuracy'] >= ACCURACY_THRESHOLD):.2f}ms")
    
    print(f"\nTuning parameters:")
    print(f"  Accuracy threshold: {ACCURACY_THRESHOLD}")
    print(f"  Time budget: {TUNING_TIME_BUDGET}s")
    print(f"  Test n values: {test_n_values}")
    print(f"  Backend: {BACKEND}")
    
    # Parse initial parameters
    manager = ar.MLIRConfigManager()
    params = manager.parse_annotations(mlir_annotated)
    print(f"\nTunable parameters found:")
    for name, param in params.items():
        print(f"  {name}: [{param.lower}, {param.upper}] (initial: {param.current})")
    
    print(f"\nStarting tuning...")
    
    # Run the tuner
    result = ar.tune(
        mlir_source=mlir_annotated,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=ACCURACY_THRESHOLD,
        time_budget=TUNING_TIME_BUDGET,
        result_callback=result_callback,
    )
    
    print(f"\n" + "-" * 40)
    print("Tuning Results:")
    print(f"  Best time: {result['best_time']:.2f}ms")
    print(f"  Best accuracy: {result['best_accuracy']:.4f}")
    print(f"  Best config: {result['best_config']}")
    print(f"  Total configs explored: {len(all_results)}")
    
    # Save the tuned MLIR
    with open("func_substitute_tuned.mlir", "w") as f:
        f.write(result['best_mlir'])
    print(f"\nSaved tuned MLIR: func_substitute_tuned.mlir")
    
    # Analysis of results
    if all_results:
        valid_results = [r for r in all_results if r['accuracy'] >= ACCURACY_THRESHOLD]
        if valid_results:
            print(f"\nValid configurations (accuracy >= {ACCURACY_THRESHOLD}): {len(valid_results)}")
            
            # Show Pareto-optimal points
            pareto = []
            for r in sorted(valid_results, key=lambda x: x['time_ms']):
                if not pareto or r['accuracy'] > pareto[-1]['accuracy']:
                    pareto.append(r)
            
            print(f"Pareto-optimal configurations:")
            for p in pareto[:5]:  # Show top 5
                print(f"  time={p['time_ms']:.2f}ms, accuracy={p['accuracy']:.4f}")
    
    return result


def run_comparison(mlir_annotated: str, tuned_mlir: str, pipeline: list):
    """Compare original vs tuned kernel performance (Step 7)."""
    print("\n" + "=" * 70)
    print("Comparison: Original vs Tuned")
    print("=" * 70)
    
    test_data = np.random.randn(1024).astype(np.float32)
    test_data_jax = jnp.array(test_data)
    ground_truth = compute_ground_truth(test_data)
    
    def benchmark(mlir_text: str, name: str):
        compiled = ar.compile(mlir_text, passes=pipeline, approx_opt_path=APPROX_OPT)
        vmfb = ar.compile_to_iree(compiled, backend=BACKEND)
        modules, device = ar.load_module(vmfb, backend=BACKEND)
        
        test_n_values = [50, 150, 300, 600, 800]
        
        # Warmup
        for n_val in test_n_values:
            n_tensor = jnp.array(n_val, dtype=jnp.int32)
            _ = modules.module["my_kernel"](n_tensor, test_data_jax)
        
        # Benchmark
        total_time = 0.0
        total_error = 0.0
        num_runs = 50
        
        for n_val in test_n_values:
            n_tensor = jnp.array(n_val, dtype=jnp.int32)
            
            start = time.perf_counter()
            for _ in range(num_runs):
                result = modules.module["my_kernel"](n_tensor, test_data_jax)
            elapsed = (time.perf_counter() - start) / num_runs * 1000
            total_time += elapsed
            
            result_np = np.array(result.to_host())
            mse = np.mean((result_np - ground_truth) ** 2)
            total_error += mse
        
        avg_time = total_time / len(test_n_values)
        avg_error = total_error / len(test_n_values)
        accuracy = 1.0 / (1.0 + avg_error)
        
        print(f"\n{name}:")
        print(f"  Avg time: {avg_time:.3f}ms")
        print(f"  Accuracy: {accuracy:.4f}")
        
        return avg_time, accuracy
    
    orig_time, orig_acc = benchmark(mlir_annotated, "Original")
    tuned_time, tuned_acc = benchmark(tuned_mlir, "Tuned")
    
    speedup = orig_time / tuned_time if tuned_time > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Accuracy change: {tuned_acc - orig_acc:+.4f}")


def main():
    global TUNING_TIME_BUDGET, ACCURACY_THRESHOLD, BACKEND
    
    parser = argparse.ArgumentParser(description="ApproxMLIR Example with Auto-tuning")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip the tuning phase")
    parser.add_argument("--tuning-time", type=int, default=TUNING_TIME_BUDGET,
                        help=f"Tuning time budget in seconds (default: {TUNING_TIME_BUDGET})")
    parser.add_argument("--accuracy", type=float, default=ACCURACY_THRESHOLD,
                        help=f"Accuracy threshold (default: {ACCURACY_THRESHOLD})")
    parser.add_argument("--backend", type=str, default=BACKEND,
                        choices=["cuda", "llvm-cpu"],
                        help=f"IREE backend (default: {BACKEND})")
    args = parser.parse_args()
    
    # Update globals from args
    TUNING_TIME_BUDGET = args.tuning_time
    ACCURACY_THRESHOLD = args.accuracy
    BACKEND = args.backend
    
    # Step 1-5: Compilation demo
    mlir_annotated, pipeline, config = run_compilation_demo()
    
    if mlir_annotated is None:
        print("Compilation failed, exiting.")
        return
    
    # Step 6: Auto-tuning
    if not args.skip_tuning:
        tuning_result = run_tuning(mlir_annotated, pipeline, config)
        
        # Step 7: Comparison
        if tuning_result and tuning_result['best_mlir']:
            run_comparison(mlir_annotated, tuning_result['best_mlir'], pipeline)
    else:
        print("\n(Skipping tuning phase)")
    
    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()