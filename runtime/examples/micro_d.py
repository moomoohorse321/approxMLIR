#!/usr/bin/env python3
"""Microbenchmark: Decision Tree Overhead

This microbenchmark measures the runtime overhead of ApproxMLIR's
decision tree mechanism for dynamic approximation selection.

What we measure:
- Direct kernel call (baseline)
- Kernel with decision tree (single threshold)
- Kernel with decision tree (multiple thresholds)

The overhead = (with_decision_tree - baseline) shows the cost of:
1. get_state() function call
2. Threshold comparisons (arith.cmpi, arith.select)
3. stablehlo.case / scf.index_switch dispatch
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import approx_runtime as ar

# =============================================================================
# Configuration
# =============================================================================

APPROX_OPT = os.environ.get(
    "APPROX_OPT_PATH",
    os.path.expanduser("~/iree-build/approxMLIR/build/bin/approx-opt")
)

DATA_SIZE = 1024
WARMUP_RUNS = 50
BENCHMARK_RUNS = 1000

print("=" * 70)
print("Microbenchmark: Decision Tree Overhead")
print("=" * 70)
print(f"\nConfig: DATA_SIZE={DATA_SIZE}, WARMUP={WARMUP_RUNS}, RUNS={BENCHMARK_RUNS}")
print(f"approx-opt: {APPROX_OPT}")

# =============================================================================
# Kernel Definitions
# =============================================================================

def get_state(n: int) -> int:
    """State function - just returns n."""
    return n


def simple_kernel(n: int, data: jax.Array) -> jax.Array:
    """Simple kernel - baseline without decision tree."""
    dummy = (n * 0).astype(data.dtype)  # Keep n as dependency
    return data * 2.0 + dummy


def approx_kernel_1(n: int, data: jax.Array) -> jax.Array:
    """Approximate kernel v1."""
    dummy = (n * 0).astype(data.dtype)
    return data * 1.9 + dummy


def approx_kernel_2(n: int, data: jax.Array) -> jax.Array:
    """Approximate kernel v2."""
    dummy = (n * 0).astype(data.dtype)
    return data * 1.8 + dummy


def approx_kernel_3(n: int, data: jax.Array) -> jax.Array:
    """Approximate kernel v3."""
    dummy = (n * 0).astype(data.dtype)
    return data * 1.7 + dummy


def approx_kernel_4(n: int, data: jax.Array) -> jax.Array:
    """Approximate kernel v4."""
    dummy = (n * 0).astype(data.dtype)
    return data * 1.6 + dummy


# Kernel with 1 threshold (2 branches)
@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=get_state,
    state_indices=[0],
    thresholds=[100],              # 1 threshold -> 2 branches
    decisions=[0, 1],              # exact or approx_1
    transform_type="func_substitute",
    approx_kernels={1: approx_kernel_1},
))
def kernel_1_threshold(n: int, data: jax.Array) -> jax.Array:
    """Kernel with 1 threshold decision tree."""
    dummy = (n * 0).astype(data.dtype)
    return data * 2.0 + dummy


# Kernel with 2 thresholds (3 branches)
@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=get_state,
    state_indices=[0],
    thresholds=[100, 500],         # 2 thresholds -> 3 branches
    decisions=[0, 1, 2],           # exact, approx_1, approx_2
    transform_type="func_substitute",
    approx_kernels={1: approx_kernel_1, 2: approx_kernel_2},
))
def kernel_2_thresholds(n: int, data: jax.Array) -> jax.Array:
    """Kernel with 2 thresholds decision tree."""
    dummy = (n * 0).astype(data.dtype)
    return data * 2.0 + dummy


# Kernel with 4 thresholds (5 branches)
@ar.Knob(decision_tree=ar.DecisionTree(
    state_function=get_state,
    state_indices=[0],
    thresholds=[100, 300, 500, 700],  # 4 thresholds -> 5 branches
    decisions=[0, 1, 2, 3, 4],         # exact, approx_1, approx_2, approx_3, approx_4
    transform_type="func_substitute",
    approx_kernels={
        1: approx_kernel_1,
        2: approx_kernel_2,
        3: approx_kernel_3,
        4: approx_kernel_4,
    },
))
def kernel_4_thresholds(n: int, data: jax.Array) -> jax.Array:
    """Kernel with 4 thresholds decision tree."""
    dummy = (n * 0).astype(data.dtype)
    return data * 2.0 + dummy


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_function(fn, args, warmup=WARMUP_RUNS, runs=BENCHMARK_RUNS):
    """Benchmark a single function, returns timing statistics in nanoseconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    
    # Timed runs
    times_ns = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn(*args)
        end = time.perf_counter_ns()
        times_ns.append(end - start)
    
    return {
        'mean_ns': np.mean(times_ns),
        'std_ns': np.std(times_ns),
        'min_ns': np.min(times_ns),
        'max_ns': np.max(times_ns),
        'median_ns': np.median(times_ns),
        'p99_ns': np.percentile(times_ns, 99),
    }


def print_results(name, stats):
    """Pretty print benchmark results."""
    print(f"\n{name}:")
    print(f"  Mean:   {stats['mean_ns']:>10.1f} ns")
    print(f"  Median: {stats['median_ns']:>10.1f} ns")
    print(f"  Std:    {stats['std_ns']:>10.1f} ns")
    print(f"  Min:    {stats['min_ns']:>10.1f} ns")
    print(f"  Max:    {stats['max_ns']:>10.1f} ns")
    print(f"  P99:    {stats['p99_ns']:>10.1f} ns")


def compile_and_load(name, kernel_fn, config, functions_dict):
    """Compile kernel with approx-opt and load via IREE."""
    print(f"\nCompiling {name}...")
    
    # Export to MLIR
    mlir = ar.export_module(functions_dict)
    
    # Inject annotations if config exists
    if config:
        mlir = ar.inject_annotations(mlir, name, config)
        pipeline = ar.get_pipeline_for_config(config)
    else:
        pipeline = ar.PASS_PIPELINE
    
    # Compile with approx-opt
    try:
        transformed = ar.compile(mlir, passes=pipeline, approx_opt_path=APPROX_OPT)
    except ar.CompilationError as e:
        print(f"  Compilation failed: {e}")
        return None, None
    
    # Compile to IREE and load
    vmfb = ar.compile_to_iree(transformed, backend='cuda')
    modules, device = ar.load_module(vmfb, backend='cuda')
    
    print(f"  ✓ Compiled and loaded successfully")
    return modules.module, device  # Return modules.module to access functions directly


def build_functions_dict(name, kernel_fn, config):
    """Build the functions dictionary for export_module."""
    DATA_SHAPE = jax.ShapeDtypeStruct((DATA_SIZE,), jnp.float32)
    N_SHAPE = jax.ShapeDtypeStruct((), jnp.int32)
    
    functions = {
        name: (kernel_fn, (N_SHAPE, DATA_SHAPE)),
    }
    
    if config:
        helpers = ar.collect_helper_functions(name, config)
        for helper_name, helper_fn in helpers:
            if helper_name == "get_state":
                functions[helper_name] = (helper_fn, (N_SHAPE,))
            else:
                # Approx kernels have same signature as main kernel
                functions[helper_name] = (helper_fn, (N_SHAPE, DATA_SHAPE))
    
    return functions


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    DATA_SHAPE = jax.ShapeDtypeStruct((DATA_SIZE,), jnp.float32)
    N_SHAPE = jax.ShapeDtypeStruct((), jnp.int32)
    
    # 1. Compile baseline (no decision tree)
    baseline_functions = {
        "simple_kernel": (simple_kernel, (N_SHAPE, DATA_SHAPE)),
    }
    baseline_module, baseline_device = compile_and_load(
        "simple_kernel", simple_kernel, None, baseline_functions
    )
    
    # 2. Compile kernel with 1 threshold
    config_1t = ar.get_config(kernel_1_threshold)
    functions_1t = build_functions_dict("kernel_1_threshold", kernel_1_threshold, config_1t)
    module_1t, device_1t = compile_and_load(
        "kernel_1_threshold", kernel_1_threshold, config_1t, functions_1t
    )
    
    # 3. Compile kernel with 2 thresholds
    config_2t = ar.get_config(kernel_2_thresholds)
    functions_2t = build_functions_dict("kernel_2_thresholds", kernel_2_thresholds, config_2t)
    module_2t, device_2t = compile_and_load(
        "kernel_2_thresholds", kernel_2_thresholds, config_2t, functions_2t
    )
    
    # 4. Compile kernel with 4 thresholds
    config_4t = ar.get_config(kernel_4_thresholds)
    functions_4t = build_functions_dict("kernel_4_thresholds", kernel_4_thresholds, config_4t)
    module_4t, device_4t = compile_and_load(
        "kernel_4_thresholds", kernel_4_thresholds, config_4t, functions_4t
    )
    
    # Prepare test data
    test_data = np.ones(DATA_SIZE, dtype=np.float32)
    n_value = np.array(50, dtype=np.int32)  # Will select branch 0 (exact) in all cases
    
    print("\n" + "=" * 70)
    print("Running Benchmarks")
    print("=" * 70)
    
    results = {}
    
    # Benchmark baseline
    if baseline_module:
        results['baseline'] = benchmark_function(
            baseline_module.simple_kernel, (n_value, test_data)
        )
        print_results("Baseline (no decision tree)", results['baseline'])
    
    # Benchmark 1 threshold
    if module_1t:
        results['1_threshold'] = benchmark_function(
            module_1t.kernel_1_threshold, (n_value, test_data)
        )
        print_results("Decision Tree (1 threshold, 2 branches)", results['1_threshold'])
    
    # Benchmark 2 thresholds
    if module_2t:
        results['2_thresholds'] = benchmark_function(
            module_2t.kernel_2_thresholds, (n_value, test_data)
        )
        print_results("Decision Tree (2 thresholds, 3 branches)", results['2_thresholds'])
    
    # Benchmark 4 thresholds
    if module_4t:
        results['4_thresholds'] = benchmark_function(
            module_4t.kernel_4_thresholds, (n_value, test_data)
        )
        print_results("Decision Tree (4 thresholds, 5 branches)", results['4_thresholds'])
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Decision Tree Overhead")
    print("=" * 70)
    
    baseline_ns = results.get('baseline', {}).get('mean_ns', 0)
    
    if baseline_ns > 0:
        print(f"\n{'Configuration':<40} {'Overhead (ns)':<15} {'Overhead (%)':<15}")
        print("-" * 70)
        
        for key, label in [
            ('1_threshold', '1 threshold (2 branches)'),
            ('2_thresholds', '2 thresholds (3 branches)'),
            ('4_thresholds', '4 thresholds (5 branches)'),
        ]:
            if key in results:
                overhead_ns = results[key]['mean_ns'] - baseline_ns
                overhead_pct = (overhead_ns / baseline_ns) * 100
                print(f"{label:<40} {overhead_ns:>+10.1f} ns   {overhead_pct:>+10.1f}%")
    
    # Per-threshold cost analysis
    print("\n" + "-" * 70)
    print("Per-threshold cost analysis:")
    
    if '1_threshold' in results and '2_thresholds' in results:
        cost_1_to_2 = results['2_thresholds']['mean_ns'] - results['1_threshold']['mean_ns']
        print(f"  Cost of adding 1 more threshold (1→2): {cost_1_to_2:+.1f} ns")
    
    if '2_thresholds' in results and '4_thresholds' in results:
        cost_2_to_4 = results['4_thresholds']['mean_ns'] - results['2_thresholds']['mean_ns']
        cost_per_threshold = cost_2_to_4 / 2
        print(f"  Cost of adding 2 more thresholds (2→4): {cost_2_to_4:+.1f} ns ({cost_per_threshold:+.1f} ns/threshold)")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()