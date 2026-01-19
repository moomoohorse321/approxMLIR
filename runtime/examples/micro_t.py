#!/usr/bin/env python3
"""Microbenchmark: Safety Contract Overhead

This microbenchmark measures the runtime overhead of ApproxMLIR's
try-check-recover safety contract mechanism.

What we measure:
- Direct kernel call (baseline)
- Kernel with safety contract (check always passes)
- Kernel with safety contract (check always fails -> recovery)

The overhead = (with_safety_contract - baseline) shows the cost of:
1. Inlined checker function call
2. scf.if branch evaluation
3. Recovery path (when check fails)
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
print("Microbenchmark: Safety Contract Overhead")
print("=" * 70)
print(f"\nConfig: DATA_SIZE={DATA_SIZE}, WARMUP={WARMUP_RUNS}, RUNS={BENCHMARK_RUNS}")
print(f"approx-opt: {APPROX_OPT}")

# =============================================================================
# Kernel Definitions
# =============================================================================

def simple_kernel(data: jax.Array) -> jax.Array:
    """Simple kernel - baseline without safety contract."""
    return data * 2.0 + 1.0


def checker_always_pass(result: jax.Array, data: jax.Array) -> bool:
    """Checker that always returns True (valid)."""
    # Simple check: result should be positive (always true for our kernel)
    return True


def checker_always_fail(result: jax.Array, data: jax.Array) -> bool:
    """Checker that always returns False (invalid) - triggers recovery."""
    return False


def recover_kernel(data: jax.Array) -> jax.Array:
    """Recovery function - recomputes with exact method."""
    return data * 2.0 + 1.0


# Kernel with safety contract (checker passes)
@ar.Knob(safety_contract=ar.SafetyContract(
    checker=checker_always_pass,
    recover=recover_kernel,
))
def kernel_with_safety_pass(data: jax.Array) -> jax.Array:
    """Kernel with safety contract where check always passes."""
    return data * 2.0 + 1.0


# Kernel with safety contract (checker fails)
@ar.Knob(safety_contract=ar.SafetyContract(
    checker=checker_always_fail,
    recover=recover_kernel,
))
def kernel_with_safety_fail(data: jax.Array) -> jax.Array:
    """Kernel with safety contract where check always fails (recovery triggered)."""
    return data * 2.0 + 1.0


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
    vmfb = ar.compile_to_iree(transformed, backend='llvm-cpu')
    modules, device = ar.load_module(vmfb, backend='llvm-cpu')
    
    print(f"  ✓ Compiled and loaded successfully")
    return modules.module, device  # Return modules.module to access functions directly


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    # Shapes
    DATA_SHAPE = jax.ShapeDtypeStruct((DATA_SIZE,), jnp.float32)
    
    # 1. Compile baseline (no safety contract)
    baseline_functions = {
        "simple_kernel": (simple_kernel, (DATA_SHAPE,)),
    }
    baseline_module, baseline_device = compile_and_load(
        "simple_kernel", simple_kernel, None, baseline_functions
    )
    
    # 2. Compile kernel with safety contract (check passes)
    config_pass = ar.get_config(kernel_with_safety_pass)
    helpers_pass = ar.collect_helper_functions("kernel_with_safety_pass", config_pass)
    
    safety_pass_functions = {
        "kernel_with_safety_pass": (kernel_with_safety_pass, (DATA_SHAPE,)),
    }
    for helper_name, helper_fn in helpers_pass:
        if helper_name == "checker_always_pass":
            # Checker: (result, data) -> bool
            safety_pass_functions[helper_name] = (helper_fn, (DATA_SHAPE, DATA_SHAPE))
        else:
            # Recover: (data) -> result
            safety_pass_functions[helper_name] = (helper_fn, (DATA_SHAPE,))
    
    safety_pass_module, safety_pass_device = compile_and_load(
        "kernel_with_safety_pass", kernel_with_safety_pass, config_pass, safety_pass_functions
    )
    
    # 3. Compile kernel with safety contract (check fails)
    config_fail = ar.get_config(kernel_with_safety_fail)
    helpers_fail = ar.collect_helper_functions("kernel_with_safety_fail", config_fail)
    
    safety_fail_functions = {
        "kernel_with_safety_fail": (kernel_with_safety_fail, (DATA_SHAPE,)),
    }
    for helper_name, helper_fn in helpers_fail:
        if helper_name == "checker_always_fail":
            safety_fail_functions[helper_name] = (helper_fn, (DATA_SHAPE, DATA_SHAPE))
        else:
            safety_fail_functions[helper_name] = (helper_fn, (DATA_SHAPE,))
    
    safety_fail_module, safety_fail_device = compile_and_load(
        "kernel_with_safety_fail", kernel_with_safety_fail, config_fail, safety_fail_functions
    )
    
    # Prepare test data
    test_data = np.ones(DATA_SIZE, dtype=np.float32)
    
    print("\n" + "=" * 70)
    print("Running Benchmarks")
    print("=" * 70)
    
    results = {}
    
    # Benchmark baseline
    if baseline_module:
        results['baseline'] = benchmark_function(
            baseline_module.simple_kernel, (test_data,)
        )
        print_results("Baseline (no safety contract)", results['baseline'])
    
    # Benchmark safety contract (check passes)
    if safety_pass_module:
        results['safety_pass'] = benchmark_function(
            safety_pass_module.kernel_with_safety_pass, (test_data,)
        )
        print_results("Safety Contract (check PASSES)", results['safety_pass'])
    
    # Benchmark safety contract (check fails)
    if safety_fail_module:
        results['safety_fail'] = benchmark_function(
            safety_fail_module.kernel_with_safety_fail, (test_data,)
        )
        print_results("Safety Contract (check FAILS → recovery)", results['safety_fail'])
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Safety Contract Overhead")
    print("=" * 70)
    
    if 'baseline' in results and 'safety_pass' in results:
        overhead_pass = results['safety_pass']['mean_ns'] - results['baseline']['mean_ns']
        overhead_pct = (overhead_pass / results['baseline']['mean_ns']) * 100
        print(f"\nOverhead (check passes): {overhead_pass:+.1f} ns ({overhead_pct:+.1f}%)")
    
    if 'baseline' in results and 'safety_fail' in results:
        overhead_fail = results['safety_fail']['mean_ns'] - results['baseline']['mean_ns']
        overhead_pct = (overhead_fail / results['baseline']['mean_ns']) * 100
        print(f"Overhead (check fails):  {overhead_fail:+.1f} ns ({overhead_pct:+.1f}%)")
    
    if 'safety_pass' in results and 'safety_fail' in results:
        recovery_cost = results['safety_fail']['mean_ns'] - results['safety_pass']['mean_ns']
        print(f"Recovery path cost:      {recovery_cost:+.1f} ns")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()