#!/usr/bin/env python3
"""
Test driver for JAX → MLIR → IREE kernel compilation and execution.

Tests:
1. matmul kernel on CPU
2. matmul kernel on CUDA (if available)
3. branch kernel on CPU (all 4 branches)
"""

import numpy as np
import jax.numpy as jnp
from runtime.kernel_runtime import (
    CompiledKernel,
    matmul_kernel,
    branch_kernel,
    check_cuda_available,
    compile_to_mlir,
    compile_to_iree,
    load_iree_module,
    save_mlir,
)
from iree import runtime as ireert


def test_matmul_cpu():
    """Test matmul kernel on CPU backend."""
    print("\n" + "=" * 60)
    print("TEST: matmul_kernel on CPU")
    print("=" * 60)
    
    # Define shapes
    M, K, N = 128, 256, 64
    a = jnp.ones((M, K), dtype=jnp.float32) * 2.0
    b = jnp.ones((K, N), dtype=jnp.float32) * 3.0
    
    # Compile
    kernel = CompiledKernel(matmul_kernel, a, b, backend="cpu", name="matmul_cpu")
    
    # Run
    result = kernel(a, b)
    
    # Verify: (2 * 3) * K = 6 * 256 = 1536
    expected = np.full((M, N), 6.0 * K, dtype=np.float32)
    
    if np.allclose(result, expected):
        print(f"✅ PASSED: matmul_cpu")
        print(f"   Shape: {result.shape}, Expected value: {expected[0, 0]}, Got: {result[0, 0]}")
        return True
    else:
        print(f"❌ FAILED: matmul_cpu")
        print(f"   Expected: {expected[0, 0]}, Got: {result[0, 0]}")
        return False


def test_matmul_cuda():
    """Test matmul kernel on CUDA backend."""
    print("\n" + "=" * 60)
    print("TEST: matmul_kernel on CUDA")
    print("=" * 60)
    
    if not check_cuda_available():
        print("⏭️  SKIPPED: CUDA not available")
        return None
    
    # Define shapes
    M, K, N = 128, 256, 64
    a = jnp.ones((M, K), dtype=jnp.float32) * 2.0
    b = jnp.ones((K, N), dtype=jnp.float32) * 3.0
    
    # Compile
    kernel = CompiledKernel(matmul_kernel, a, b, backend="cuda", name="matmul_cuda")
    
    # Run
    result = kernel(a, b)
    
    # Verify
    expected = np.full((M, N), 6.0 * K, dtype=np.float32)
    
    if np.allclose(result, expected):
        print(f"✅ PASSED: matmul_cuda")
        print(f"   Shape: {result.shape}, Expected value: {expected[0, 0]}, Got: {result[0, 0]}")
        return True
    else:
        print(f"❌ FAILED: matmul_cuda")
        print(f"   Expected: {expected[0, 0]}, Got: {result[0, 0]}")
        return False


def test_branch_cpu():
    """Test branch kernel on CPU backend with all 4 branches."""
    print("\n" + "=" * 60)
    print("TEST: branch_kernel on CPU (all 4 branches)")
    print("=" * 60)
    
    # Input array
    x = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    
    # We need to compile with example inputs
    # condition is a scalar int, x is a 1D array
    example_condition = jnp.array(0, dtype=jnp.int32)
    
    # Compile once (condition value doesn't matter for tracing, just shape/dtype)
    kernel = CompiledKernel(branch_kernel, example_condition, x, backend="cpu", name="branch_cpu")
    
    # Test all branches
    test_cases = [
        (0, x * 2.0, "double"),           # branch 0: x * 2
        (1, x + 100.0, "shift"),          # branch 1: x + 100
        (2, -x, "negate"),                # branch 2: -x
        (3, x ** 2, "square"),            # branch 3: x ** 2
    ]
    
    all_passed = True
    for branch_id, expected, name in test_cases:
        condition = jnp.array(branch_id, dtype=jnp.int32)
        result = kernel(condition, x)
        
        if np.allclose(result, expected):
            print(f"  ✅ Branch {branch_id} ({name}): {result.tolist()}")
        else:
            print(f"  ❌ Branch {branch_id} ({name}): expected {expected.tolist()}, got {result.tolist()}")
            all_passed = False
    
    if all_passed:
        print(f"✅ PASSED: branch_cpu (all branches)")
    else:
        print(f"❌ FAILED: branch_cpu")
    
    return all_passed


def test_mlir_output():
    """Test that MLIR output is generated correctly."""
    print("\n" + "=" * 60)
    print("TEST: MLIR generation")
    print("=" * 60)
    
    # Generate MLIR for matmul
    a = jnp.zeros((4, 8), dtype=jnp.float32)
    b = jnp.zeros((8, 16), dtype=jnp.float32)
    
    mlir = compile_to_mlir(matmul_kernel, a, b)
    
    # Check it contains expected patterns
    checks = [
        ("module" in mlir, "has module declaration"),
        ("stablehlo" in mlir.lower() or "func" in mlir, "has function/stablehlo ops"),
        ("f32" in mlir, "has f32 type"),
    ]
    
    all_passed = True
    for check, desc in checks:
        if check:
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ {desc}")
            all_passed = False
    
    # Save for inspection
    save_mlir(mlir, "/home/claude/matmul.mlir")
    
    if all_passed:
        print(f"✅ PASSED: MLIR generation")
    return all_passed


def main():
    """Run all tests."""
    print("=" * 60)
    print("JAX → MLIR → IREE Kernel Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test MLIR generation
    results["mlir_generation"] = test_mlir_output()
    
    # Test matmul on CPU
    results["matmul_cpu"] = test_matmul_cpu()
    
    # Test matmul on CUDA
    results["matmul_cuda"] = test_matmul_cuda()
    
    # Test branch kernel on CPU
    results["branch_cpu"] = test_branch_cpu()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for name, result in results.items():
        status = "✅ PASS" if result is True else ("❌ FAIL" if result is False else "⏭️  SKIP")
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)