"""Tests for export.py - JAX to MLIR export."""

import pytest
from approx_runtime import get_function_name
from approx_runtime.export import JAX_AVAILABLE, _extract_and_rename_func

# Skip JAX tests if not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")


class TestGetFunctionName:
    def test_simple_function(self):
        def my_func(x):
            return x
        
        assert get_function_name(my_func) == "my_func"
    
    def test_lambda(self):
        f = lambda x: x
        assert get_function_name(f) == "<lambda>"


class TestExtractAndRenameFunc:
    """Test the MLIR function extraction and renaming logic."""
    
    def test_rename_simple_func(self):
        mlir = '''module @test {
  func.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}'''
        result = _extract_and_rename_func(mlir, "my_kernel")
        assert result is not None
        assert "@my_kernel" in result
        assert "@main" not in result
        assert "tensor<4xf32>" in result
    
    def test_rename_with_operations(self):
        mlir = '''module {
  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %cst : tensor<f32>
    return %0 : tensor<f32>
  }
}'''
        result = _extract_and_rename_func(mlir, "scale_fn")
        assert result is not None
        assert "@scale_fn" in result
        assert "stablehlo.multiply" in result


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestExportToMlir:
    def test_export_simple_function(self):
        import jax.numpy as jnp
        from approx_runtime import export_to_mlir
        
        def add(x, y):
            return x + y
        
        mlir = export_to_mlir(add, jnp.array(1.0), jnp.array(2.0))
        
        assert "module" in mlir
        assert "stablehlo" in mlir or "func" in mlir
    
    def test_export_with_knob_decorator(self):
        import jax.numpy as jnp
        from approx_runtime import export_to_mlir, Knob, StaticTransform
        
        @Knob(static_transform=StaticTransform("loop_perforate", 2))
        def my_kernel(x):
            return jnp.sum(x)
        
        mlir = export_to_mlir(my_kernel, jnp.zeros(100))
        
        assert "module" in mlir


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available") 
class TestExportModule:
    def test_export_single_function(self):
        import jax
        import jax.numpy as jnp
        from approx_runtime import export_module
        
        def my_func(x):
            return x * 2.0
        
        functions = {
            "my_func": (my_func, (jax.ShapeDtypeStruct((4,), jnp.float32),))
        }
        mlir = export_module(functions)
        
        assert "module" in mlir
        assert "@my_func" in mlir
    
    def test_export_multiple_functions(self):
        import jax
        import jax.numpy as jnp
        from approx_runtime import export_module
        
        def kernel(data):
            return data * 2.0
        
        def get_state(n):
            return n
        
        functions = {
            "my_kernel": (kernel, (jax.ShapeDtypeStruct((1024,), jnp.float32),)),
            "get_state": (get_state, (jax.ShapeDtypeStruct((), jnp.int32),)),
        }
        mlir = export_module(functions)
        
        assert "module" in mlir
        assert "@my_kernel" in mlir
        assert "@get_state" in mlir
    
    def test_export_empty_returns_empty_module(self):
        from approx_runtime import export_module
        
        mlir = export_module({})
        assert mlir == "module {}\n"
    
    def test_export_with_module_name(self):
        import jax
        import jax.numpy as jnp
        from approx_runtime import export_module
        
        def f(x):
            return x
        
        functions = {"f": (f, (jax.ShapeDtypeStruct((), jnp.float32),))}
        mlir = export_module(functions, module_name="my_module")
        
        assert "@my_module" in mlir
