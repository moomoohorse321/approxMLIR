"""Tests for iree.py - IREE deployment."""

import pytest
from approx_runtime.iree import IREE_AVAILABLE


@pytest.mark.skipif(not IREE_AVAILABLE, reason="IREE not available")
class TestCompileToIree:
    def test_compile_simple_module(self):
        from approx_runtime import compile_to_iree
        
        # Simple identity function in core MLIR dialects
        mlir = '''
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}'''
        vmfb = compile_to_iree(mlir, backend="llvm-cpu", input_type="auto")
        
        assert isinstance(vmfb, bytes)
        assert len(vmfb) > 0


@pytest.mark.skipif(not IREE_AVAILABLE, reason="IREE not available")  
class TestLoadModule:
    def test_load_and_run(self):
        import numpy as np
        from iree import runtime as ireert
        from approx_runtime import compile_to_iree, load_module
        
        mlir = '''
module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}'''
        vmfb = compile_to_iree(mlir, backend="llvm-cpu", input_type="auto")
        modules, device = load_module(vmfb, backend="cpu")
        
        input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        device_input = ireert.asdevicearray(device, input_data)
        result = modules.module.main(device_input)
        
        np.testing.assert_array_equal(np.array(result), input_data)


class TestIreeNotAvailable:
    def test_compile_raises_without_iree(self):
        if IREE_AVAILABLE:
            pytest.skip("IREE is available")
        
        from approx_runtime import compile_to_iree
        
        with pytest.raises(ImportError, match="IREE compiler"):
            compile_to_iree("module {}")
    
    def test_load_raises_without_iree(self):
        if IREE_AVAILABLE:
            pytest.skip("IREE is available")
        
        from approx_runtime import load_module
        
        with pytest.raises(ImportError, match="IREE runtime"):
            load_module(b"fake")
