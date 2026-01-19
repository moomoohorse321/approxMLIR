"""Tests for mlir_gen.py - MLIR annotation generation."""

import pytest
from approx_runtime import (
    DecisionTree,
    SafetyContract,
    StaticTransform,
    generate_annotations,
    inject_annotations,
    collect_helper_functions,
    needs_pre_emit_transform,
)


# Mock functions
def get_state(x):
    return x

def checker(result, x):
    return True

def recover(x):
    return x

def approx_v1(x):
    return x

def approx_v2(x):
    return x * 2


class TestGenerateAnnotations:
    def test_decision_tree_annotation(self):
        config = {
            'decision_tree': DecisionTree(
                state_function=get_state,
                state_indices=[0, 1],
                thresholds=[100, 500],
                decisions=[0, 1, 2],
                transform_type="loop_perforate",
            ),
            'safety_contract': None,
            'static_transform': None,
        }
        
        result = generate_annotations("my_func", config)
        
        assert 'approx.util.annotation.decision_tree' in result
        assert 'func_name = "my_func"' in result
        assert 'transform_type = "loop_perforate"' in result
        assert 'state_function = "get_state"' in result
        assert 'state_indices = array<i64: 0, 1>' in result
        assert 'thresholds = array<i32: 100, 500>' in result
        assert 'decisions = array<i32: 0, 1, 2>' in result
    
    def test_safety_contract_annotation(self):
        config = {
            'decision_tree': None,
            'safety_contract': SafetyContract(
                checker=checker,
                recover=recover,
            ),
            'static_transform': None,
        }
        
        result = generate_annotations("my_func", config)
        
        assert 'approx.util.annotation.try' in result
        assert 'func_name = "my_func"' in result
        assert 'checker = "checker"' in result
        assert 'recover = "recover"' in result
    
    def test_static_transform_annotation(self):
        config = {
            'decision_tree': None,
            'safety_contract': None,
            'static_transform': StaticTransform(
                transform_type="loop_perforate",
                knob_val=3,
            ),
        }
        
        result = generate_annotations("kernel", config)
        
        assert 'approx.util.annotation.transform' in result
        assert 'func_name = "kernel"' in result
        assert 'transform_type = "loop_perforate"' in result
        assert 'knob_val = 3 : i32' in result
    
    def test_combined_annotations(self):
        config = {
            'decision_tree': DecisionTree(
                state_function=get_state,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],
                transform_type="loop_perforate",
            ),
            'safety_contract': SafetyContract(
                checker=checker,
                recover=recover,
            ),
            'static_transform': None,
        }
        
        result = generate_annotations("my_func", config)
        
        assert 'approx.util.annotation.decision_tree' in result
        assert 'approx.util.annotation.try' in result
    
    def test_empty_config(self):
        config = {
            'decision_tree': None,
            'safety_contract': None,
            'static_transform': None,
        }
        
        result = generate_annotations("my_func", config)
        assert result == ""
    
    def test_func_substitute_emits_convert_to_call(self):
        """func_substitute should emit convert_to_call annotation first."""
        config = {
            'decision_tree': DecisionTree(
                state_function=get_state,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],
                transform_type="func_substitute",
                approx_kernels={1: approx_v1},
            ),
            'safety_contract': None,
            'static_transform': None,
        }
        
        result = generate_annotations("my_func", config)
        
        # Should have convert_to_call BEFORE decision_tree
        assert 'approx.util.annotation.convert_to_call' in result
        assert 'approx.util.annotation.decision_tree' in result
        
        convert_pos = result.index('convert_to_call')
        decision_pos = result.index('decision_tree')
        assert convert_pos < decision_pos


class TestCollectHelperFunctions:
    def test_collect_state_function(self):
        config = {
            'decision_tree': DecisionTree(
                state_function=get_state,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],
                transform_type="loop_perforate",
            ),
            'safety_contract': None,
            'static_transform': None,
        }
        
        helpers = collect_helper_functions("my_func", config)
        
        names = [name for name, fn in helpers]
        assert "get_state" in names
    
    def test_collect_safety_contract_functions(self):
        config = {
            'decision_tree': None,
            'safety_contract': SafetyContract(
                checker=checker,
                recover=recover,
            ),
            'static_transform': None,
        }
        
        helpers = collect_helper_functions("my_func", config)
        
        names = [name for name, fn in helpers]
        assert "checker" in names
        assert "recover" in names
    
    def test_collect_approx_kernels(self):
        config = {
            'decision_tree': DecisionTree(
                state_function=get_state,
                state_indices=[0],
                thresholds=[100, 200],  # 2 thresholds -> 3 decisions
                decisions=[0, 1, 2],
                transform_type="func_substitute",
                approx_kernels={1: approx_v1, 2: approx_v2},
            ),
            'safety_contract': None,
            'static_transform': None,
        }
        
        helpers = collect_helper_functions("my_kernel", config)
        
        names = [name for name, fn in helpers]
        assert "approx_my_kernel_1" in names
        assert "approx_my_kernel_2" in names
    
    def test_collect_static_approx_kernel(self):
        config = {
            'decision_tree': None,
            'safety_contract': None,
            'static_transform': StaticTransform(
                transform_type="func_substitute",
                knob_val=1,
                approx_kernel=approx_v1,
            ),
        }
        
        helpers = collect_helper_functions("my_func", config)
        
        names = [name for name, fn in helpers]
        assert "approx_my_func_1" in names


class TestNeedsPreEmitTransform:
    def test_loop_perforate_no_pre_emit(self):
        config = {
            'decision_tree': DecisionTree(
                state_function=get_state,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],
                transform_type="loop_perforate",
            ),
            'safety_contract': None,
            'static_transform': None,
        }
        
        assert not needs_pre_emit_transform(config)
    
    def test_func_substitute_needs_pre_emit(self):
        config = {
            'decision_tree': DecisionTree(
                state_function=get_state,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],
                transform_type="func_substitute",
                approx_kernels={1: approx_v1},
            ),
            'safety_contract': None,
            'static_transform': None,
        }
        
        assert needs_pre_emit_transform(config)
    
    def test_static_func_substitute_needs_pre_emit(self):
        config = {
            'decision_tree': None,
            'safety_contract': None,
            'static_transform': StaticTransform(
                transform_type="func_substitute",
                knob_val=1,
                approx_kernel=approx_v1,
            ),
        }
        
        assert needs_pre_emit_transform(config)


class TestInjectAnnotations:
    def test_inject_into_module(self):
        mlir = '''module {
  func.func @my_func(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}'''
        config = {
            'decision_tree': None,
            'safety_contract': None,
            'static_transform': StaticTransform("loop_perforate", 2),
        }
        
        result = inject_annotations(mlir, "my_func", config)
        
        assert 'approx.util.annotation.transform' in result
        assert result.index('approx.util.annotation.transform') < result.index('func.func')
    
    def test_inject_into_named_module(self):
        mlir = '''module @mymodule {
  func.func @kernel() {
    return
  }
}'''
        config = {
            'decision_tree': None,
            'safety_contract': None,
            'static_transform': StaticTransform("task_skipping", 1),
        }
        
        result = inject_annotations(mlir, "kernel", config)
        
        assert 'approx.util.annotation.transform' in result
    
    def test_inject_empty_config(self):
        mlir = '''module {
  func.func @f() { return }
}'''
        
        result = inject_annotations(mlir, "f", None)
        assert result == mlir
        
        result = inject_annotations(mlir, "f", {})
        assert result == mlir
    
    def test_full_decision_tree_injection(self):
        mlir = '''module {
  func.func @process(%n: i32, %data: tensor<1000xf32>) -> tensor<1000xf32> {
    return %data : tensor<1000xf32>
  }
}'''
        config = {
            'decision_tree': DecisionTree(
                state_function=get_state,
                state_indices=[0],
                thresholds=[100, 500],
                decisions=[0, 1, 2],
                transform_type="loop_perforate",
                thresholds_lower=[0],
                thresholds_upper=[1000],
            ),
            'safety_contract': None,
            'static_transform': None,
        }
        
        result = inject_annotations(mlir, "process", config)
        
        # Check all required fields are present
        assert 'func_name = "process"' in result
        assert 'num_thresholds = 2' in result
        assert 'thresholds_uppers = array<i32: 1000>' in result
        assert 'thresholds_lowers = array<i32: 0>' in result
    
    def test_inject_into_module_with_attributes(self):
        """Test injection into JAX-style module with attributes."""
        mlir = '''#loc1 = loc("data")
module @jit_my_kernel attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32} {
  func.func public @main(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    return %arg0 : tensor<1024xf32>
  }
}'''
        config = {
            'decision_tree': None,
            'safety_contract': None,
            'static_transform': StaticTransform("loop_perforate", 2),
        }
        
        result = inject_annotations(mlir, "my_func", config)
        
        # Annotation should be inside the module
        assert 'approx.util.annotation.transform' in result
        module_idx = result.index('module @jit_my_kernel attributes')
        annotation_idx = result.index('approx.util.annotation.transform')
        func_idx = result.index('func.func')
        
        # Annotation should be after module { but before func.func
        assert module_idx < annotation_idx < func_idx
