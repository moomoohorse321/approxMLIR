"""Tests for knob.py - decorators and configuration."""

import pytest
from approx_runtime import (
    DecisionTree,
    SafetyContract,
    StaticTransform,
    Knob,
    get_config,
)


# Mock functions for testing
def mock_state_fn(x):
    return x

def mock_checker(result, x):
    return result > 0

def mock_recover(x):
    return x * 2

def mock_approx_v1(x):
    return x

def mock_approx_v2(x):
    return x * 2


class TestDecisionTree:
    def test_valid_decision_tree(self):
        dt = DecisionTree(
            state_function=mock_state_fn,
            state_indices=[0],
            thresholds=[100, 500],
            decisions=[0, 1, 2],
            transform_type="loop_perforate",
        )
        assert dt.state_function == mock_state_fn
        assert dt.thresholds == [100, 500]
        assert dt.decisions == [0, 1, 2]
        assert dt.transform_type == "loop_perforate"
    
    def test_decisions_length_mismatch(self):
        with pytest.raises(ValueError, match="decisions length"):
            DecisionTree(
                state_function=mock_state_fn,
                state_indices=[0],
                thresholds=[100, 500],
                decisions=[0, 1],  # Should be 3
                transform_type="loop_perforate",
            )
    
    def test_invalid_transform_type(self):
        with pytest.raises(ValueError, match="transform_type"):
            DecisionTree(
                state_function=mock_state_fn,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],
                transform_type="invalid_type",
            )
    
    def test_all_transform_types(self):
        for tt in ["loop_perforate", "task_skipping"]:
            dt = DecisionTree(
                state_function=mock_state_fn,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],
                transform_type=tt,
            )
            assert dt.transform_type == tt
    
    def test_func_substitute_requires_approx_kernels(self):
        """func_substitute with non-zero decisions needs approx_kernels."""
        with pytest.raises(ValueError, match="approx_kernels"):
            DecisionTree(
                state_function=mock_state_fn,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],  # 1 is non-zero, needs approx_kernel
                transform_type="func_substitute",
            )
    
    def test_func_substitute_with_approx_kernels(self):
        """func_substitute works with proper approx_kernels."""
        dt = DecisionTree(
            state_function=mock_state_fn,
            state_indices=[0],
            thresholds=[100],
            decisions=[0, 1],
            transform_type="func_substitute",
            approx_kernels={1: mock_approx_v1},
        )
        assert dt.approx_kernels[1] == mock_approx_v1
    
    def test_func_substitute_zero_decisions_no_kernels_needed(self):
        """func_substitute with only decision=0 doesn't need kernels."""
        dt = DecisionTree(
            state_function=mock_state_fn,
            state_indices=[0],
            thresholds=[100],
            decisions=[0, 0],  # All zeros = exact, no approx needed
            transform_type="func_substitute",
        )
        assert dt.transform_type == "func_substitute"


class TestSafetyContract:
    def test_safety_contract(self):
        sc = SafetyContract(
            checker=mock_checker,
            recover=mock_recover,
        )
        assert sc.checker == mock_checker
        assert sc.recover == mock_recover


class TestStaticTransform:
    def test_valid_static_transform(self):
        st = StaticTransform(
            transform_type="loop_perforate",
            knob_val=2,
        )
        assert st.transform_type == "loop_perforate"
        assert st.knob_val == 2
    
    def test_invalid_transform_type(self):
        with pytest.raises(ValueError, match="transform_type"):
            StaticTransform(
                transform_type="invalid",
                knob_val=1,
            )
    
    def test_func_substitute_requires_approx_kernel(self):
        """func_substitute with knob_val > 0 needs approx_kernel."""
        with pytest.raises(ValueError, match="approx_kernel"):
            StaticTransform(
                transform_type="func_substitute",
                knob_val=1,
            )
    
    def test_func_substitute_zero_no_kernel_needed(self):
        """func_substitute with knob_val=0 (exact) doesn't need kernel."""
        st = StaticTransform(
            transform_type="func_substitute",
            knob_val=0,
        )
        assert st.knob_val == 0
    
    def test_func_substitute_with_kernel(self):
        """func_substitute with approx_kernel works."""
        st = StaticTransform(
            transform_type="func_substitute",
            knob_val=1,
            approx_kernel=mock_approx_v1,
        )
        assert st.approx_kernel == mock_approx_v1


class TestKnobDecorator:
    def test_decorator_attaches_config(self):
        @Knob(static_transform=StaticTransform("loop_perforate", 2))
        def my_func(x):
            return x
        
        config = get_config(my_func)
        assert config is not None
        assert config['static_transform'].knob_val == 2
    
    def test_decision_tree_decorator(self):
        @Knob(decision_tree=DecisionTree(
            state_function=mock_state_fn,
            state_indices=[0],
            thresholds=[100],
            decisions=[0, 1],
            transform_type="loop_perforate",
        ))
        def my_func(n, data):
            return data
        
        config = get_config(my_func)
        assert config['decision_tree'] is not None
        assert config['decision_tree'].thresholds == [100]
    
    def test_safety_contract_decorator(self):
        @Knob(safety_contract=SafetyContract(
            checker=mock_checker,
            recover=mock_recover,
        ))
        def my_func(x):
            return x
        
        config = get_config(my_func)
        assert config['safety_contract'] is not None
        assert config['safety_contract'].checker == mock_checker
    
    def test_combined_decision_tree_and_safety(self):
        @Knob(
            decision_tree=DecisionTree(
                state_function=mock_state_fn,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],
                transform_type="loop_perforate",
            ),
            safety_contract=SafetyContract(
                checker=mock_checker,
                recover=mock_recover,
            ),
        )
        def my_func(n, data):
            return data
        
        config = get_config(my_func)
        assert config['decision_tree'] is not None
        assert config['safety_contract'] is not None
    
    def test_mutually_exclusive_decision_tree_and_static(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            @Knob(
                decision_tree=DecisionTree(
                    state_function=mock_state_fn,
                    state_indices=[0],
                    thresholds=[100],
                    decisions=[0, 1],
                    transform_type="loop_perforate",
                ),
                static_transform=StaticTransform("loop_perforate", 2),
            )
            def my_func(x):
                return x
    
    def test_undecorated_function(self):
        def plain_func(x):
            return x
        
        assert get_config(plain_func) is None
    
    def test_function_still_callable(self):
        @Knob(static_transform=StaticTransform("loop_perforate", 2))
        def add_one(x):
            return x + 1
        
        assert add_one(5) == 6
