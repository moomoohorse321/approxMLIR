"""Unit tests for the approx_runtime tuning module.

These tests verify:
1. MLIR parsing and modification (MLIRConfigManager)
2. OpenTuner integration (ApproxTunerInterface) with mocked evaluation
3. Configuration validation (ascending thresholds constraint)

Requirements:
    pip install --user opentuner pytest

Run tests:
    pytest test_tuner.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the tuning module components directly (not via __init__.py)
from tuner_config import MLIRConfigManager, TunableParam

# Try to import OpenTuner-dependent components
try:
    from tuner import ApproxTunerInterface, tune, create_tuner_arg_parser
    OPENTUNER_AVAILABLE = True
except ImportError:
    OPENTUNER_AVAILABLE = False


# ============================================================================
# Sample MLIR for testing (uses the current dialect syntax)
# ============================================================================

SAMPLE_MLIR_SINGLE_FUNC = '''
module {
  "approx.util.annotation.decision_tree"() <{
      func_name = "my_func",
      transform_type = "loop_perforate",
      state_indices = array<i64: 0>,
      state_function = "get_state",
      num_thresholds = 2 : i32,
      thresholds_uppers = array<i32: 100, 200>,
      thresholds_lowers = array<i32: 0, 50>,
      decision_values = array<i32: 0, 1, 2>,
      thresholds = array<i32: 50, 150>,
      decisions = array<i32: 0, 1, 2>
  }> : () -> ()
  
  func.func @my_func(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}
'''

SAMPLE_MLIR_MULTI_FUNC = '''
module {
  "approx.util.annotation.decision_tree"() <{
      func_name = "parse_embedding",
      transform_type = "func_substitute",
      state_indices = array<i64: 2>,
      state_function = "get_state",
      num_thresholds = 1 : i32,
      thresholds_uppers = array<i32: 4>,
      thresholds_lowers = array<i32: 1>,
      decision_values = array<i32: 0, 1, 2>,
      thresholds = array<i32: 3>,
      decisions = array<i32: 1, 2>
  }> : () -> ()

  "approx.util.annotation.decision_tree"() <{
      func_name = "compute_similarities",
      transform_type = "func_substitute",
      state_indices = array<i64: 0>,
      state_function = "get_compute_state",
      num_thresholds = 2 : i32,
      thresholds_uppers = array<i32: 10, 20>,
      thresholds_lowers = array<i32: 1, 5>,
      decision_values = array<i32: 0, 1, 2, 3>,
      thresholds = array<i32: 5, 15>,
      decisions = array<i32: 0, 1, 2>
  }> : () -> ()
  
  func.func @parse_embedding(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
  
  func.func @compute_similarities(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}
'''

SAMPLE_MLIR_NO_ANNOTATIONS = '''
module {
  func.func @my_func(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}
'''


# ============================================================================
# MLIRConfigManager Tests
# ============================================================================

class TestMLIRConfigManager:
    """Test MLIR parsing and modification - no OpenTuner dependency."""
    
    def test_parse_single_annotation(self):
        """Test parsing a single decision_tree annotation."""
        manager = MLIRConfigManager()
        params = manager.parse_annotations(SAMPLE_MLIR_SINGLE_FUNC)
        
        # Should have 2 thresholds + 3 decisions = 5 parameters
        assert len(params) == 5
        
        # Verify threshold parameters
        assert "my_func_threshold_0" in params
        t0 = params["my_func_threshold_0"]
        assert t0.lower == 0
        assert t0.upper == 100
        assert t0.current == 50
        assert t0.param_type == "threshold"
        assert t0.func_name == "my_func"
        assert t0.index == 0
        
        assert "my_func_threshold_1" in params
        t1 = params["my_func_threshold_1"]
        assert t1.lower == 50
        assert t1.upper == 200
        assert t1.current == 150
        
        # Verify decision parameters
        assert "my_func_decision_0" in params
        d0 = params["my_func_decision_0"]
        assert d0.lower == 0  # min(decision_values)
        assert d0.upper == 2  # max(decision_values)
        assert d0.current == 0
        assert d0.param_type == "decision"
        
        assert "my_func_decision_1" in params
        assert "my_func_decision_2" in params
    
    def test_parse_multiple_annotations(self):
        """Test parsing multiple decision_tree annotations."""
        manager = MLIRConfigManager()
        params = manager.parse_annotations(SAMPLE_MLIR_MULTI_FUNC)
        
        # parse_embedding: 1 threshold + 2 decisions = 3
        # compute_similarities: 2 thresholds + 3 decisions = 5
        # Total: 8 parameters
        assert len(params) == 8
        
        # Verify parse_embedding params
        assert "parse_embedding_threshold_0" in params
        assert params["parse_embedding_threshold_0"].lower == 1
        assert params["parse_embedding_threshold_0"].upper == 4
        
        # Verify compute_similarities params
        assert "compute_similarities_threshold_0" in params
        assert "compute_similarities_threshold_1" in params
        assert "compute_similarities_decision_0" in params
    
    def test_parse_no_annotations(self):
        """Test parsing MLIR with no annotations returns empty dict."""
        manager = MLIRConfigManager()
        params = manager.parse_annotations(SAMPLE_MLIR_NO_ANNOTATIONS)
        
        assert len(params) == 0
    
    def test_apply_config_single_param(self):
        """Test applying a single parameter change."""
        manager = MLIRConfigManager()
        config = {"my_func_threshold_0": 75}
        
        modified = manager.apply_config(SAMPLE_MLIR_SINGLE_FUNC, config)
        
        # Re-parse and verify
        new_params = manager.parse_annotations(modified)
        assert new_params["my_func_threshold_0"].current == 75
        
        # Other params should be unchanged
        assert new_params["my_func_threshold_1"].current == 150
        assert new_params["my_func_decision_0"].current == 0
    
    def test_apply_config_multiple_params(self):
        """Test applying multiple parameter changes."""
        manager = MLIRConfigManager()
        config = {
            "my_func_threshold_0": 60,
            "my_func_threshold_1": 180,
            "my_func_decision_1": 2,
        }
        
        modified = manager.apply_config(SAMPLE_MLIR_SINGLE_FUNC, config)
        
        # Re-parse and verify
        new_params = manager.parse_annotations(modified)
        assert new_params["my_func_threshold_0"].current == 60
        assert new_params["my_func_threshold_1"].current == 180
        assert new_params["my_func_decision_1"].current == 2
        
        # Unchanged params
        assert new_params["my_func_decision_0"].current == 0
        assert new_params["my_func_decision_2"].current == 2
    
    def test_apply_config_multi_func(self):
        """Test applying config to MLIR with multiple functions."""
        manager = MLIRConfigManager()
        config = {
            "parse_embedding_threshold_0": 2,
            "compute_similarities_threshold_0": 8,
            "compute_similarities_decision_1": 3,
        }
        
        modified = manager.apply_config(SAMPLE_MLIR_MULTI_FUNC, config)
        
        # Re-parse and verify
        new_params = manager.parse_annotations(modified)
        assert new_params["parse_embedding_threshold_0"].current == 2
        assert new_params["compute_similarities_threshold_0"].current == 8
        assert new_params["compute_similarities_decision_1"].current == 3
    
    def test_validate_ascending_thresholds_valid(self):
        """Test validation passes for ascending thresholds."""
        manager = MLIRConfigManager()
        params = manager.parse_annotations(SAMPLE_MLIR_SINGLE_FUNC)
        
        # Valid: 50 < 150 (ascending)
        config = {
            "my_func_threshold_0": 50,
            "my_func_threshold_1": 150,
            "my_func_decision_0": 0,
            "my_func_decision_1": 1,
            "my_func_decision_2": 2,
        }
        
        assert manager.validate_config(config, params) is True
    
    def test_validate_ascending_thresholds_invalid(self):
        """Test validation fails for non-ascending thresholds."""
        manager = MLIRConfigManager()
        params = manager.parse_annotations(SAMPLE_MLIR_SINGLE_FUNC)
        
        # Invalid: 180 > 100 (not ascending)
        config = {
            "my_func_threshold_0": 180,
            "my_func_threshold_1": 100,
            "my_func_decision_0": 0,
            "my_func_decision_1": 1,
            "my_func_decision_2": 2,
        }
        
        assert manager.validate_config(config, params) is False
    
    def test_validate_ascending_thresholds_equal(self):
        """Test validation fails for equal thresholds (must be strictly ascending)."""
        manager = MLIRConfigManager()
        params = manager.parse_annotations(SAMPLE_MLIR_SINGLE_FUNC)
        
        # Invalid: 100 == 100 (not strictly ascending)
        config = {
            "my_func_threshold_0": 100,
            "my_func_threshold_1": 100,
        }
        
        assert manager.validate_config(config, params) is False
    
    def test_validate_bounds(self):
        """Test validation checks parameter bounds."""
        manager = MLIRConfigManager()
        params = manager.parse_annotations(SAMPLE_MLIR_SINGLE_FUNC)
        
        # Invalid: threshold_0 has bounds [0, 100], 150 is out of range
        config = {
            "my_func_threshold_0": 150,  # upper bound is 100
            "my_func_threshold_1": 160,
        }
        
        assert manager.validate_config(config, params) is False
        
        # Also invalid: below lower bound
        config = {
            "my_func_threshold_0": 50,
            "my_func_threshold_1": 40,  # lower bound is 50
        }
        
        assert manager.validate_config(config, params) is False
    
    def test_validate_single_threshold(self):
        """Test validation with single threshold (no ascending check needed)."""
        manager = MLIRConfigManager()
        params = manager.parse_annotations(SAMPLE_MLIR_MULTI_FUNC)
        
        # parse_embedding has only 1 threshold
        config = {"parse_embedding_threshold_0": 2}
        
        assert manager.validate_config(config, params) is True
    
    def test_roundtrip_preserves_structure(self):
        """Test that apply_config preserves MLIR structure."""
        manager = MLIRConfigManager()
        config = {"my_func_threshold_0": 75}
        
        modified = manager.apply_config(SAMPLE_MLIR_SINGLE_FUNC, config)
        
        # Should still contain the function definition
        assert "func.func @my_func" in modified
        assert "return %arg0 : i32" in modified
        
        # Should still have all annotation attributes
        assert 'state_function = "get_state"' in modified
        assert "transform_type" in modified


# ============================================================================
# OpenTuner Interface Tests (require OpenTuner)
# ============================================================================

@pytest.mark.skipif(not OPENTUNER_AVAILABLE, reason="OpenTuner not installed")
class TestApproxTunerInterface:
    """Test OpenTuner integration with mocked evaluation."""
    
    def test_interface_creation(self):
        """Test that interface can be created with valid MLIR."""
        mock_evaluate = Mock(return_value=(100.0, 0.95))
        
        # Create minimal args
        args = Mock()
        args.database = ":memory:"
        args.output_config = "test_config.json"
        args.output_mlir = "test_output.mlir"
        
        tuner = ApproxTunerInterface(
            args=args,
            mlir_source=SAMPLE_MLIR_SINGLE_FUNC,
            evaluate_fn=mock_evaluate,
            accuracy_threshold=0.9
        )
        
        assert len(tuner.tunable_params) == 5
        assert tuner.accuracy_threshold == 0.9
    
    def test_interface_creation_no_annotations(self):
        """Test that interface raises error for MLIR without annotations."""
        mock_evaluate = Mock(return_value=(100.0, 0.95))
        
        args = Mock()
        args.database = ":memory:"
        
        with pytest.raises(ValueError, match="No tunable parameters found"):
            ApproxTunerInterface(
                args=args,
                mlir_source=SAMPLE_MLIR_NO_ANNOTATIONS,
                evaluate_fn=mock_evaluate,
                accuracy_threshold=0.9
            )
    
    def test_manipulator_creation(self):
        """Test that manipulator has correct parameters."""
        mock_evaluate = Mock(return_value=(100.0, 0.95))
        
        args = Mock()
        args.database = ":memory:"
        args.output_config = "test_config.json"
        args.output_mlir = "test_output.mlir"
        
        tuner = ApproxTunerInterface(
            args=args,
            mlir_source=SAMPLE_MLIR_SINGLE_FUNC,
            evaluate_fn=mock_evaluate,
            accuracy_threshold=0.9
        )
        
        interface = tuner.create_interface()
        manipulator = interface.manipulator()
        
        # Check that parameters exist with correct bounds
        params = manipulator.params
        assert len(params) == 5
        
        # Verify parameter names exist
        param_names = [p.name for p in params]
        assert "my_func_threshold_0" in param_names
        assert "my_func_threshold_1" in param_names
        assert "my_func_decision_0" in param_names


# ============================================================================
# Integration Tests (mocked compilation, real tuning logic)
# ============================================================================

@pytest.mark.skipif(not OPENTUNER_AVAILABLE, reason="OpenTuner not installed")
class TestTuningIntegration:
    """Test the full tuning flow with mocked compilation."""
    
    def test_evaluate_fn_receives_config_dict(self):
        """Verify evaluate_fn receives config dict (not MLIR)."""
        received_configs = []
        
        def capture_evaluate(config: dict):
            received_configs.append(config.copy())
            return (100.0, 0.95)
        
        args = Mock()
        args.database = ":memory:"
        args.output_config = "test_config.json"
        args.output_mlir = "test_output.mlir"
        
        tuner = ApproxTunerInterface(
            args=args,
            mlir_source=SAMPLE_MLIR_SINGLE_FUNC,
            evaluate_fn=capture_evaluate,
            accuracy_threshold=0.9
        )
        
        interface = tuner.create_interface()
        
        # Simulate a run with specific config
        desired_result = Mock()
        desired_result.configuration.data = {
            "my_func_threshold_0": 60,
            "my_func_threshold_1": 160,
            "my_func_decision_0": 0,
            "my_func_decision_1": 1,
            "my_func_decision_2": 2,
        }
        
        result = interface.run(desired_result, input=None, limit=10000)
        
        # Verify evaluate was called with config dict
        assert len(received_configs) == 1
        assert received_configs[0]["my_func_threshold_0"] == 60
        assert received_configs[0]["my_func_threshold_1"] == 160
        assert received_configs[0]["my_func_decision_0"] == 0
    
    def test_invalid_config_rejected(self):
        """Test that invalid configs are rejected without calling evaluate."""
        call_count = [0]
        
        def counting_evaluate(config: dict):
            call_count[0] += 1
            return (100.0, 0.95)
        
        args = Mock()
        args.database = ":memory:"
        args.output_config = "test_config.json"
        args.output_mlir = "test_output.mlir"
        
        tuner = ApproxTunerInterface(
            args=args,
            mlir_source=SAMPLE_MLIR_SINGLE_FUNC,
            evaluate_fn=counting_evaluate,
            accuracy_threshold=0.9
        )
        
        interface = tuner.create_interface()
        
        # Invalid config: threshold_0 > threshold_1
        desired_result = Mock()
        desired_result.configuration.data = {
            "my_func_threshold_0": 180,
            "my_func_threshold_1": 100,  # Invalid: less than threshold_0
            "my_func_decision_0": 0,
            "my_func_decision_1": 1,
            "my_func_decision_2": 2,
        }
        
        result = interface.run(desired_result, input=None, limit=10000)
        
        # Should return ERROR state
        assert result.state == "ERROR"
        assert result.time == float("inf")
        assert result.accuracy == 0.0
        
        # evaluate_fn should NOT have been called
        assert call_count[0] == 0
    
    def test_result_callback_invoked(self):
        """Test that result_callback is called for each evaluation."""
        callback_results = []
        
        def my_callback(config, time, accuracy):
            callback_results.append((config.copy(), time, accuracy))
        
        def mock_evaluate(config: dict):
            return (50.0, 0.92)
        
        args = Mock()
        args.database = ":memory:"
        args.output_config = "test_config.json"
        args.output_mlir = "test_output.mlir"
        
        tuner = ApproxTunerInterface(
            args=args,
            mlir_source=SAMPLE_MLIR_SINGLE_FUNC,
            evaluate_fn=mock_evaluate,
            accuracy_threshold=0.9,
            result_callback=my_callback
        )
        
        interface = tuner.create_interface()
        
        desired_result = Mock()
        desired_result.configuration.data = {
            "my_func_threshold_0": 50,
            "my_func_threshold_1": 150,
            "my_func_decision_0": 0,
            "my_func_decision_1": 1,
            "my_func_decision_2": 2,
        }
        
        interface.run(desired_result, input=None, limit=10000)
        
        # Callback should have been called
        assert len(callback_results) == 1
        config, time, accuracy = callback_results[0]
        assert time == 50.0
        assert accuracy == 0.92
    
    def test_best_tracking(self):
        """Test that best result is tracked correctly."""
        call_count = [0]
        
        def varying_evaluate(config: dict):
            call_count[0] += 1
            # Return different results based on call
            if call_count[0] == 1:
                return (100.0, 0.95)
            elif call_count[0] == 2:
                return (80.0, 0.92)  # Better time, meets threshold
            else:
                return (70.0, 0.85)  # Best time but below threshold
        
        args = Mock()
        args.database = ":memory:"
        args.output_config = "test_config.json"
        args.output_mlir = "test_output.mlir"
        
        tuner = ApproxTunerInterface(
            args=args,
            mlir_source=SAMPLE_MLIR_SINGLE_FUNC,
            evaluate_fn=varying_evaluate,
            accuracy_threshold=0.9
        )
        
        interface = tuner.create_interface()
        
        # Run three configs
        for i in range(3):
            desired_result = Mock()
            desired_result.configuration.data = {
                "my_func_threshold_0": 50 + i * 10,
                "my_func_threshold_1": 150 + i * 10,
                "my_func_decision_0": 0,
                "my_func_decision_1": 1,
                "my_func_decision_2": 2,
            }
            interface.run(desired_result, input=None, limit=10000)
        
        # Best should be the second config (80ms, 0.92 accuracy >= 0.9 threshold)
        assert tuner.best_time == 80.0
        assert tuner.best_accuracy == 0.92


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def test_malformed_mlir(self):
        """Test handling of malformed MLIR."""
        manager = MLIRConfigManager()
        
        # Missing closing brace
        malformed = '''
        "approx.util.annotation.decision_tree"() <{
            func_name = "test"
        '''
        
        params = manager.parse_annotations(malformed)
        assert len(params) == 0  # Should gracefully return empty
    
    def test_missing_arrays(self):
        """Test handling of annotation with missing arrays."""
        manager = MLIRConfigManager()
        
        # Missing thresholds array
        incomplete = '''
        module {
          "approx.util.annotation.decision_tree"() <{
              func_name = "test",
              transform_type = "loop_perforate",
              decisions = array<i32: 0, 1>
          }> : () -> ()
        }
        '''
        
        params = manager.parse_annotations(incomplete)
        # Should still parse decisions
        assert "test_decision_0" not in params  # No decision_values, so can't determine bounds


# ============================================================================
# Main entry point for running tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
