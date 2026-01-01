"""Tests for compiler.py - approx-opt integration."""

import pytest
import subprocess
from unittest.mock import patch, MagicMock
from approx_runtime import (
    compile, 
    compile_file, 
    PASS_PIPELINE, 
    FUNC_SUBSTITUTE_PIPELINE,
    CompilationError,
    get_pipeline_for_config,
    DecisionTree,
    StaticTransform,
)


# Mock functions
def mock_state_fn(x):
    return x

def mock_approx(x):
    return x


class TestPassPipeline:
    def test_pipeline_order(self):
        expected = [
            "emit-approx",
            "emit-management",
            "config-approx",
            "transform-approx",
            "finalize-approx",
        ]
        assert PASS_PIPELINE == expected
    
    def test_func_substitute_pipeline_order(self):
        expected = [
            "pre-emit-transform",
            "emit-approx",
            "emit-management",
            "config-approx",
            "transform-approx",
            "finalize-approx",
        ]
        assert FUNC_SUBSTITUTE_PIPELINE == expected


class TestGetPipelineForConfig:
    def test_loop_perforate_uses_default(self):
        config = {
            'decision_tree': DecisionTree(
                state_function=mock_state_fn,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],
                transform_type="loop_perforate",
            ),
            'safety_contract': None,
            'static_transform': None,
        }
        
        assert get_pipeline_for_config(config) == PASS_PIPELINE
    
    def test_func_substitute_uses_pre_emit(self):
        config = {
            'decision_tree': DecisionTree(
                state_function=mock_state_fn,
                state_indices=[0],
                thresholds=[100],
                decisions=[0, 1],
                transform_type="func_substitute",
                approx_kernels={1: mock_approx},
            ),
            'safety_contract': None,
            'static_transform': None,
        }
        
        assert get_pipeline_for_config(config) == FUNC_SUBSTITUTE_PIPELINE
    
    def test_static_func_substitute_uses_pre_emit(self):
        config = {
            'decision_tree': None,
            'safety_contract': None,
            'static_transform': StaticTransform(
                transform_type="func_substitute",
                knob_val=1,
                approx_kernel=mock_approx,
            ),
        }
        
        assert get_pipeline_for_config(config) == FUNC_SUBSTITUTE_PIPELINE
    
    def test_none_config_uses_default(self):
        assert get_pipeline_for_config(None) == PASS_PIPELINE
        assert get_pipeline_for_config({}) == PASS_PIPELINE


class TestCompile:
    @patch('subprocess.run')
    def test_compile_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="transformed mlir output",
            stderr="",
        )
        
        result = compile("module { }")
        
        assert result == "transformed mlir output"
        mock_run.assert_called_once()
        
        # Check passes were included
        call_args = mock_run.call_args[0][0]
        assert '--emit-approx' in call_args
        assert '--finalize-approx' in call_args
    
    @patch('subprocess.run')
    def test_compile_with_custom_passes(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="output",
            stderr="",
        )
        
        compile("module { }", passes=["emit-approx", "config-approx"])
        
        call_args = mock_run.call_args[0][0]
        assert '--emit-approx' in call_args
        assert '--config-approx' in call_args
        assert '--emit-management' not in call_args
    
    @patch('subprocess.run')
    def test_compile_with_func_substitute_pipeline(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="output",
            stderr="",
        )
        
        compile("module { }", passes=FUNC_SUBSTITUTE_PIPELINE)
        
        call_args = mock_run.call_args[0][0]
        assert '--pre-emit-transform' in call_args
        assert '--emit-approx' in call_args
    
    @patch('subprocess.run')
    def test_compile_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error: unknown pass",
        )
        
        with pytest.raises(CompilationError, match="approx-opt failed"):
            compile("module { }")
    
    @patch('subprocess.run')
    def test_compile_custom_binary_path(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="output",
            stderr="",
        )
        
        compile("module { }", approx_opt_path="/custom/path/approx-opt")
        
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "/custom/path/approx-opt"


class TestCompileFile:
    @patch('subprocess.run')
    def test_compile_file_to_stdout(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="output mlir",
            stderr="",
        )
        
        result = compile_file("input.mlir")
        
        assert result == "output mlir"
        call_args = mock_run.call_args[0][0]
        assert "input.mlir" in call_args
    
    @patch('subprocess.run')
    def test_compile_file_to_file(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )
        
        result = compile_file("input.mlir", output_path="output.mlir")
        
        assert result == "output.mlir"
        call_args = mock_run.call_args[0][0]
        assert '-o' in call_args
        assert 'output.mlir' in call_args
    
    @patch('subprocess.run')
    def test_compile_file_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="file not found",
        )
        
        with pytest.raises(CompilationError):
            compile_file("nonexistent.mlir")
