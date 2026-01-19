"""Tests for toolchain configuration."""

from approx_runtime.toolchain import ToolchainConfig, WorkloadType
from approx_runtime.knob import DecisionTree


def _mock_state(x):
    return x


def test_get_opt_path():
    tc = ToolchainConfig(ml_opt_path="ml-opt", cpp_opt_path="cpp-opt")
    assert tc.get_opt_path(WorkloadType.ML) == "ml-opt"
    assert tc.get_opt_path(WorkloadType.CPP) == "cpp-opt"


def test_get_pipeline_pre_emit_transform():
    tc = ToolchainConfig(
        ml_pipeline=["emit-approx"],
        cpp_pipeline=["emit-approx"],
    )
    config = {
        "decision_tree": DecisionTree(
            state_function=_mock_state,
            state_indices=[0],
            thresholds=[1],
            decisions=[0, 1],
            transform_type="func_substitute",
            approx_kernels={1: _mock_state},
        )
    }
    pipeline = tc.get_pipeline(WorkloadType.ML, config)
    assert pipeline[0] == "pre-emit-transform"
    assert "emit-approx" in pipeline


def test_get_pipeline_safety_contract_removes_legalize():
    tc = ToolchainConfig(
        ml_pipeline=["emit-approx", "legalize-to-stablehlo"],
        cpp_pipeline=["emit-approx"],
    )
    config = {"safety_contract": object()}
    pipeline = tc.get_pipeline(WorkloadType.ML, config)
    assert "legalize-to-stablehlo" not in pipeline
