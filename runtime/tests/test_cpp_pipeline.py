"""Tests for C/C++ pipeline integration (cgeist -> annotations -> approx-opt-cpp)."""

from unittest.mock import MagicMock, patch

from approx_runtime.cpp_pipeline import CgeistConfig, compile_cpp_source
from approx_runtime.toolchain import ToolchainConfig


def test_compile_cpp_source_calls_cgeist_and_approx_opt():
    cpp_source = """
// @approx:decision_tree {
//   transform_type: loop_perforate
//   thresholds: [1]
//   decisions: [0, 1]
// }
int f(int x, int state) { return x; }
"""
    cgeist_out = "module {\n  func.func @f(%arg0: i32, %arg1: i32) -> i32 { return %arg0 : i32 }\n}\n"
    approx_out = "module { // transformed }\n"

    def run_side_effect(cmd, capture_output, text):
        exe = cmd[0]
        mock = MagicMock()
        if "cgeist" in exe:
            mock.returncode = 0
            mock.stdout = cgeist_out
            mock.stderr = ""
        else:
            mock.returncode = 0
            mock.stdout = approx_out
            mock.stderr = ""
        return mock

    toolchain = ToolchainConfig(
        ml_opt_path="approx-opt-ml",
        cpp_opt_path="approx-opt-cpp",
        cpp_pipeline=["emit-approx", "finalize-approx"],
    )

    with patch("subprocess.run", side_effect=run_side_effect) as mock_run:
        output = compile_cpp_source(
            cpp_source,
            cgeist_config=CgeistConfig(cgeist_path="cgeist"),
            toolchain=toolchain,
        )

    assert output == approx_out
    assert mock_run.call_count == 2
    cgeist_cmd = mock_run.call_args_list[0][0][0]
    approx_cmd = mock_run.call_args_list[1][0][0]
    assert cgeist_cmd[0] == "cgeist"
    assert approx_cmd[0] == "approx-opt-cpp"
    assert "--emit-approx" in approx_cmd
    assert "--finalize-approx" in approx_cmd
