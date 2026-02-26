"""Integration tests for Triton plugin passes on tt.* IR."""

from pathlib import Path

import pytest

import approx_runtime as ar


def _default_plugin_path() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return str(
        repo_root / "approx-triton-plugin" / "build" / "lib" / "libApproxTritonPlugin.so"
    )


def _plugin_path_or_skip() -> str:
    if not ar.TRITON_AVAILABLE:
        pytest.skip("triton python bindings are not available in this test context")
    plugin_path = _default_plugin_path()
    if not Path(plugin_path).exists():
        pytest.skip(f"plugin not built: {plugin_path}")
    return plugin_path


def test_pre_emit_transform_on_tt_func():
    plugin_path = _plugin_path_or_skip()
    mlir = """
module {
  "approx.util.annotation.convert_to_call"() <{func_name = "foo"}> : () -> ()
  tt.func @foo(%x: i32) -> i32 {
    tt.return %x : i32
  }
}
"""
    out = ar.compile_with_triton_plugin(
        mlir,
        passes=["pre-emit-transform"],
        plugin_path=plugin_path,
    )
    assert "tt.func @__internal_foo" in out
    assert "tt.call @__internal_foo" in out


def test_transform_substitute_rewrites_tt_call():
    plugin_path = _plugin_path_or_skip()
    mlir = """
module {
  tt.func @__internal_foo(%x: i32) -> i32 {
    tt.return %x : i32
  }
  tt.func @approx_foo_1(%x: i32) -> i32 {
    tt.return %x : i32
  }
  tt.func @foo(%x: i32) -> i32 {
    %0 = tt.call @__internal_foo(%x) : (i32) -> i32
    "approx.transform"() <{knob_val = 1 : i32, transform_type = "func_substitute"}> : () -> ()
    tt.return %0 : i32
  }
}
"""
    out = ar.compile_with_triton_plugin(
        mlir,
        passes=["transform-approx"],
        plugin_path=plugin_path,
    )
    assert "tt.call @approx_foo_1" in out
