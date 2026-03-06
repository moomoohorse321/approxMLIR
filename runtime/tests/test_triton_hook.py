"""Tests for Triton stage-hook integration."""

import sys
import types
from pathlib import Path

import pytest

from approx_runtime.triton_compiler import TritonCompilationError
from approx_runtime.triton_hook import make_triton_stages_hook


def _install_fake_triton(monkeypatch):
    calls = {"added": [], "ran": False, "parsed_text": None}

    class FakePM:
        def __init__(self, _ctx):
            self.ctx = _ctx

        def enable_debug(self):
            return None

        def run(self, _mod, _name):
            calls["ran"] = True

    def pass_manager(ctx):
        return FakePM(ctx)

    plugin_ns = types.SimpleNamespace()

    def _mk_pass(name):
        def _f(pm):
            calls["added"].append(name)
            return pm

        return _f

    setattr(plugin_ns, "pre-emit-transform", _mk_pass("pre-emit-transform"))
    setattr(plugin_ns, "emit-approx", _mk_pass("emit-approx"))
    setattr(plugin_ns, "transform-approx", _mk_pass("transform-approx"))
    setattr(plugin_ns, "finalize-approx", _mk_pass("finalize-approx"))

    class ParsedModule:
        def __init__(self, context):
            self.context = context

        def __str__(self):
            return calls["parsed_text"] or "module {}"

    def parse_mlir_module(path, ctx):
        calls["parsed_text"] = Path(path).read_text(encoding="utf-8")
        return ParsedModule(ctx)

    fake_libtriton = types.SimpleNamespace(
        ir=types.SimpleNamespace(pass_manager=pass_manager, parse_mlir_module=parse_mlir_module),
        passes=types.SimpleNamespace(plugin=plugin_ns),
    )
    fake_triton_c = types.SimpleNamespace(libtriton=fake_libtriton)
    fake_triton = types.SimpleNamespace(_C=fake_triton_c)

    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton._C", fake_triton_c)
    monkeypatch.setitem(sys.modules, "triton._C.libtriton", fake_libtriton)
    return calls


def test_make_triton_stages_hook_keyhash():
    hook = make_triton_stages_hook(passes=["emit-approx"])
    key, key_hash = hook()
    assert "approx_triton_hook" in key
    assert len(key_hash) == 64


def test_make_triton_stages_hook_runs_passes(monkeypatch):
    calls = _install_fake_triton(monkeypatch)
    hook = make_triton_stages_hook(
        passes=["emit-approx", "finalize-approx"],
        plugin_path="/tmp/plugin.so",
    )

    class FakeCompiler:
        def make_ttir(self, mod, _metadata, _opt, _cap):
            return mod

    module = types.SimpleNamespace(context=object())
    stages = {}
    hook(self=FakeCompiler(), stages=stages, options=None, language=None, capability=None)
    stages["ttir"](module, {})
    assert calls["added"] == ["emit-approx", "finalize-approx"]
    assert calls["ran"] is True


def test_make_triton_stages_hook_missing_pass(monkeypatch):
    _install_fake_triton(monkeypatch)
    hook = make_triton_stages_hook(
        passes=["emit-approx", "does-not-exist"],
        plugin_path="/tmp/plugin.so",
    )

    class FakeCompiler:
        def make_ttir(self, mod, _metadata, _opt, _cap):
            return mod

    module = types.SimpleNamespace(context=object())
    stages = {}
    hook(self=FakeCompiler(), stages=stages, options=None, language=None, capability=None)
    with pytest.raises(TritonCompilationError):
        stages["ttir"](module, {})


def test_make_triton_stages_hook_injects_annotations(monkeypatch):
    calls = _install_fake_triton(monkeypatch)
    hook = make_triton_stages_hook(
        passes=["emit-approx", "finalize-approx"],
        plugin_path="/tmp/plugin.so",
        func_name="foo",
        config={"static_transform": types.SimpleNamespace(transform_type="task_skipping", knob_val=0)},
    )

    class FakeModule:
        def __init__(self):
            self.context = object()

        def __str__(self):
            return "module { tt.func @foo() { tt.return } }"

    class FakeCompiler:
        def make_ttir(self, mod, _metadata, _opt, _cap):
            return mod

    module = FakeModule()
    stages = {}
    hook(self=FakeCompiler(), stages=stages, options=None, language=None, capability=None)
    stages["ttir"](module, {"name": "foo"})
    assert calls["parsed_text"] is not None
    assert "approx.util.annotation.transform" in calls["parsed_text"]


def test_make_triton_stages_hook_auto_prepends_pre_emit(monkeypatch):
    calls = _install_fake_triton(monkeypatch)
    def __state_fn():
        return 0

    hook = make_triton_stages_hook(
        passes=["emit-approx", "finalize-approx"],
        plugin_path="/tmp/plugin.so",
        func_name="foo",
        config={
                "decision_tree": types.SimpleNamespace(
                    state_function=__state_fn,
                state_indices=[0],
                thresholds=[1],
                decisions=[0, 1],
                thresholds_lower=[0],
                thresholds_upper=[4],
                decision_values=[0, 1],
                transform_type="func_substitute",
            ),
            "safety_contract": None,
            "static_transform": None,
        },
    )

    class FakeModule:
        def __init__(self):
            self.context = object()

        def __str__(self):
            return """
module {
  tt.func @foo() {
    tt.return
  }
  tt.func @approx_foo_1() {
    tt.return
  }
  tt.func @__state_fn() -> i32 {
    %c0 = arith.constant 0 : i32
    tt.return %c0 : i32
  }
}
"""

    class FakeCompiler:
        def make_ttir(self, mod, _metadata, _opt, _cap):
            return mod

    module = FakeModule()
    stages = {}
    hook(self=FakeCompiler(), stages=stages, options=None, language=None, capability=None)
    stages["ttir"](module, {"name": "foo"})
    assert calls["added"][0] == "pre-emit-transform"


def test_make_triton_stages_hook_missing_func_substitute_helper(monkeypatch):
    _install_fake_triton(monkeypatch)
    hook = make_triton_stages_hook(
        passes=["emit-approx", "transform-approx", "finalize-approx"],
        plugin_path="/tmp/plugin.so",
        func_name="foo",
        config={
            "decision_tree": None,
            "safety_contract": None,
            "static_transform": types.SimpleNamespace(transform_type="func_substitute", knob_val=1),
        },
    )

    class FakeModule:
        def __init__(self):
            self.context = object()

        def __str__(self):
            return """
module {
  tt.func @foo() {
    tt.return
  }
}
"""

    class FakeCompiler:
        def make_ttir(self, mod, _metadata, _opt, _cap):
            return mod

    module = FakeModule()
    stages = {}
    hook(self=FakeCompiler(), stages=stages, options=None, language=None, capability=None)
    with pytest.raises(TritonCompilationError, match="missing"):
        stages["ttir"](module, {"name": "foo"})


def test_make_triton_stages_hook_uses_provided_extra_ttir(monkeypatch):
    calls = _install_fake_triton(monkeypatch)
    hook = make_triton_stages_hook(
        passes=["emit-approx", "transform-approx", "finalize-approx"],
        plugin_path="/tmp/plugin.so",
        func_name="foo",
        config={
            "decision_tree": None,
            "safety_contract": None,
            "static_transform": types.SimpleNamespace(transform_type="func_substitute", knob_val=1),
        },
        extra_ttir_texts=[
            """
module {
  tt.func @approx_foo_1() {
    %c42 = arith.constant 42 : i32 loc(#loc10)
    tt.return loc(#loc11)
  }
}
#loc10 = loc("extra.mlir":1:1)
#loc11 = loc("extra.mlir":1:2)
"""
        ],
    )

    class FakeModule:
        def __init__(self):
            self.context = object()

        def __str__(self):
            return """
module {
  tt.func @foo() {
    tt.return
  }
}
"""

    class FakeCompiler:
        def make_ttir(self, mod, _metadata, _opt, _cap):
            return mod

    module = FakeModule()
    stages = {}
    hook(self=FakeCompiler(), stages=stages, options=None, language=None, capability=None)
    stages["ttir"](module, {"name": "foo"})
    assert "tt.func @foo" in calls["parsed_text"]
    assert "tt.func @approx_foo_1" in calls["parsed_text"]
    assert "%c42 = arith.constant 42 : i32" in calls["parsed_text"]
    assert "loc(#loc10)" not in calls["parsed_text"]


def test_make_triton_stages_hook_missing_state_helper(monkeypatch):
    _install_fake_triton(monkeypatch)

    def _state_fn():
        return 0

    hook = make_triton_stages_hook(
        passes=["emit-approx"],
        plugin_path="/tmp/plugin.so",
        func_name="foo",
        config={
            "decision_tree": types.SimpleNamespace(
                state_function=_state_fn,
                state_indices=[0],
                thresholds=[1],
                decisions=[0, 1],
                thresholds_lower=[0],
                thresholds_upper=[4],
                decision_values=[0, 1],
                transform_type="func_substitute",
            ),
            "safety_contract": None,
            "static_transform": None,
        },
    )

    class FakeModule:
        def __init__(self):
            self.context = object()

        def __str__(self):
            return """
module {
  tt.func @foo() {
    tt.return
  }
}
"""

    class FakeCompiler:
        def make_ttir(self, mod, _metadata, _opt, _cap):
            return mod

    module = FakeModule()
    stages = {}
    hook(self=FakeCompiler(), stages=stages, options=None, language=None, capability=None)
    with pytest.raises(TritonCompilationError, match="state function"):
        stages["ttir"](module, {"name": "foo"})
