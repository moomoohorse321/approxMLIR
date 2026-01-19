"""Tests for annotate.py non-invasive API."""

import pytest

from approx_runtime.annotate import with_config, export_with_config
from approx_runtime.knob import DecisionTree
from approx_runtime.export import JAX_AVAILABLE


def _mock_state(x):
    return x


def test_with_config_attaches_config():
    def f(x):
        return x

    wrapped = with_config(
        f,
        decision_tree=DecisionTree(
            state_function=_mock_state,
            state_indices=[0],
            thresholds=[1],
            decisions=[0, 1],
            transform_type="loop_perforate",
        ),
    )
    assert hasattr(wrapped, "_approx_config")
    assert wrapped._approx_config["decision_tree"] is not None
    assert wrapped(3) == 3


def test_export_with_config_rejects_safety_contract():
    def f(x):
        return x

    config = {"safety_contract": object()}
    with pytest.raises(ValueError):
        export_with_config(f, (1,), config)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_export_with_config_basic():
    import jax.numpy as jnp

    def f(x):
        return x + 1

    config = {
        "decision_tree": DecisionTree(
            state_function=_mock_state,
            state_indices=[0],
            thresholds=[1],
            decisions=[0, 1],
            transform_type="loop_perforate",
        )
    }

    mlir = export_with_config(f, (jnp.array(1),), config)
    assert "approx.util.annotation.decision_tree" in mlir
