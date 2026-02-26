"""Toolchain selection and configuration for ApproxMLIR."""

from dataclasses import dataclass, field
from enum import Enum
import os
from typing import List, Optional

from .mlir_gen import needs_pre_emit_transform

__all__ = [
    "WorkloadType",
    "ToolchainConfig",
    "get_toolchain",
    "set_toolchain",
]


class WorkloadType(Enum):
    """Type of workload for compiler selection."""

    ML = "ml"    # JAX/StableHLO -> IREE
    CPP = "cpp"  # C++/Polygeist -> native
    TRITON = "triton"  # Triton TTIR -> plugin pass manager


@dataclass
class ToolchainConfig:
    """Configuration for ApproxMLIR toolchains."""

    ml_opt_path: str = field(
        default_factory=lambda: os.environ.get("APPROX_OPT_ML", "approx-opt")
    )
    cpp_opt_path: str = field(
        default_factory=lambda: os.environ.get("APPROX_OPT_CPP", "approx-opt-cpp")
    )
    triton_plugin_path: str = field(
        default_factory=lambda: os.environ.get("TRITON_PASS_PLUGIN_PATH", "")
    )

    ml_pipeline: List[str] = field(
        default_factory=lambda: [
            "emit-approx",
            "emit-management",
            "config-approx",
            "transform-approx",
            "finalize-approx",
            "legalize-to-stablehlo",
        ]
    )
    cpp_pipeline: List[str] = field(
        default_factory=lambda: [
            "emit-approx",
            "emit-management",
            "config-approx",
            "transform-approx",
            "finalize-approx",
        ]
    )
    triton_pipeline: List[str] = field(
        default_factory=lambda: [
            "emit-approx",
            "emit-management",
            "config-approx",
            "transform-approx",
            "finalize-approx",
        ]
    )

    def get_opt_path(self, workload: WorkloadType) -> str:
        """Get approx-opt path for workload type."""
        if workload == WorkloadType.ML:
            return self.ml_opt_path
        if workload == WorkloadType.CPP:
            return self.cpp_opt_path
        raise ValueError(
            "WorkloadType.TRITON does not use approx-opt; "
            "use plugin pass manager integration instead."
        )

    def get_pipeline(
        self, workload: WorkloadType, config: Optional[dict] = None
    ) -> List[str]:
        """Get pass pipeline for workload type, adjusted for config."""
        if workload == WorkloadType.ML:
            base = self.ml_pipeline
        elif workload == WorkloadType.CPP:
            base = self.cpp_pipeline
        elif workload == WorkloadType.TRITON:
            base = self.triton_pipeline
        else:
            raise ValueError(f"Unsupported workload type: {workload}")
        pipeline = list(base)

        if config and needs_pre_emit_transform(config):
            pipeline = ["pre-emit-transform"] + pipeline

        sc = config.get("safety_contract") if config else None
        if sc:
            pipeline = [p for p in pipeline if p != "legalize-to-stablehlo"]

        return pipeline


_DEFAULT_TOOLCHAIN = ToolchainConfig()


def get_toolchain() -> ToolchainConfig:
    """Get the current toolchain configuration."""
    return _DEFAULT_TOOLCHAIN


def set_toolchain(config: ToolchainConfig) -> None:
    """Set the global toolchain configuration."""
    global _DEFAULT_TOOLCHAIN
    _DEFAULT_TOOLCHAIN = config
