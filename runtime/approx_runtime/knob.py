"""Approximation knob decorators and configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any, Dict

__all__ = ['DecisionTree', 'SafetyContract', 'StaticTransform', 'Knob', 'get_config']


@dataclass
class DecisionTree:
    """Dynamic approximation based on runtime state.
    
    Attributes:
        state_function: User function (state_args...) -> i32 that computes runtime state
        state_indices: Which function arguments are passed to state_function
        thresholds: Decision boundaries (N thresholds -> N+1 regions)
        decisions: Knob values for each region (len = len(thresholds) + 1)
        transform_type: One of "loop_perforate", "func_substitute", "task_skipping"
        approx_kernels: For func_substitute: {knob_val: callable} mapping knob values
                        to approximate kernel implementations
    """
    state_function: Callable
    state_indices: List[int]
    thresholds: List[int]
    decisions: List[int]
    transform_type: str
    
    # For func_substitute: map knob_val -> approximate implementation
    # e.g., {1: approx_v1, 2: approx_v2} where knob_val=0 means exact
    approx_kernels: Dict[int, Callable] = field(default_factory=dict)
    
    # Optional metadata for autotuning
    thresholds_lower: List[int] = field(default_factory=lambda: [0])
    thresholds_upper: List[int] = field(default_factory=lambda: [100])
    decision_values: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.decisions) != len(self.thresholds) + 1:
            raise ValueError(
                f"decisions length ({len(self.decisions)}) must be "
                f"thresholds length + 1 ({len(self.thresholds) + 1})"
            )
        valid_transforms = {"loop_perforate", "func_substitute", "task_skipping"}
        if self.transform_type not in valid_transforms:
            raise ValueError(f"transform_type must be one of {valid_transforms}")
        
        # Validate approx_kernels for func_substitute
        if self.transform_type == "func_substitute":
            non_zero_decisions = [d for d in self.decisions if d != 0]
            for knob_val in non_zero_decisions:
                if knob_val not in self.approx_kernels:
                    raise ValueError(
                        f"func_substitute requires approx_kernels[{knob_val}] "
                        f"for decision value {knob_val}"
                    )
        if self.decision_values and len(self.decision_values) < len(self.decisions):
            raise ValueError("decision_values length must be >= decisions length")


@dataclass
class SafetyContract:
    """Try-check-recover pattern for error detection and recovery.
    
    Attributes:
        checker: Function (results..., inputs...) -> bool that validates results
        recover: Function (inputs...) -> results that provides fallback computation
    """
    checker: Callable
    recover: Callable


@dataclass
class StaticTransform:
    """Fixed approximation without runtime decision.
    
    Attributes:
        transform_type: One of "loop_perforate", "func_substitute", "task_skipping"
        knob_val: Approximation intensity (0 = exact, higher = more approximate)
        approx_kernel: For func_substitute with knob_val > 0: the approximate implementation
    """
    transform_type: str
    knob_val: int
    approx_kernel: Optional[Callable] = None
    
    def __post_init__(self):
        valid_transforms = {"loop_perforate", "func_substitute", "task_skipping"}
        if self.transform_type not in valid_transforms:
            raise ValueError(f"transform_type must be one of {valid_transforms}")
        
        # Validate approx_kernel for func_substitute
        if self.transform_type == "func_substitute" and self.knob_val > 0:
            if self.approx_kernel is None:
                raise ValueError(
                    f"func_substitute with knob_val={self.knob_val} requires approx_kernel"
                )


def Knob(
    decision_tree: Optional[DecisionTree] = None,
    safety_contract: Optional[SafetyContract] = None,
    static_transform: Optional[StaticTransform] = None,
):
    """Decorator that attaches approximation configuration to a function.
    
    Args:
        decision_tree: Dynamic approximation based on runtime state
        safety_contract: Try-check-recover error handling
        static_transform: Fixed approximation (mutually exclusive with decision_tree)
    
    Example:
        @Knob(decision_tree=DecisionTree(
            state_function=get_state,
            state_indices=[0],
            thresholds=[100],
            decisions=[0, 2],
            transform_type="loop_perforate"
        ))
        def my_kernel(n, data):
            return jnp.sum(data[:n])
    """
    if decision_tree and static_transform:
        raise ValueError("Cannot specify both decision_tree and static_transform")
    
    def decorator(func: Callable) -> Callable:
        func._approx_config = {
            'decision_tree': decision_tree,
            'safety_contract': safety_contract,
            'static_transform': static_transform,
        }
        return func
    
    return decorator


def get_config(func: Callable) -> Optional[dict]:
    """Get approximation config attached to a function, or None if not decorated."""
    return getattr(func, '_approx_config', None)
