"""OpenTuner interface for ApproxMLIR auto-tuning.

This module provides the integration with OpenTuner for automatically
searching the configuration space defined by approx.util.annotation.decision_tree
operations in MLIR.

The user provides an evaluation callback that receives a config dict and handles:
- Applying config to MLIR file(s) via MLIRConfigManager.apply_config()
- Compilation (via approx-opt and optionally IREE)
- Execution with test inputs
- Accuracy measurement

The tuner handles:
- Parsing MLIR to extract the search space
- Managing the OpenTuner search process
- Tracking best results

Example:
    from approx_runtime import tune, MLIRConfigManager
    
    # User sets up their environment via closure
    manager = MLIRConfigManager()
    mlir_source = open("my_kernel.mlir").read()
    
    def evaluate(config: Dict[str, int]) -> Tuple[float, float]:
        # Apply config to MLIR(s)
        modified_mlir = manager.apply_config(mlir_source, config)
        
        # Compile and deploy
        compiled = compile(modified_mlir, passes=PIPELINE)
        vmfb = compile_to_iree(compiled)
        module = load_module(vmfb)
        
        # Run and measure
        start = time.time()
        output = module["my_kernel"](test_input)
        time_ms = (time.time() - start) * 1000
        
        accuracy = my_accuracy_metric(output, ground_truth)
        return time_ms, accuracy
    
    result = tune(mlir_source, evaluate, accuracy_threshold=0.9)
"""

import argparse
import json
import logging
from typing import Callable, Dict, Tuple, Optional, Any

from .tuner_config import MLIRConfigManager, TunableParam
from .annotate import export_with_config
from .cpp_annotation import parse_and_generate
from .mlir_gen import inject_annotations_text

__all__ = [
    'ApproxTunerInterface',
    'tune',
    'create_tuner_arg_parser',
    'tune_jax',
    'tune_cpp',
]

log = logging.getLogger("approx_tuner")

# Lazy import of OpenTuner to allow module to be imported without it
_opentuner_imported = False
_MeasurementInterface = None
_ConfigurationManipulator = None
_IntegerParameter = None
_FixedInputManager = None
_ThresholdAccuracyMinimizeTime = None
_Result = None


def _import_opentuner():
    """Lazy import of OpenTuner modules."""
    global _opentuner_imported
    global _MeasurementInterface, _ConfigurationManipulator, _IntegerParameter
    global _FixedInputManager, _ThresholdAccuracyMinimizeTime, _Result
    
    if _opentuner_imported:
        return True
    
    try:
        import opentuner
        from opentuner import MeasurementInterface, ConfigurationManipulator, IntegerParameter
        from opentuner.measurement.inputmanager import FixedInputManager
        from opentuner.search.objective import ThresholdAccuracyMinimizeTime
        import opentuner.resultsdb.models as models
        
        _MeasurementInterface = MeasurementInterface
        _ConfigurationManipulator = ConfigurationManipulator
        _IntegerParameter = IntegerParameter
        _FixedInputManager = FixedInputManager
        _ThresholdAccuracyMinimizeTime = ThresholdAccuracyMinimizeTime
        _Result = models.Result
        
        _opentuner_imported = True
        return True
    except ImportError as e:
        log.error(f"OpenTuner not available: {e}")
        log.error("Install with: pip install --user opentuner")
        return False


def create_tuner_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser with OpenTuner arguments.
    
    Returns:
        ArgumentParser with OpenTuner's standard arguments added
    """
    if not _import_opentuner():
        raise ImportError("OpenTuner is required: pip install --user opentuner")
    
    import opentuner
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    parser.add_argument("--accuracy-threshold", type=float, default=0.9,
                        help="Minimum acceptable accuracy (default: 0.9)")
    parser.add_argument("--mlir-file", type=str, default=None,
                        help="Path to MLIR file with annotations")
    parser.add_argument("--output-config", type=str, default="approx_tuned_config.json",
                        help="Output file for best configuration")
    parser.add_argument("--output-mlir", type=str, default="approx_tuned.mlir",
                        help="Output file for tuned MLIR")
    return parser


class ApproxTunerInterface:
    """OpenTuner interface for ApproxMLIR auto-tuning.
    
    This class wraps OpenTuner's MeasurementInterface to provide auto-tuning
    for MLIR modules with approx.util.annotation.decision_tree operations.
    
    The user provides:
    - mlir_source: MLIR text with annotations (used to extract search space)
    - evaluate_fn: Callback (config: Dict[str, int]) -> (time_ms: float, accuracy: float)
    - accuracy_threshold: Target accuracy for optimization
    
    Example:
        manager = ar.MLIRConfigManager()
        mlir_source = open("kernel.mlir").read()
        
        def my_evaluate(config: Dict[str, int]) -> Tuple[float, float]:
            # Apply config to MLIR
            modified = manager.apply_config(mlir_source, config)
            # Compile, run, measure...
            return time_ms, accuracy
        
        tuner = ApproxTunerInterface(
            args=args,
            mlir_source=mlir_source,
            evaluate_fn=my_evaluate,
            accuracy_threshold=0.95
        )
        result = tuner.run_tuning()
    """
    
    def __init__(
        self,
        args: argparse.Namespace,
        mlir_source: str,
        evaluate_fn: Callable[[Dict[str, int]], Tuple[float, float]],
        accuracy_threshold: float = 0.9,
        result_callback: Optional[Callable[[Dict[str, int], float, float], None]] = None,
    ):
        """Initialize the tuner interface.
        
        Args:
            args: Parsed arguments (from create_tuner_arg_parser())
            mlir_source: MLIR text with decision_tree annotations (for extracting search space)
            evaluate_fn: User callback that takes a config dict and returns
                         (execution_time_ms, accuracy_score). User is responsible
                         for applying config, compiling, running, and measuring.
            accuracy_threshold: Minimum acceptable accuracy (0.0 to 1.0)
            result_callback: Optional callback for each result: (config, time, accuracy)
        """
        if not _import_opentuner():
            raise ImportError("OpenTuner is required: pip install --user opentuner")
        
        self.mlir_source = mlir_source
        self.evaluate_fn = evaluate_fn
        self.accuracy_threshold = accuracy_threshold
        self.result_callback = result_callback
        
        # Parse MLIR to extract tunable parameters
        self.config_manager = MLIRConfigManager()
        self.tunable_params = self.config_manager.parse_annotations(mlir_source)
        
        if not self.tunable_params:
            raise ValueError("No tunable parameters found in MLIR. "
                           "Ensure it contains approx.util.annotation.decision_tree ops.")
        
        log.info(f"Found {len(self.tunable_params)} tunable parameters")
        for name, param in self.tunable_params.items():
            log.info(f"  {name}: [{param.lower}, {param.upper}] (current: {param.current})")
        
        # Store args for later
        self._args = args
        
        # Best result tracking
        self.best_config: Optional[Dict[str, int]] = None
        self.best_time: float = float('inf')
        self.best_accuracy: float = 0.0
    
    def create_interface(self) -> '_MeasurementInterface':
        """Create the OpenTuner MeasurementInterface instance.
        
        Returns:
            MeasurementInterface subclass instance
        """
        outer_self = self
        
        class _Interface(_MeasurementInterface):
            def __init__(self, args):
                objective = _ThresholdAccuracyMinimizeTime(outer_self.accuracy_threshold)
                input_manager = _FixedInputManager()
                super().__init__(
                    args,
                    program_name="approx_tuner",
                    objective=objective,
                    input_manager=input_manager
                )
            
            def manipulator(self):
                """Build search space from parsed MLIR annotations."""
                manipulator = _ConfigurationManipulator()
                for name, param in outer_self.tunable_params.items():
                    manipulator.add_parameter(
                        _IntegerParameter(name, param.lower, param.upper)
                    )
                return manipulator
            
            def run(self, desired_result, input, limit):
                """Execute one tuning iteration."""
                config = desired_result.configuration.data
                
                # Validate configuration
                if not outer_self.config_manager.validate_config(config, outer_self.tunable_params):
                    result = _Result()
                    result.state = "ERROR"
                    result.time = float("inf")
                    result.accuracy = 0.0
                    log.debug(f"Invalid config (constraint violation): {config}")
                    return result
                
                # Call user's evaluation function with config dict
                # User is responsible for applying config to MLIR, compiling, running
                try:
                    time_ms, accuracy = outer_self.evaluate_fn(config)
                    
                    result = _Result()
                    result.time = time_ms
                    result.accuracy = accuracy
                    result.state = "OK" if time_ms < limit else "TIMEOUT"
                    
                    # Track best result
                    if accuracy >= outer_self.accuracy_threshold:
                        if time_ms < outer_self.best_time:
                            outer_self.best_config = config.copy()
                            outer_self.best_time = time_ms
                            outer_self.best_accuracy = accuracy
                            log.info(f"New best: time={time_ms:.2f}ms, accuracy={accuracy:.4f}")
                    
                    # Call result callback if provided
                    if outer_self.result_callback:
                        outer_self.result_callback(config, time_ms, accuracy)
                    
                    return result
                    
                except Exception as e:
                    log.error(f"Evaluation failed: {e}")
                    result = _Result()
                    result.state = "ERROR"
                    result.time = float("inf")
                    result.accuracy = 0.0
                    return result
            
            def save_final_config(self, configuration):
                """Save optimal configuration to file."""
                config_file = getattr(outer_self._args, 'output_config', 'approx_tuned_config.json')
                self.manipulator().save_to_file(configuration.data, config_file)
                log.info(f"Saved best configuration to {config_file}")
                
                # Also save the tuned MLIR
                mlir_file = getattr(outer_self._args, 'output_mlir', 'approx_tuned.mlir')
                tuned_mlir = outer_self.config_manager.apply_config(
                    outer_self.mlir_source, configuration.data
                )
                with open(mlir_file, 'w') as f:
                    f.write(tuned_mlir)
                log.info(f"Saved tuned MLIR to {mlir_file}")
        
        return _Interface(self._args)
    
    def run_tuning(self) -> Dict[str, Any]:
        """Run the auto-tuning process.
        
        Returns:
            Dictionary with results:
            {
                "best_config": Dict[str, int],
                "best_mlir": str,
                "best_time": float,
                "best_accuracy": float,
            }
        """
        interface = self.create_interface()
        interface.main(self._args)
        
        # Get final results
        if self.best_config:
            best_mlir = self.config_manager.apply_config(self.mlir_source, self.best_config)
        else:
            best_mlir = self.mlir_source
            self.best_config = {name: p.current for name, p in self.tunable_params.items()}
        
        return {
            "best_config": self.best_config,
            "best_mlir": best_mlir,
            "best_time": self.best_time,
            "best_accuracy": self.best_accuracy,
        }


def tune(
    mlir_source: str,
    evaluate_fn: Callable[[Dict[str, int]], Tuple[float, float]],
    accuracy_threshold: float = 0.9,
    time_budget: int = 3600,
    database: str = "opentuner.db",
    parallelism: int = 1,
    result_callback: Optional[Callable[[Dict[str, int], float, float], None]] = None,
) -> Dict[str, Any]:
    """High-level auto-tuning entry point.
    
    This function provides a simple interface for auto-tuning without
    needing to manage OpenTuner's argument parsing directly.
    
    Args:
        mlir_source: MLIR text with approx.util.annotation.decision_tree ops
                     (used to extract the search space)
        evaluate_fn: User callback (config: Dict[str, int]) -> (time_ms, accuracy)
                     User is responsible for:
                     - Applying config to MLIR via MLIRConfigManager.apply_config()
                     - Compilation, execution, and accuracy measurement
        accuracy_threshold: Minimum acceptable accuracy (0.0 to 1.0)
        time_budget: Tuning time in seconds
        database: OpenTuner database file path
        parallelism: Number of parallel evaluations
        result_callback: Optional callback for each result: (config, time, accuracy)
    
    Returns:
        Dictionary with:
        {
            "best_config": Dict[str, int],  # Parameter name -> value
            "best_mlir": str,                # MLIR with best config applied
            "best_time": float,              # Best execution time in ms
            "best_accuracy": float,          # Best accuracy achieved
        }
    
    Example:
        import approx_runtime as ar
        
        # Setup: user creates manager and loads MLIR
        manager = ar.MLIRConfigManager()
        mlir_source = open("my_kernel.mlir").read()
        
        def my_evaluate(config: Dict[str, int]) -> Tuple[float, float]:
            # User applies config to their MLIR file(s)
            modified_mlir = manager.apply_config(mlir_source, config)
            
            # User handles compilation
            compiled = ar.compile(modified_mlir, passes=ar.PASS_PIPELINE)
            vmfb = ar.compile_to_iree(compiled)
            module, _ = ar.load_module(vmfb)
            
            # User handles execution and timing
            start = time.time()
            output = module["my_kernel"](test_input)
            elapsed_ms = (time.time() - start) * 1000
            
            # User computes accuracy with their own metric
            accuracy = my_accuracy_metric(output, ground_truth)
            return elapsed_ms, accuracy
        
        result = ar.tune(
            mlir_source=mlir_source,
            evaluate_fn=my_evaluate,
            accuracy_threshold=0.95,
            time_budget=1800,  # 30 minutes
        )
        
        print(f"Best config: {result['best_config']}")
        print(f"Best time: {result['best_time']:.2f}ms")
    """
    if not _import_opentuner():
        raise ImportError("OpenTuner is required: pip install --user opentuner")
    
    # Build args namespace programmatically
    parser = create_tuner_arg_parser()
    args = parser.parse_args([
        f"--stop-after={time_budget}",
        f"--database={database}",
        f"--parallelism={parallelism}",
        "--no-dups",
    ])
    args.accuracy_threshold = accuracy_threshold
    
    # Create and run tuner
    tuner = ApproxTunerInterface(
        args=args,
        mlir_source=mlir_source,
        evaluate_fn=evaluate_fn,
        accuracy_threshold=accuracy_threshold,
        result_callback=result_callback,
    )
    
    return tuner.run_tuning()


def tune_jax(
    func,
    example_args,
    config: dict,
    evaluate_fn: Callable[[Dict[str, int]], Tuple[float, float]],
    accuracy_threshold: float = 0.9,
    **kwargs,
) -> Dict[str, Any]:
    """Auto-tune a JAX function with approximation config."""
    mlir_source = export_with_config(func, example_args, config)
    return tune(mlir_source, evaluate_fn, accuracy_threshold, **kwargs)


def tune_cpp(
    cpp_source: str,
    mlir_text: str,
    evaluate_fn: Callable[[Dict[str, int]], Tuple[float, float]],
    accuracy_threshold: float = 0.9,
    **kwargs,
) -> Dict[str, Any]:
    """Auto-tune C++ code with auto-generated annotations."""
    annotations = parse_and_generate(cpp_source)
    mlir_annotated = inject_annotations_text(mlir_text, annotations)
    return tune(mlir_annotated, evaluate_fn, accuracy_threshold, **kwargs)
