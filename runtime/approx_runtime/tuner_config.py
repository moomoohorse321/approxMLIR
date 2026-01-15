"""MLIR configuration manager for auto-tuning.

This module handles parsing and modification of approx.util.annotation.decision_tree
operations in MLIR files. It extracts tunable parameters (thresholds and decisions)
and applies new configurations during the auto-tuning process.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = ['TunableParam', 'MLIRConfigManager']


@dataclass
class TunableParam:
    """A single tunable parameter extracted from MLIR.
    
    Attributes:
        name: Unique identifier (e.g., "my_func_threshold_0")
        lower: Minimum allowed value (from thresholds_lowers or min(decision_values))
        upper: Maximum allowed value (from thresholds_uppers or max(decision_values))
        current: Current value in the MLIR
        param_type: Either "threshold" or "decision"
        func_name: Target function name from the annotation
        index: Parameter index within its type
    """
    name: str
    lower: int
    upper: int
    current: int
    param_type: str  # "threshold" or "decision"
    func_name: str
    index: int


class MLIRConfigManager:
    """Parse and modify approx.util.annotation.decision_tree operations.
    
    This class provides functionality to:
    1. Parse MLIR files and extract tunable parameters
    2. Apply new configurations to MLIR text
    3. Validate configurations (e.g., ascending thresholds constraint)
    
    Example:
        manager = MLIRConfigManager()
        params = manager.parse_annotations(mlir_text)
        
        # params is a dict like:
        # {
        #     "my_func_threshold_0": TunableParam(...),
        #     "my_func_decision_0": TunableParam(...),
        #     ...
        # }
        
        new_config = {"my_func_threshold_0": 75, "my_func_decision_0": 1}
        modified_mlir = manager.apply_config(mlir_text, new_config)
    """
    
    # Pattern to match the entire decision_tree annotation
    ANNOTATION_PATTERN = re.compile(
        r'"approx\.util\.annotation\.decision_tree"\(\)\s*<\{([^}]+)\}>\s*:\s*\(\)\s*->\s*\(\)',
        re.DOTALL
    )
    
    def parse_annotations(self, mlir_text: str) -> Dict[str, TunableParam]:
        """Extract tunable parameters from MLIR annotations.
        
        Args:
            mlir_text: MLIR module text containing decision_tree annotations
            
        Returns:
            Dictionary mapping parameter names to TunableParam objects
        """
        params = {}
        
        for match in self.ANNOTATION_PATTERN.finditer(mlir_text):
            annotation_content = match.group(1)
            
            # Extract func_name
            func_name = self._extract_string_attr(annotation_content, 'func_name')
            if not func_name:
                continue
            
            # Extract threshold bounds and current values
            thresholds_lowers = self._extract_i32_array(annotation_content, 'thresholds_lowers')
            thresholds_uppers = self._extract_i32_array(annotation_content, 'thresholds_uppers')
            thresholds = self._extract_i32_array(annotation_content, 'thresholds')
            
            # Extract decision bounds and current values
            decision_values = self._extract_i32_array(annotation_content, 'decision_values')
            decisions = self._extract_i32_array(annotation_content, 'decisions')
            
            # Create threshold parameters
            if thresholds and thresholds_lowers and thresholds_uppers:
                for i, current_val in enumerate(thresholds):
                    lower = thresholds_lowers[i] if i < len(thresholds_lowers) else 0
                    upper = thresholds_uppers[i] if i < len(thresholds_uppers) else 100
                    
                    param_name = f"{func_name}_threshold_{i}"
                    params[param_name] = TunableParam(
                        name=param_name,
                        lower=lower,
                        upper=upper,
                        current=current_val,
                        param_type="threshold",
                        func_name=func_name,
                        index=i
                    )
            
            # Create decision parameters
            if decisions and decision_values:
                min_decision = min(decision_values)
                max_decision = max(decision_values)
                
                for i, current_val in enumerate(decisions):
                    param_name = f"{func_name}_decision_{i}"
                    params[param_name] = TunableParam(
                        name=param_name,
                        lower=min_decision,
                        upper=max_decision,
                        current=current_val,
                        param_type="decision",
                        func_name=func_name,
                        index=i
                    )
        
        return params
    
    def apply_config(self, mlir_text: str, config: Dict[str, int]) -> str:
        """Apply a configuration to MLIR text.
        
        Args:
            mlir_text: Original MLIR text
            config: Dictionary mapping parameter names to new values
            
        Returns:
            Modified MLIR text with updated parameter values
        """
        # Group config by function name
        func_configs: Dict[str, Dict[str, int]] = {}
        for param_name, value in config.items():
            # Parse param_name: func_name_type_index
            parts = param_name.rsplit('_', 2)
            if len(parts) >= 3:
                func_name = '_'.join(parts[:-2])
                param_type = parts[-2]
                index = int(parts[-1])
                
                if func_name not in func_configs:
                    func_configs[func_name] = {'thresholds': {}, 'decisions': {}}
                
                if param_type == 'threshold':
                    func_configs[func_name]['thresholds'][index] = value
                elif param_type == 'decision':
                    func_configs[func_name]['decisions'][index] = value
        
        # Apply modifications for each function
        modified_mlir = mlir_text
        for func_name, changes in func_configs.items():
            modified_mlir = self._modify_annotation(
                modified_mlir, 
                func_name, 
                changes['thresholds'], 
                changes['decisions']
            )
        
        return modified_mlir
    
    def validate_config(self, config: Dict[str, int], params: Optional[Dict[str, TunableParam]] = None) -> bool:
        """Validate a configuration.
        
        Checks:
        1. All values are within bounds (if params provided)
        2. Thresholds are in ascending order for each function
        
        Args:
            config: Configuration to validate
            params: Optional TunableParam dict for bounds checking
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # Group thresholds by function
        func_thresholds: Dict[str, List[Tuple[int, int]]] = {}
        
        for param_name, value in config.items():
            # Check bounds if params provided
            if params and param_name in params:
                param = params[param_name]
                if value < param.lower or value > param.upper:
                    return False
            
            # Collect thresholds for ascending check
            parts = param_name.rsplit('_', 2)
            if len(parts) >= 3 and parts[-2] == 'threshold':
                func_name = '_'.join(parts[:-2])
                index = int(parts[-1])
                
                if func_name not in func_thresholds:
                    func_thresholds[func_name] = []
                func_thresholds[func_name].append((index, value))
        
        # Check ascending order for each function
        for func_name, threshold_list in func_thresholds.items():
            if len(threshold_list) <= 1:
                continue
            
            # Sort by index and extract values
            threshold_list.sort(key=lambda x: x[0])
            values = [v for _, v in threshold_list]
            
            # Check strictly ascending
            for i in range(1, len(values)):
                if values[i] <= values[i-1]:
                    return False
        
        return True
    
    def _extract_string_attr(self, content: str, attr_name: str) -> Optional[str]:
        """Extract a string attribute value from annotation content."""
        pattern = rf'{attr_name}\s*=\s*"([^"]+)"'
        match = re.search(pattern, content)
        return match.group(1) if match else None
    
    def _extract_i32_array(self, content: str, attr_name: str) -> List[int]:
        """Extract an i32 array attribute from annotation content."""
        pattern = rf'{attr_name}\s*=\s*array<i32:\s*([^>]+)>'
        match = re.search(pattern, content)
        if not match:
            return []
        
        values_str = match.group(1)
        try:
            return [int(v.strip()) for v in values_str.split(',') if v.strip()]
        except ValueError:
            return []
    
    def _modify_annotation(
        self, 
        mlir_text: str, 
        func_name: str, 
        threshold_changes: Dict[int, int],
        decision_changes: Dict[int, int]
    ) -> str:
        """Modify a specific function's annotation.
        
        Args:
            mlir_text: Original MLIR text
            func_name: Target function name
            threshold_changes: Dict mapping threshold index to new value
            decision_changes: Dict mapping decision index to new value
            
        Returns:
            Modified MLIR text
        """
        # Find the annotation for this function
        pattern = re.compile(
            rf'("approx\.util\.annotation\.decision_tree"\(\)\s*<\{{)([^}}]*func_name\s*=\s*"{re.escape(func_name)}"[^}}]*)(\}}>\s*:\s*\(\)\s*->\s*\(\))',
            re.DOTALL
        )
        
        match = pattern.search(mlir_text)
        if not match:
            return mlir_text
        
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)
        
        # Modify thresholds array
        if threshold_changes:
            content = self._update_array_values(content, 'thresholds', threshold_changes)
        
        # Modify decisions array
        if decision_changes:
            content = self._update_array_values(content, 'decisions', decision_changes)
        
        # Reconstruct the annotation
        new_annotation = prefix + content + suffix
        
        # Replace in original text
        return mlir_text[:match.start()] + new_annotation + mlir_text[match.end():]
    
    def _update_array_values(
        self, 
        content: str, 
        array_name: str, 
        changes: Dict[int, int]
    ) -> str:
        """Update specific values in an array attribute.
        
        Args:
            content: Annotation content string
            array_name: Name of the array attribute (e.g., 'thresholds')
            changes: Dict mapping index to new value
            
        Returns:
            Modified content string
        """
        pattern = rf'({array_name}\s*=\s*array<i32:\s*)([^>]+)(>)'
        match = re.search(pattern, content)
        if not match:
            return content
        
        # Parse current values
        values_str = match.group(2)
        try:
            values = [int(v.strip()) for v in values_str.split(',') if v.strip()]
        except ValueError:
            return content
        
        # Apply changes
        for index, new_value in changes.items():
            if 0 <= index < len(values):
                values[index] = new_value
        
        # Reconstruct
        new_values_str = ', '.join(str(v) for v in values)
        new_array = match.group(1) + new_values_str + match.group(3)
        
        return content[:match.start()] + new_array + content[match.end():]
