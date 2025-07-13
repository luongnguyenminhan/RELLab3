"""
Hyperparameters Utilities Module.

This module provides utilities for loading, validating, and managing hyperparameters in the Modular DRL Framework, as described in "Deep Reinforcement Learning: A Survey" by Wang et al. and the project engineering blueprint.

Detailed Description:
The hyperparameters module ensures reproducibility and configurability by centralizing the management of all experiment and algorithm parameters. It supports loading hyperparameters from YAML files, validating their types and ranges, and providing easy access throughout the framework. This design enables systematic experimentation and comparison of DRL algorithms.

Key Concepts/Algorithms:
- YAML-based configuration management with hierarchical structure
- Hyperparameter validation with type checking and range validation
- Support for parameter search spaces and sampling
- Configuration merging and overriding capabilities
- Experiment reproducibility through parameter logging
- Dynamic parameter updates during training

Important Parameters/Configurations:
- config_path: Path to YAML config files (e.g., `configs/dqn_config.yaml`)
- validation_schema: Schema for parameter validation
- default_values: Default parameter values for fallback
- search_spaces: Parameter search space definitions

Expected Inputs/Outputs:
- Inputs: YAML configuration files, parameter dictionaries, validation schemas
- Outputs: Validated hyperparameter objects, parameter samples, configuration files

Dependencies:
- PyYAML: YAML file parsing and generation
- numpy: Numerical operations and random sampling
- os, pathlib: File operations and path handling
- typing: Type hints for better code documentation

Author: REL Project Team
Date: 2025-07-13
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from pathlib import Path
import copy
import json


class HyperparameterManager:
    """
    Comprehensive Hyperparameter Management for DRL Experiments.

    This class provides centralized management of hyperparameters for deep reinforcement learning
    experiments, supporting YAML-based configuration, validation, and parameter search capabilities.
    It ensures reproducibility and configurability across the entire framework.

    Detailed Description:
    The HyperparameterManager class implements a robust system for handling all aspects of
    hyperparameter management, from loading and validation to search space definition and
    sampling. It supports hierarchical parameter structures, type validation, range checking,
    and dynamic parameter updates during training.

    Key Features:
    - YAML-based configuration loading and saving
    - Hierarchical parameter access with dot notation
    - Type and range validation with custom validators
    - Parameter search space definition and sampling
    - Configuration merging and overriding
    - Experiment reproducibility through parameter logging
    - Dynamic parameter updates and scheduling

    Args:
        config_path (Optional[str]): Path to YAML configuration file
        defaults (Optional[Dict]): Default parameter values
        validation_schema (Optional[Dict]): Parameter validation schema

    Attributes:
        params (Dict[str, Any]): Current parameter values
        defaults (Dict[str, Any]): Default parameter values
        validation_schema (Dict[str, Any]): Validation rules
        search_spaces (Dict[str, Dict]): Parameter search space definitions

    Example:
        ```python
        # Initialize with config file
        manager = HyperparameterManager('configs/dqn_config.yaml')

        # Access parameters
        lr = manager.get('learning_rate', 0.001)
        batch_size = manager.get('training.batch_size', 32)

        # Set parameters
        manager.set('learning_rate', 0.0005)
        manager.set('network.hidden_sizes', [64, 64])

        # Validate parameters
        manager.validate_range('learning_rate', min_val=0.0, max_val=1.0)

        # Define search space
        search_space = {
            'learning_rate': {'type': 'uniform', 'low': 0.0001, 'high': 0.01},
            'batch_size': {'type': 'choice', 'choices': [16, 32, 64, 128]}
        }
        manager.define_search_space(search_space)

        # Sample parameters
        sampled_params = manager.sample_parameters(seed=42)
        ```

    Dependencies:
    - PyYAML: YAML configuration parsing
    - numpy: Numerical operations and sampling

    Author: REL Project Team
    Date: 2025-07-13
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        defaults: Optional[Dict[str, Any]] = None,
        validation_schema: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the HyperparameterManager.

        Args:
            config_path: Path to YAML configuration file
            defaults: Default parameter values
            validation_schema: Parameter validation schema
        """
        self.params = {}
        self.defaults = defaults or {}
        self.validation_schema = validation_schema or {}
        self.search_spaces = {}
        
        # Load configuration if path provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # Merge with defaults
        self._merge_defaults()

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as f:
                loaded_params = yaml.safe_load(f) or {}
            
            self.params.update(loaded_params)
            
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {e}")

    def save_config(self, save_path: str) -> None:
        """
        Save current configuration to YAML file.

        Args:
            save_path: Path to save configuration
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.params, f, default_flow_style=False, indent=2)

    def _merge_defaults(self) -> None:
        """Merge default values with current parameters."""
        def merge_dicts(target: Dict, source: Dict) -> Dict:
            """Recursively merge dictionaries."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    merge_dicts(target[key], value)
                elif key not in target:
                    target[key] = copy.deepcopy(value)
            return target
        
        merge_dicts(self.params, self.defaults)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get parameter value using dot notation for nested access.

        Args:
            key: Parameter key (supports dot notation, e.g., 'network.hidden_sizes')
            default: Default value if parameter not found

        Returns:
            Parameter value or default
        """
        keys = key.split('.')
        value = self.params
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set parameter value using dot notation for nested access.

        Args:
            key: Parameter key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        current = self.params
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value

    def update(self, params: Dict[str, Any]) -> None:
        """
        Update multiple parameters at once.

        Args:
            params: Dictionary of parameter updates
        """
        def merge_dicts(target: Dict, source: Dict) -> Dict:
            """Recursively merge dictionaries."""
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    merge_dicts(target[key], value)
                else:
                    target[key] = value
            return target
        
        merge_dicts(self.params, params)

    def get_all(self) -> Dict[str, Any]:
        """
        Get all parameters as a dictionary.

        Returns:
            Complete parameter dictionary
        """
        return copy.deepcopy(self.params)

    def validate_range(
        self, 
        key: str, 
        min_val: Optional[Union[int, float]] = None,
        max_val: Optional[Union[int, float]] = None
    ) -> bool:
        """
        Validate that a parameter value is within specified range.

        Args:
            key: Parameter key
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            True if value is within range, False otherwise
        """
        value = self.get(key)
        
        if value is None:
            return False
        
        if min_val is not None and value < min_val:
            return False
        
        if max_val is not None and value > max_val:
            return False
        
        return True

    def validate_type(self, key: str, expected_type: type) -> bool:
        """
        Validate that a parameter has the expected type.

        Args:
            key: Parameter key
            expected_type: Expected parameter type

        Returns:
            True if type matches, False otherwise
        """
        value = self.get(key)
        return isinstance(value, expected_type)

    def validate_choices(self, key: str, valid_choices: List[Any]) -> bool:
        """
        Validate that a parameter value is in the list of valid choices.

        Args:
            key: Parameter key
            valid_choices: List of valid values

        Returns:
            True if value is valid, False otherwise
        """
        value = self.get(key)
        return value in valid_choices

    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Validate all parameters against the validation schema.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for param_key, validation_rules in self.validation_schema.items():
            value = self.get(param_key)
            
            # Check if required parameter exists
            if validation_rules.get('required', False) and value is None:
                errors.append(f"Required parameter '{param_key}' is missing")
                continue
            
            if value is None:
                continue  # Skip validation for optional missing parameters
            
            # Type validation
            if 'type' in validation_rules:
                expected_type = validation_rules['type']
                if not isinstance(value, expected_type):
                    errors.append(f"Parameter '{param_key}' has wrong type. Expected {expected_type.__name__}, got {type(value).__name__}")
            
            # Range validation
            if 'min' in validation_rules or 'max' in validation_rules:
                min_val = validation_rules.get('min')
                max_val = validation_rules.get('max')
                if not self.validate_range(param_key, min_val, max_val):
                    errors.append(f"Parameter '{param_key}' value {value} is out of range [{min_val}, {max_val}]")
            
            # Choices validation
            if 'choices' in validation_rules:
                valid_choices = validation_rules['choices']
                if not self.validate_choices(param_key, valid_choices):
                    errors.append(f"Parameter '{param_key}' value {value} is not in valid choices {valid_choices}")
        
        return len(errors) == 0, errors

    def define_search_space(self, search_spaces: Dict[str, Dict[str, Any]]) -> None:
        """
        Define parameter search spaces for hyperparameter optimization.

        Args:
            search_spaces: Dictionary defining search spaces for parameters
                Format: {
                    'param_name': {
                        'type': 'uniform|choice|log_uniform|normal',
                        'low': float,     # for uniform, log_uniform
                        'high': float,    # for uniform, log_uniform
                        'choices': list,  # for choice
                        'mean': float,    # for normal
                        'std': float      # for normal
                    }
                }
        """
        self.search_spaces.update(search_spaces)

    def sample_parameters(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Sample parameters from defined search spaces.

        Args:
            seed: Random seed for reproducible sampling

        Returns:
            Dictionary of sampled parameter values
        """
        if seed is not None:
            np.random.seed(seed)
        
        sampled = {}
        
        for param_name, space_def in self.search_spaces.items():
            space_type = space_def.get('type', 'uniform')
            
            if space_type == 'uniform':
                low = space_def['low']
                high = space_def['high']
                sampled[param_name] = np.random.uniform(low, high)
            
            elif space_type == 'log_uniform':
                low = space_def['low']
                high = space_def['high']
                log_low = np.log10(low)
                log_high = np.log10(high)
                sampled[param_name] = 10 ** np.random.uniform(log_low, log_high)
            
            elif space_type == 'choice':
                choices = space_def['choices']
                sampled[param_name] = np.random.choice(choices)
            
            elif space_type == 'normal':
                mean = space_def['mean']
                std = space_def['std']
                sampled[param_name] = np.random.normal(mean, std)
            
            else:
                raise ValueError(f"Unknown search space type: {space_type}")
        
        return sampled

    def apply_sampled_parameters(self, sampled_params: Dict[str, Any]) -> None:
        """
        Apply sampled parameters to current configuration.

        Args:
            sampled_params: Dictionary of sampled parameter values
        """
        for key, value in sampled_params.items():
            self.set(key, value)

    def create_parameter_schedule(
        self, 
        param_key: str, 
        schedule_type: str,
        **kwargs
    ) -> Callable[[int], float]:
        """
        Create a parameter schedule function for dynamic parameter updates.

        Args:
            param_key: Parameter key to schedule
            schedule_type: Type of schedule ('linear', 'exponential', 'step', 'cosine')
            **kwargs: Schedule-specific parameters

        Returns:
            Schedule function that takes step and returns parameter value
        """
        initial_value = self.get(param_key)
        
        if schedule_type == 'linear':
            final_value = kwargs.get('final_value', initial_value)
            total_steps = kwargs.get('total_steps', 1000)
            
            def schedule(step: int) -> float:
                progress = min(step / total_steps, 1.0)
                return initial_value + (final_value - initial_value) * progress
        
        elif schedule_type == 'exponential':
            decay_rate = kwargs.get('decay_rate', 0.99)
            
            def schedule(step: int) -> float:
                return initial_value * (decay_rate ** step)
        
        elif schedule_type == 'step':
            step_size = kwargs.get('step_size', 100)
            decay_factor = kwargs.get('decay_factor', 0.1)
            
            def schedule(step: int) -> float:
                return initial_value * (decay_factor ** (step // step_size))
        
        elif schedule_type == 'cosine':
            min_value = kwargs.get('min_value', initial_value * 0.1)
            total_steps = kwargs.get('total_steps', 1000)
            
            def schedule(step: int) -> float:
                progress = min(step / total_steps, 1.0)
                return min_value + (initial_value - min_value) * (1 + np.cos(np.pi * progress)) / 2
        
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        return schedule

    def save(self, filepath: str) -> None:
        """
        Save hyperparameters to file.

        Args:
            filepath: Path to save the parameters
        """
        self.save_config(filepath)

    def __str__(self) -> str:
        """String representation of hyperparameters."""
        return f"HyperparameterManager({len(self.params)} parameters)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"HyperparameterManager(params={self.params})"


# Export main classes
__all__ = ['HyperparameterManager']