"""
Base Agent Utilities Module.

This module provides the abstract base class for all utility agents in the Modular DRL Framework,
as described in "Deep Reinforcement Learning: A Survey" by Wang et al. and the project engineering
blueprint. The base class encapsulates common functionality for configuration management, logging,
hyperparameter handling, and experiment reproducibility.

Detailed Description:
The BaseAgent class serves as the foundation for all utility agents, providing a consistent
interface and shared functionality. It supports YAML-based configuration management, integrated
logging, hyperparameter validation, and experiment reproducibility through deterministic seeding.
The design follows the Abstract Base Class pattern to ensure consistent interfaces while allowing
agent-specific implementations through abstract methods.

Key Concepts/Design Patterns:
- Abstract Base Class (ABC) pattern for consistent interfaces
- Template Method pattern for common initialization and lifecycle management
- Strategy pattern for different configuration and logging approaches
- YAML-based configuration management for reproducibility
- Comprehensive logging and metrics tracking
- Deterministic behavior through seed management

Key Components:
- BaseAgent: Abstract base class for all utility agents
- Configuration management with validation and defaults
- Logging integration with metrics tracking
- Hyperparameter management and validation
- Experiment reproducibility through seeding
- Lifecycle management (setup, reset, cleanup)

Agent Features:
- YAML-based configuration with validation
- Automatic logging setup and metrics tracking
- Hyperparameter loading and management
- Deterministic behavior with seed control
- Model/state persistence and loading
- Performance monitoring and debugging

Expected Usage:
- Utility agents inherit from BaseAgent and implement abstract methods
- All agent parameters are configured via YAML files in `configs/`
- Agents support both training and evaluation modes
- Experiment tracking and reproducibility are built-in

Dependencies:
- abc: Abstract base class functionality
- yaml: Configuration file parsing
- logging: Logging and metrics tracking
- numpy: Numerical operations and random seeding
- typing: Type hints and annotations
- os, sys: File and system operations

Author: REL Project Team
Date: 2025-07-13
"""

import os
import sys
import yaml
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import time
from collections import defaultdict, deque

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from utils.logger import Logger
    from utils.hyperparameters import HyperparameterManager
except ImportError:
    # Fallback implementations if utils modules are not available
    import logging
    
    class Logger:
        """Fallback logger implementation."""
        def __init__(self, experiment_name: str = "BaseAgent", log_dir: str = "./logs"):
            self.experiment_name = experiment_name
            self.log_dir = log_dir
            self.logger = logging.getLogger(experiment_name)
            self.metrics = defaultdict(list)
            self.episodes = []
            self.metadata = {}
            
            # Setup basic logging
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        def info(self, msg: str): self.logger.info(msg)
        def warning(self, msg: str): self.logger.warning(msg)
        def error(self, msg: str): self.logger.error(msg)
        def log_scalar(self, key: str, value: float, step: int = 0): pass
        def log_episode(self, data: Dict[str, Any]): pass
        def log_hyperparameters(self, params: Dict[str, Any]): pass
        def save_metrics(self) -> str: return ""
        def get_metrics_summary(self) -> Dict[str, Any]: return {}

    class HyperparameterManager:
        """Fallback hyperparameter manager implementation."""
        def __init__(self, config_path: Optional[str] = None):
            self.params = {}
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.params = yaml.safe_load(f) or {}
        
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            value = self.params
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        
        def set(self, key: str, value: Any) -> None:
            keys = key.split('.')
            current = self.params
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        
        def get_all(self) -> Dict[str, Any]: return self.params
        def update(self, params: Dict[str, Any]) -> None: self.params.update(params)


class BaseAgent(ABC):
    """
    Abstract Base Class for Utility Agents in the Modular DRL Framework.

    This class provides a unified interface and common functionality for all utility agents
    in the framework. It encapsulates shared patterns including configuration management,
    logging integration, hyperparameter handling, and experiment reproducibility while
    allowing agent-specific implementations through abstract methods.

    Detailed Description:
    The BaseAgent follows the Abstract Base Class pattern to ensure consistent interfaces
    across all utility agent types while providing concrete implementations for common
    functionality. It supports YAML-based configuration management, integrated logging,
    hyperparameter validation, and deterministic behavior through seed management. Each
    utility agent inherits from this base class and implements agent-specific methods.

    Key Concepts/Design Patterns:
    - Abstract Base Class (ABC) pattern for interface consistency
    - Template Method pattern for initialization and lifecycle management
    - Configuration-driven design with YAML support
    - Comprehensive logging and experiment tracking
    - Deterministic behavior through seed management

    Common Functionality Provided:
    - Configuration loading and validation from YAML files
    - Logging setup with metrics tracking and experiment management
    - Hyperparameter management with validation and defaults
    - Deterministic seeding for reproducibility
    - Agent lifecycle management (setup, reset, cleanup)
    - Performance monitoring and debugging utilities
    - Model/state persistence and loading capabilities

    Abstract Methods (must be implemented by subclasses):
    - _get_default_config(): Provide default configuration parameters
    - _validate_config(): Validate loaded configuration parameters
    - _setup_agent(): Setup agent-specific components and state
    - process(): Main agent processing method

    Args:
        agent_name (str): Name of the agent for identification and logging
        config_path (Optional[str]): Path to YAML configuration file
        log_dir (str): Directory for logging and experiment tracking
        seed (Optional[int]): Random seed for reproducible behavior
        **kwargs: Additional keyword arguments

    Attributes:
        agent_name (str): Agent identifier for logging and tracking
        config (Dict[str, Any]): Agent configuration parameters
        logger (Logger): Logger instance for this agent
        hyperparams (HyperparameterManager): Hyperparameter manager
        seed (Optional[int]): Random seed for reproducibility
        is_setup (bool): Whether agent has been properly initialized
        metrics (Dict[str, Any]): Performance and debugging metrics
        state (Dict[str, Any]): Current agent state information

    Expected Usage:
    ```python
    # Inherit from BaseAgent to create specific utility agents
    class CustomUtilityAgent(BaseAgent):
        def _get_default_config(self):
            return {"param1": "value1", "param2": 42}

        def _validate_config(self):
            # Validate configuration parameters
            pass

        def _setup_agent(self):
            # Setup agent-specific components
            pass

        def process(self, data):
            # Main agent processing logic
            pass

    # Use the agent
    agent = CustomUtilityAgent(
        agent_name="my_utility_agent",
        config_path="configs/utility_config.yaml",
        seed=42
    )
    result = agent.process(input_data)
    ```

    Dependencies:
    - abc: Abstract base class functionality
    - yaml: Configuration file parsing
    - logging: Logging and metrics tracking
    - numpy: Random seeding and numerical operations
    - typing: Type hints and annotations

    Author: REL Project Team
    Date: 2025-07-13
    """

    def __init__(
        self,
        agent_name: str,
        config_path: Optional[str] = None,
        log_dir: str = "./logs",
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the base agent with common parameters and infrastructure.

        Args:
            agent_name: Name of the agent for identification and logging
            config_path: Path to YAML configuration file
            log_dir: Directory for logging and experiment tracking
            seed: Random seed for reproducible behavior
            **kwargs: Additional keyword arguments
        """
        # Store basic parameters
        self.agent_name = agent_name
        self.seed = seed
        self.log_dir = log_dir
        self.is_setup = False
        
        # Initialize state tracking
        self.state = {}
        self.metrics = defaultdict(list)
        self._step_count = 0
        self._episode_count = 0
        self._start_time = time.time()
        
        # Set random seed for reproducibility
        if seed is not None:
            self._set_seed(seed)
        
        # Load and merge configuration
        self.config = self._load_and_merge_config(config_path)
        
        # Validate configuration
        self._validate_config()
        
        # Setup logging and hyperparameter management
        self._setup_logging()
        self._setup_hyperparameters(config_path)
        
        # Setup agent-specific components
        self._setup_agent()
        
        # Mark as properly initialized
        self.is_setup = True
        
        self.logger.info(f"Initialized {agent_name} with seed={seed}")

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        # Note: Additional seeds (torch, etc.) would be set here if available

    def _load_and_merge_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load and merge configuration with defaults."""
        # Start with default configuration
        config = self._get_default_config()
        
        # Load and merge YAML configuration
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}
                
                # Extract agent-specific config if it exists
                agent_config = yaml_config.get(self.agent_name, {})
                if not agent_config:
                    # Fallback to generic 'agent' section
                    agent_config = yaml_config.get('agent', {})
                
                # Merge configurations (YAML overrides defaults)
                config.update(agent_config)
                
            except Exception as e:
                self._log_warning(f"Could not load config from {config_path}: {e}")
        
        return config

    def _setup_logging(self) -> None:
        """Setup logging infrastructure."""
        try:
            self.logger = Logger(
                experiment_name=self.agent_name,
                log_dir=self.log_dir,
                log_level=getattr(logging, self.config.get('log_level', 'INFO'))
            )
        except Exception:
            # Fallback to basic logging
            self.logger = Logger(experiment_name=self.agent_name, log_dir=self.log_dir)

    def _setup_hyperparameters(self, config_path: Optional[str]) -> None:
        """Setup hyperparameter management."""
        try:
            self.hyperparams = HyperparameterManager(config_path=config_path)
        except Exception:
            # Fallback to basic hyperparameter management
            self.hyperparams = HyperparameterManager(config_path=config_path)

    def _log_warning(self, message: str) -> None:
        """Log warning message safely."""
        if hasattr(self, 'logger'):
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the agent to initial state.

        Args:
            seed: Optional new random seed
        """
        if seed is not None:
            self.seed = seed
            self._set_seed(seed)
        
        # Reset counters and metrics
        self._step_count = 0
        self._episode_count = 0
        self._start_time = time.time()
        self.state.clear()
        
        # Agent-specific reset
        self._reset_agent()
        
        self.logger.info(f"Reset {self.agent_name} with seed={seed}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration parameter by key.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set_config(self, key: str, value: Any) -> None:
        """
        Set configuration parameter.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value

    def log_metric(self, key: str, value: Union[float, int], step: Optional[int] = None) -> None:
        """
        Log a metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number (uses internal counter if not provided)
        """
        if step is None:
            step = self._step_count
        
        self.metrics[key].append((step, value))
        
        if hasattr(self.logger, 'log_scalar'):
            self.logger.log_scalar(key, value, step)

    def log_episode_data(self, data: Dict[str, Any]) -> None:
        """
        Log episode-level data.

        Args:
            data: Episode data dictionary
        """
        data['episode'] = self._episode_count
        data['timestamp'] = time.time() - self._start_time
        
        if hasattr(self.logger, 'log_episode'):
            self.logger.log_episode(data)
        
        self._episode_count += 1

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for logged metrics.

        Returns:
            Dictionary containing metric summaries
        """
        summary = {}
        
        for key, values in self.metrics.items():
            if values:
                metric_values = [v[1] for v in values]  # Extract values (ignore steps)
                summary[key] = {
                    'count': len(metric_values),
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values),
                    'latest': metric_values[-1]
                }
        
        return summary

    def save_state(self, save_path: str) -> None:
        """
        Save agent state to disk.

        Args:
            save_path: Path to save the agent state
        """
        state_dict = {
            'agent_name': self.agent_name,
            'config': self.config,
            'state': self.state,
            'metrics': dict(self.metrics),
            'step_count': self._step_count,
            'episode_count': self._episode_count,
            'seed': self.seed
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)
        
        self.logger.info(f"Agent state saved to {save_path}")

    def load_state(self, load_path: str) -> None:
        """
        Load agent state from disk.

        Args:
            load_path: Path to load the agent state from
        """
        with open(load_path, 'r') as f:
            state_dict = json.load(f)
        
        # Restore state
        self.config = state_dict.get('config', {})
        self.state = state_dict.get('state', {})
        self.metrics = defaultdict(list, state_dict.get('metrics', {}))
        self._step_count = state_dict.get('step_count', 0)
        self._episode_count = state_dict.get('episode_count', 0)
        self.seed = state_dict.get('seed', None)
        
        self.logger.info(f"Agent state loaded from {load_path}")

    def get_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the agent.

        Returns:
            Dictionary containing agent information
        """
        return {
            'agent_name': self.agent_name,
            'is_setup': self.is_setup,
            'step_count': self._step_count,
            'episode_count': self._episode_count,
            'config': self.config,
            'uptime': time.time() - self._start_time,
            'metrics_count': len(self.metrics),
            'seed': self.seed
        }

    def cleanup(self) -> None:
        """Cleanup agent resources and save final state."""
        try:
            # Save metrics if logger supports it
            if hasattr(self.logger, 'save_metrics'):
                self.logger.save_metrics()
            
            # Perform agent-specific cleanup
            self._cleanup_agent()
            
            self.logger.info(f"Cleaned up {self.agent_name}")
        except Exception as e:
            self._log_warning(f"Error during cleanup: {e}")

    # ==================== Abstract Methods ====================

    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for this agent type.
        
        Returns:
            Dictionary containing default configuration parameters
        """
        pass

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the loaded configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def _setup_agent(self) -> None:
        """
        Setup agent-specific components and state.
        
        This method is called during initialization after configuration
        loading and logging setup.
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Main agent processing method.
        
        This method contains the core logic of the utility agent.
        
        Args:
            *args: Variable positional arguments
            **kwargs: Variable keyword arguments
            
        Returns:
            Processing result (type depends on agent implementation)
        """
        pass

    # ==================== Optional Override Methods ====================

    def _reset_agent(self) -> None:
        """
        Reset agent-specific state.
        
        Override this method to implement agent-specific reset logic.
        """
        pass

    def _cleanup_agent(self) -> None:
        """
        Cleanup agent-specific resources.
        
        Override this method to implement agent-specific cleanup logic.
        """
        pass

    def _step(self) -> None:
        """Increment step counter and update internal state."""
        self._step_count += 1


# Factory function for creating agents
def create_agent(
    agent_type: str,
    agent_name: str,
    config_path: Optional[str] = None,
    log_dir: str = "./logs",
    seed: Optional[int] = None,
    **kwargs
) -> BaseAgent:
    """
    Factory function to create utility agents based on type.

    Args:
        agent_type: Type of agent to create
        agent_name: Name of the agent instance
        config_path: Path to configuration file
        log_dir: Directory for logging
        seed: Random seed for reproducibility
        **kwargs: Additional arguments

    Returns:
        BaseAgent: The created agent instance

    Raises:
        ValueError: If agent_type is not recognized
        ImportError: If agent module cannot be imported
    """
    # Registry of available agent types
    agent_registry = {
        'data_processor': 'data_processor.DataProcessorAgent',
        'experiment_tracker': 'experiment_tracker.ExperimentTrackerAgent',
        'hyperparameter_optimizer': 'hyperparameter_optimizer.HyperparameterOptimizerAgent',
        'model_evaluator': 'model_evaluator.ModelEvaluatorAgent',
        # Add more agent types as they are implemented
    }
    
    if agent_type not in agent_registry:
        available_types = ', '.join(agent_registry.keys())
        raise ValueError(f"Unknown agent type: {agent_type}. Available types: {available_types}")
    
    try:
        module_path, class_name = agent_registry[agent_type].rsplit('.', 1)
        module = __import__(f'utils.{module_path}', fromlist=[class_name])
        agent_class = getattr(module, class_name)
        
        return agent_class(
            agent_name=agent_name,
            config_path=config_path,
            log_dir=log_dir,
            seed=seed,
            **kwargs
        )
    except ImportError as e:
        raise ImportError(f"Could not import {agent_type}: {e}")


# Export main classes and functions
__all__ = [
    'BaseAgent',
    'create_agent'
]