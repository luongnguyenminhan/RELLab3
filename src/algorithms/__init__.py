"""
Algorithms Module.

This package contains the core Deep Reinforcement Learning (DRL) algorithm implementations for the Modular DRL Framework, as classified in "Deep Reinforcement Learning: A Survey" by Wang et al.

Key Algorithms:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- DDPG (Deep Deterministic Policy Gradient)
- SAC (Soft Actor-Critic)

Each algorithm integrates canonical improvements (e.g., experience replay, target networks, trust region, entropy regularization) and is designed for modularity and extensibility.

Expected Usage:
- Import and instantiate algorithm classes for training and evaluation
- Algorithms are configured via YAML files in `configs/`

Dependencies:
- torch, numpy, gym, PyYAML

Author: REL Project Team
Date: 2025-07-13
"""

import os
import sys
import yaml
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils.logger import Logger
except ImportError:
    # Fallback logger if utils.logger is not available
    import logging
    class Logger:
        def __init__(self, name: str):
            self.logger = logging.getLogger(name)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        def info(self, msg: str): self.logger.info(msg)
        def warning(self, msg: str): self.logger.warning(msg)
        def error(self, msg: str): self.logger.error(msg)


class BaseAgent(ABC):
    """
    Abstract Base Class for Deep Reinforcement Learning Agents.
    
    This class provides a unified interface and common functionality for all DRL algorithms
    in the Modular DRL Framework. It encapsulates shared patterns including configuration
    management, training infrastructure, metrics tracking, and persistence while allowing
    algorithm-specific implementations through abstract methods.
    
    Detailed Description:
    The BaseAgent follows the Abstract Base Class pattern to ensure consistent interfaces
    across all algorithms while providing concrete implementations for common functionality.
    It supports YAML-based configuration management, comprehensive metrics tracking,
    model persistence, and standardized logging. Each algorithm inherits from this base
    class and implements algorithm-specific methods for network initialization, action
    selection, and learning procedures.
    
    Key Concepts/Design Patterns:
    - Abstract Base Class (ABC) pattern for interface consistency
    - Template Method pattern for configuration and training flow
    - Strategy pattern for algorithm-specific implementations
    - YAML-based configuration management
    - Comprehensive metrics tracking and logging
    
    Common Functionality Provided:
    - Configuration loading and validation
    - Training metrics collection and reporting
    - Model save/load functionality
    - Step and episode counting
    - Logging infrastructure
    - Device management
    
    Abstract Methods (must be implemented by subclasses):
    - _init_networks(): Initialize algorithm-specific neural networks
    - _get_default_config(): Provide default configuration parameters
    - act(): Select actions based on current policy
    - _learn(): Perform learning updates using collected experiences
    
    Args:
        state_size (int): Dimension of the state/observation space
        action_size (int): Size of the action space (discrete: num_actions, continuous: action_dim)
        config_path (str): Path to YAML configuration file
        device (str): Computing device ('cpu', 'cuda', 'cuda:0', etc.)
        algorithm_name (str): Name of the algorithm for logging purposes
    
    Attributes:
        state_size (int): State space dimension
        action_size (int): Action space size
        device (torch.device): PyTorch device for computations
        config (Dict[str, Any]): Algorithm configuration parameters
        step_count (int): Total number of training steps
        episode_count (int): Total number of episodes
        training_metrics (defaultdict): Collection of training metrics
        logger (Logger): Logger instance for this agent
    
    Expected Usage:
    ```python
    # Inherit from BaseAgent to create new algorithms
    class MyAgent(BaseAgent):
        def _init_networks(self):
            # Initialize algorithm-specific networks
            pass
        
        def _get_default_config(self):
            # Return default configuration
            return {...}
        
        def act(self, state):
            # Implement action selection
            pass
        
        def _learn(self):
            # Implement learning procedure
            pass
    
    # Use the agent
    agent = MyAgent(state_size=4, action_size=2)
    action = agent.act(state)
    agent.step(state, action, reward, next_state, done)
    ```
    
    Dependencies:
    - torch: Neural network computations
    - numpy: Numerical operations
    - yaml: Configuration file parsing
    - abc: Abstract base class functionality
    - typing: Type hints
    - collections: defaultdict for metrics
    
    Author: REL Project Team
    Date: 2025-07-13
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config_path: str,
        device: str = "cpu",
        algorithm_name: str = "BaseAgent"
    ):
        """
        Initialize the base agent with common parameters and infrastructure.
        
        Args:
            state_size: Dimension of the state/observation space
            action_size: Size of the action space
            config_path: Path to YAML configuration file
            device: Computing device for PyTorch operations
            algorithm_name: Name of the algorithm for logging
        """
        # Store basic parameters
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.algorithm_name = algorithm_name
        
        # Initialize training counters
        self.step_count: int = 0
        self.episode_count: int = 0
        
        # Initialize metrics tracking
        self.training_metrics: defaultdict = defaultdict(list)
        
        # Initialize logger
        self.logger = Logger(algorithm_name)
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.logger.info(f"Initialized {algorithm_name} with config from {config_path}")
        
        # Initialize algorithm-specific networks (abstract method)
        self._init_networks()
        
        # Log device and configuration info
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"State size: {state_size}, Action size: {action_size}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with fallback to defaults.
        
        This method attempts to load configuration from the specified YAML file.
        If the file is not found or contains invalid YAML, it falls back to the
        default configuration provided by the subclass.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            ValueError: If the configuration contains invalid parameters
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
                # Handle case where YAML file contains only documentation strings
                if config is None or isinstance(config, str):
                    self.logger.warning(f"Config file {config_path} contains no valid configuration. Using defaults.")
                    config = self._get_default_config()
                else:
                    # Merge with defaults to ensure all required parameters are present
                    default_config = self._get_default_config()
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                            self.logger.info(f"Added missing config parameter: {key} = {value}")
                
                return config
                
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML config file {config_path}: {e}")
            self.logger.warning("Falling back to default configuration.")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Unexpected error loading config from {config_path}: {e}")
            self.logger.warning("Falling back to default configuration.")
            return self._get_default_config()
    
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration parameters for the algorithm.
        
        This abstract method must be implemented by each algorithm to provide
        sensible default values for all configuration parameters. These defaults
        serve as fallbacks when configuration files are missing or incomplete.
        
        Returns:
            Dictionary containing default configuration parameters
            
        Note:
            Each algorithm should provide comprehensive defaults that enable
            basic functionality without requiring external configuration.
        """
        pass
    
    @abstractmethod
    def _init_networks(self):
        """
        Initialize algorithm-specific neural networks.
        
        This abstract method must be implemented by each algorithm to set up
        the required neural networks (e.g., Q-networks, policy networks,
        value networks, target networks). The method should create all networks
        needed for the algorithm and move them to the appropriate device.
        
        Note:
            - Networks should be stored as instance attributes
            - Networks should be moved to self.device
            - Target networks should be initialized with same weights as main networks
        """
        pass
    
    @abstractmethod
    def act(
        self,
        state: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Union[int, np.ndarray, torch.Tensor]:
        """
        Select an action based on the current state and policy.
        
        This abstract method must be implemented by each algorithm to define
        how actions are selected given the current state. The method should
        handle both exploration and exploitation appropriately for the algorithm.
        
        Args:
            state: Current state/observation
            **kwargs: Algorithm-specific parameters (e.g., deterministic, add_noise)
            
        Returns:
            Selected action (format depends on action space type)
            
        Note:
            - For discrete action spaces: return action index (int)
            - For continuous action spaces: return action vector (np.ndarray)
            - Handle exploration vs exploitation based on algorithm design
        """
        pass
    
    @abstractmethod
    def _learn(self):
        """
        Perform one learning/training step.
        
        This abstract method must be implemented by each algorithm to define
        the core learning procedure. It should update the agent's policy/networks
        using collected experiences and update training metrics.
        
        Note:
            - Should only be called when sufficient data is available
            - Should update self.training_metrics with relevant metrics
            - Should handle gradient computation and optimization
        """
        pass
    
    def step(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[int, np.ndarray, torch.Tensor],
        reward: float,
        next_state: Union[np.ndarray, torch.Tensor],
        done: bool,
        **kwargs
    ):
        """
        Process one environment step and trigger learning if appropriate.
        
        This method provides a standard interface for processing environment
        transitions. It updates counters, stores experiences (if applicable),
        and triggers learning when appropriate conditions are met.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is complete
            **kwargs: Algorithm-specific parameters
        """
        # Update step counter
        self.step_count += 1
        
        # Update episode counter if episode is done
        if done:
            self.episode_count += 1
        
        # Algorithm-specific step processing should be implemented in subclasses
        # This base implementation provides the interface and basic counting
        
    def save(self, filepath: str, **kwargs):
        """
        Save agent state to file.
        
        This method provides a standard interface for saving agent state,
        including networks, optimizers, counters, and configuration.
        
        Args:
            filepath: Path where to save the agent state
            **kwargs: Algorithm-specific save parameters
            
        Note:
            Subclasses should override this method to save algorithm-specific
            components while calling super().save() for common components.
        """
        # Common save data that all algorithms should include
        save_dict = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'algorithm_name': self.algorithm_name
        }
        
        # Add any additional save data from kwargs
        save_dict.update(kwargs)
        
        try:
            torch.save(save_dict, filepath)
            self.logger.info(f"Agent state saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save agent state to {filepath}: {e}")
            raise
    
    def load(self, filepath: str, **kwargs):
        """
        Load agent state from file.
        
        This method provides a standard interface for loading agent state,
        including networks, optimizers, counters, and configuration.
        
        Args:
            filepath: Path to the saved agent state
            **kwargs: Algorithm-specific load parameters
            
        Note:
            Subclasses should override this method to load algorithm-specific
            components while calling super().load() for common components.
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load common components
            self.step_count = checkpoint.get('step_count', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            
            # Optionally update config if saved config is different
            if 'config' in checkpoint:
                self.logger.info("Loaded configuration from checkpoint")
                # Note: Config validation/merging can be implemented here if needed
            
            self.logger.info(f"Agent state loaded from {filepath}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load agent state from {filepath}: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current training metrics.
        
        Returns:
            Dictionary containing all collected training metrics
        """
        return dict(self.training_metrics)
    
    def reset_metrics(self):
        """Reset all training metrics."""
        self.training_metrics.clear()
        self.logger.info("Training metrics reset")
    
    def log_metrics(self, metrics: Optional[Dict[str, Any]] = None):
        """
        Log training metrics.
        
        Args:
            metrics: Optional custom metrics to log. If None, logs current training metrics.
        """
        if metrics is None:
            metrics = self.get_metrics()
        
        for key, values in metrics.items():
            if values:  # Only log if there are values
                if isinstance(values, list) and len(values) > 0:
                    latest_value = values[-1]
                    avg_value = np.mean(values[-100:])  # Average of last 100 values
                    self.logger.info(f"{key}: latest={latest_value:.4f}, avg_100={avg_value:.4f}")
                else:
                    self.logger.info(f"{key}: {values}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get general information about the agent.
        
        Returns:
            Dictionary containing agent information
        """
        return {
            'algorithm_name': self.algorithm_name,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'device': str(self.device),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"{self.algorithm_name}(state_size={self.state_size}, "
                f"action_size={self.action_size}, device={self.device}, "
                f"steps={self.step_count}, episodes={self.episode_count})")


# Import concrete algorithm implementations
try:
    from .dqn import DQNAgent
    from .ddpg import DDPGAgent
    from .ppo import PPOAgent
    from .sac import SACAgent
    
    __all__ = ['BaseAgent', 'DQNAgent', 'DDPGAgent', 'PPOAgent', 'SACAgent']
    
except ImportError as e:
    # If concrete implementations are not available, just export the base class
    __all__ = ['BaseAgent']
    import warnings
    warnings.warn(f"Some algorithm implementations could not be imported: {e}")


def create_agent(
    algorithm: str,
    state_size: int,
    action_size: int,
    config_path: Optional[str] = None,
    device: str = "cpu",
    **kwargs
) -> BaseAgent:
    """
    Factory function to create agents by algorithm name.
    
    This utility function provides a convenient way to instantiate agents
    without directly importing specific classes.
    
    Args:
        algorithm: Name of the algorithm ('dqn', 'ddpg', 'ppo', 'sac')
        state_size: Dimension of the state space
        action_size: Size of the action space
        config_path: Path to configuration file (optional)
        device: Computing device
        **kwargs: Additional arguments passed to the agent constructor
        
    Returns:
        Instantiated agent of the specified algorithm
        
    Raises:
        ValueError: If algorithm name is not recognized
        
    Example:
        ```python
        agent = create_agent('dqn', state_size=4, action_size=2)
        ```
    """
    algorithm = algorithm.lower()
    
    # Set default config paths if not provided
    if config_path is None:
        config_path = f"configs/{algorithm}_config.yaml"
    
    try:
        if algorithm == 'dqn':
            return DQNAgent(state_size, action_size, config_path, device, **kwargs)
        elif algorithm == 'ddpg':
            return DDPGAgent(state_size, action_size, config_path=config_path, device=device, **kwargs)
        elif algorithm == 'ppo':
            return PPOAgent(state_size, action_size, config_path=config_path, device=device, **kwargs)
        elif algorithm == 'sac':
            return SACAgent(state_size, action_size, config_path=config_path, device=device, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported algorithms: dqn, ddpg, ppo, sac")
    
    except NameError:
        raise ImportError(f"Algorithm '{algorithm}' is not available. Check if the implementation is properly imported.")
