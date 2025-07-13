"""
Networks Module.

This package defines neural network architectures used by DRL algorithms in the Modular DRL Framework, as described in the survey by Wang et al. The module provides a unified BaseNetwork abstract base class that enforces consistent interfaces across all network types while enabling algorithm-specific implementations.

Detailed Description:
The networks module implements the foundation for all neural network architectures used in the framework, including Q-Networks (value-based methods), Policy Networks (actor-critic and policy-based methods), Value Networks (critic estimation), and Dueling Networks (advanced Q-value estimation). Each network type inherits from the BaseNetwork class, ensuring modularity, consistency, and extensibility.

Key Concepts/Design Patterns:
- Abstract Base Class (ABC) pattern for consistent interfaces
- Template Method pattern for common initialization flows
- Factory Method pattern for network instantiation
- Configuration-driven architecture definition via YAML
- PyTorch nn.Module integration

Key Components:
- BaseNetwork: Abstract base class for all neural network architectures
- Q-Networks: For value-based methods (DQN and variants)
- Policy Networks: For actor-critic and policy-based methods (PPO, DDPG, SAC)
- Value Networks: For critic estimation in actor-critic methods
- Dueling Networks: For advanced Q-value decomposition

Network Features:
- YAML-based configuration management
- Automatic device management (CPU/GPU)
- Parameter initialization schemes
- Model saving and loading
- Architecture validation and debugging

Expected Usage:
- Networks are instantiated by algorithm modules for function approximation
- All architecture parameters are configured via YAML files in `configs/`
- Networks support both training and evaluation modes
- Model checkpointing and restoration for reproducibility

Dependencies:
- torch: Neural network computations and automatic differentiation
- numpy: Numerical operations and array manipulation
- yaml: Configuration file parsing
- abc: Abstract base class functionality
- typing: Type hints and annotations

Author: REL Project Team
Date: 2025-07-13
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import OrderedDict

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from utils.logger import Logger
except ImportError:
    # Fallback logger if utils.logger is not available
    import logging
    
    class Logger:
        def __init__(self, name: str = "BaseNetwork"):
            self.logger = logging.getLogger(name)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        def info(self, msg: str):
            self.logger.info(msg)

        def warning(self, msg: str):
            self.logger.warning(msg)

        def error(self, msg: str):
            self.logger.error(msg)


class BaseNetwork(nn.Module, ABC):
    """
    Abstract Base Class for Deep Reinforcement Learning Neural Networks.

    This class provides a unified interface and common functionality for all neural network
    architectures in the Modular DRL Framework. It encapsulates shared patterns including
    configuration management, device handling, parameter initialization, and model persistence
    while allowing network-specific implementations through abstract methods.

    Detailed Description:
    The BaseNetwork follows the Abstract Base Class pattern to ensure consistent interfaces
    across all network types while providing concrete implementations for common functionality.
    It supports YAML-based configuration management, automatic device detection and management,
    various parameter initialization schemes, and standardized model saving/loading. Each
    network type inherits from this base class and implements network-specific methods for
    architecture definition and forward computation.

    Key Concepts/Design Patterns:
    - Abstract Base Class (ABC) pattern for interface consistency
    - Template Method pattern for initialization and configuration flow
    - Strategy pattern for different initialization schemes
    - YAML-based configuration management for reproducibility
    - Comprehensive logging and debugging support

    Common Functionality Provided:
    - Configuration loading and validation from YAML files
    - Device management (CPU/GPU) with automatic detection
    - Parameter initialization with multiple schemes (Xavier, He, orthogonal)
    - Model persistence (save/load) with full state preservation
    - Parameter counting and architecture debugging
    - Layer creation utilities for common architectures

    Abstract Methods (must be implemented by subclasses):
    - _build_architecture(): Define the specific network architecture
    - _get_default_config(): Provide default configuration parameters
    - forward(): Implement the forward pass computation

    Args:
        input_size (int): Dimension of the input features
        output_size (int): Dimension of the output features
        config_path (Optional[str]): Path to YAML configuration file
        device (str): Computing device ('cpu', 'cuda', 'cuda:0', etc.)
        network_name (str): Name of the network for logging purposes
        seed (Optional[int]): Random seed for reproducible initialization

    Attributes:
        input_size (int): Input feature dimension
        output_size (int): Output feature dimension
        device (torch.device): PyTorch device for computations
        config (Dict[str, Any]): Network configuration parameters
        network_name (str): Network identifier for logging
        logger (Logger): Logger instance for this network
        _parameter_count (int): Total number of trainable parameters

    Expected Usage:
    ```python
    # Inherit from BaseNetwork to create specific architectures
    class QNetwork(BaseNetwork):
        def _build_architecture(self):
            # Define Q-network specific layers
            pass

        def _get_default_config(self):
            # Return default configuration
            return {...}

        def forward(self, x):
            # Implement forward pass
            pass

    # Use the network
    q_net = QNetwork(input_size=4, output_size=2, config_path="configs/dqn_config.yaml")
    output = q_net(input_tensor)
    ```

    Dependencies:
    - torch: Neural network computations and device management
    - yaml: Configuration file parsing
    - abc: Abstract base class functionality
    - typing: Type hints and annotations

    Author: REL Project Team
    Date: 2025-07-13
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config_path: Optional[str] = None,
        device: str = "auto",
        network_name: str = "BaseNetwork",
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the base network with common parameters and infrastructure.

        Args:
            input_size: Dimension of the input features
            output_size: Dimension of the output features
            config_path: Path to YAML configuration file
            device: Computing device for PyTorch operations
            network_name: Name of the network for logging
            seed: Random seed for reproducible initialization
            **kwargs: Additional keyword arguments
        """
        super(BaseNetwork, self).__init__()
        
        # Store basic parameters
        self.input_size = input_size
        self.output_size = output_size
        self.network_name = network_name
        self.logger = Logger(network_name)
        
        # Set random seed for reproducibility
        if seed is not None:
            self._set_seed(seed)
        
        # Load and merge configuration
        self.config = self._load_and_merge_config(config_path)
        
        # Setup device
        self.device = self._setup_device(device)
        
        # Build the network architecture
        self._build_architecture()
        
        # Move to device
        self.to(self.device)
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Count parameters
        self._parameter_count = self._count_parameters()
        
        self.logger.info(f"Initialized {network_name} with {self._parameter_count} parameters on {self.device}")

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None or not os.path.exists(config_path):
            return {}
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return {}

    def _load_and_merge_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load and merge configuration with defaults."""
        # Start with default configuration
        config = self._get_default_config()
        
        # Load and merge YAML configuration
        if config_path:
            yaml_config = self._load_config(config_path)
            # Extract network-specific config if it exists
            network_config = yaml_config.get('network', {})
            config.update(network_config)
            
        return config

    def _setup_device(self, device: str) -> torch.device:
        """Setup and validate the computing device."""
        if device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = device
            
        device_obj = torch.device(device_str)
        
        # Validate device availability
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            device_obj = torch.device("cpu")
            
        return device_obj

    def _count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _initialize_parameters(self) -> None:
        """Initialize network parameters using the specified scheme."""
        init_scheme = self.config.get('initialization', 'xavier_uniform')
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_scheme == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif init_scheme == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif init_scheme == 'he_uniform':
                    nn.init.kaiming_uniform_(module.weight)
                elif init_scheme == 'he_normal':
                    nn.init.kaiming_normal_(module.weight)
                elif init_scheme == 'orthogonal':
                    nn.init.orthogonal_(module.weight)
                else:
                    self.logger.warning(f"Unknown initialization scheme: {init_scheme}")
                
                # Initialize bias to zeros
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.Conv2d):
                if init_scheme in ['he_uniform', 'he_normal']:
                    if init_scheme == 'he_uniform':
                        nn.init.kaiming_uniform_(module.weight)
                    else:
                        nn.init.kaiming_normal_(module.weight)
                else:
                    nn.init.xavier_uniform_(module.weight)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def create_mlp(
        self, 
        input_size: int, 
        output_size: int, 
        hidden_sizes: List[int],
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False,
        output_activation: Optional[str] = None
    ) -> nn.Sequential:
        """
        Create a Multi-Layer Perceptron (MLP) with specified architecture.

        Args:
            input_size: Size of input features
            output_size: Size of output features
            hidden_sizes: List of hidden layer sizes
            activation: Activation function name
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            output_activation: Activation for output layer

        Returns:
            nn.Sequential: The constructed MLP
        """
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        
        activation_fn = self._get_activation_function(activation)
        
        for i in range(len(sizes) - 1):
            # Linear layer
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            
            # Skip activation and other layers for the output layer unless specified
            if i < len(sizes) - 2:
                # Batch normalization
                if batch_norm:
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                
                # Activation
                layers.append(activation_fn)
                
                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            elif output_activation:
                # Output activation if specified
                output_fn = self._get_activation_function(output_activation)
                layers.append(output_fn)
        
        return nn.Sequential(*layers)

    @staticmethod
    def _get_activation_function(name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'softplus': nn.Softplus(),
            'identity': nn.Identity()
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation function: {name}")
        
        return activations[name.lower()]

    def save_model(self, save_path: str) -> None:
        """
        Save the model state to disk.

        Args:
            save_path: Path to save the model
        """
        save_dict = {
            'state_dict': self.state_dict(),
            'config': self.config,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'network_name': self.network_name,
            'parameter_count': self._parameter_count
        }
        
        torch.save(save_dict, save_path)
        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str) -> None:
        """
        Load the model state from disk.

        Args:
            load_path: Path to load the model from
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.load_state_dict(checkpoint['state_dict'])
        self.config = checkpoint['config']
        
        self.logger.info(f"Model loaded from {load_path}")

    def get_architecture_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the network architecture.

        Returns:
            Dict containing architecture information
        """
        info = {
            'network_name': self.network_name,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'parameter_count': self._parameter_count,
            'device': str(self.device),
            'config': self.config,
            'modules': []
        }
        
        for name, module in self.named_modules():
            if name:  # Skip the root module
                module_info = {
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                }
                info['modules'].append(module_info)
        
        return info

    def reset_parameters(self) -> None:
        """Reset all network parameters to their initial values."""
        self._initialize_parameters()
        self.logger.info("Network parameters reset")

    @abstractmethod
    def _build_architecture(self) -> None:
        """
        Build the specific network architecture.
        
        This method must be implemented by subclasses to define the layers
        and structure of the neural network.
        """
        pass

    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for this network type.
        
        Returns:
            Dict containing default configuration parameters
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass


# Factory function for creating networks
def create_network(
    network_type: str,
    input_size: int,
    output_size: int,
    config_path: Optional[str] = None,
    device: str = "auto",
    **kwargs
) -> BaseNetwork:
    """
    Factory function to create neural networks based on type.

    Args:
        network_type: Type of network to create ('q_network', 'policy_network', etc.)
        input_size: Input feature dimension
        output_size: Output feature dimension
        config_path: Path to configuration file
        device: Computing device
        **kwargs: Additional arguments

    Returns:
        BaseNetwork: The created network instance

    Raises:
        ValueError: If network_type is not recognized
    """
    network_registry = {
        'q_network': lambda: __import__('q_network', fromlist=['QNetwork']).QNetwork,
        'policy_network': lambda: __import__('policy_network', fromlist=['PolicyNetwork']).PolicyNetwork,
        'value_network': lambda: __import__('value_network', fromlist=['ValueNetwork']).ValueNetwork,
        'dueling_network': lambda: __import__('dueling_network', fromlist=['DuelingNetwork']).DuelingNetwork,
    }
    
    if network_type not in network_registry:
        available_types = ', '.join(network_registry.keys())
        raise ValueError(f"Unknown network type: {network_type}. Available types: {available_types}")
    
    try:
        network_class = network_registry[network_type]()
        return network_class(
            input_size=input_size,
            output_size=output_size,
            config_path=config_path,
            device=device,
            **kwargs
        )
    except ImportError as e:
        raise ImportError(f"Could not import {network_type}: {e}")


# Export main classes and functions
__all__ = [
    'BaseNetwork',
    'create_network',
    'Logger'
]