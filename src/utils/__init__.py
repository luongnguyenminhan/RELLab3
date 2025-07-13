"""
Utils Module.

This package provides utility functions and helper classes for the Modular DRL Framework, supporting logging, noise generation, hyperparameter management, and base agent functionality as described in the survey by Wang et al.

Key Components:
- BaseAgent: Abstract base class for all utility agents
- Logger: Comprehensive experiment logging and metrics tracking
- HyperparameterManager: YAML-based configuration management and validation
- Noise processes: Gaussian and Ornstein-Uhlenbeck noise for exploration
- Agent factory for creating utility agents

Expected Usage:
- Utility agents inherit from BaseAgent for consistent interfaces
- Logger provides experiment tracking and reproducibility
- HyperparameterManager handles all configuration management
- Noise generators provide exploration for continuous control
- Configuration-driven design with YAML support

Dependencies:
- numpy, PyYAML, logging, abc, typing

Author: REL Project Team
Date: 2025-07-13
"""

# Import main components with graceful fallback
__all__ = []

# BaseAgent and factory
try:
    from .base_agent import BaseAgent, create_agent
    __all__.extend(['BaseAgent', 'create_agent'])
except ImportError:
    pass

# Logger utilities
try:
    from .logger import Logger
    __all__.extend(['Logger'])
except ImportError:
    pass

# Hyperparameter management
try:
    from .hyperparameters import HyperparameterManager
    __all__.extend(['HyperparameterManager'])
except ImportError:
    pass

# Noise generators
try:
    from .noise import (
        BaseNoise,
        GaussianNoise, 
        OrnsteinUhlenbeckNoise,
        AdaptiveNoise,
        create_noise_from_config
    )
    __all__.extend([
        'BaseNoise',
        'GaussianNoise', 
        'OrnsteinUhlenbeckNoise',
        'AdaptiveNoise',
        'create_noise_from_config'
    ])
except ImportError:
    pass

# Remove duplicates and sort
__all__ = sorted(list(set(__all__)))