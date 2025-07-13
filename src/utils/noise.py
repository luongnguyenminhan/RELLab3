"""
Noise Utilities Module.

This module provides noise process implementations for exploration in DRL algorithms, as described in "Deep Reinforcement Learning: A Survey" by Wang et al. and the Modular DRL Framework engineering blueprint.

Detailed Description:
The noise module implements stochastic processes such as Ornstein-Uhlenbeck and Gaussian noise, which are used to encourage exploration in continuous action spaces. These utilities are essential for algorithms like DDPG, TD3, and SAC. The design is modular, allowing easy extension and configuration via YAML files while maintaining reproducibility through proper seeding.

Key Concepts/Algorithms:
- Ornstein-Uhlenbeck process for temporally correlated noise with mean reversion
- Gaussian (Normal) noise for uncorrelated exploration
- Noise decay and annealing schedules for adaptive exploration
- Configurable noise parameters for reproducibility
- Support for multi-dimensional action spaces

Important Parameters/Configurations:
- Noise type and distribution parameters (mean, std, theta, sigma)
- Decay rates and minimum values for noise annealing
- Random seeds for reproducible noise generation
- Action space dimensions and bounds
- All parameters configurable via YAML (e.g., `configs/ddpg_config.yaml`)

Expected Inputs/Outputs:
- Inputs: action dimension, current action (optional), time step
- Outputs: noise sample (numpy.ndarray or float)

Dependencies:
- numpy: Numerical operations and random number generation
- typing: Type hints for better code documentation

Author: REL Project Team
Date: 2025-07-13
"""

import numpy as np
from typing import Union, Optional, Tuple
import copy


class BaseNoise:
    """
    Abstract Base Class for Noise Processes.

    This class provides a common interface for all noise implementations used in
    deep reinforcement learning for exploration. It defines the basic structure
    and common functionality that all noise processes should implement.

    Args:
        size (Union[int, Tuple[int, ...]]): Dimension(s) of the noise
        seed (Optional[int]): Random seed for reproducibility

    Attributes:
        size: Noise dimensions
        seed: Random seed used for initialization
    """

    def __init__(self, size: Union[int, Tuple[int, ...]], seed: Optional[int] = None):
        """
        Initialize base noise process.

        Args:
            size: Dimension(s) of the noise
            seed: Random seed for reproducibility
        """
        self.size = size if isinstance(size, tuple) else (size,)
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)

    def sample(self) -> np.ndarray:
        """
        Sample noise from the process.

        Returns:
            Noise sample as numpy array
        """
        raise NotImplementedError("Subclasses must implement sample method")

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the noise process to initial state.

        Args:
            seed: Optional new random seed
        """
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)


class GaussianNoise(BaseNoise):
    """
    Gaussian (Normal) Noise for Exploration.

    This class implements uncorrelated Gaussian noise for exploration in continuous
    action spaces. The noise is independently sampled at each time step with
    configurable mean and standard deviation. Supports noise decay for adaptive
    exploration schedules.

    Detailed Description:
    Gaussian noise provides simple, memoryless exploration by adding random values
    sampled from a normal distribution to actions. This is useful for algorithms
    that require uncorrelated exploration noise. The class supports both scalar
    and multi-dimensional noise generation with optional decay scheduling.

    Args:
        mean (float): Mean of the Gaussian distribution
        std (float): Standard deviation of the Gaussian distribution
        size (Union[int, Tuple[int, ...]]): Dimension(s) of the noise
        decay_rate (float): Rate of standard deviation decay per step
        min_std (float): Minimum standard deviation (decay floor)
        seed (Optional[int]): Random seed for reproducibility

    Attributes:
        mean (float): Current mean value
        std (float): Current standard deviation
        initial_std (float): Initial standard deviation value
        decay_rate (float): Decay rate for std
        min_std (float): Minimum allowed std

    Example:
        ```python
        # Initialize Gaussian noise
        noise = GaussianNoise(
            mean=0.0,
            std=0.1,
            size=4,  # For 4-dimensional action space
            decay_rate=0.995,
            min_std=0.01,
            seed=42
        )

        # Sample noise
        noise_sample = noise.sample()  # Shape: (4,)

        # Apply decay
        noise.decay()

        # Reset with new parameters
        noise.reset()
        noise.set_std(0.2)
        ```

    Dependencies:
    - numpy: Random number generation and array operations

    Author: REL Project Team
    Date: 2025-07-13
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.1,
        size: Union[int, Tuple[int, ...]] = 1,
        decay_rate: float = 1.0,
        min_std: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize Gaussian noise process.

        Args:
            mean: Mean of the Gaussian distribution
            std: Standard deviation of the Gaussian distribution
            size: Dimension(s) of the noise
            decay_rate: Rate of standard deviation decay per step
            min_std: Minimum standard deviation (decay floor)
            seed: Random seed for reproducibility
        """
        super().__init__(size, seed)
        
        self.mean = mean
        self.std = std
        self.initial_std = std
        self.decay_rate = decay_rate
        self.min_std = min_std

    def sample(self) -> np.ndarray:
        """
        Sample Gaussian noise.

        Returns:
            Gaussian noise sample with shape matching self.size
        """
        noise = np.random.normal(self.mean, self.std, self.size)
        return noise.squeeze() if len(self.size) == 1 and self.size[0] == 1 else noise

    def decay(self) -> None:
        """Apply decay to the standard deviation."""
        self.std = max(self.std * self.decay_rate, self.min_std)

    def set_std(self, new_std: float) -> None:
        """
        Set new standard deviation.

        Args:
            new_std: New standard deviation value
        """
        self.std = max(new_std, self.min_std)

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset noise to initial state.

        Args:
            seed: Optional new random seed
        """
        super().reset(seed)
        self.std = self.initial_std

    def __repr__(self) -> str:
        """String representation of the noise process."""
        return f"GaussianNoise(mean={self.mean}, std={self.std}, size={self.size})"


class OrnsteinUhlenbeckNoise(BaseNoise):
    """
    Ornstein-Uhlenbeck Noise for Temporally Correlated Exploration.

    This class implements the Ornstein-Uhlenbeck process, which generates temporally
    correlated noise with mean reversion properties. This is particularly useful for
    continuous control tasks where smooth, correlated exploration is beneficial.

    Detailed Description:
    The Ornstein-Uhlenbeck process is a stochastic process that exhibits mean reversion,
    making it ideal for exploration in continuous action spaces. The process generates
    noise that is correlated over time, leading to more consistent exploration patterns
    compared to uncorrelated Gaussian noise. This is especially useful in robotics and
    continuous control domains.

    Mathematical formulation:
    dx_t = theta * (mu - x_t) * dt + sigma * dW_t

    Where:
    - theta: Mean reversion rate (how quickly the process returns to the mean)
    - mu: Long-term mean of the process
    - sigma: Volatility (noise intensity)
    - dW_t: Wiener process (random component)
    - dt: Time step

    Args:
        size (Union[int, Tuple[int, ...]]): Dimension(s) of the noise
        mu (float): Long-term mean of the process
        theta (float): Mean reversion rate
        sigma (float): Volatility/noise intensity
        dt (float): Time step size
        seed (Optional[int]): Random seed for reproducibility

    Attributes:
        mu (float): Long-term mean
        theta (float): Mean reversion rate
        sigma (float): Volatility
        dt (float): Time step
        state (np.ndarray): Current state of the process

    Example:
        ```python
        # Initialize OU noise for continuous control
        ou_noise = OrnsteinUhlenbeckNoise(
            size=2,        # 2D action space
            mu=0.0,        # Zero mean
            theta=0.15,    # Moderate mean reversion
            sigma=0.2,     # Moderate volatility
            dt=1e-2,       # Small time step
            seed=42
        )

        # Generate noise sequence
        for step in range(1000):
            noise = ou_noise.sample()
            # Use noise for exploration...

        # Reset process
        ou_noise.reset()
        ```

    Dependencies:
    - numpy: Numerical operations and random generation

    Author: REL Project Team
    Date: 2025-07-13
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, ...]] = 1,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        seed: Optional[int] = None
    ):
        """
        Initialize Ornstein-Uhlenbeck noise process.

        Args:
            size: Dimension(s) of the noise
            mu: Long-term mean of the process
            theta: Mean reversion rate
            sigma: Volatility/noise intensity
            dt: Time step size
            seed: Random seed for reproducibility
        """
        super().__init__(size, seed)
        
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        
        # Initialize state
        self.state = np.ones(self.size) * self.mu

    def sample(self) -> np.ndarray:
        """
        Sample from Ornstein-Uhlenbeck process.

        Returns:
            OU noise sample with temporal correlation
        """
        # OU process update: dx = theta * (mu - x) * dt + sigma * sqrt(dt) * dW
        dx = (
            self.theta * (self.mu - self.state) * self.dt +
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        
        self.state += dx
        
        return self.state.squeeze() if len(self.size) == 1 and self.size[0] == 1 else self.state.copy()

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the OU process to initial state.

        Args:
            seed: Optional new random seed
        """
        super().reset(seed)
        self.state = np.ones(self.size) * self.mu

    def __repr__(self) -> str:
        """String representation of the noise process."""
        return (f"OrnsteinUhlenbeckNoise(size={self.size}, mu={self.mu}, "
                f"theta={self.theta}, sigma={self.sigma}, dt={self.dt})")


class AdaptiveNoise:
    """
    Adaptive Noise with Scheduling Capabilities.

    This class provides a wrapper around base noise processes with additional
    functionality for adaptive exploration strategies. It supports various
    scheduling methods for dynamically adjusting noise parameters during training.

    Args:
        base_noise (BaseNoise): Base noise process to wrap
        schedule_type (str): Type of scheduling ('linear', 'exponential', 'step')
        schedule_params (dict): Parameters for the scheduling function

    Attributes:
        base_noise: The underlying noise process
        schedule_type: Type of parameter scheduling
        schedule_params: Scheduling parameters
        step_count: Current step counter

    Example:
        ```python
        # Create adaptive Gaussian noise with linear decay
        base_noise = GaussianNoise(mean=0.0, std=0.3, size=4)
        adaptive_noise = AdaptiveNoise(
            base_noise=base_noise,
            schedule_type='linear',
            schedule_params={
                'initial_value': 0.3,
                'final_value': 0.05,
                'total_steps': 100000
            }
        )

        # Use in training loop
        for step in range(100000):
            noise = adaptive_noise.sample()
            adaptive_noise.update_schedule(step)
        ```
    """

    def __init__(
        self,
        base_noise: BaseNoise,
        schedule_type: str = 'constant',
        schedule_params: Optional[dict] = None
    ):
        """
        Initialize adaptive noise wrapper.

        Args:
            base_noise: Base noise process to wrap
            schedule_type: Type of scheduling
            schedule_params: Parameters for scheduling
        """
        self.base_noise = base_noise
        self.schedule_type = schedule_type
        self.schedule_params = schedule_params or {}
        self.step_count = 0

    def sample(self) -> np.ndarray:
        """Sample from the base noise process."""
        return self.base_noise.sample()

    def update_schedule(self, step: int) -> None:
        """
        Update noise parameters based on schedule.

        Args:
            step: Current training step
        """
        self.step_count = step
        
        if self.schedule_type == 'constant':
            return
        
        elif self.schedule_type == 'linear':
            initial_val = self.schedule_params.get('initial_value', 1.0)
            final_val = self.schedule_params.get('final_value', 0.1)
            total_steps = self.schedule_params.get('total_steps', 10000)
            
            progress = min(step / total_steps, 1.0)
            current_val = initial_val + (final_val - initial_val) * progress
            
            if hasattr(self.base_noise, 'set_std'):
                self.base_noise.set_std(current_val)
            elif hasattr(self.base_noise, 'sigma'):
                self.base_noise.sigma = current_val
        
        elif self.schedule_type == 'exponential':
            initial_val = self.schedule_params.get('initial_value', 1.0)
            decay_rate = self.schedule_params.get('decay_rate', 0.995)
            min_val = self.schedule_params.get('min_value', 0.01)
            
            current_val = max(initial_val * (decay_rate ** step), min_val)
            
            if hasattr(self.base_noise, 'set_std'):
                self.base_noise.set_std(current_val)
            elif hasattr(self.base_noise, 'sigma'):
                self.base_noise.sigma = current_val
        
        elif self.schedule_type == 'step':
            initial_val = self.schedule_params.get('initial_value', 1.0)
            step_size = self.schedule_params.get('step_size', 1000)
            decay_factor = self.schedule_params.get('decay_factor', 0.5)
            
            current_val = initial_val * (decay_factor ** (step // step_size))
            
            if hasattr(self.base_noise, 'set_std'):
                self.base_noise.set_std(current_val)
            elif hasattr(self.base_noise, 'sigma'):
                self.base_noise.sigma = current_val

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the noise process."""
        self.base_noise.reset(seed)
        self.step_count = 0

    def __repr__(self) -> str:
        """String representation."""
        return f"AdaptiveNoise({self.base_noise}, schedule={self.schedule_type})"


# Convenience function for creating noise from configuration
def create_noise_from_config(config: dict, action_dim: int, seed: Optional[int] = None) -> BaseNoise:
    """
    Create noise process from configuration dictionary.

    Args:
        config: Configuration dictionary with noise parameters
        action_dim: Dimension of the action space
        seed: Random seed for reproducibility

    Returns:
        Configured noise process

    Example:
        ```python
        config = {
            'type': 'ou',
            'theta': 0.15,
            'sigma': 0.2,
            'mu': 0.0,
            'dt': 1e-2
        }
        noise = create_noise_from_config(config, action_dim=4, seed=42)
        ```
    """
    noise_type = config.get('type', 'gaussian').lower()
    
    if noise_type == 'gaussian' or noise_type == 'normal':
        return GaussianNoise(
            mean=config.get('mean', 0.0),
            std=config.get('std', 0.1),
            size=action_dim,
            decay_rate=config.get('decay_rate', 1.0),
            min_std=config.get('min_std', 0.01),
            seed=seed
        )
    
    elif noise_type == 'ou' or noise_type == 'ornstein_uhlenbeck':
        return OrnsteinUhlenbeckNoise(
            size=action_dim,
            mu=config.get('mu', 0.0),
            theta=config.get('theta', 0.15),
            sigma=config.get('sigma', 0.2),
            dt=config.get('dt', 1e-2),
            seed=seed
        )
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


# Export main classes and functions
__all__ = [
    'BaseNoise',
    'GaussianNoise', 
    'OrnsteinUhlenbeckNoise',
    'AdaptiveNoise',
    'create_noise_from_config'
]