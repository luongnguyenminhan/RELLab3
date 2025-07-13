"""
Buffers Module.

This package provides experience replay buffer implementations for DRL algorithms, as described in the Modular DRL Framework and "Deep Reinforcement Learning: A Survey" by Wang et al.

Detailed Description:
Experience replay buffers are fundamental components of off-policy DRL algorithms, enabling temporal correlation breaking and data efficiency improvements. This module implements a modular buffer hierarchy with a base abstract class and specialized implementations for uniform and prioritized sampling strategies. All buffers support reproducibility through seeding and are configurable via YAML files.

Key Concepts/Algorithms:
- Experience Replay (uniform sampling)
- Prioritized Experience Replay (importance-based sampling)
- Abstract base class for extensibility
- Efficient storage and sampling mechanisms

Important Parameters/Configurations:
- Buffer size (maximum number of transitions)
- Batch size (number of samples per mini-batch)
- Device specification (CPU/GPU)
- Prioritization parameters (alpha, beta for prioritized replay)
- All parameters loaded from YAML configs (e.g., `configs/dqn_config.yaml`)

Expected Inputs/Outputs:
- Inputs: transitions (state, action, reward, next_state, done), priorities (for prioritized)
- Outputs: sampled mini-batches, importance weights (for prioritized)

Dependencies:
- numpy, torch, collections, abc, typing
- Compatible with PyTorch tensor operations

Author: REL Project Team
Date: 2025-07-13
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
import torch

# Type aliases for better code readability
ObsType = Union[np.ndarray, torch.Tensor]
ActionType = Union[np.ndarray, torch.Tensor, int, float]
RewardType = Union[float, np.ndarray, torch.Tensor]
DoneType = Union[bool, np.ndarray, torch.Tensor]
BufferSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
PrioritizedBufferSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]


class BaseBuffer(ABC):
    """
    Abstract base class for experience replay buffers.
    
    This class defines the common interface and shared functionality for all buffer implementations
    in the Modular DRL Framework. It provides the foundation for uniform and prioritized replay
    buffers, ensuring consistent behavior and easy extensibility.
    
    Args:
        buffer_size (int): Maximum number of transitions to store
        batch_size (int): Number of transitions to sample per batch
        device (str): Device for tensor operations ('cpu' or 'cuda')
        seed (Optional[int]): Random seed for reproducible sampling
        
    Attributes:
        buffer_size (int): Maximum buffer capacity
        batch_size (int): Sampling batch size
        device (torch.device): PyTorch device for tensor operations
        pos (int): Current position in the buffer (circular)
        full (bool): Whether the buffer has been filled once
    """
    
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.pos = 0
        self.full = False
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize storage (to be defined by concrete classes)
        self._init_storage()
    
    @abstractmethod
    def _init_storage(self) -> None:
        """Initialize storage arrays. Must be implemented by concrete classes."""
        pass
    
    @abstractmethod
    def add(
        self,
        obs: ObsType,
        action: ActionType,
        reward: RewardType,
        next_obs: ObsType,
        done: DoneType,
        **kwargs,
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Episode termination flag
            **kwargs: Additional arguments for specialized buffers
        """
        pass
    
    @abstractmethod
    def sample(self, **kwargs) -> Union[BufferSample, PrioritizedBufferSample]:
        """
        Sample a batch of transitions.
        
        Returns:
            Sampled batch of transitions
        """
        pass    
    def size(self) -> int:
        """
        Get the current number of transitions in the buffer.
        
        Returns:
            Number of stored transitions
        """
        return self.buffer_size if self.full else self.pos
    
    def can_sample(self) -> bool:
        """
        Check if the buffer has enough transitions to sample a batch.
        
        Returns:
            True if sampling is possible, False otherwise
        """
        return self.size() >= self.batch_size
    
    def clear(self) -> None:
        """Clear the buffer and reset position."""
        self.pos = 0
        self.full = False
        self._init_storage()
    
    def _to_tensor(self, array: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.
        
        Args:
            array: Input array or tensor
            
        Returns:
            PyTorch tensor on the specified device
        """
        if isinstance(array, torch.Tensor):
            return array.to(self.device)
        return torch.as_tensor(array, device=self.device)
    
    def _get_samples(self, batch_indices: np.ndarray) -> BufferSample:
        """
        Get transition samples for given indices.
        
        Args:
            batch_indices: Indices of transitions to sample
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        # To be implemented by concrete classes based on their storage format
        raise NotImplementedError("Concrete classes must implement _get_samples")


# Import concrete buffer implementations
try:
    from .replay_buffer import ReplayBuffer
    from .prioritized_replay_buffer import PrioritizedReplayBuffer
    
    __all__ = ["BaseBuffer", "ReplayBuffer", "PrioritizedReplayBuffer"]
    
except ImportError as e:
    # Fallback if concrete implementations are not yet available
    __all__ = ["BaseBuffer"]
    print(f"Warning: Some buffer implementations not available: {e}")


def create_buffer(
    buffer_type: str,
    buffer_size: int,
    batch_size: int,
    device: str = "cpu",
    seed: Optional[int] = None,
    **kwargs,
) -> BaseBuffer:
    """
    Factory function to create buffer instances.
    
    Args:
        buffer_type: Type of buffer ('replay' or 'prioritized')
        buffer_size: Maximum number of transitions to store
        batch_size: Number of transitions to sample per batch
        device: Device for tensor operations
        seed: Random seed for reproducibility
        **kwargs: Additional arguments for specific buffer types
        
    Returns:
        Buffer instance
        
    Raises:
        ValueError: If buffer_type is not recognized
    """
    if buffer_type.lower() == "replay":
        from .replay_buffer import ReplayBuffer
        return ReplayBuffer(buffer_size, batch_size, device, seed, **kwargs)
    elif buffer_type.lower() == "prioritized":
        from .prioritized_replay_buffer import PrioritizedReplayBuffer
        return PrioritizedReplayBuffer(buffer_size, batch_size, device, seed, **kwargs)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")