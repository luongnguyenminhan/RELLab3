"""
Replay Buffer Implementation.

This module implements the standard uniform experience replay buffer for off-policy DRL algorithms, as described in the Modular DRL Framework and "Deep Reinforcement Learning: A Survey" by Wang et al.

Detailed Description:
The replay buffer stores agent-environment transitions (state, action, reward, next_state, done) and enables random sampling for training, breaking temporal correlations and improving data efficiency. It implements a circular buffer with uniform random sampling and supports both numpy arrays and PyTorch tensors. The buffer is designed for reproducibility and is configurable via YAML files. Used by DQN, DDPG, SAC, and other off-policy algorithms.

Key Concepts/Algorithms:
- Uniform experience replay
- Circular buffer implementation
- Random sampling for mini-batch updates
- Efficient memory management
- Support for various data types

Important Parameters/Configurations:
- Buffer size (maximum capacity)
- Batch size (sampling size)
- Device specification (CPU/GPU)
- All parameters loaded from relevant YAML configs (e.g., `configs/dqn_config.yaml`)

Expected Inputs/Outputs:
- Inputs: transitions (state, action, reward, next_state, done)
- Outputs: sampled mini-batches as PyTorch tensors

Dependencies:
- numpy, torch, collections
- Inherits from BaseBuffer in __init__.py

Author: REL Project Team
Date: 2025-07-13
"""
import numpy as np
import torch
from typing import Optional, Union

from . import BaseBuffer, BufferSample, ObsType, ActionType, RewardType, DoneType


class ReplayBuffer(BaseBuffer):
    """
    Standard uniform experience replay buffer.
    
    This buffer stores transitions in a circular fashion and samples uniformly at random.
    It's the foundation for off-policy learning in algorithms like DQN, DDPG, and SAC.
    
    Args:
        buffer_size (int): Maximum number of transitions to store
        batch_size (int): Number of transitions to sample per batch
        device (str): Device for tensor operations ('cpu' or 'cuda')
        seed (Optional[int]): Random seed for reproducible sampling
        
    Example:
        >>> buffer = ReplayBuffer(buffer_size=10000, batch_size=32, device='cpu')
        >>> buffer.add(obs, action, reward, next_obs, done)
        >>> if buffer.can_sample():
        ...     batch = buffer.sample()
    """
    
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__(buffer_size, batch_size, device, seed)
    
    def _init_storage(self) -> None:
        """Initialize storage arrays for transitions."""
        # Storage will be initialized when first transition is added
        # This allows for dynamic sizing based on observation/action shapes
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
        self._initialized = False
    
    def _init_storage_arrays(
        self,
        obs: ObsType,
        action: ActionType,
        reward: RewardType,
        next_obs: ObsType,
        done: DoneType,
    ) -> None:
        """
        Initialize storage arrays based on the first transition.
        
        Args:
            obs: First observation to determine shape and dtype
            action: First action to determine shape and dtype
            reward: First reward to determine dtype
            next_obs: First next observation
            done: First done flag
        """
        # Convert to numpy arrays for shape and dtype inference
        obs_array = np.asarray(obs)
        action_array = np.asarray(action)
        reward_array = np.asarray(reward)
        next_obs_array = np.asarray(next_obs)
        done_array = np.asarray(done)
        
        # Initialize storage arrays
        self.observations = np.zeros((self.buffer_size,) + obs_array.shape, dtype=obs_array.dtype)
        self.actions = np.zeros((self.buffer_size,) + action_array.shape, dtype=action_array.dtype)
        self.rewards = np.zeros((self.buffer_size,) + reward_array.shape, dtype=reward_array.dtype)
        self.next_observations = np.zeros((self.buffer_size,) + next_obs_array.shape, dtype=next_obs_array.dtype)
        self.dones = np.zeros((self.buffer_size,) + done_array.shape, dtype=done_array.dtype)
        
        self._initialized = True    
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
            **kwargs: Additional arguments (ignored for standard replay buffer)
        """
        # Initialize storage on first transition
        if not self._initialized:
            self._init_storage_arrays(obs, action, reward, next_obs, done)
        
        # Store transition
        self.observations[self.pos] = np.asarray(obs)
        self.actions[self.pos] = np.asarray(action)
        self.rewards[self.pos] = np.asarray(reward)
        self.next_observations[self.pos] = np.asarray(next_obs)
        self.dones[self.pos] = np.asarray(done)
        
        # Update position and full flag
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
    
    def sample(self, **kwargs) -> BufferSample:
        """
        Sample a batch of transitions uniformly at random.
        
        Args:
            **kwargs: Additional arguments (ignored for standard replay buffer)
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
            
        Raises:
            AssertionError: If buffer doesn't have enough transitions
        """
        assert self.can_sample(), f"Cannot sample {self.batch_size} transitions from buffer with {self.size()} transitions"
        
        # Sample random indices
        indices = np.random.randint(0, self.size(), size=self.batch_size)
        
        return self._get_samples(indices)
    
    def _get_samples(self, batch_indices: np.ndarray) -> BufferSample:
        """
        Get transition samples for given indices.
        
        Args:
            batch_indices: Indices of transitions to sample
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones) as tensors
        """
        # Extract samples and convert to tensors
        obs_batch = self._to_tensor(self.observations[batch_indices])
        action_batch = self._to_tensor(self.actions[batch_indices])
        reward_batch = self._to_tensor(self.rewards[batch_indices])
        next_obs_batch = self._to_tensor(self.next_observations[batch_indices])
        done_batch = self._to_tensor(self.dones[batch_indices])
        
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
    
    def get_transition(self, index: int) -> BufferSample:
        """
        Get a single transition by index.
        
        Args:
            index: Index of the transition
            
        Returns:
            Single transition as tensors
        """
        assert 0 <= index < self.size(), f"Index {index} out of range [0, {self.size()})"
        return self._get_samples(np.array([index]))
    
    def extend(self, other_buffer: 'ReplayBuffer') -> None:
        """
        Extend this buffer with transitions from another buffer.
        
        Args:
            other_buffer: Another ReplayBuffer to copy transitions from
        """
        if other_buffer.size() == 0:
            return
            
        for i in range(other_buffer.size()):
            idx = i if not other_buffer.full else (other_buffer.pos + i) % other_buffer.buffer_size
            self.add(
                other_buffer.observations[idx],
                other_buffer.actions[idx],
                other_buffer.rewards[idx],
                other_buffer.next_observations[idx],
                other_buffer.dones[idx],
            )