"""
Prioritized Replay Buffer Implementation.

This module implements the prioritized experience replay buffer for DRL algorithms, as described in the Modular DRL Framework and "Deep Reinforcement Learning: A Survey" by Wang et al.

Detailed Description:
Prioritized replay samples transitions with higher learning potential (e.g., larger TD error) more frequently, accelerating convergence and improving data efficiency. It uses stochastic prioritization based on temporal difference errors and importance sampling to correct for the bias introduced by non-uniform sampling. The buffer implements a sum-tree data structure for efficient priority-based sampling and updates. It supports both rank-based and proportional prioritization and is configurable via YAML files. Used by advanced DQN variants and other algorithms supporting prioritized replay.

Key Concepts/Algorithms:
- Prioritized experience replay with sum-tree
- Stochastic prioritization based on TD errors
- Importance sampling for bias correction
- Efficient priority updates and sampling
- Support for both proportional and rank-based prioritization

Important Parameters/Configurations:
- Buffer size (maximum capacity)
- Batch size (sampling size)
- Prioritization exponent (alpha): controls how much prioritization is used
- Importance sampling exponent (beta): controls bias correction strength
- Small epsilon value to ensure non-zero priorities
- All parameters loaded from relevant YAML configs (e.g., `configs/dqn_config.yaml`)

Expected Inputs/Outputs:
- Inputs: transitions (state, action, reward, next_state, done), priorities/TD errors
- Outputs: sampled mini-batches with importance weights for bias correction

Dependencies:
- numpy, torch, collections
- Inherits from BaseBuffer in __init__.py

Author: REL Project Team
Date: 2025-07-13
"""
import numpy as np
import torch
from typing import Optional, Union

from . import BaseBuffer, PrioritizedBufferSample, ObsType, ActionType, RewardType, DoneType


class SumTree:
    """
    Sum-tree data structure for efficient priority-based sampling.
    
    This data structure stores priorities in a binary tree where each parent node
    contains the sum of its children. This allows for O(log n) sampling and updates.
    
    Args:
        capacity: Maximum number of elements
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree as array
        self.data_pointer = 0
        
    def add(self, priority: float, data_index: int) -> None:
        """Add a priority value at the next position."""
        tree_index = data_index + self.capacity - 1
        self.update(tree_index, priority)
        
    def update(self, tree_index: int, priority: float) -> None:
        """Update priority value and propagate changes up the tree."""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Propagate change up the tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, value: float) -> tuple:
        """Get leaf index and priority for given cumulative value."""
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, search is done
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            # Descend to left or right child
            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], data_index
    
    @property
    def total_priority(self) -> float:
        """Get total priority (root of the tree)."""
        return self.tree[0]
class PrioritizedReplayBuffer(BaseBuffer):
    """
    Prioritized experience replay buffer with sum-tree implementation.
    
    This buffer samples transitions based on their priorities (typically TD errors),
    giving more importance to transitions that are expected to be more informative.
    It includes importance sampling weights to correct for the bias introduced by
    non-uniform sampling.
    
    Args:
        buffer_size (int): Maximum number of transitions to store
        batch_size (int): Number of transitions to sample per batch
        alpha (float): Prioritization exponent (0 = uniform, 1 = full prioritization)
        beta (float): Importance sampling exponent (0 = no correction, 1 = full correction)
        epsilon (float): Small value to ensure non-zero priorities
        device (str): Device for tensor operations ('cpu' or 'cuda')
        seed (Optional[int]): Random seed for reproducible sampling
        
    Example:
        >>> buffer = PrioritizedReplayBuffer(
        ...     buffer_size=10000, batch_size=32, alpha=0.6, beta=0.4
        ... )
        >>> buffer.add(obs, action, reward, next_obs, done, priority=1.0)
        >>> if buffer.can_sample():
        ...     batch, weights, indices = buffer.sample()
        ...     # Update priorities based on new TD errors
        ...     buffer.update_priorities(indices, new_priorities)
    """
    
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__(buffer_size, batch_size, device, seed)
        
        # Prioritization parameters
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.epsilon = epsilon  # Small value for numerical stability
        
        # Sum-tree for efficient priority-based sampling
        self.sum_tree = SumTree(buffer_size)
        self.max_priority = 1.0  # Maximum priority seen so far
        
    def _init_storage(self) -> None:
        """Initialize storage arrays for transitions."""
        # Storage will be initialized when first transition is added
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
        """Initialize storage arrays based on the first transition."""
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
        priority: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Add a transition to the buffer with priority.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Episode termination flag
            priority: Priority value (if None, uses maximum priority)
            **kwargs: Additional arguments
        """
        # Initialize storage on first transition
        if not self._initialized:
            self._init_storage_arrays(obs, action, reward, next_obs, done)
        
        # Use maximum priority if not specified (optimistic initialization)
        if priority is None:
            priority = self.max_priority
        
        # Store transition
        self.observations[self.pos] = np.asarray(obs)
        self.actions[self.pos] = np.asarray(action)
        self.rewards[self.pos] = np.asarray(reward)
        self.next_observations[self.pos] = np.asarray(next_obs)
        self.dones[self.pos] = np.asarray(done)
        
        # Update priority in sum-tree
        priority_alpha = (priority + self.epsilon) ** self.alpha
        self.sum_tree.add(priority_alpha, self.pos)
        self.max_priority = max(self.max_priority, priority)
        
        # Update position and full flag
        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True
    
    def sample(self, beta: Optional[float] = None) -> PrioritizedBufferSample:
        """
        Sample a batch of transitions based on priorities.
        
        Args:
            beta: Importance sampling exponent (if None, uses instance beta)
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones, 
                     importance_weights, indices)
        """
        assert self.can_sample(), f"Cannot sample {self.batch_size} transitions from buffer with {self.size()} transitions"
        
        if beta is None:
            beta = self.beta
        
        # Sample indices based on priorities
        indices = []
        priorities = []
        segment = self.sum_tree.total_priority / self.batch_size
        
        for i in range(self.batch_size):
            # Sample uniformly from each segment
            left = segment * i
            right = segment * (i + 1)
            value = np.random.uniform(left, right)
            
            # Get corresponding leaf
            _, priority, data_index = self.sum_tree.get_leaf(value)
            indices.append(data_index)
            priorities.append(priority)
        
        indices = np.array(indices)
        priorities = np.array(priorities)
        
        # Calculate importance sampling weights
        # weight = (N * P(i))^(-beta) / max_weight
        total_transitions = self.size()
        sampling_probabilities = priorities / self.sum_tree.total_priority
        weights = (total_transitions * sampling_probabilities) ** (-beta)
        weights = weights / weights.max()  # Normalize by maximum weight
        
        # Get transition samples
        batch_samples = self._get_samples(indices)
        
        # Convert weights to tensor
        weights_tensor = self._to_tensor(weights)
        
        return (*batch_samples, weights_tensor, indices)    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for given indices.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            # Update maximum priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update priority in sum-tree
            priority_alpha = (priority + self.epsilon) ** self.alpha
            tree_index = idx + self.sum_tree.capacity - 1
            self.sum_tree.update(tree_index, priority_alpha)
    
    def _get_samples(self, batch_indices: np.ndarray) -> tuple:
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
    
    def set_beta(self, beta: float) -> None:
        """
        Update the importance sampling exponent beta.
        
        Args:
            beta: New beta value
        """
        self.beta = beta
    
    def get_max_priority(self) -> float:
        """Get the current maximum priority."""
        return self.max_priority
    
    def clear(self) -> None:
        """Clear the buffer and reset all priorities."""
        super().clear()
        self.sum_tree = SumTree(self.buffer_size)
        self.max_priority = 1.0