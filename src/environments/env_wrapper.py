"""
Environment Wrapper Implementation.

This module provides standardized wrappers for RL environments, as required by the Modular DRL Framework and described in "Deep Reinforcement Learning: A Survey" by Wang et al.

Detailed Description:
The environment wrapper ensures a consistent interface for agent-environment interaction, supporting OpenAI Gym and compatible APIs. It can preprocess observations, rewards, and actions, and is designed for extensibility to support custom or multi-agent environments. This abstraction enables seamless integration of new environments and facilitates reproducible experimentation.

Key Concepts/Algorithms:
- Standardized environment interface
- Observation and reward preprocessing
- Compatibility with Gym and custom environments

Important Parameters/Configurations:
- Environment name or specification (from config)
- Preprocessing options (e.g., normalization, frame stacking)
- All parameters are loaded from the relevant YAML config (e.g., `configs/dqn_config.yaml`)

Expected Inputs/Outputs:
- Inputs: environment step and reset calls, agent actions
- Outputs: processed observations, rewards, done flags, info dicts

Dependencies:
- gym, numpy

Author: REL Project Team
Date: 2025-07-13
"""
import numpy as np
import torch
from typing import Any, Dict, Optional, Union
from . import BaseEnvironmentWrapper, ObservationType, ActionType, RewardType, InfoType


class StandardEnvironmentWrapper(BaseEnvironmentWrapper):
    """
    Standard environment wrapper implementation for common RL environments.
    
    This wrapper provides basic preprocessing functionality including observation normalization,
    action clipping, and reward scaling. It serves as a concrete implementation of the
    BaseEnvironmentWrapper and can be used directly or extended for specialized needs.
    
    Args:
        env_id (str): Environment identifier for gym.make()
        config_path (Optional[str]): Path to YAML configuration file
        device (str): Device for tensor operations ('cpu' or 'cuda')
        seed (Optional[int]): Random seed for reproducible experiments
        render_mode (Optional[str]): Rendering mode for the environment
        normalize_obs (bool): Whether to normalize observations
        normalize_rewards (bool): Whether to normalize rewards
        clip_actions (bool): Whether to clip actions to action space bounds
        reward_scale (float): Scaling factor for rewards
        **env_kwargs: Additional keyword arguments for environment creation
        
    Attributes:
        normalize_obs (bool): Observation normalization flag
        normalize_rewards (bool): Reward normalization flag  
        clip_actions (bool): Action clipping flag
        reward_scale (float): Reward scaling factor
        obs_rms (RunningMeanStd): Running statistics for observation normalization
        reward_rms (RunningMeanStd): Running statistics for reward normalization
    """
    
    def __init__(
        self,
        env_id: str,
        config_path: Optional[str] = None,
        device: str = "cpu", 
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        normalize_obs: bool = False,
        normalize_rewards: bool = False,
        clip_actions: bool = True,
        reward_scale: float = 1.0,
        **env_kwargs
    ):
        # Store wrapper-specific parameters
        self.normalize_obs = normalize_obs
        self.normalize_rewards = normalize_rewards
        self.clip_actions = clip_actions
        self.reward_scale = reward_scale
        
        # Initialize base class
        super().__init__(
            env_id=env_id,
            config_path=config_path,
            device=device,
            seed=seed,
            render_mode=render_mode,
            **env_kwargs
        )
    
    def _init_wrapper_components(self) -> None:
        """Initialize wrapper-specific components."""
        # Initialize running statistics for normalization
        if self.normalize_obs:
            self.obs_rms = RunningMeanStd(shape=self.env.observation_space.shape)
        
        if self.normalize_rewards:
            self.reward_rms = RunningMeanStd(shape=())
            self.reward_history = []
    
    def _preprocess_observation(self, observation: ObservationType) -> ObservationType:
        """
        Preprocess observations with optional normalization.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Preprocessed observation
        """
        if self.normalize_obs and hasattr(self, 'obs_rms'):
            # Convert to numpy if necessary
            obs_array = np.array(observation) if not isinstance(observation, np.ndarray) else observation
            
            # Update running statistics
            self.obs_rms.update(obs_array)
            
            # Normalize observation
            normalized_obs = (obs_array - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            
            # Convert to tensor if using GPU
            if self.device.type == 'cuda':
                return torch.tensor(normalized_obs, dtype=torch.float32, device=self.device)
            
            return normalized_obs
        
        # Convert to tensor if using GPU and no normalization
        if self.device.type == 'cuda' and isinstance(observation, np.ndarray):
            return torch.tensor(observation, dtype=torch.float32, device=self.device)
        
        return observation
    
    def _transform_action(self, action: ActionType) -> ActionType:
        """
        Transform actions with optional clipping.
        
        Args:
            action: Action from agent
            
        Returns:
            Transformed action for environment
        """
        # Convert tensor to numpy if necessary
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # Clip actions to action space bounds
        if self.clip_actions and hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        return action
    
    def _shape_reward(self, reward: RewardType, info: InfoType) -> RewardType:
        """
        Apply reward shaping including scaling and normalization.
        
        Args:
            reward: Raw reward from environment
            info: Environment info dictionary
            
        Returns:
            Shaped/normalized reward
        """
        # Apply reward scaling
        shaped_reward = float(reward) * self.reward_scale
        
        # Apply reward normalization if enabled
        if self.normalize_rewards and hasattr(self, 'reward_rms'):
            self.reward_history.append(shaped_reward)
            
            # Update running statistics periodically
            if len(self.reward_history) >= 100:
                self.reward_rms.update(np.array(self.reward_history))
                self.reward_history = []
            
            # Normalize reward
            shaped_reward = shaped_reward / np.sqrt(self.reward_rms.var + 1e-8)
        
        return shaped_reward


class RunningMeanStd:
    """
    Tracks the running mean and standard deviation of a stream of values.
    
    This class implements Welford's online algorithm for computing running statistics,
    which is numerically stable and memory efficient.
    
    Args:
        epsilon (float): Small value to prevent division by zero
        shape (tuple): Shape of the values being tracked
        
    Attributes:
        mean (np.ndarray): Running mean
        var (np.ndarray): Running variance
        count (int): Number of samples seen
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with new batch of samples.
        
        Args:
            x: New batch of samples
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """
        Update running statistics from batch moments.
        
        Args:
            batch_mean: Mean of the batch
            batch_var: Variance of the batch
            batch_count: Number of samples in the batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


# Additional specialized wrappers can be implemented here
class AtariWrapper(StandardEnvironmentWrapper):
    """
    Specialized wrapper for Atari environments with common preprocessing.
    
    This wrapper applies standard Atari preprocessing including frame skipping,
    grayscale conversion, and frame stacking commonly used in DRL papers.
    """
    
    def __init__(self, env_id: str, **kwargs):
        # Set Atari-specific defaults
        kwargs.setdefault('normalize_obs', False)
        kwargs.setdefault('normalize_rewards', False) 
        kwargs.setdefault('clip_actions', False)
        kwargs.setdefault('reward_scale', 1.0)
        
        super().__init__(env_id, **kwargs)
    
    def _init_wrapper_components(self) -> None:
        """Initialize Atari-specific preprocessing components."""
        super()._init_wrapper_components()
        # Additional Atari-specific initialization can be added here
        pass


class ContinuousControlWrapper(StandardEnvironmentWrapper):
    """
    Specialized wrapper for continuous control environments.
    
    This wrapper applies preprocessing commonly used for continuous control tasks
    such as action normalization and observation scaling.
    """
    
    def __init__(self, env_id: str, **kwargs):
        # Set continuous control defaults
        kwargs.setdefault('normalize_obs', True)
        kwargs.setdefault('normalize_rewards', True)
        kwargs.setdefault('clip_actions', True)
        kwargs.setdefault('reward_scale', 1.0)
        
        super().__init__(env_id, **kwargs)


# Export classes for use by other modules
__all__ = [
    'StandardEnvironmentWrapper',
    'AtariWrapper', 
    'ContinuousControlWrapper',
    'RunningMeanStd'
]