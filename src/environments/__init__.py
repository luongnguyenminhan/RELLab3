"""
Environments Module.

This package provides wrappers and adapters for standardizing interaction with RL environments, as required by the Modular DRL Framework and described in "Deep Reinforcement Learning: A Survey" by Wang et al.

Detailed Description:
The environments module implements a modular hierarchy of environment wrappers that ensure consistent interfaces for agent-environment interaction across different RL algorithms. The base wrapper class provides fundamental functionality for observation preprocessing, action transformation, reward shaping, and episode management. This abstraction enables seamless integration of OpenAI Gym/Gymnasium environments, custom environments, and multi-agent systems while maintaining compatibility with the framework's configuration system.

Key Concepts/Algorithms:
- Environment abstraction and standardization
- Observation preprocessing (normalization, stacking, filtering)
- Action space transformation and clipping
- Reward shaping and normalization
- Episode statistics tracking and logging
- Reproducible environment seeding

Important Parameters/Configurations:
- Environment name/ID and creation parameters
- Preprocessing options (observation transforms, frame stacking)
- Action space modifications (clipping, rescaling)
- Reward processing (normalization, clipping)
- Episode limits and timeout handling
- All parameters loaded from YAML configs (e.g., `configs/dqn_config.yaml`)

Expected Inputs/Outputs:
- Inputs: Environment IDs, configuration dictionaries, agent actions
- Outputs: Preprocessed observations, shaped rewards, episode info, done flags
- Compatible with standard Gymnasium step/reset interface

Dependencies:
- gymnasium, numpy, torch, typing, abc, yaml
- Compatible with OpenAI Gym environments for backward compatibility

Author: REL Project Team
Date: 2025-07-13
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import yaml
import os

# Type aliases for better code readability and consistency
ObservationType = Union[np.ndarray, torch.Tensor, Dict[str, Any]]
ActionType = Union[np.ndarray, torch.Tensor, int, float, List]
RewardType = Union[float, np.ndarray, torch.Tensor]
DoneType = Union[bool, np.ndarray, torch.Tensor]
InfoType = Dict[str, Any]
EnvStepReturn = Tuple[ObservationType, RewardType, DoneType, DoneType, InfoType]
EnvResetReturn = Tuple[ObservationType, InfoType]


class BaseEnvironmentWrapper(ABC):
    """
    Abstract base class for environment wrappers in the Modular DRL Framework.
    
    This class defines the common interface and shared functionality for all environment wrapper
    implementations. It provides foundation for observation preprocessing, action transformation,
    reward shaping, and episode management while ensuring compatibility with Gymnasium environments
    and the framework's configuration system.
    
    The wrapper follows the Gymnasium Wrapper pattern but adds framework-specific functionality
    such as YAML configuration loading, tensor device management, and enhanced logging capabilities.
    
    Args:
        env_id (str): Environment identifier for gym.make()
        config_path (Optional[str]): Path to YAML configuration file
        device (str): Device for tensor operations ('cpu' or 'cuda')
        seed (Optional[int]): Random seed for reproducible experiments
        render_mode (Optional[str]): Rendering mode for the environment
        **env_kwargs: Additional keyword arguments for environment creation
        
    Attributes:
        env (gym.Env): The wrapped Gymnasium environment
        env_id (str): Environment identifier
        device (torch.device): PyTorch device for tensor operations
        config (Dict[str, Any]): Loaded configuration parameters
        episode_count (int): Number of completed episodes
        total_steps (int): Total number of environment steps
        episode_rewards (List[float]): History of episode rewards
        episode_lengths (List[int]): History of episode lengths
    """
    
    def __init__(
        self,
        env_id: str,
        config_path: Optional[str] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        **env_kwargs
    ):
        self.env_id = env_id
        self.device = torch.device(device)
        self.config = self._load_config(config_path) if config_path else {}
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        
        # Current episode tracking
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        
        # Initialize the base environment
        self.env = self._create_environment(env_id, render_mode, **env_kwargs)
        
        # Set random seed for reproducibility
        if seed is not None:
            self._set_seed(seed)
        
        # Initialize wrapper-specific components
        self._init_wrapper_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file {config_path}: {e}")
    
    def _create_environment(
        self, 
        env_id: str, 
        render_mode: Optional[str] = None,
        **env_kwargs
    ) -> gym.Env:
        """
        Create the base Gymnasium environment.
        
        Args:
            env_id: Environment identifier
            render_mode: Rendering mode for the environment
            **env_kwargs: Additional environment creation arguments
            
        Returns:
            Created Gymnasium environment
        """
        try:
            env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
            return env
        except Exception as e:
            raise RuntimeError(f"Failed to create environment '{env_id}': {e}")
    
    def _set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        elif hasattr(self.env, 'reset'):
            # For newer Gymnasium versions
            self.env.reset(seed=seed)
    
    @abstractmethod
    def _init_wrapper_components(self) -> None:
        """
        Initialize wrapper-specific components.
        
        This method should be implemented by concrete wrapper classes to set up
        any specialized functionality, preprocessing pipelines, or additional
        tracking mechanisms.
        """
        pass
    
    @abstractmethod
    def _preprocess_observation(self, observation: ObservationType) -> ObservationType:
        """
        Preprocess observations before returning to agent.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Preprocessed observation
        """
        pass
    
    @abstractmethod
    def _transform_action(self, action: ActionType) -> ActionType:
        """
        Transform actions before passing to environment.
        
        Args:
            action: Action from agent
            
        Returns:
            Transformed action for environment
        """
        pass
    
    @abstractmethod
    def _shape_reward(self, reward: RewardType, info: InfoType) -> RewardType:
        """
        Apply reward shaping or normalization.
        
        Args:
            reward: Raw reward from environment
            info: Environment info dictionary
            
        Returns:
            Shaped/normalized reward
        """
        pass    
    def step(self, action: ActionType) -> EnvStepReturn:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Transform action if needed
        transformed_action = self._transform_action(action)
        
        # Execute environment step
        observation, reward, terminated, truncated, info = self.env.step(transformed_action)
        
        # Apply preprocessing and shaping
        processed_observation = self._preprocess_observation(observation)
        shaped_reward = self._shape_reward(reward, info)
        
        # Update episode tracking
        self._current_episode_reward += float(shaped_reward)
        self._current_episode_length += 1
        self.total_steps += 1
        
        # Handle episode completion
        if terminated or truncated:
            self._handle_episode_end(info)
        
        # Add wrapper-specific info
        info = self._update_info(info)
        
        return processed_observation, shaped_reward, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> EnvResetReturn:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional seed for episode randomization
            options: Optional reset options
            
        Returns:
            Tuple of (initial_observation, info)
        """
        # Reset base environment
        observation, info = self.env.reset(seed=seed, options=options)
        
        # Preprocess initial observation
        processed_observation = self._preprocess_observation(observation)
        
        # Reset episode tracking
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        
        # Add wrapper-specific info
        info = self._update_info(info)
        
        return processed_observation, info
    
    def render(self) -> Any:
        """
        Render the environment.
        
        Returns:
            Rendered frame or None
        """
        return self.env.render()
    
    def close(self) -> None:
        """Close the environment and cleanup resources."""
        self.env.close()
    
    def _handle_episode_end(self, info: InfoType) -> None:
        """
        Handle end of episode bookkeeping.
        
        Args:
            info: Environment info dictionary
        """
        self.episode_rewards.append(self._current_episode_reward)
        self.episode_lengths.append(self._current_episode_length)
        self.episode_count += 1
        
        # Add episode statistics to info
        info['episode'] = {
            'r': self._current_episode_reward,
            'l': self._current_episode_length,
            't': self.total_steps
        }
    
    def _update_info(self, info: InfoType) -> InfoType:
        """
        Add wrapper-specific information to info dictionary.
        
        Args:
            info: Original info dictionary
            
        Returns:
            Updated info dictionary with wrapper statistics
        """
        info['wrapper_stats'] = {
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'current_episode_length': self._current_episode_length,
            'current_episode_reward': self._current_episode_reward
        }
        return info
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive episode statistics.
        
        Returns:
            Dictionary containing episode statistics
        """
        if not self.episode_rewards:
            return {
                'total_episodes': 0,
                'mean_reward': 0.0,
                'mean_length': 0.0,
                'total_steps': self.total_steps
            }
        
        return {
            'total_episodes': self.episode_count,
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
            'total_steps': self.total_steps,
            'recent_rewards': self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        }
    
    def save_statistics(self, filepath: str) -> None:
        """
        Save episode statistics to file.
        
        Args:
            filepath: Path to save statistics
        """
        stats = self.get_episode_statistics()
        stats['episode_rewards'] = self.episode_rewards
        stats['episode_lengths'] = self.episode_lengths
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(stats, f, default_flow_style=False)
        except Exception as e:
            print(f"Failed to save statistics to {filepath}: {e}")
    
    # Properties for Gymnasium compatibility
    @property
    def action_space(self) -> spaces.Space:
        """Get the action space of the environment."""
        return self.env.action_space
    
    @property
    def observation_space(self) -> spaces.Space:
        """Get the observation space of the environment."""
        return self.env.observation_space
    
    @property
    def spec(self) -> Optional[gym.envs.registration.EnvSpec]:
        """Get the environment specification."""
        return getattr(self.env, 'spec', None)
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the environment metadata."""
        return getattr(self.env, 'metadata', {})
    
    @property
    def np_random(self) -> Optional[np.random.Generator]:
        """Get the environment's random number generator."""
        return getattr(self.env, 'np_random', None)
    
    @property
    def unwrapped(self) -> gym.Env:
        """Get the base environment without any wrappers."""
        if hasattr(self.env, 'unwrapped'):
            return self.env.unwrapped
        return self.env


# Export the base class for use by concrete implementations
__all__ = [
    'BaseEnvironmentWrapper',
    'ObservationType',
    'ActionType', 
    'RewardType',
    'DoneType',
    'InfoType',
    'EnvStepReturn',
    'EnvResetReturn'
]