"""
DDPG (Deep Deterministic Policy Gradient) Algorithm Implementation.

This module implements the Deep Deterministic Policy Gradient (DDPG) algorithm, a canonical policy-based DRL method as described in Section III.B of "Deep Reinforcement Learning: A Survey" by Wang et al. and the Modular DRL Framework engineering blueprint.

Detailed Description:
DDPG is an off-policy, actor-critic algorithm designed for continuous action spaces. It leverages deterministic policy gradients, experience replay, and target networks (with soft updates) to achieve stable and efficient learning. The algorithm integrates Ornstein-Uhlenbeck noise for exploration and supports modular configuration via YAML files. DDPG is foundational for advanced methods such as TD3 and is a key representative of policy-based DRL in this framework.

Key Concepts/Algorithms:
- Deterministic Policy Gradient (DPG)
- Actor-Critic architecture
- Experience Replay buffer
- Target Network with soft updates (Polyak averaging)
- Ornstein-Uhlenbeck noise for exploration

Important Parameters/Configurations:
- Learning rate, discount factor (gamma), batch size
- Actor and Critic network architectures
- Replay buffer size and sampling
- Soft update parameter (tau)
- Noise process parameters
- All parameters are loaded from `configs/ddpg_config.yaml`

Expected Inputs/Outputs:
- Inputs: state (numpy.ndarray or torch.Tensor), action, reward, next_state, done
- Outputs: updated policy and value networks, training metrics

Dependencies:
- torch, numpy, gym, PyYAML
- src/networks/policy_network.py, src/networks/q_network.py
- src/buffers/replay_buffer.py, src/utils/noise.py

Author: REL Project Team
Date: 2025-07-13
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from buffers.replay_buffer import ReplayBuffer
from networks.policy_network import PolicyNetwork
from networks.q_network import QNetwork
from utils.logger import Logger
from utils.noise import OrnsteinUhlenbeckNoise


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) Agent Implementation.
    
    This class implements the DDPG algorithm with support for:
    - Deterministic policy gradient
    - Soft target network updates (Polyak averaging)
    - Experience replay
    - Ornstein-Uhlenbeck noise for exploration
    - Continuous action spaces
    
    Args:
        state_size (int): Dimension of the state space
        action_size (int): Dimension of the action space
        action_low (float): Lower bound of action space
        action_high (float): Upper bound of action space
        config_path (str): Path to YAML configuration file
        device (str): Device to run computations ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        config_path: str = "configs/ddpg_config.yaml",
        device: str = "cpu"
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.device = torch.device(device)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize networks
        self._init_networks()
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(
            capacity=self.config['buffer_size'],
            batch_size=self.config['batch_size']
        )
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config['actor_lr']
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config['critic_lr']
        )
        
        # Initialize noise process
        self.noise = OrnsteinUhlenbeckNoise(
            size=action_size,
            mu=self.config['noise_mu'],
            theta=self.config['noise_theta'],
            sigma=self.config['noise_sigma']
        )
        
        # Training parameters
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']
        
        # Training tracking
        self.step_count = 0
        self.episode_count = 0
        self.training_metrics = defaultdict(list)
        
        # Initialize logger
        self.logger = Logger("DDPG")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                # Extract the actual config from the documented YAML
                if isinstance(config, str) or config is None:
                    config = self._get_default_config()
                return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default DDPG configuration."""
        return {
            'actor_lr': 0.001,
            'critic_lr': 0.002,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 128,
            'buffer_size': 1000000,
            'actor_hidden_sizes': [400, 300],
            'critic_hidden_sizes': [400, 300],
            'noise_mu': 0.0,
            'noise_theta': 0.15,
            'noise_sigma': 0.2
        }
    
    def _init_networks(self):
        """Initialize actor, critic, and target networks."""
        # Actor networks
        self.actor = PolicyNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            hidden_sizes=self.config['actor_hidden_sizes'],
            output_activation='tanh'
        ).to(self.device)
        
        self.actor_target = PolicyNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            hidden_sizes=self.config['actor_hidden_sizes'],
            output_activation='tanh'
        ).to(self.device)
        
        # Critic networks (Q-networks for continuous actions)
        self.critic = QNetwork(
            input_size=self.state_size + self.action_size,
            output_size=1,
            hidden_sizes=self.config['critic_hidden_sizes']
        ).to(self.device)
        
        self.critic_target = QNetwork(
            input_size=self.state_size + self.action_size,
            output_size=1,
            hidden_sizes=self.config['critic_hidden_sizes']
        ).to(self.device)
        
        # Initialize target networks with same weights
        self.soft_update(self.actor, self.actor_target, tau=1.0)
        self.soft_update(self.critic, self.critic_target, tau=1.0)
    
    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float):
        """Soft update model parameters using Polyak averaging."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action using deterministic policy with optional exploration noise.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            
        Returns:
            Selected action (scaled to action bounds)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy()[0]
        self.actor.train()
        
        if add_noise:
            action += self.noise.sample()
        
        # Scale action to bounds
        action = np.clip(action, -1, 1)
        scaled_action = self.action_low + (action + 1) * 0.5 * (self.action_high - self.action_low)
        
        return scaled_action
    
    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store experience and perform learning step if ready.
        
        Args:
            state: Current state
            action: Action taken (in original scale)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Normalize action to [-1, 1] for storage
        normalized_action = 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
        
        # Store experience in replay buffer
        self.memory.add(state, normalized_action, reward, next_state, done)
        
        self.step_count += 1
        
        # Learn if enough samples are available
        if len(self.memory) >= self.config['batch_size']:
            self._learn()
        
        # Reset noise at episode end
        if done:
            self.episode_count += 1
            self.noise.reset()
    
    def _learn(self):
        """Perform one learning step using sampled experiences."""
        experiences = self.memory.sample()
        if experiences is None:
            return
        
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q_values = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        current_q_values = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, predicted_actions], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)
        
        # Log metrics
        self.training_metrics['critic_loss'].append(critic_loss.item())
        self.training_metrics['actor_loss'].append(actor_loss.item())
        self.training_metrics['q_values_mean'].append(current_q_values.mean().item())
    
    def save(self, filepath: str):
        """Save model and training state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return dict(self.training_metrics)
    
    def reset_metrics(self):
        """Reset training metrics."""
        self.training_metrics.clear()
