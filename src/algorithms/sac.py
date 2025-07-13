"""
SAC (Soft Actor-Critic) Algorithm Implementation.

This module implements the Soft Actor-Critic (SAC) algorithm, a maximum entropy-based DRL method as described in Section III.C of "Deep Reinforcement Learning: A Survey" by Wang et al. and the Modular DRL Framework blueprint.

Detailed Description:
SAC is an off-policy, actor-critic algorithm that maximizes both expected reward and policy entropy, encouraging robust exploration and optimal stochastic policies. It integrates temperature (entropy regularization), experience replay, and target networks for stable and efficient learning. The module is designed for modularity, reproducibility, and extensibility, with all hyperparameters managed via YAML configuration.

Key Concepts/Algorithms:
- Maximum entropy RL objective
- Actor-Critic architecture
- Temperature (entropy) parameter
- Experience Replay buffer
- Target Network

Important Parameters/Configurations:
- Learning rate, discount factor (gamma), batch size
- Temperature parameter (alpha)
- Actor and Critic network architectures
- Replay buffer size and sampling
- All parameters are loaded from `configs/sac_config.yaml`

Expected Inputs/Outputs:
- Inputs: state (numpy.ndarray or torch.Tensor), action, reward, next_state, done
- Outputs: updated policy and value networks, training metrics

Dependencies:
- torch, numpy, gym, PyYAML
- src/networks/policy_network.py, src/networks/q_network.py
- src/buffers/replay_buffer.py

Author: REL Project Team
Date: 2025-07-13
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import yaml
import os
from typing import Dict, Any
from collections import defaultdict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from networks.policy_network import PolicyNetwork
from networks.q_network import QNetwork
from buffers.replay_buffer import ReplayBuffer
from utils.logger import Logger


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent Implementation.
    
    This class implements the SAC algorithm with support for:
    - Maximum entropy objective
    - Automatic temperature tuning
    - Twin Q-networks (delayed updates)
    - Experience replay
    - Continuous action spaces
    
    Args:
        state_size (int): Dimension of the state space
        action_size (int): Dimension of the action space
        config_path (str): Path to YAML configuration file
        device (str): Device to run computations ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config_path: str = "configs/sac_config.yaml",
        device: str = "cpu"
    ):
        self.state_size = state_size
        self.action_size = action_size
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
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=self.config['critic_lr']
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=self.config['critic_lr']
        )
        
        # Temperature parameter
        self.automatic_entropy_tuning = self.config['automatic_entropy_tuning']
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_size])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config['alpha_lr'])
        else:
            self.alpha = self.config['alpha']
        
        # Training parameters
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']
        
        # Training tracking
        self.step_count = 0
        self.episode_count = 0
        self.training_metrics = defaultdict(list)
        
        # Initialize logger
        self.logger = Logger("SAC")
        
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
        """Default SAC configuration."""
        return {
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'alpha_lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 256,
            'buffer_size': 1000000,
            'alpha': 0.2,
            'automatic_entropy_tuning': True,
            'actor_hidden_sizes': [256, 256],
            'critic_hidden_sizes': [256, 256]
        }
    
    def _init_networks(self):
        """Initialize actor and critic networks."""
        # Actor network (stochastic policy)
        self.actor = PolicyNetwork(
            input_size=self.state_size,
            output_size=self.action_size * 2,  # mean and log_std
            hidden_sizes=self.config['actor_hidden_sizes'],
            output_activation='none'
        ).to(self.device)
        
        # Twin Q-networks
        self.critic1 = QNetwork(
            input_size=self.state_size + self.action_size,
            output_size=1,
            hidden_sizes=self.config['critic_hidden_sizes']
        ).to(self.device)
        
        self.critic2 = QNetwork(
            input_size=self.state_size + self.action_size,
            output_size=1,
            hidden_sizes=self.config['critic_hidden_sizes']
        ).to(self.device)
        
        # Target Q-networks
        self.critic1_target = QNetwork(
            input_size=self.state_size + self.action_size,
            output_size=1,
            hidden_sizes=self.config['critic_hidden_sizes']
        ).to(self.device)
        
        self.critic2_target = QNetwork(
            input_size=self.state_size + self.action_size,
            output_size=1,
            hidden_sizes=self.config['critic_hidden_sizes']
        ).to(self.device)
        
        # Initialize target networks
        self.soft_update(self.critic1, self.critic1_target, tau=1.0)
        self.soft_update(self.critic2, self.critic2_target, tau=1.0)
    
    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float):
        """Soft update model parameters using Polyak averaging."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False):
        """
        Get action and log probability from current policy.
        
        Args:
            state: State tensor
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, log_prob, mean)
        """
        policy_output = self.actor(state)
        mean, log_std = torch.chunk(policy_output, 2, dim=-1)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action_tanh = torch.tanh(action)
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(action) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action_tanh, log_prob, mean
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.get_action_and_log_prob(state_tensor, deterministic)
        
        return action.cpu().numpy()[0]
    
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
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        self.step_count += 1
        
        # Learn if enough samples are available
        if len(self.memory) >= self.config['batch_size']:
            self._learn()
        
        if done:
            self.episode_count += 1
    
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
        
        # Get current alpha value
        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha
        
        # Update Q-functions
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.get_action_and_log_prob(next_states)
            next_q1 = self.critic1_target(torch.cat([next_states, next_actions], dim=1))
            next_q2 = self.critic2_target(torch.cat([next_states, next_actions], dim=1))
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + (self.gamma * next_q * ~dones.unsqueeze(1))
        
        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs, _ = self.get_action_and_log_prob(states)
        q1_new = self.critic1(torch.cat([states, new_actions], dim=1))
        q2_new = self.critic2(torch.cat([states, new_actions], dim=1))
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature parameter
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.training_metrics['alpha_loss'].append(alpha_loss.item())
            self.training_metrics['alpha'].append(alpha.item())
        
        # Soft update target networks
        self.soft_update(self.critic1, self.critic1_target, self.tau)
        self.soft_update(self.critic2, self.critic2_target, self.tau)
        
        # Log metrics
        self.training_metrics['critic1_loss'].append(critic1_loss.item())
        self.training_metrics['critic2_loss'].append(critic2_loss.item())
        self.training_metrics['actor_loss'].append(actor_loss.item())
        self.training_metrics['q_values_mean'].append(current_q1.mean().item())
        self.training_metrics['log_probs_mean'].append(log_probs.mean().item())
    
    def save(self, filepath: str):
        """Save model and training state."""
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }
        
        if self.automatic_entropy_tuning:
            save_dict.update({
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict()
            })
        
        torch.save(save_dict, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return dict(self.training_metrics)
    
    def reset_metrics(self):
        """Reset training metrics."""
        self.training_metrics.clear()
