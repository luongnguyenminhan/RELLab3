"""
PPO (Proximal Policy Optimization) Algorithm Implementation.

This module implements the Proximal Policy Optimization (PPO) algorithm, a canonical policy-based DRL method as described in Section III.B of "Deep Reinforcement Learning: A Survey" by Wang et al. and the Modular DRL Framework blueprint.

Detailed Description:
PPO is an on-policy, actor-critic algorithm that uses a clipped surrogate objective to ensure stable and monotonic policy updates. It supports both discrete and continuous action spaces and leverages advantage estimation for efficient learning. The module is designed for modularity, reproducibility, and extensibility, with all hyperparameters managed via YAML configuration.

Key Concepts/Algorithms:
- Actor-Critic architecture
- Clipped surrogate objective (trust region)
- Advantage estimation (GAE)
- Policy and value networks

Important Parameters/Configurations:
- Learning rate, discount factor (gamma), batch size
- Clipping parameter (epsilon)
- Advantage estimation parameters (lambda)
- Policy and value network architectures
- All parameters are loaded from `configs/ppo_config.yaml`

Expected Inputs/Outputs:
- Inputs: state (numpy.ndarray or torch.Tensor), action, reward, next_state, done
- Outputs: updated policy and value networks, training metrics

Dependencies:
- torch, numpy, gym, PyYAML
- src/networks/policy_network.py, src/networks/value_network.py

Author: REL Project Team
Date: 2025-07-13
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.distributions import Categorical, Normal

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from networks.policy_network import PolicyNetwork
from networks.value_network import ValueNetwork
from utils.logger import Logger


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent Implementation.
    
    This class implements the PPO algorithm with support for:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Both discrete and continuous action spaces
    - Multiple epochs of optimization per batch
    - Entropy regularization
    
    Args:
        state_size (int): Dimension of the state space
        action_size (int): Number of actions (discrete) or dimension (continuous)
        action_type (str): Type of action space ('discrete' or 'continuous')
        config_path (str): Path to YAML configuration file
        device (str): Device to run computations ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_type: str = "discrete",
        config_path: str = "configs/ppo_config.yaml",
        device: str = "cpu"
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.action_type = action_type
        self.device = torch.device(device)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize networks
        self._init_networks()
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config['policy_lr']
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=self.config['value_lr']
        )
        
        # Training parameters
        self.gamma = self.config['gamma']
        self.gae_lambda = self.config['gae_lambda']
        self.clip_epsilon = self.config['clip_epsilon']
        self.entropy_coef = self.config['entropy_coef']
        self.value_loss_coef = self.config['value_loss_coef']
        self.ppo_epochs = self.config['ppo_epochs']
        self.mini_batch_size = self.config['mini_batch_size']
        
        # Storage for rollout data
        self.reset_rollout()
        
        # Training tracking
        self.step_count = 0
        self.episode_count = 0
        self.training_metrics = defaultdict(list)
        
        # Initialize logger
        self.logger = Logger("PPO")
        
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
        """Default PPO configuration."""
        return {
            'policy_lr': 3e-4,
            'value_lr': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'entropy_coef': 0.01,
            'value_loss_coef': 0.5,
            'ppo_epochs': 10,
            'mini_batch_size': 64,
            'rollout_length': 2048,
            'policy_hidden_sizes': [64, 64],
            'value_hidden_sizes': [64, 64]
        }
    
    def _init_networks(self):
        """Initialize policy and value networks."""
        if self.action_type == "discrete":
            # Discrete action space - output probabilities
            self.policy = PolicyNetwork(
                input_size=self.state_size,
                output_size=self.action_size,
                hidden_sizes=self.config['policy_hidden_sizes'],
                output_activation='softmax'
            ).to(self.device)
        else:
            # Continuous action space - output mean and std
            self.policy = PolicyNetwork(
                input_size=self.state_size,
                output_size=self.action_size * 2,  # mean and log_std
                hidden_sizes=self.config['policy_hidden_sizes'],
                output_activation='none'
            ).to(self.device)
        
        self.value_net = ValueNetwork(
            input_size=self.state_size,
            hidden_sizes=self.config['value_hidden_sizes']
        ).to(self.device)
    
    def reset_rollout(self):
        """Reset rollout storage."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_values = []
    
    def act(self, state: np.ndarray) -> tuple:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get policy output
            policy_output = self.policy(state_tensor)
            
            # Get action distribution
            if self.action_type == "discrete":
                dist = Categorical(policy_output)
            else:
                mean, log_std = torch.chunk(policy_output, 2, dim=-1)
                std = torch.exp(log_std.clamp(-20, 2))  # Clamp for numerical stability
                dist = Normal(mean, std)
            
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            
            # Get value estimate
            value = self.value_net(state_tensor)
        
        if self.action_type == "discrete":
            return action.item(), log_prob.item(), value.item()
        else:
            return action.cpu().numpy()[0], log_prob.item(), value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store a transition in the rollout buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
        if done:
            self.episode_count += 1
    
    def compute_gae(self, next_value: float = 0.0) -> tuple:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value of the next state (for terminal states)
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        # Convert to tensors
        values = torch.FloatTensor(self.values + [next_value]).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.BoolTensor(self.dones).to(self.device)
        
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * ~dones[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * ~dones[step] * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        advantages = torch.stack(advantages)
        returns = torch.stack(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, next_value: float = 0.0):
        """
        Update policy and value networks using PPO algorithm.
        
        Args:
            next_value: Value of the next state
        """
        if len(self.states) == 0:
            return
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert data to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        if self.action_type == "discrete":
            actions = torch.LongTensor(self.actions).to(self.device)
        else:
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        
        # PPO update for multiple epochs
        dataset_size = len(states)
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute current policy probabilities
                policy_output = self.policy(batch_states)
                
                if self.action_type == "discrete":
                    dist = Categorical(policy_output)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                else:
                    mean, log_std = torch.chunk(policy_output, 2, dim=-1)
                    std = torch.exp(log_std.clamp(-20, 2))
                    dist = Normal(mean, std)
                    log_probs = dist.log_prob(batch_actions).sum(-1)
                    entropy = dist.entropy().sum(-1).mean()
                
                # Compute ratio and clipped surrogate objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                values = self.value_net(batch_states).squeeze()
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Update policy
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Log metrics
                self.training_metrics['policy_loss'].append(policy_loss.item())
                self.training_metrics['value_loss'].append(value_loss.item())
                self.training_metrics['entropy'].append(entropy.item())
                self.training_metrics['advantages_mean'].append(batch_advantages.mean().item())
        
        # Reset rollout buffer
        self.reset_rollout()
    
    def save(self, filepath: str):
        """Save model and training state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return dict(self.training_metrics)
    
    def reset_metrics(self):
        """Reset training metrics."""
        self.training_metrics.clear()
