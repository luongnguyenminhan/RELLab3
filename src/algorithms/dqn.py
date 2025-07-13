"""
DQN (Deep Q-Network) Algorithm Implementation.

This module implements the Deep Q-Network (DQN) algorithm, a foundational value-based DRL method as described in Section III.A of "Deep Reinforcement Learning: A Survey" by Wang et al. and the Modular DRL Framework blueprint.

Detailed Description:
DQN combines Q-learning with deep neural networks to approximate the action-value function for high-dimensional state spaces. It integrates key improvements such as experience replay, target networks, and ϵ-greedy exploration to stabilize training and improve data efficiency. Double DQN (DDQN) is supported as a configurable option to mitigate Q-value overestimation. The module is designed for modularity and extensibility, supporting reproducible research and practical applications.

Key Concepts/Algorithms:
- Q-learning and Deep Q-Networks
- Experience Replay buffer
- Target Network
- ϵ-greedy exploration
- Double DQN (configurable)

Important Parameters/Configurations:
- Learning rate, discount factor (gamma), batch size
- Replay buffer size and sampling
- Target network update frequency
- ϵ-greedy parameters (epsilon, decay)
- All parameters are loaded from `configs/dqn_config.yaml`

Expected Inputs/Outputs:
- Inputs: state (numpy.ndarray or torch.Tensor), action, reward, next_state, done
- Outputs: updated Q-network, training metrics

Dependencies:
- torch, numpy, gym, PyYAML
- src/networks/q_network.py, src/buffers/replay_buffer.py

Author: REL Project Team
Date: 2025-07-13
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import yaml
import os
from typing import Dict, Any
from collections import defaultdict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from networks.q_network import QNetwork, DuelingQNetwork
from buffers.replay_buffer import ReplayBuffer
from utils.logger import Logger


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent Implementation.
    
    This class implements the DQN algorithm with support for:
    - Experience Replay
    - Target Networks with periodic updates
    - ϵ-greedy exploration with decay
    - Double DQN (optional)
    - Dueling DQN (optional)
    
    Args:
        state_size (int): Dimension of the state space
        action_size (int): Number of discrete actions
        config_path (str): Path to YAML configuration file
        device (str): Device to run computations ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config_path: str = "configs/dqn_config.yaml",
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
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Initialize exploration parameters
        self.epsilon = self.config['epsilon_start']
        self.epsilon_min = self.config['epsilon_end']
        self.epsilon_decay = self.config['epsilon_decay']
        
        # Training parameters
        self.gamma = self.config['gamma']
        self.target_update_freq = self.config['target_update_freq']
        self.double_dqn = self.config.get('double_dqn', False)
        
        # Training tracking
        self.step_count = 0
        self.episode_count = 0
        self.training_metrics = defaultdict(list)
        
        # Initialize logger
        self.logger = Logger("DQN")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                # Extract the actual config from the documented YAML
                if isinstance(config, str):
                    # If it's just a string (documentation), use defaults
                    config = self._get_default_config()
                elif config is None:
                    config = self._get_default_config()
                return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default DQN configuration."""
        return {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'batch_size': 32,
            'buffer_size': 100000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update_freq': 1000,
            'double_dqn': False,
            'dueling_dqn': False,
            'hidden_sizes': [128, 128]
        }
    
    def _init_networks(self):
        """Initialize Q-network and target network."""
        network_class = DuelingQNetwork if self.config.get('dueling_dqn', False) else QNetwork
        
        self.q_network = network_class(
            input_size=self.state_size,
            output_size=self.action_size,
            hidden_sizes=self.config['hidden_sizes']
        ).to(self.device)
        
        self.target_network = network_class(
            input_size=self.state_size,
            output_size=self.action_size,
            hidden_sizes=self.config['hidden_sizes']
        ).to(self.device)
        
        # Initialize target network with same weights
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ϵ-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def step(
        self,
        state: np.ndarray,
        action: int,
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
        
        # Update target network periodically
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if done:
            self.episode_count += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _learn(self):
        """Perform one learning step using sampled experiences."""
        experiences = self.memory.sample()
        if experiences is None:
            return
        
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network for action selection, target for evaluation
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Log metrics
        self.training_metrics['loss'].append(loss.item())
        self.training_metrics['q_values_mean'].append(current_q_values.mean().item())
        self.training_metrics['epsilon'].append(self.epsilon)
    
    def save(self, filepath: str):
        """Save model and training state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return dict(self.training_metrics)
    
    def reset_metrics(self):
        """Reset training metrics."""
        self.training_metrics.clear()
