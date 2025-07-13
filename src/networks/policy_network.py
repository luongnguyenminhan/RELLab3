"""
Policy Network Implementation.

This module implements policy network architectures for actor-critic and policy-based DRL algorithms, as described in "Deep Reinforcement Learning: A Survey" by Wang et al. and the Modular DRL Framework engineering blueprint.

Detailed Description:
The policy network is responsible for mapping observed states to actions, supporting both deterministic and stochastic policies. It is used in algorithms such as DDPG, PPO, and SAC. The design is modular, allowing for easy customization of network depth, activation functions, and output distributions. Configuration is managed via YAML files for reproducibility and experiment tracking.

Key Concepts/Algorithms:
- Deterministic and stochastic policy parameterization
- Action sampling and squashing (e.g., tanh for bounded actions)
- Support for continuous and discrete action spaces
- Gaussian and categorical policy distributions
- PyTorch-based modular implementation with BaseNetwork inheritance

Important Parameters/Configurations:
- Network architecture (hidden layer sizes, activations)
- Policy type (deterministic, stochastic, categorical)
- Output distribution parameters (mean, std for Gaussian)
- Action bounds and scaling for continuous actions
- All parameters are loaded from the relevant YAML config (e.g., `configs/ppo_config.yaml`)

Expected Inputs/Outputs:
- Inputs: state (torch.Tensor) - shape (batch_size, state_dim)
- Outputs: actions, log probabilities, entropy (depending on policy type)

Dependencies:
- torch: Neural network computations and distributions
- numpy: Numerical operations
- Base classes from networks.__init__

Author: REL Project Team
Date: 2025-07-13
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, Independent
from typing import Dict, Any, List, Optional, Tuple
from . import BaseNetwork
import numpy as np


class PolicyNetwork(BaseNetwork):
    """
    Policy Network for Actor-Critic and Policy-Based Deep Reinforcement Learning.

    This class implements policy networks that map states to actions for both continuous
    and discrete action spaces. It supports deterministic and stochastic policies with
    various probability distributions, inheriting from BaseNetwork for common functionality.

    Detailed Description:
    The Policy Network is a core component in policy-based and actor-critic DRL algorithms.
    For discrete action spaces, it outputs action probabilities using a categorical
    distribution. For continuous action spaces, it can output deterministic actions or
    parameterize a Gaussian distribution with mean and standard deviation. The network
    architecture is highly configurable through YAML configuration files.

    Key Features:
    - Support for both discrete and continuous action spaces
    - Deterministic and stochastic policy modes
    - Configurable probability distributions (Gaussian, Categorical)
    - Action bounds enforcement for continuous actions
    - Entropy computation for exploration
    - Log probability calculation for policy gradient methods

    Policy Types:
    - Deterministic: Direct action output (used in DDPG)
    - Gaussian: Continuous actions with learnable/fixed standard deviation
    - Categorical: Discrete action probabilities (used in PPO, A2C)

    Args:
        input_size (int): Dimension of the state space
        output_size (int): Dimension of the action space
        config_path (Optional[str]): Path to YAML configuration file
        action_space_type (str): Type of action space ('continuous' or 'discrete')
        device (str): Computing device ('cpu', 'cuda', etc.)
        **kwargs: Additional arguments passed to BaseNetwork

    Attributes:
        action_space_type (str): Type of action space
        policy_type (str): Type of policy (deterministic, stochastic, categorical)
        action_bounds (Tuple): Min and max action values for continuous spaces
        policy_head (nn.Sequential): Main policy network
        log_std (nn.Parameter): Learnable log standard deviation for Gaussian policies
        
    Expected Usage:
    ```python
    # Continuous action space (DDPG, SAC)
    policy_net = PolicyNetwork(
        input_size=8,
        output_size=2,
        action_space_type='continuous',
        config_path="configs/ddpg_config.yaml"
    )
    
    # Discrete action space (PPO, A2C)
    policy_net = PolicyNetwork(
        input_size=4,
        output_size=3,
        action_space_type='discrete',
        config_path="configs/ppo_config.yaml"
    )
    
    states = torch.randn(32, 8)
    actions, log_probs, entropy = policy_net.sample_action(states)
    ```

    Configuration Parameters:
    - hidden_sizes: List of hidden layer dimensions
    - activation: Activation function name
    - policy_type: 'deterministic', 'stochastic', or 'categorical'
    - log_std_init: Initial log standard deviation for Gaussian policies
    - learnable_std: Whether standard deviation is learnable
    - action_bounds: [min, max] for continuous actions

    Author: REL Project Team
    Date: 2025-07-13
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        action_space_type: str = 'continuous',
        config_path: Optional[str] = None,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize the Policy Network.

        Args:
            input_size: Dimension of the state space
            output_size: Dimension of the action space
            action_space_type: Type of action space ('continuous' or 'discrete')
            config_path: Path to YAML configuration file
            device: Computing device
            **kwargs: Additional arguments
        """
        self.action_space_type = action_space_type.lower()
        
        super(PolicyNetwork, self).__init__(
            input_size=input_size,
            output_size=output_size,
            config_path=config_path,
            device=device,
            network_name="PolicyNetwork",
            **kwargs
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for Policy Network.

        Returns:
            Dict containing default Policy Network parameters
        """
        base_config = {
            'hidden_sizes': [256, 256],
            'activation': 'relu',
            'dropout': 0.0,
            'batch_norm': False,
            'initialization': 'xavier_uniform'
        }
        
        if self.action_space_type == 'continuous':
            base_config.update({
                'policy_type': 'stochastic',  # 'deterministic' or 'stochastic'
                'log_std_init': -0.5,
                'learnable_std': True,
                'action_bounds': [-1.0, 1.0],
                'squash_actions': True  # Use tanh to bound actions
            })
        else:  # discrete
            base_config.update({
                'policy_type': 'categorical'
            })
        
        return base_config

    def _build_architecture(self) -> None:
        """Build the Policy Network architecture based on configuration."""
        hidden_sizes = self.config.get('hidden_sizes', [256, 256])
        activation = self.config.get('activation', 'relu')
        dropout = self.config.get('dropout', 0.0)
        batch_norm = self.config.get('batch_norm', False)
        
        self.policy_type = self.config.get('policy_type', 'stochastic')
        
        if self.action_space_type == 'continuous':
            self._build_continuous_policy(hidden_sizes, activation, dropout, batch_norm)
        else:
            self._build_discrete_policy(hidden_sizes, activation, dropout, batch_norm)

    def _build_continuous_policy(
        self, 
        hidden_sizes: List[int], 
        activation: str, 
        dropout: float, 
        batch_norm: bool
    ) -> None:
        """Build policy network for continuous action spaces."""
        self.action_bounds = tuple(self.config.get('action_bounds', [-1.0, 1.0]))
        self.squash_actions = self.config.get('squash_actions', True)
        
        if self.policy_type == 'deterministic':
            # Deterministic policy outputs actions directly
            self.policy_head = self.create_mlp(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm,
                output_activation='tanh' if self.squash_actions else None
            )
        else:
            # Stochastic policy outputs mean (and optionally std)
            self.policy_head = self.create_mlp(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm
            )
            
            # Standard deviation parameter
            learnable_std = self.config.get('learnable_std', True)
            log_std_init = self.config.get('log_std_init', -0.5)
            
            if learnable_std:
                # Learnable log standard deviation
                self.log_std = nn.Parameter(
                    torch.ones(self.output_size) * log_std_init
                )
            else:
                # Fixed log standard deviation
                self.register_buffer(
                    'log_std', 
                    torch.ones(self.output_size) * log_std_init
                )

    def _build_discrete_policy(
        self, 
        hidden_sizes: List[int], 
        activation: str, 
        dropout: float, 
        batch_norm: bool
    ) -> None:
        """Build policy network for discrete action spaces."""
        # Categorical policy outputs logits for each action
        self.policy_head = self.create_mlp(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Policy Network.

        Args:
            x: Input state tensor of shape (batch_size, input_size)

        Returns:
            Policy output (actions for deterministic, logits/mean for stochastic)
        """
        return self.policy_head(x)

    def get_action_distribution(self, states: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the action distribution for given states.

        Args:
            states: Batch of states

        Returns:
            Torch distribution object
        """
        if self.action_space_type == 'continuous':
            if self.policy_type == 'deterministic':
                raise ValueError("Cannot get distribution for deterministic policy")
            
            mean = self.forward(states)
            
            if self.squash_actions:
                mean = torch.tanh(mean)
            
            std = torch.exp(self.log_std).expand_as(mean)
            return Independent(Normal(mean, std), 1)
        
        else:  # discrete
            logits = self.forward(states)
            return Categorical(logits=logits)

    def sample_action(
        self, 
        states: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.

        Args:
            states: Batch of states
            deterministic: Whether to use deterministic action selection

        Returns:
            Tuple of (actions, log_probabilities, entropy)
        """
        if self.action_space_type == 'continuous':
            return self._sample_continuous_action(states, deterministic)
        else:
            return self._sample_discrete_action(states, deterministic)

    def _sample_continuous_action(
        self, 
        states: torch.Tensor, 
        deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample continuous actions."""
        if self.policy_type == 'deterministic':
            actions = self.forward(states)
            # Scale actions to bounds if not using tanh squashing
            if not self.squash_actions:
                actions = self._scale_actions(actions)
            log_probs = torch.zeros(states.shape[0], device=self.device)
            entropy = torch.zeros(states.shape[0], device=self.device)
            return actions, log_probs, entropy
        
        else:  # stochastic
            mean = self.forward(states)
            std = torch.exp(self.log_std).expand_as(mean)
            
            if deterministic:
                actions = mean
            else:
                dist = Normal(mean, std)
                actions = dist.rsample()  # Reparameterization trick
            
            # Apply tanh squashing if enabled
            if self.squash_actions:
                # Compute log prob before squashing
                log_probs = Normal(mean, std).log_prob(actions).sum(dim=-1)
                # Apply tanh squashing
                actions = torch.tanh(actions)
                # Correct log prob for tanh transformation
                log_probs -= torch.sum(
                    torch.log(1 - actions.pow(2) + 1e-7), dim=-1
                )
            else:
                log_probs = Normal(mean, std).log_prob(actions).sum(dim=-1)
                actions = self._scale_actions(actions)
            
            # Compute entropy
            entropy = Normal(mean, std).entropy().sum(dim=-1)
            
            return actions, log_probs, entropy

    def _sample_discrete_action(
        self, 
        states: torch.Tensor, 
        deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample discrete actions."""
        logits = self.forward(states)
        dist = Categorical(logits=logits)
        
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return actions, log_probs, entropy

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """
        Get single action for a given state.

        Args:
            state: Single state tensor
            deterministic: Whether to use deterministic action selection

        Returns:
            Action as numpy array
        """
        with torch.no_grad():
            actions, _, _ = self.sample_action(state.unsqueeze(0), deterministic)
            return actions.squeeze(0).cpu().numpy()

    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given state-action pairs.

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            Tuple of (log_probabilities, entropy)
        """
        if self.action_space_type == 'continuous':
            if self.policy_type == 'deterministic':
                raise ValueError("Cannot evaluate actions for deterministic policy")
            
            mean = self.forward(states)
            std = torch.exp(self.log_std).expand_as(mean)
            
            # Add numerical stability
            std = torch.clamp(std, min=1e-6, max=10.0)
            
            if self.squash_actions:
                # Clamp actions to valid range for atanh
                actions_clamped = torch.clamp(actions, -0.9999, 0.9999)
                actions_pretanh = torch.atanh(actions_clamped)
                
                # Compute log probabilities with better numerical stability
                normal_dist = Normal(mean, std)
                log_probs = normal_dist.log_prob(actions_pretanh).sum(dim=-1)
                
                # Correct for tanh transformation with numerical stability
                correction = torch.sum(
                    torch.log(torch.clamp(1 - actions_clamped.pow(2), min=1e-6)), dim=-1
                )
                log_probs -= correction
            else:
                # Unscale actions if not using tanh
                actions_unscaled = self._unscale_actions(actions)
                normal_dist = Normal(mean, std)
                log_probs = normal_dist.log_prob(actions_unscaled).sum(dim=-1)
            
            # Compute entropy with clamped std
            entropy = Normal(mean, std).entropy().sum(dim=-1)
            
        else:  # discrete
            logits = self.forward(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        
        return log_probs, entropy

    def _scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale actions to action bounds."""
        low, high = self.action_bounds
        return low + (actions + 1.0) * 0.5 * (high - low)

    def _unscale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Unscale actions from action bounds."""
        low, high = self.action_bounds
        return 2.0 * (actions - low) / (high - low) - 1.0

    def get_deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic actions (mean for stochastic policies).

        Args:
            states: Batch of states

        Returns:
            Deterministic actions
        """
        if self.action_space_type == 'continuous':
            if self.policy_type == 'deterministic':
                return self.forward(states)
            else:
                mean = self.forward(states)
                if self.squash_actions:
                    return torch.tanh(mean)
                else:
                    return self._scale_actions(mean)
        else:
            logits = self.forward(states)
            return torch.argmax(logits, dim=-1)

    def compute_kl_divergence(
        self, 
        states: torch.Tensor, 
        old_policy: 'PolicyNetwork'
    ) -> torch.Tensor:
        """
        Compute KL divergence between current and old policy.

        Args:
            states: Batch of states
            old_policy: Old policy network

        Returns:
            KL divergence
        """
        current_dist = self.get_action_distribution(states)
        
        with torch.no_grad():
            old_dist = old_policy.get_action_distribution(states)
        
        return torch.distributions.kl_divergence(current_dist, old_dist)


class ActorNetwork(PolicyNetwork):
    """
    Actor Network for Actor-Critic algorithms.
    
    This is an alias for PolicyNetwork to maintain compatibility with
    actor-critic terminology while providing the same functionality.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Actor Network."""
        super(ActorNetwork, self).__init__(*args, **kwargs)
        self.network_name = "ActorNetwork"