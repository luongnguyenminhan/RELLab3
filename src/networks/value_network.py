"""
Value Network Implementation.

This module implements value network architectures for critic estimation in actor-critic and maximum entropy-based DRL algorithms, as described in "Deep Reinforcement Learning: A Survey" by Wang et al. and the Modular DRL Framework engineering blueprint.

Detailed Description:
The value network estimates the state-value function, which is used for baseline estimation, advantage calculation, and policy evaluation. It is a key component in algorithms such as PPO and SAC. The design is modular and configurable, supporting different network depths and activation functions, with all parameters managed via YAML files for reproducibility.

Key Concepts/Algorithms:
- State-value function approximation V(s)
- Baseline estimation for policy gradients to reduce variance
- Advantage computation: A(s,a) = Q(s,a) - V(s)
- Target value estimation for temporal difference learning
- PyTorch-based modular implementation with BaseNetwork inheritance

Important Parameters/Configurations:
- Network architecture (hidden layer sizes, activations)
- Initialization schemes (Xavier, He, orthogonal)
- Dropout and batch normalization options
- Output activation (typically none for value estimation)
- All parameters are loaded from the relevant YAML config (e.g., `configs/ppo_config.yaml`)

Expected Inputs/Outputs:
- Inputs: state (torch.Tensor) - shape (batch_size, state_dim)
- Outputs: state value (torch.Tensor) - shape (batch_size, 1)

Dependencies:
- torch: Neural network computations
- numpy: Numerical operations
- Base classes from networks.__init__

Author: REL Project Team
Date: 2025-07-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from . import BaseNetwork


class ValueNetwork(BaseNetwork):
    """
    Value Network for State-Value Function Estimation in Deep Reinforcement Learning.

    This class implements a value network that estimates the state-value function V(s),
    representing the expected cumulative reward from a given state under the current policy.
    It inherits from BaseNetwork to leverage common functionality while implementing
    value-specific architecture and computation methods.

    Detailed Description:
    The Value Network is a fundamental component in actor-critic and policy gradient methods.
    It serves multiple purposes: (1) providing baseline estimates to reduce variance in
    policy gradient calculations, (2) computing advantage estimates for improved learning,
    and (3) enabling temporal difference learning through value function approximation.
    The network outputs a single scalar value for each input state.

    Key Features:
    - Configurable MLP architecture with customizable hidden layers
    - Support for different activation functions and regularization
    - Baseline estimation for variance reduction in policy gradients
    - Advantage computation capabilities
    - Target value network support for stable learning
    - Integration with actor-critic and policy gradient algorithms

    Network Architecture:
    - Input: State representation (various dimensionalities supported)
    - Hidden Layers: Configurable depth and width
    - Output: Single scalar value (no activation function typically)
    - Regularization: Optional dropout and batch normalization

    Args:
        input_size (int): Dimension of the state space
        config_path (Optional[str]): Path to YAML configuration file
        device (str): Computing device ('cpu', 'cuda', etc.)
        **kwargs: Additional arguments passed to BaseNetwork

    Attributes:
        value_network (nn.Sequential): The main value network layers
        output_activation (Optional[str]): Output activation function if any

    Expected Usage:
    ```python
    value_net = ValueNetwork(
        input_size=8,
        config_path="configs/ppo_config.yaml"
    )
    
    states = torch.randn(32, 8)  # Batch of states
    values = value_net(states)   # State values
    ```

    Configuration Parameters:
    - hidden_sizes: List of hidden layer dimensions
    - activation: Activation function name
    - dropout: Dropout probability
    - batch_norm: Whether to use batch normalization
    - output_activation: Output layer activation (typically None)
    - initialization: Parameter initialization scheme

    Author: REL Project Team
    Date: 2025-07-13
    """

    def __init__(
        self,
        input_size: int,
        config_path: Optional[str] = None,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize the Value Network.

        Args:
            input_size: Dimension of the state space
            config_path: Path to YAML configuration file
            device: Computing device
            **kwargs: Additional arguments
        """
        # Value networks always output a single scalar value
        super(ValueNetwork, self).__init__(
            input_size=input_size,
            output_size=1,
            config_path=config_path,
            device=device,
            network_name="ValueNetwork",
            **kwargs
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for Value Network.

        Returns:
            Dict containing default Value Network parameters
        """
        return {
            'hidden_sizes': [256, 256],
            'activation': 'relu',
            'dropout': 0.0,
            'batch_norm': False,
            'output_activation': None,  # Typically no activation for value output
            'initialization': 'xavier_uniform'
        }

    def _build_architecture(self) -> None:
        """Build the Value Network architecture based on configuration."""
        hidden_sizes = self.config.get('hidden_sizes', [256, 256])
        activation = self.config.get('activation', 'relu')
        dropout = self.config.get('dropout', 0.0)
        batch_norm = self.config.get('batch_norm', False)
        output_activation = self.config.get('output_activation', None)

        self.value_network = self.create_mlp(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            output_activation=output_activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Value Network.

        Args:
            x: Input state tensor of shape (batch_size, input_size)

        Returns:
            Value estimates tensor of shape (batch_size, 1)
        """
        return self.value_network(x)

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get value estimates for given states.

        Args:
            states: Batch of states

        Returns:
            Value estimates
        """
        return self.forward(states).squeeze(-1)  # Remove last dimension for convenience

    def compute_advantage(
        self, 
        states: torch.Tensor, 
        next_states: torch.Tensor, 
        rewards: torch.Tensor,
        dones: torch.Tensor, 
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Compute advantage estimates using temporal difference.

        Args:
            states: Current states
            next_states: Next states
            rewards: Rewards received
            dones: Done flags
            gamma: Discount factor

        Returns:
            Advantage estimates
        """
        with torch.no_grad():
            current_values = self.get_value(states)
            next_values = self.get_value(next_states)
            
            # TD target: r + gamma * V(s') * (1 - done)
            td_targets = rewards + gamma * next_values * (1 - dones)
            
            # Advantage: TD_target - V(s)
            advantages = td_targets - current_values
            
        return advantages

    def compute_gae_advantages(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lambda_gae: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            states: Current states
            next_states: Next states  
            rewards: Rewards received
            dones: Done flags
            gamma: Discount factor
            lambda_gae: GAE lambda parameter

        Returns:
            Tuple of (advantages, value_targets)
        """
        with torch.no_grad():
            values = self.get_value(states)
            next_values = self.get_value(next_states)
            
            # Compute deltas
            deltas = rewards + gamma * next_values * (1 - dones) - values
            
            # Compute GAE advantages
            advantages = torch.zeros_like(rewards)
            advantage = 0
            
            # Compute advantages backwards through time
            for t in reversed(range(len(rewards))):
                advantage = deltas[t] + gamma * lambda_gae * advantage * (1 - dones[t])
                advantages[t] = advantage
            
            # Value targets for value function update
            value_targets = advantages + values
            
        return advantages, value_targets

    def compute_value_loss(
        self,
        states: torch.Tensor,
        value_targets: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        clip_range: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute value function loss.

        Args:
            states: Current states
            value_targets: Target values
            old_values: Old value estimates (for clipped loss)
            clip_range: Clipping range for PPO-style value loss

        Returns:
            Value loss
        """
        current_values = self.get_value(states)
        
        if clip_range is not None and old_values is not None:
            # PPO-style clipped value loss
            value_clipped = old_values + torch.clamp(
                current_values - old_values, -clip_range, clip_range
            )
            
            loss_unclipped = F.mse_loss(current_values, value_targets)
            loss_clipped = F.mse_loss(value_clipped, value_targets)
            
            value_loss = torch.max(loss_unclipped, loss_clipped)
        else:
            # Standard MSE loss
            value_loss = F.mse_loss(current_values, value_targets)
        
        return value_loss

    def update_target_network(self, target_network: 'ValueNetwork', tau: float = 1.0) -> None:
        """
        Update target network parameters using soft update.

        Args:
            target_network: Target value network to update
            tau: Soft update coefficient (1.0 for hard update)
        """
        for target_param, param in zip(target_network.parameters(), self.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def compute_bellman_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        target_network: Optional['ValueNetwork'] = None
    ) -> torch.Tensor:
        """
        Compute Bellman targets for value function updates.

        Args:
            rewards: Rewards received
            next_states: Next states
            dones: Done flags  
            gamma: Discount factor
            target_network: Target network for stable learning

        Returns:
            Bellman targets
        """
        with torch.no_grad():
            if target_network is not None:
                next_values = target_network.get_value(next_states)
            else:
                next_values = self.get_value(next_states)
            
            targets = rewards + gamma * next_values * (1 - dones)
            
        return targets

    def get_value_single(self, state: torch.Tensor) -> float:
        """
        Get value estimate for a single state.

        Args:
            state: Single state tensor

        Returns:
            Value estimate as float
        """
        with torch.no_grad():
            value = self.forward(state.unsqueeze(0))
            return value.squeeze().item()


class CriticNetwork(ValueNetwork):
    """
    Critic Network for Actor-Critic algorithms.
    
    This is an alias for ValueNetwork to maintain compatibility with
    actor-critic terminology while providing the same functionality.
    In some contexts, critics may have slightly different configurations
    or additional methods specific to the critic role.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Critic Network."""
        super(CriticNetwork, self).__init__(*args, **kwargs)
        self.network_name = "CriticNetwork"


class StateActionValueNetwork(BaseNetwork):
    """
    State-Action Value Network (Q-Network variant) for Critic estimation.
    
    This network estimates Q(s,a) values by taking both states and actions
    as input. It's commonly used in algorithms like SAC and TD3 where the
    critic needs to evaluate state-action pairs.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config_path: Optional[str] = None,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize State-Action Value Network.

        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space
            config_path: Path to YAML configuration file
            device: Computing device
            **kwargs: Additional arguments
        """
        self.state_size = state_size
        self.action_size = action_size
        
        super(StateActionValueNetwork, self).__init__(
            input_size=state_size + action_size,
            output_size=1,
            config_path=config_path,
            device=device,
            network_name="StateActionValueNetwork",
            **kwargs
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for State-Action Value Network."""
        return {
            'hidden_sizes': [256, 256],
            'activation': 'relu',
            'dropout': 0.0,
            'batch_norm': False,
            'output_activation': None,
            'initialization': 'xavier_uniform'
        }

    def _build_architecture(self) -> None:
        """Build the State-Action Value Network architecture."""
        hidden_sizes = self.config.get('hidden_sizes', [256, 256])
        activation = self.config.get('activation', 'relu')
        dropout = self.config.get('dropout', 0.0)
        batch_norm = self.config.get('batch_norm', False)

        self.value_network = self.create_mlp(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the State-Action Value Network.

        Args:
            states: State tensor of shape (batch_size, state_size)
            actions: Action tensor of shape (batch_size, action_size)

        Returns:
            Q-values tensor of shape (batch_size, 1)
        """
        # Concatenate states and actions
        state_action = torch.cat([states, actions], dim=-1)
        return self.value_network(state_action)

    def get_q_value(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for given state-action pairs.

        Args:
            states: Batch of states
            actions: Batch of actions

        Returns:
            Q-values
        """
        return self.forward(states, actions).squeeze(-1)