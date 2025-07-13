"""
Q-Network Implementation.

This module implements Q-network architectures for value-based DRL algorithms, as described in "Deep Reinforcement Learning: A Survey" by Wang et al. and the Modular DRL Framework engineering blueprint.

Detailed Description:
The Q-network estimates the action-value function, mapping state-action pairs to expected returns. It is a core component of algorithms such as DQN and its variants. The module is designed for flexibility, supporting different network depths, activation functions, and integration with target networks. All configurations are managed via YAML files for reproducibility.

Key Concepts/Algorithms:
- Q-value function approximation using deep neural networks
- Support for both discrete and continuous action spaces
- Target network compatibility for stable learning
- Configurable architecture via YAML configuration
- PyTorch-based modular implementation with BaseNetwork inheritance

Important Parameters/Configurations:
- Network architecture (hidden layer sizes, activations)
- Initialization schemes (Xavier, He, orthogonal)
- Dropout and batch normalization options
- All parameters are loaded from the relevant YAML config (e.g., `configs/dqn_config.yaml`)

Expected Inputs/Outputs:
- Inputs: state (torch.Tensor) - shape (batch_size, state_dim)
- Outputs: Q-values for all actions (torch.Tensor) - shape (batch_size, action_dim)

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
from typing import Dict, Any, List, Optional
from . import BaseNetwork


class QNetwork(BaseNetwork):
    """
    Q-Network for Value-Based Deep Reinforcement Learning.

    This class implements a Q-network that estimates action-value functions for discrete
    action spaces. It inherits from BaseNetwork to leverage common functionality while
    implementing Q-network specific architecture and forward pass logic.

    Detailed Description:
    The Q-Network is a fundamental component in value-based DRL algorithms like DQN.
    It approximates the Q-function Q(s,a) which represents the expected cumulative
    reward when taking action 'a' in state 's' and following the current policy
    thereafter. This implementation supports various network architectures through
    configuration files and provides flexibility for different state and action spaces.

    Key Features:
    - Configurable MLP architecture with customizable hidden layers
    - Support for different activation functions
    - Optional dropout and batch normalization
    - Target network compatibility
    - Gradient clipping support
    - Dueling architecture option (when enabled in config)

    Architecture Options:
    - Standard Q-Network: Direct mapping from states to Q-values
    - Dueling Q-Network: Separate value and advantage streams
    - Noisy Networks: Parameter space noise for exploration

    Args:
        input_size (int): Dimension of the state space
        output_size (int): Number of discrete actions
        config_path (Optional[str]): Path to YAML configuration file
        device (str): Computing device ('cpu', 'cuda', etc.)
        **kwargs: Additional arguments passed to BaseNetwork

    Attributes:
        q_network (nn.Sequential): The main Q-network layers
        dueling_enabled (bool): Whether dueling architecture is enabled
        value_stream (nn.Sequential): Value stream for dueling networks
        advantage_stream (nn.Sequential): Advantage stream for dueling networks

    Expected Usage:
    ```python
    q_net = QNetwork(
        input_size=4,
        output_size=2,
        config_path="configs/dqn_config.yaml"
    )
    
    states = torch.randn(32, 4)  # Batch of states
    q_values = q_net(states)     # Q-values for all actions
    ```

    Configuration Parameters:
    - hidden_sizes: List of hidden layer dimensions
    - activation: Activation function name
    - dropout: Dropout probability
    - batch_norm: Whether to use batch normalization
    - dueling: Whether to use dueling architecture
    - initialization: Parameter initialization scheme

    Author: REL Project Team
    Date: 2025-07-13
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config_path: Optional[str] = None,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize the Q-Network.

        Args:
            input_size: Dimension of the state space
            output_size: Number of discrete actions
            config_path: Path to YAML configuration file
            device: Computing device
            **kwargs: Additional arguments
        """
        super(QNetwork, self).__init__(
            input_size=input_size,
            output_size=output_size,
            config_path=config_path,
            device=device,
            network_name="QNetwork",
            **kwargs
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for Q-Network.

        Returns:
            Dict containing default Q-Network parameters
        """
        return {
            'hidden_sizes': [128, 128],
            'activation': 'relu',
            'dropout': 0.0,
            'batch_norm': False,
            'dueling': False,
            'initialization': 'xavier_uniform',
            'gradient_clipping': None
        }

    def _build_architecture(self) -> None:
        """Build the Q-Network architecture based on configuration."""
        hidden_sizes = self.config.get('hidden_sizes', [128, 128])
        activation = self.config.get('activation', 'relu')
        dropout = self.config.get('dropout', 0.0)
        batch_norm = self.config.get('batch_norm', False)
        self.dueling_enabled = self.config.get('dueling', False)

        if self.dueling_enabled:
            self._build_dueling_architecture(hidden_sizes, activation, dropout, batch_norm)
        else:
            self._build_standard_architecture(hidden_sizes, activation, dropout, batch_norm)

    def _build_standard_architecture(
        self, 
        hidden_sizes: List[int], 
        activation: str, 
        dropout: float, 
        batch_norm: bool
    ) -> None:
        """Build standard Q-Network architecture."""
        self.q_network = self.create_mlp(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def _build_dueling_architecture(
        self, 
        hidden_sizes: List[int], 
        activation: str, 
        dropout: float, 
        batch_norm: bool
    ) -> None:
        """Build dueling Q-Network architecture with separate value and advantage streams."""
        # Shared feature extractor
        feature_size = hidden_sizes[-1] if hidden_sizes else self.input_size
        
        if len(hidden_sizes) > 1:
            self.feature_extractor = self.create_mlp(
                input_size=self.input_size,
                output_size=feature_size,
                hidden_sizes=hidden_sizes[:-1],
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm
            )
        else:
            self.feature_extractor = nn.Identity()
            feature_size = self.input_size

        # Value stream (outputs single value)
        self.value_stream = self.create_mlp(
            input_size=feature_size,
            output_size=1,
            hidden_sizes=[hidden_sizes[-1]] if hidden_sizes else [64],
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

        # Advantage stream (outputs advantage for each action)
        self.advantage_stream = self.create_mlp(
            input_size=feature_size,
            output_size=self.output_size,
            hidden_sizes=[hidden_sizes[-1]] if hidden_sizes else [64],
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-Network.

        Args:
            x: Input state tensor of shape (batch_size, input_size)

        Returns:
            Q-values tensor of shape (batch_size, output_size)
        """
        if self.dueling_enabled:
            return self._forward_dueling(x)
        else:
            return self._forward_standard(x)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        """Standard Q-Network forward pass."""
        return self.q_network(x)

    def _forward_dueling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dueling Q-Network forward pass.
        
        Combines value and advantage streams using the dueling formula:
        Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        """
        features = self.feature_extractor(x)
        
        # Get value and advantage estimates
        value = self.value_stream(features)  # Shape: (batch_size, 1)
        advantage = self.advantage_stream(features)  # Shape: (batch_size, action_dim)
        
        # Dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

    def get_q_values(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for given states.

        Args:
            states: Batch of states

        Returns:
            Q-values for all actions
        """
        return self.forward(states)

    def get_max_q_values(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get maximum Q-values for given states.

        Args:
            states: Batch of states

        Returns:
            Maximum Q-values across actions
        """
        q_values = self.forward(states)
        return torch.max(q_values, dim=1)[0]

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.

        Args:
            state: Single state tensor
            epsilon: Exploration probability

        Returns:
            Selected action index
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.output_size, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return torch.argmax(q_values, dim=1).item()

    def update_target_network(self, target_network: 'QNetwork', tau: float = 1.0) -> None:
        """
        Update target network parameters using soft update.

        Args:
            target_network: Target Q-network to update
            tau: Soft update coefficient (1.0 for hard update)
        """
        for target_param, param in zip(target_network.parameters(), self.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def compute_loss(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor,
        next_states: torch.Tensor, 
        dones: torch.Tensor, 
        target_network: 'QNetwork',
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Compute Q-learning loss (Bellman equation).

        Args:
            states: Current states
            actions: Actions taken
            rewards: Rewards received
            next_states: Next states
            dones: Done flags
            target_network: Target Q-network
            gamma: Discount factor

        Returns:
            MSE loss between current and target Q-values
        """
        # Current Q-values
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = target_network.forward(next_states).max(1)[0]
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        return loss


class DoubleDQNNetwork(QNetwork):
    """
    Double DQN Network implementation.
    
    Extends QNetwork to support Double DQN algorithm which reduces
    overestimation bias by using the online network for action selection
    and the target network for Q-value estimation.
    """
    
    def compute_double_dqn_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        target_network: 'QNetwork',
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Compute Double DQN loss.
        
        Args:
            states: Current states
            actions: Actions taken
            rewards: Rewards received
            next_states: Next states
            dones: Done flags
            target_network: Target Q-network
            gamma: Discount factor
            
        Returns:
            MSE loss for Double DQN
        """
        # Current Q-values
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use online network for action selection
        with torch.no_grad():
            # Online network selects actions
            next_actions = self.forward(next_states).argmax(1)
            # Target network evaluates the selected actions
            next_q_values = target_network.forward(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        return loss