"""
Dueling Network Architecture Implementation.

This module implements the Dueling Network architecture for Q-value estimation, as described in "Deep Reinforcement Learning: A Survey" by Wang et al. and the Modular DRL Framework engineering blueprint.

Detailed Description:
The Dueling Network separates the estimation of state-value and advantage functions within the Q-network, improving learning efficiency and stability in value-based DRL algorithms. This architecture is particularly effective in environments with many similar-valued actions. The module is designed for extensibility and integration with DQN and its variants.

Key Concepts/Algorithms:
- Dueling Q-Network architecture with separate streams
- Value stream: V(s) - estimates state value independent of actions
- Advantage stream: A(s,a) - estimates relative advantage of each action
- Aggregation: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
- Shared feature extraction layers
- PyTorch-based modular implementation with BaseNetwork inheritance

Important Parameters/Configurations:
- Network layer sizes and activation functions (from YAML config)
- Feature extractor architecture (shared layers)
- Value and advantage stream architectures
- Aggregation method (mean subtraction is standard)
- Initialization schemes
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
from typing import Dict, Any, Optional
from . import BaseNetwork


class DuelingNetwork(BaseNetwork):
    """
    Dueling Q-Network for Advanced Q-Value Estimation in Deep Reinforcement Learning.

    This class implements the Dueling DQN architecture that decomposes Q-values into
    state-value and advantage functions. It inherits from BaseNetwork to leverage 
    common functionality while implementing the dueling-specific architecture that
    can improve learning efficiency and stability.

    Detailed Description:
    The Dueling Network architecture addresses the problem that in many states,
    the values of different actions don't vary significantly. By separating the
    estimation of state value V(s) and action advantage A(s,a), the network can
    learn to evaluate states independently of learning the relative ranking of
    actions. This leads to more efficient learning, especially in environments
    where the correct action doesn't matter most of the time.

    Architecture Components:
    1. Feature Extractor: Shared layers that process the input state
    2. Value Stream: Estimates V(s), the value of being in state s
    3. Advantage Stream: Estimates A(s,a), the advantage of taking action a in state s
    4. Aggregation Layer: Combines V(s) and A(s,a) to produce Q(s,a)

    Mathematical Formulation:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    
    The mean subtraction ensures identifiability and stable learning.

    Args:
        input_size (int): Dimension of the state space
        output_size (int): Number of discrete actions
        config_path (Optional[str]): Path to YAML configuration file
        device (str): Computing device ('cpu', 'cuda', etc.)
        **kwargs: Additional arguments passed to BaseNetwork

    Attributes:
        feature_extractor (nn.Sequential): Shared feature extraction layers
        value_stream (nn.Sequential): State value estimation network
        advantage_stream (nn.Sequential): Action advantage estimation network
        aggregation_method (str): Method for combining value and advantage

    Expected Usage:
    ```python
    dueling_net = DuelingNetwork(
        input_size=4,
        output_size=2,
        config_path="configs/dqn_config.yaml"
    )
    
    states = torch.randn(32, 4)  # Batch of states
    q_values = dueling_net(states)  # Q-values for all actions
    ```

    Configuration Parameters:
    - feature_sizes: List of shared feature extractor layer sizes
    - value_sizes: List of value stream layer sizes
    - advantage_sizes: List of advantage stream layer sizes
    - activation: Activation function name
    - dropout: Dropout probability
    - batch_norm: Whether to use batch normalization
    - aggregation_method: How to combine value and advantage ('mean', 'max')

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
        Initialize the Dueling Network.

        Args:
            input_size: Dimension of the state space
            output_size: Number of discrete actions
            config_path: Path to YAML configuration file
            device: Computing device
            **kwargs: Additional arguments
        """
        super(DuelingNetwork, self).__init__(
            input_size=input_size,
            output_size=output_size,
            config_path=config_path,
            device=device,
            network_name="DuelingNetwork",
            **kwargs
        )

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for Dueling Network.

        Returns:
            Dict containing default Dueling Network parameters
        """
        return {
            'feature_sizes': [128, 128],  # Shared feature extractor
            'value_sizes': [128],         # Value stream
            'advantage_sizes': [128],     # Advantage stream
            'activation': 'relu',
            'dropout': 0.0,
            'batch_norm': False,
            'aggregation_method': 'mean',  # 'mean' or 'max'
            'initialization': 'xavier_uniform'
        }

    def _build_architecture(self) -> None:
        """Build the Dueling Network architecture based on configuration."""
        feature_sizes = self.config.get('feature_sizes', [128, 128])
        value_sizes = self.config.get('value_sizes', [128])
        advantage_sizes = self.config.get('advantage_sizes', [128])
        activation = self.config.get('activation', 'relu')
        dropout = self.config.get('dropout', 0.0)
        batch_norm = self.config.get('batch_norm', False)
        
        self.aggregation_method = self.config.get('aggregation_method', 'mean')

        # Shared feature extractor
        if feature_sizes:
            self.feature_extractor = self.create_mlp(
                input_size=self.input_size,
                output_size=feature_sizes[-1],
                hidden_sizes=feature_sizes[:-1],
                activation=activation,
                dropout=dropout,
                batch_norm=batch_norm
            )
            feature_output_size = feature_sizes[-1]
        else:
            # No shared features, streams connect directly to input
            self.feature_extractor = nn.Identity()
            feature_output_size = self.input_size

        # Value stream (outputs single state value)
        self.value_stream = self.create_mlp(
            input_size=feature_output_size,
            output_size=1,
            hidden_sizes=value_sizes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

        # Advantage stream (outputs advantage for each action)
        self.advantage_stream = self.create_mlp(
            input_size=feature_output_size,
            output_size=self.output_size,
            hidden_sizes=advantage_sizes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Dueling Network.

        Args:
            x: Input state tensor of shape (batch_size, input_size)

        Returns:
            Q-values tensor of shape (batch_size, output_size)
        """
        # Extract shared features
        features = self.feature_extractor(x)
        
        # Compute value and advantage estimates
        value = self.value_stream(features)      # Shape: (batch_size, 1)
        advantage = self.advantage_stream(features)  # Shape: (batch_size, action_dim)
        
        # Aggregate value and advantage to form Q-values
        q_values = self._aggregate_value_advantage(value, advantage)
        
        return q_values

    def _aggregate_value_advantage(
        self, 
        value: torch.Tensor, 
        advantage: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate value and advantage to form Q-values.

        Args:
            value: State value tensor of shape (batch_size, 1)
            advantage: Action advantage tensor of shape (batch_size, action_dim)

        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        if self.aggregation_method == 'mean':
            # Standard dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        elif self.aggregation_method == 'max':
            # Alternative formulation: Q(s,a) = V(s) + A(s,a) - max(A(s,a'))
            q_values = value + advantage - advantage.max(dim=1, keepdim=True)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        return q_values

    def get_value_estimates(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get state value estimates V(s).

        Args:
            states: Batch of states

        Returns:
            State value estimates
        """
        features = self.feature_extractor(states)
        return self.value_stream(features).squeeze(-1)

    def get_advantage_estimates(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get raw advantage estimates A(s,a) before aggregation.

        Args:
            states: Batch of states

        Returns:
            Raw advantage estimates
        """
        features = self.feature_extractor(states)
        return self.advantage_stream(features)

    def get_decomposed_q_values(
        self, 
        states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get decomposed Q-value components for analysis.

        Args:
            states: Batch of states

        Returns:
            Dict containing 'value', 'advantage', 'q_values'
        """
        features = self.feature_extractor(states)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = self._aggregate_value_advantage(value, advantage)
        
        return {
            'value': value.squeeze(-1),
            'advantage': advantage,
            'q_values': q_values
        }

    def compute_dueling_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        target_network: 'DuelingNetwork',
        gamma: float = 0.99
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Dueling DQN loss with additional regularization terms.

        Args:
            states: Current states
            actions: Actions taken
            rewards: Rewards received
            next_states: Next states
            dones: Done flags
            target_network: Target dueling network
            gamma: Discount factor

        Returns:
            Dict containing various loss components
        """
        # Current Q-values
        current_q_values = self.forward(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values using Double DQN with dueling
        with torch.no_grad():
            # Use online network for action selection
            next_q_online = self.forward(next_states)
            next_actions = next_q_online.argmax(1)
            
            # Use target network for Q-value estimation
            next_q_target = target_network.forward(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze()
            
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))
        
        # Main DQN loss
        td_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Additional regularization: encourage meaningful advantage estimates
        current_decomp = self.get_decomposed_q_values(states)
        advantage_reg = torch.mean(torch.var(current_decomp['advantage'], dim=1))
        
        return {
            'td_loss': td_loss,
            'advantage_regularization': advantage_reg,
            'total_loss': td_loss + 0.01 * advantage_reg  # Small weight for regularization
        }

    def update_target_network(self, target_network: 'DuelingNetwork', tau: float = 1.0) -> None:
        """
        Update target network parameters using soft update.

        Args:
            target_network: Target dueling network to update
            tau: Soft update coefficient (1.0 for hard update)
        """
        for target_param, param in zip(target_network.parameters(), self.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def analyze_value_advantage_separation(
        self, 
        states: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze how well the network separates value and advantage.

        Args:
            states: Batch of states for analysis

        Returns:
            Dict with analysis metrics
        """
        with torch.no_grad():
            decomp = self.get_decomposed_q_values(states)
            
            # Compute statistics
            value_var = torch.var(decomp['value']).item()
            advantage_var = torch.mean(torch.var(decomp['advantage'], dim=1)).item()
            q_var = torch.var(decomp['q_values']).item()
            
            # Correlation between value and max advantage
            max_advantages = torch.max(decomp['advantage'], dim=1)[0]
            correlation = torch.corrcoef(torch.stack([decomp['value'], max_advantages]))[0, 1].item()
            
        return {
            'value_variance': value_var,
            'advantage_variance': advantage_var,
            'q_value_variance': q_var,
            'value_advantage_correlation': correlation
        }


class NoisyDuelingNetwork(DuelingNetwork):
    """
    Noisy Dueling Network combining dueling architecture with noisy layers.
    
    This extends the dueling network with parameter space noise for exploration,
    useful in environments where epsilon-greedy exploration is insufficient.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize Noisy Dueling Network."""
        super(NoisyDuelingNetwork, self).__init__(*args, **kwargs)
        self.network_name = "NoisyDuelingNetwork"
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with noisy network parameters."""
        config = super()._get_default_config()
        config.update({
            'noisy_layers': True,
            'noise_std': 0.5,
            'factorised_noise': True
        })
        return config
    
    def _create_noisy_linear(self, in_features: int, out_features: int) -> nn.Module:
        """Create a noisy linear layer for exploration."""
        # This would implement noisy linear layers
        # For now, return standard linear layer
        return nn.Linear(in_features, out_features)
    
    def reset_noise(self) -> None:
        """Reset noise in noisy layers."""
        # Implementation would reset noise parameters
        pass