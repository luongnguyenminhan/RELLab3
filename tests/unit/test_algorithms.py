"""
Unit Tests for Algorithms.

This module contains comprehensive unit tests for the core DRL algorithm implementations in the 
Modular DRL Framework, as described in the DRL survey and project engineering blueprint.

Detailed Description:
Tests in this module validate the correctness, stability, and reproducibility of value-based, 
policy-based, and maximum entropy-based algorithms (e.g., DQN, PPO, DDPG, SAC). Each test targets 
algorithm logic, update steps, and edge cases using mock data and controlled random seeds. The 
structure supports extensibility for new algorithms and improvements.

Key Concepts/Algorithms:
- Isolated testing of algorithm update logic
- Mocking of environment and network outputs
- Reproducibility checks via random seeds
- Parameter initialization and configuration loading
- Training step validation and convergence checks

Important Parameters/Configurations:
- Test configuration files (if needed)
- Random seed specification for reproducibility
- Mock environment and network configurations

Expected Inputs/Outputs:
- Inputs: mock transitions, hyperparameters, network outputs
- Outputs: pass/fail results, coverage reports

Dependencies:
- pytest, unittest, numpy, torch, mock (if needed)

Author: REL Project Team
Date: 2025-07-13
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import torch
import tempfile
import yaml
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from algorithms.dqn import DQNAgent
    from algorithms.ppo import PPOAgent
    from algorithms.ddpg import DDPGAgent
    from algorithms.sac import SACAgent
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

# Test configurations
TEST_DQN_CONFIG = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'buffer_size': 10000,
    'batch_size': 32,
    'target_update': 100,
    'double_dqn': False,
    'dueling': False,
    'hidden_sizes': [64, 64]
}

TEST_PPO_CONFIG = {
    'learning_rate': 0.0003,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_ratio': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'epochs': 4,
    'batch_size': 64,
    'hidden_sizes': [64, 64]
}

TEST_DDPG_CONFIG = {
    'learning_rate_actor': 0.0001,
    'learning_rate_critic': 0.001,
    'gamma': 0.99,
    'tau': 0.005,
    'buffer_size': 100000,
    'batch_size': 64,
    'noise_std': 0.2,
    'noise_clip': 0.5,
    'hidden_sizes': [64, 64]
}

TEST_SAC_CONFIG = {
    'learning_rate': 0.0003,
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.2,
    'automatic_entropy_tuning': True,
    'buffer_size': 100000,
    'batch_size': 64,
    'hidden_sizes': [64, 64]
}


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Algorithm imports not available")
class TestAlgorithmBase(unittest.TestCase):
    """Base test class for common algorithm testing utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.state_size = 4
        self.action_size = 2
        self.device = 'cpu'
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
    def create_temp_config(self, config_dict: Dict[str, Any]) -> str:
        """Create a temporary configuration file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name
        
    def generate_mock_experience(self, batch_size: int = 32):
        """Generate mock experience tuples for testing."""
        states = np.random.randn(batch_size, self.state_size).astype(np.float32)
        actions = np.random.randint(0, self.action_size, batch_size)
        rewards = np.random.randn(batch_size).astype(np.float32)
        next_states = np.random.randn(batch_size, self.state_size).astype(np.float32)
        dones = np.random.choice([True, False], batch_size)
        
        return states, actions, rewards, next_states, dones
        
    def tearDown(self):
        """Clean up after tests."""
        # Clean up any temporary files if needed
        pass


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="DQN imports not available")
class TestDQNAgent(TestAlgorithmBase):
    """Test cases for DQN Agent."""
    
    def setUp(self):
        """Set up DQN test environment."""
        super().setUp()
        self.config_file = self.create_temp_config(TEST_DQN_CONFIG)
        
    def test_dqn_initialization(self):
        """Test DQN agent initialization."""
        with patch('algorithms.dqn.QNetwork'), \
             patch('algorithms.dqn.ReplayBuffer'):
            
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            self.assertEqual(agent.state_size, self.state_size)
            self.assertEqual(agent.action_size, self.action_size)
            self.assertEqual(agent.epsilon, TEST_DQN_CONFIG['epsilon_start'])
            
    def test_dqn_config_loading(self):
        """Test configuration loading."""
        with patch('algorithms.dqn.QNetwork'), \
             patch('algorithms.dqn.ReplayBuffer'):
            
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            self.assertEqual(agent.config['learning_rate'], TEST_DQN_CONFIG['learning_rate'])
            self.assertEqual(agent.config['gamma'], TEST_DQN_CONFIG['gamma'])
            self.assertEqual(agent.config['batch_size'], TEST_DQN_CONFIG['batch_size'])
            
    def test_dqn_action_selection(self):
        """Test action selection (epsilon-greedy)."""
        with patch('algorithms.dqn.QNetwork') as mock_network, \
             patch('algorithms.dqn.ReplayBuffer'):
            
            # Mock Q-network output
            mock_q_values = torch.tensor([[1.0, 2.0, 0.5]])
            mock_network.return_value.forward.return_value = mock_q_values
            
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=3,
                config_path=self.config_file,
                device=self.device
            )
            
            # Test greedy action (epsilon = 0)
            agent.epsilon = 0.0
            state = np.random.randn(self.state_size).astype(np.float32)
            action = agent.act(state)
            self.assertEqual(action, 1)  # Should select action with highest Q-value
            
    def test_dqn_experience_storage(self):
        """Test experience storage in replay buffer."""
        with patch('algorithms.dqn.QNetwork'), \
             patch('algorithms.dqn.ReplayBuffer') as mock_buffer:
            
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            # Test storing experience
            state = np.random.randn(self.state_size).astype(np.float32)
            action = 1
            reward = 1.0
            next_state = np.random.randn(self.state_size).astype(np.float32)
            done = False
            
            agent.step(state, action, reward, next_state, done)
            
            # Verify buffer.add was called
            mock_buffer.return_value.add.assert_called_once()
            
    def test_dqn_learning_step(self):
        """Test DQN learning step."""
        with patch('algorithms.dqn.QNetwork') as mock_network, \
             patch('algorithms.dqn.ReplayBuffer') as mock_buffer:
            
            # Mock network outputs
            mock_q_values = torch.tensor([[1.0, 2.0]])
            mock_target_q_values = torch.tensor([[1.5, 1.8]])
            mock_network.return_value.forward.return_value = mock_q_values
            
            # Mock buffer sample
            states, actions, rewards, next_states, dones = self.generate_mock_experience()
            mock_buffer.return_value.sample.return_value = (
                torch.tensor(states), torch.tensor(actions),
                torch.tensor(rewards), torch.tensor(next_states),
                torch.tensor(dones)
            )
            mock_buffer.return_value.__len__.return_value = 1000  # Sufficient for learning
            
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            # Test learning step
            loss = agent.learn()
            
            # Verify learning occurred
            self.assertIsInstance(loss, (float, torch.Tensor))
            
    def test_dqn_target_network_update(self):
        """Test target network update."""
        with patch('algorithms.dqn.QNetwork') as mock_network, \
             patch('algorithms.dqn.ReplayBuffer'):
            
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            # Test target network update
            agent.update_target_network()
            
            # Should call load_state_dict
            self.assertTrue(hasattr(agent, 'target_network'))
            
    def test_dqn_epsilon_decay(self):
        """Test epsilon decay mechanism."""
        with patch('algorithms.dqn.QNetwork'), \
             patch('algorithms.dqn.ReplayBuffer'):
            
            agent = DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            initial_epsilon = agent.epsilon
            agent.decay_epsilon()
            
            # Epsilon should decrease
            self.assertLess(agent.epsilon, initial_epsilon)
            self.assertGreaterEqual(agent.epsilon, TEST_DQN_CONFIG['epsilon_end'])
            
    def tearDown(self):
        """Clean up DQN test environment."""
        super().tearDown()
        os.unlink(self.config_file)
@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="PPO imports not available")
class TestPPOAgent(TestAlgorithmBase):
    """Test cases for PPO Agent."""
    
    def setUp(self):
        """Set up PPO test environment."""
        super().setUp()
        self.config_file = self.create_temp_config(TEST_PPO_CONFIG)
        
    def test_ppo_initialization(self):
        """Test PPO agent initialization."""
        with patch('algorithms.ppo.PolicyNetwork'), \
             patch('algorithms.ppo.ValueNetwork'):
            
            agent = PPOAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            self.assertEqual(agent.state_size, self.state_size)
            self.assertEqual(agent.action_size, self.action_size)
            
    def test_ppo_action_selection(self):
        """Test PPO action selection with policy network."""
        with patch('algorithms.ppo.PolicyNetwork') as mock_policy, \
             patch('algorithms.ppo.ValueNetwork'):
            
            # Mock policy output
            mock_action_probs = torch.tensor([[0.3, 0.7]])
            mock_policy.return_value.forward.return_value = mock_action_probs
            
            agent = PPOAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            state = np.random.randn(self.state_size).astype(np.float32)
            action, log_prob = agent.act(state)
            
            self.assertIsInstance(action, int)
            self.assertIsInstance(log_prob, torch.Tensor)
            
    def test_ppo_gae_computation(self):
        """Test Generalized Advantage Estimation (GAE) computation."""
        with patch('algorithms.ppo.PolicyNetwork'), \
             patch('algorithms.ppo.ValueNetwork') as mock_value:
            
            # Mock value function outputs
            mock_values = torch.tensor([1.0, 1.2, 0.8, 1.5])
            mock_value.return_value.forward.return_value = mock_values
            
            agent = PPOAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            # Mock trajectory data
            rewards = torch.tensor([1.0, 0.5, -0.2, 1.0])
            dones = torch.tensor([False, False, False, True])
            
            advantages = agent.compute_gae(rewards, mock_values, dones)
            
            self.assertEqual(len(advantages), len(rewards))
            self.assertIsInstance(advantages, torch.Tensor)
            
    def test_ppo_policy_loss(self):
        """Test PPO policy loss computation with clipping."""
        with patch('algorithms.ppo.PolicyNetwork'), \
             patch('algorithms.ppo.ValueNetwork'):
            
            agent = PPOAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            # Mock data for loss computation
            old_log_probs = torch.tensor([0.5, 0.3, 0.7])
            new_log_probs = torch.tensor([0.6, 0.2, 0.8])
            advantages = torch.tensor([1.0, -0.5, 0.8])
            
            policy_loss = agent.compute_policy_loss(old_log_probs, new_log_probs, advantages)
            
            self.assertIsInstance(policy_loss, torch.Tensor)
            self.assertEqual(policy_loss.shape, ())  # Should be scalar
            
    def tearDown(self):
        """Clean up PPO test environment."""
        super().tearDown()
        os.unlink(self.config_file)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="DDPG imports not available")
class TestDDPGAgent(TestAlgorithmBase):
    """Test cases for DDPG Agent."""
    
    def setUp(self):
        """Set up DDPG test environment."""
        super().setUp()
        self.config_file = self.create_temp_config(TEST_DDPG_CONFIG)
        
    def test_ddpg_initialization(self):
        """Test DDPG agent initialization."""
        with patch('algorithms.ddpg.PolicyNetwork'), \
             patch('algorithms.ddpg.QNetwork'), \
             patch('algorithms.ddpg.ReplayBuffer'):
            
            agent = DDPGAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            self.assertEqual(agent.state_size, self.state_size)
            self.assertEqual(agent.action_size, self.action_size)
            
    def test_ddpg_action_selection(self):
        """Test DDPG deterministic action selection."""
        with patch('algorithms.ddpg.PolicyNetwork') as mock_actor, \
             patch('algorithms.ddpg.QNetwork'), \
             patch('algorithms.ddpg.ReplayBuffer'):
            
            # Mock actor output
            mock_action = torch.tensor([[0.5, -0.3]])
            mock_actor.return_value.forward.return_value = mock_action
            
            agent = DDPGAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            state = np.random.randn(self.state_size).astype(np.float32)
            action = agent.act(state, add_noise=False)
            
            self.assertEqual(action.shape, (self.action_size,))
            
    def test_ddpg_noise_injection(self):
        """Test noise injection for exploration."""
        with patch('algorithms.ddpg.PolicyNetwork') as mock_actor, \
             patch('algorithms.ddpg.QNetwork'), \
             patch('algorithms.ddpg.ReplayBuffer'):
            
            # Mock actor output
            mock_action = torch.tensor([[0.5, -0.3]])
            mock_actor.return_value.forward.return_value = mock_action
            
            agent = DDPGAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            state = np.random.randn(self.state_size).astype(np.float32)
            action_no_noise = agent.act(state, add_noise=False)
            action_with_noise = agent.act(state, add_noise=True)
            
            # Actions should be different when noise is added
            self.assertFalse(np.allclose(action_no_noise, action_with_noise))
            
    def test_ddpg_soft_update(self):
        """Test soft update of target networks."""
        with patch('algorithms.ddpg.PolicyNetwork'), \
             patch('algorithms.ddpg.QNetwork'), \
             patch('algorithms.ddpg.ReplayBuffer'):
            
            agent = DDPGAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            # Test soft update
            agent.soft_update_target_networks()
            
            # Should have target networks
            self.assertTrue(hasattr(agent, 'target_actor'))
            self.assertTrue(hasattr(agent, 'target_critic'))
            
    def tearDown(self):
        """Clean up DDPG test environment."""
        super().tearDown()
        os.unlink(self.config_file)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="SAC imports not available")
class TestSACAgent(TestAlgorithmBase):
    """Test cases for SAC Agent."""
    
    def setUp(self):
        """Set up SAC test environment."""
        super().setUp()
        self.config_file = self.create_temp_config(TEST_SAC_CONFIG)
        
    def test_sac_initialization(self):
        """Test SAC agent initialization."""
        with patch('algorithms.sac.PolicyNetwork'), \
             patch('algorithms.sac.QNetwork'), \
             patch('algorithms.sac.ReplayBuffer'):
            
            agent = SACAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            self.assertEqual(agent.state_size, self.state_size)
            self.assertEqual(agent.action_size, self.action_size)
            
    def test_sac_stochastic_action(self):
        """Test SAC stochastic action selection."""
        with patch('algorithms.sac.PolicyNetwork') as mock_policy, \
             patch('algorithms.sac.QNetwork'), \
             patch('algorithms.sac.ReplayBuffer'):
            
            # Mock policy output (mean and log_std)
            mock_mean = torch.tensor([[0.0, 0.0]])
            mock_log_std = torch.tensor([[-1.0, -1.0]])
            mock_policy.return_value.forward.return_value = (mock_mean, mock_log_std)
            
            agent = SACAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            state = np.random.randn(self.state_size).astype(np.float32)
            action, log_prob = agent.act(state)
            
            self.assertEqual(action.shape, (self.action_size,))
            self.assertIsInstance(log_prob, torch.Tensor)
            
    def test_sac_entropy_regularization(self):
        """Test entropy regularization in SAC."""
        with patch('algorithms.sac.PolicyNetwork'), \
             patch('algorithms.sac.QNetwork'), \
             patch('algorithms.sac.ReplayBuffer'):
            
            agent = SACAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            # Test entropy coefficient
            self.assertIsInstance(agent.alpha, (float, torch.Tensor))
            if TEST_SAC_CONFIG['automatic_entropy_tuning']:
                self.assertTrue(hasattr(agent, 'target_entropy'))
                
    def test_sac_twin_q_networks(self):
        """Test twin Q-networks in SAC."""
        with patch('algorithms.sac.PolicyNetwork'), \
             patch('algorithms.sac.QNetwork'), \
             patch('algorithms.sac.ReplayBuffer'):
            
            agent = SACAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                config_path=self.config_file,
                device=self.device
            )
            
            # Should have two Q-networks
            self.assertTrue(hasattr(agent, 'q_network1'))
            self.assertTrue(hasattr(agent, 'q_network2'))
            
    def tearDown(self):
        """Clean up SAC test environment."""
        super().tearDown()
        os.unlink(self.config_file)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Algorithm imports not available")
class TestAlgorithmIntegration(TestAlgorithmBase):
    """Integration tests for algorithm components."""
    
    def test_algorithm_reproducibility(self):
        """Test algorithm reproducibility with fixed seeds."""
        config_file = self.create_temp_config(TEST_DQN_CONFIG)
        
        try:
            with patch('algorithms.dqn.QNetwork'), \
                 patch('algorithms.dqn.ReplayBuffer'):
                
                # Create two agents with same seed
                torch.manual_seed(123)
                np.random.seed(123)
                agent1 = DQNAgent(
                    state_size=self.state_size,
                    action_size=self.action_size,
                    config_path=config_file,
                    device=self.device
                )
                
                torch.manual_seed(123)
                np.random.seed(123)
                agent2 = DQNAgent(
                    state_size=self.state_size,
                    action_size=self.action_size,
                    config_path=config_file,
                    device=self.device
                )
                
                # Both agents should have same epsilon initially
                self.assertEqual(agent1.epsilon, agent2.epsilon)
                
        finally:
            os.unlink(config_file)
            
    def test_config_validation(self):
        """Test configuration validation and error handling."""
        invalid_config = {'learning_rate': 'invalid'}
        config_file = self.create_temp_config(invalid_config)
        
        try:
            with patch('algorithms.dqn.QNetwork'), \
                 patch('algorithms.dqn.ReplayBuffer'):
                
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    DQNAgent(
                        state_size=self.state_size,
                        action_size=self.action_size,
                        config_path=config_file,
                        device=self.device
                    )
        finally:
            os.unlink(config_file)
            
    def test_device_compatibility(self):
        """Test algorithm compatibility with different devices."""
        config_file = self.create_temp_config(TEST_DQN_CONFIG)
        
        try:
            with patch('algorithms.dqn.QNetwork'), \
                 patch('algorithms.dqn.ReplayBuffer'):
                
                # Test CPU device
                agent_cpu = DQNAgent(
                    state_size=self.state_size,
                    action_size=self.action_size,
                    config_path=config_file,
                    device='cpu'
                )
                self.assertEqual(str(agent_cpu.device), 'cpu')
                
                # Test CUDA device (if available)
                if torch.cuda.is_available():
                    agent_cuda = DQNAgent(
                        state_size=self.state_size,
                        action_size=self.action_size,
                        config_path=config_file,
                        device='cuda'
                    )
                    self.assertEqual(str(agent_cuda.device), 'cuda:0')
                    
        finally:
            os.unlink(config_file)


if __name__ == '__main__':
    # Run specific test classes
    unittest.main(verbosity=2)