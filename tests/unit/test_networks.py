"""
Unit Tests for Neural Network Base Classes.

This module contains comprehensive unit tests for the BaseNetwork abstract base class and
its concrete implementations in the Modular DRL Framework, following the project engineering
blueprint and testing standards outlined in INSTRUCTION.md.

Detailed Description:
Tests in this module validate the correctness, functionality, and reproducibility of neural
network base classes including BaseNetwork abstract class and concrete implementations like
QNetwork, PolicyNetwork, ValueNetwork, and DuelingNetwork. Each test targets network
architecture, forward passes, configuration management, and edge cases using controlled
random seeds for reproducibility.

Key Concepts/Algorithms:
- Isolated testing of neural network functionality with mocked dependencies
- Configuration loading and validation testing
- Parameter initialization and counting verification
- Model saving/loading functionality testing
- Abstract base class interface compliance verification

Important Parameters/Configurations:
- Network architectures, initialization schemes, and activation functions
- Random seed specification for reproducibility
- Device management testing (CPU/GPU)

Expected Inputs/Outputs:
- Inputs: Mock network configurations, tensor inputs
- Outputs: Pass/fail results, coverage reports

Dependencies:
- pytest: Test framework and assertions
- unittest.mock: Mocking functionality
- torch: Neural network computations
- numpy: Numerical operations
- tempfile: Temporary file creation for save/load tests

Author: REL Project Team
Date: 2025-07-13
"""

import os
import sys
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import torch
import numpy as np
import yaml

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from networks import BaseNetwork
    from networks.q_network import QNetwork, DoubleDQNNetwork
    from networks.policy_network import PolicyNetwork, ActorNetwork
    from networks.value_network import ValueNetwork, CriticNetwork, StateActionValueNetwork
    from networks.dueling_network import DuelingNetwork, NoisyDuelingNetwork
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    pytestmark = pytest.mark.skip(f"Network modules not available: {e}")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Network imports not available")
class TestBaseNetwork(unittest.TestCase):
    """Test cases for BaseNetwork abstract base class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 4
        self.output_size = 2
        self.device = "cpu"
        self.seed = 42
    
    def test_base_network_is_abstract(self):
        """Test that BaseNetwork cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseNetwork(
                input_size=self.input_size,
                output_size=self.output_size,
                device=self.device
            )
    
    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = BaseNetwork.__abstractmethods__
        expected_methods = {
            '_build_architecture',
            '_get_default_config', 
            'forward'
        }
        self.assertTrue(expected_methods.issubset(abstract_methods))
    
    def test_seed_setting(self):
        """Test that random seed setting works correctly."""
        # Test static method directly
        BaseNetwork._set_seed(42)
        
        # Verify torch seed was set
        initial_tensor = torch.randn(2, 2)
        BaseNetwork._set_seed(42)
        second_tensor = torch.randn(2, 2)
        
        # Should be identical with same seed
        torch.testing.assert_close(initial_tensor, second_tensor)
    
    def test_config_loading(self):
        """Test configuration loading from YAML files."""
        # Test with non-existent file
        config = BaseNetwork._load_config("non_existent.yaml")
        self.assertEqual(config, {})
        
        # Test with valid YAML content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_config = {'network': {'hidden_sizes': [64, 32], 'activation': 'tanh'}}
            yaml.dump(test_config, f)
            config_path = f.name
        
        try:
            config = BaseNetwork._load_config(config_path)
            self.assertEqual(config, test_config)
        finally:
            os.unlink(config_path)
    
    def test_device_setup(self):
        """Test device setup and validation."""
        # Create a concrete implementation for testing
        class TestNetwork(BaseNetwork):
            def _build_architecture(self):
                self.test_layer = torch.nn.Linear(self.input_size, self.output_size)
            
            def _get_default_config(self):
                return {'activation': 'relu'}
            
            def forward(self, x):
                return self.test_layer(x)
        
        # Test auto device selection
        net = TestNetwork(self.input_size, self.output_size, device="auto")
        self.assertIsInstance(net.device, torch.device)
        
        # Test explicit CPU device
        net_cpu = TestNetwork(self.input_size, self.output_size, device="cpu")
        self.assertEqual(net_cpu.device.type, "cpu")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Network imports not available")
class TestQNetwork(unittest.TestCase):
    """Test cases for QNetwork implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 4
        self.action_size = 2
        self.batch_size = 32
        self.device = "cpu"
        self.seed = 42
    
    def test_q_network_creation(self):
        """Test Q-Network creation and basic properties."""
        q_net = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        self.assertEqual(q_net.input_size, self.state_size)
        self.assertEqual(q_net.output_size, self.action_size)
        self.assertEqual(q_net.device.type, "cpu")
        self.assertGreater(q_net._parameter_count, 0)
    
    def test_q_network_forward_pass(self):
        """Test Q-Network forward pass functionality."""
        q_net = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        # Test forward pass
        states = torch.randn(self.batch_size, self.state_size)
        q_values = q_net(states)
        
        self.assertEqual(q_values.shape, (self.batch_size, self.action_size))
        self.assertTrue(torch.isfinite(q_values).all())
    
    def test_dueling_architecture(self):
        """Test dueling Q-Network architecture."""
        # Create temporary config for dueling network
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'network': {
                    'hidden_sizes': [64, 64],
                    'dueling': True,
                    'activation': 'relu'
                }
            }
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            q_net = QNetwork(
                input_size=self.state_size,
                output_size=self.action_size,
                config_path=config_path,
                device=self.device,
                seed=self.seed
            )
            
            # Check dueling is enabled
            self.assertTrue(q_net.dueling_enabled)
            self.assertTrue(hasattr(q_net, 'value_stream'))
            self.assertTrue(hasattr(q_net, 'advantage_stream'))
            
            # Test forward pass
            states = torch.randn(self.batch_size, self.state_size)
            q_values = q_net(states)
            self.assertEqual(q_values.shape, (self.batch_size, self.action_size))
            
        finally:
            os.unlink(config_path)
    
    def test_epsilon_greedy_action(self):
        """Test epsilon-greedy action selection."""
        q_net = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        state = torch.randn(self.state_size)
        
        # Test deterministic action (epsilon=0)
        action = q_net.get_action(state, epsilon=0.0)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_size)
        
        # Test with exploration (epsilon=1.0 should always explore)
        torch.manual_seed(self.seed)
        action_explore = q_net.get_action(state, epsilon=1.0)
        self.assertIsInstance(action_explore, int)
        self.assertTrue(0 <= action_explore < self.action_size)
    
    def test_target_network_update(self):
        """Test target network soft update functionality."""
        q_net = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        target_net = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed + 100  # Ensure different initialization
        )
        
        # Manually ensure parameters are different
        with torch.no_grad():
            for param in target_net.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Store original target parameters
        original_params = [p.clone() for p in target_net.parameters()]
        
        # Perform soft update
        q_net.update_target_network(target_net, tau=0.1)
        
        # Check parameters changed
        updated_params = list(target_net.parameters())
        for orig, updated in zip(original_params, updated_params):
            self.assertFalse(torch.equal(orig, updated))


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Network imports not available")
class TestPolicyNetwork(unittest.TestCase):
    """Test cases for PolicyNetwork implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 8
        self.action_size = 2
        self.batch_size = 16
        self.device = "cpu"
        self.seed = 42
    
    def test_continuous_policy_creation(self):
        """Test continuous action space policy creation."""
        policy_net = PolicyNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            action_space_type='continuous',
            device=self.device,
            seed=self.seed
        )
        
        self.assertEqual(policy_net.action_space_type, 'continuous')
        self.assertEqual(policy_net.policy_type, 'stochastic')
        self.assertTrue(hasattr(policy_net, 'log_std'))
    
    def test_discrete_policy_creation(self):
        """Test discrete action space policy creation."""
        policy_net = PolicyNetwork(
            input_size=self.state_size,
            output_size=3,  # 3 discrete actions
            action_space_type='discrete',
            device=self.device,
            seed=self.seed
        )
        
        self.assertEqual(policy_net.action_space_type, 'discrete')
        self.assertEqual(policy_net.policy_type, 'categorical')
    
    def test_continuous_action_sampling(self):
        """Test continuous action sampling functionality."""
        policy_net = PolicyNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            action_space_type='continuous',
            device=self.device,
            seed=self.seed
        )
        
        states = torch.randn(self.batch_size, self.state_size)
        
        # Test stochastic sampling
        actions, log_probs, entropy = policy_net.sample_action(states, deterministic=False)
        
        self.assertEqual(actions.shape, (self.batch_size, self.action_size))
        self.assertEqual(log_probs.shape, (self.batch_size,))
        self.assertEqual(entropy.shape, (self.batch_size,))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(entropy).all())
        
        # Test deterministic sampling
        det_actions, _, _ = policy_net.sample_action(states, deterministic=True)
        self.assertEqual(det_actions.shape, (self.batch_size, self.action_size))
    
    def test_discrete_action_sampling(self):
        """Test discrete action sampling functionality."""
        action_size = 3
        policy_net = PolicyNetwork(
            input_size=self.state_size,
            output_size=action_size,
            action_space_type='discrete',
            device=self.device,
            seed=self.seed
        )
        
        states = torch.randn(self.batch_size, self.state_size)
        actions, log_probs, entropy = policy_net.sample_action(states)
        
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertEqual(log_probs.shape, (self.batch_size,))
        self.assertEqual(entropy.shape, (self.batch_size,))
        
        # Check actions are valid discrete values
        self.assertTrue((actions >= 0).all())
        self.assertTrue((actions < action_size).all())
    
    def test_action_evaluation(self):
        """Test action evaluation for given state-action pairs."""
        policy_net = PolicyNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            action_space_type='continuous',
            device=self.device,
            seed=self.seed
        )
        
        states = torch.randn(self.batch_size, self.state_size)
        actions = torch.randn(self.batch_size, self.action_size)
        
        log_probs, entropy = policy_net.evaluate_actions(states, actions)
        
        self.assertEqual(log_probs.shape, (self.batch_size,))
        self.assertEqual(entropy.shape, (self.batch_size,))
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(entropy).all())


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Network imports not available")
class TestValueNetwork(unittest.TestCase):
    """Test cases for ValueNetwork implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 8
        self.batch_size = 16
        self.device = "cpu"
        self.seed = 42
    
    def test_value_network_creation(self):
        """Test Value Network creation and basic properties."""
        value_net = ValueNetwork(
            input_size=self.state_size,
            device=self.device,
            seed=self.seed
        )
        
        self.assertEqual(value_net.input_size, self.state_size)
        self.assertEqual(value_net.output_size, 1)  # Always outputs single value
        self.assertGreater(value_net._parameter_count, 0)
    
    def test_value_network_forward_pass(self):
        """Test Value Network forward pass functionality."""
        value_net = ValueNetwork(
            input_size=self.state_size,
            device=self.device,
            seed=self.seed
        )
        
        states = torch.randn(self.batch_size, self.state_size)
        values = value_net(states)
        
        self.assertEqual(values.shape, (self.batch_size, 1))
        self.assertTrue(torch.isfinite(values).all())
        
        # Test get_value method
        value_estimates = value_net.get_value(states)
        self.assertEqual(value_estimates.shape, (self.batch_size,))
    
    def test_advantage_computation(self):
        """Test advantage computation functionality."""
        value_net = ValueNetwork(
            input_size=self.state_size,
            device=self.device,
            seed=self.seed
        )
        
        states = torch.randn(self.batch_size, self.state_size)
        next_states = torch.randn(self.batch_size, self.state_size)
        rewards = torch.randn(self.batch_size)
        dones = torch.zeros(self.batch_size)  # No episodes done
        
        advantages = value_net.compute_advantage(states, next_states, rewards, dones)
        
        self.assertEqual(advantages.shape, (self.batch_size,))
        self.assertTrue(torch.isfinite(advantages).all())
    
    def test_gae_computation(self):
        """Test Generalized Advantage Estimation (GAE) computation."""
        value_net = ValueNetwork(
            input_size=self.state_size,
            device=self.device,
            seed=self.seed
        )
        
        states = torch.randn(self.batch_size, self.state_size)
        next_states = torch.randn(self.batch_size, self.state_size)
        rewards = torch.randn(self.batch_size)
        dones = torch.zeros(self.batch_size)
        
        advantages, value_targets = value_net.compute_gae_advantages(
            states, next_states, rewards, dones, gamma=0.99, lambda_gae=0.95
        )
        
        self.assertEqual(advantages.shape, (self.batch_size,))
        self.assertEqual(value_targets.shape, (self.batch_size,))
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertTrue(torch.isfinite(value_targets).all())


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Network imports not available")
class TestDuelingNetwork(unittest.TestCase):
    """Test cases for DuelingNetwork implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 4
        self.action_size = 3
        self.batch_size = 16
        self.device = "cpu"
        self.seed = 42
    
    def test_dueling_network_creation(self):
        """Test Dueling Network creation and architecture."""
        dueling_net = DuelingNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        self.assertTrue(hasattr(dueling_net, 'feature_extractor'))
        self.assertTrue(hasattr(dueling_net, 'value_stream'))
        self.assertTrue(hasattr(dueling_net, 'advantage_stream'))
        self.assertEqual(dueling_net.aggregation_method, 'mean')
    
    def test_dueling_forward_pass(self):
        """Test Dueling Network forward pass and decomposition."""
        dueling_net = DuelingNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        states = torch.randn(self.batch_size, self.state_size)
        q_values = dueling_net(states)
        
        self.assertEqual(q_values.shape, (self.batch_size, self.action_size))
        self.assertTrue(torch.isfinite(q_values).all())
        
        # Test decomposed Q-values
        decomp = dueling_net.get_decomposed_q_values(states)
        self.assertIn('value', decomp)
        self.assertIn('advantage', decomp)
        self.assertIn('q_values', decomp)
        
        self.assertEqual(decomp['value'].shape, (self.batch_size,))
        self.assertEqual(decomp['advantage'].shape, (self.batch_size, self.action_size))
        self.assertEqual(decomp['q_values'].shape, (self.batch_size, self.action_size))
    
    def test_value_advantage_separation_analysis(self):
        """Test value-advantage separation analysis."""
        dueling_net = DuelingNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        states = torch.randn(self.batch_size, self.state_size)
        analysis = dueling_net.analyze_value_advantage_separation(states)
        
        required_keys = ['value_variance', 'advantage_variance', 'q_value_variance', 'value_advantage_correlation']
        for key in required_keys:
            self.assertIn(key, analysis)
            self.assertIsInstance(analysis[key], float)
            self.assertTrue(np.isfinite(analysis[key]))


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Network imports not available")
class TestNetworkPersistence(unittest.TestCase):
    """Test cases for network saving and loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 4
        self.action_size = 2
        self.device = "cpu"
        self.seed = 42
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_load_q_network(self):
        """Test saving and loading Q-Network."""
        # Create and save network
        q_net = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        save_path = os.path.join(self.temp_dir, "q_network.pth")
        q_net.save_model(save_path)
        self.assertTrue(os.path.exists(save_path))
        
        # Create new network and load
        q_net_loaded = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed + 1  # Different seed
        )
        
        q_net_loaded.load_model(save_path)
        
        # Test that loaded network produces same outputs
        test_input = torch.randn(1, self.state_size)
        with torch.no_grad():
            original_output = q_net(test_input)
            loaded_output = q_net_loaded(test_input)
            torch.testing.assert_close(original_output, loaded_output)
    
    def test_architecture_info(self):
        """Test network architecture information extraction."""
        q_net = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        info = q_net.get_architecture_info()
        
        required_keys = ['network_name', 'input_size', 'output_size', 'parameter_count', 'device', 'config', 'modules']
        for key in required_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['input_size'], self.state_size)
        self.assertEqual(info['output_size'], self.action_size)
        self.assertGreater(info['parameter_count'], 0)
        self.assertIsInstance(info['modules'], list)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Network imports not available")
class TestNetworkReproducibility(unittest.TestCase):
    """Test cases for ensuring reproducible network behavior."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 4
        self.action_size = 2
        self.device = "cpu"
        self.seed = 42
    
    def test_deterministic_initialization(self):
        """Test that networks are initialized deterministically with same seed."""
        # Create two networks with same seed
        q_net1 = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        q_net2 = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        # Compare parameters
        for p1, p2 in zip(q_net1.parameters(), q_net2.parameters()):
            torch.testing.assert_close(p1, p2)
    
    def test_deterministic_forward_pass(self):
        """Test that forward passes are deterministic with same input."""
        q_net = QNetwork(
            input_size=self.state_size,
            output_size=self.action_size,
            device=self.device,
            seed=self.seed
        )
        
        test_input = torch.randn(1, self.state_size)
        
        # Multiple forward passes should produce identical results
        with torch.no_grad():
            output1 = q_net(test_input)
            output2 = q_net(test_input)
            torch.testing.assert_close(output1, output2)


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)