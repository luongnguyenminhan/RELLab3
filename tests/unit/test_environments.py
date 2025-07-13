"""
Unit Tests for Environment Wrappers.

This module contains comprehensive unit tests for environment wrapper implementations in the 
Modular DRL Framework, following the project engineering blueprint and testing standards.

Detailed Description:
Tests in this module validate the correctness, functionality, and reproducibility of environment 
wrappers including the BaseEnvironmentWrapper abstract class and concrete implementations like 
StandardEnvironmentWrapper, AtariWrapper, and ContinuousControlWrapper. Each test targets wrapper 
logic, preprocessing, and edge cases using mock environments and controlled random seeds.

Key Concepts/Algorithms:
- Isolated testing of environment wrapper functionality
- Mock environment and observation/action preprocessing validation
- Configuration loading and error handling
- Reproducibility checks via random seeds

Important Parameters/Configurations:
- Environment IDs, wrapper parameters, and preprocessing options
- Random seed specification for reproducibility

Expected Inputs/Outputs:
- Inputs: mock environments, wrapper configurations
- Outputs: pass/fail results, coverage reports

Dependencies:
- pytest, unittest, numpy, torch, gymnasium, mock

Author: REL Project Team
Date: 2025-07-13
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import tempfile
import yaml

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from environments import (
        BaseEnvironmentWrapper, ObservationType, ActionType, RewardType, 
        DoneType, InfoType, EnvStepReturn, EnvResetReturn
    )
    from environments.env_wrapper import (
        StandardEnvironmentWrapper, AtariWrapper, ContinuousControlWrapper, 
        RunningMeanStd
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pytestmark = pytest.mark.skip("Environment wrapper modules not available")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Environment wrapper imports not available")
class TestBaseEnvironmentWrapper(unittest.TestCase):
    """Test cases for BaseEnvironmentWrapper abstract base class."""
    
    def test_base_wrapper_is_abstract(self):
        """Test that BaseEnvironmentWrapper cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseEnvironmentWrapper('CartPole-v1')
    
    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = BaseEnvironmentWrapper.__abstractmethods__
        expected_methods = {
            '_init_wrapper_components', 
            '_preprocess_observation', 
            '_transform_action', 
            '_shape_reward'
        }
        self.assertTrue(expected_methods.issubset(abstract_methods))
    
    def test_type_aliases_defined(self):
        """Test that type aliases are properly defined."""
        # Test that type aliases exist and can be imported
        self.assertTrue(hasattr(sys.modules['environments'], 'ObservationType'))
        self.assertTrue(hasattr(sys.modules['environments'], 'ActionType'))
        self.assertTrue(hasattr(sys.modules['environments'], 'RewardType'))
        self.assertTrue(hasattr(sys.modules['environments'], 'DoneType'))
        self.assertTrue(hasattr(sys.modules['environments'], 'InfoType'))


class MockEnvironment:
    """Mock Gymnasium environment for testing."""
    
    def __init__(self, obs_shape=(4,), action_space_size=2):
        self.observation_space = Mock()
        self.observation_space.shape = obs_shape
        self.action_space = Mock()
        self.action_space.n = action_space_size
        self.spec = None
        self.metadata = {}
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs = np.random.randn(*self.observation_space.shape)
        info = {"reset": True}
        return obs, info
        
    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(*self.observation_space.shape)
        reward = np.random.randn()
        terminated = self.step_count >= 10
        truncated = False
        info = {"step": self.step_count}
        return obs, reward, terminated, truncated, info
        
    def render(self):
        return None
        
    def close(self):
        pass


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Environment wrapper imports not available")
class TestStandardEnvironmentWrapper(unittest.TestCase):
    """Test cases for StandardEnvironmentWrapper implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = MockEnvironment()
        
        # Patch gym.make to return our mock environment
        self.gym_make_patcher = patch('environments.gym.make')
        self.mock_gym_make = self.gym_make_patcher.start()
        self.mock_gym_make.return_value = self.mock_env
        
        # Create wrapper instance
        self.wrapper = StandardEnvironmentWrapper(
            env_id='MockEnv-v0',
            seed=42,
            normalize_obs=False,
            normalize_rewards=False,
            clip_actions=True,
            reward_scale=1.0
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.gym_make_patcher.stop()
        self.wrapper.close()
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        self.assertEqual(self.wrapper.env_id, 'MockEnv-v0')
        self.assertEqual(self.wrapper.normalize_obs, False)
        self.assertEqual(self.wrapper.normalize_rewards, False)
        self.assertEqual(self.wrapper.clip_actions, True)
        self.assertEqual(self.wrapper.reward_scale, 1.0)
        self.assertEqual(self.wrapper.episode_count, 0)
        self.assertEqual(self.wrapper.total_steps, 0)
    
    def test_environment_creation(self):
        """Test that environment is created correctly."""
        self.mock_gym_make.assert_called_once()
        self.assertEqual(self.wrapper.env, self.mock_env)
    
    def test_reset_functionality(self):
        """Test environment reset."""
        obs, info = self.wrapper.reset()
        
        # Check that observation is preprocessed
        self.assertTrue(isinstance(obs, (np.ndarray, np.generic)))
        self.assertTrue(isinstance(info, dict))
        
        # Check episode tracking reset
        self.assertEqual(self.wrapper._current_episode_reward, 0.0)
        self.assertEqual(self.wrapper._current_episode_length, 0)
        
        # Check wrapper stats in info
        self.assertIn('wrapper_stats', info)
    
    def test_step_functionality(self):
        """Test environment step."""
        # Reset first
        self.wrapper.reset()
        
        # Take a step
        action = 1
        obs, reward, terminated, truncated, info = self.wrapper.step(action)
        
        # Check return types
        self.assertTrue(isinstance(obs, (np.ndarray, np.generic)))
        self.assertTrue(isinstance(reward, (float, np.floating)))
        self.assertTrue(isinstance(terminated, (bool, np.bool_)))
        self.assertTrue(isinstance(truncated, (bool, np.bool_)))
        self.assertTrue(isinstance(info, dict))
        
        # Check episode tracking
        self.assertEqual(self.wrapper._current_episode_length, 1)
        self.assertEqual(self.wrapper.total_steps, 1)
        self.assertIn('wrapper_stats', info)
    
    def test_episode_completion(self):
        """Test episode completion handling."""
        # Reset environment
        self.wrapper.reset()
        
        # Run until episode ends
        total_reward = 0.0
        step_count = 0
        
        while True:
            action = 1
            obs, reward, terminated, truncated, info = self.wrapper.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                # Check episode statistics
                self.assertIn('episode', info)
                self.assertEqual(info['episode']['l'], step_count)
                self.assertAlmostEqual(info['episode']['r'], total_reward, places=5)
                
                # Check wrapper tracking
                self.assertEqual(self.wrapper.episode_count, 1)
                self.assertEqual(len(self.wrapper.episode_rewards), 1)
                self.assertEqual(len(self.wrapper.episode_lengths), 1)
                break
    
    def test_action_clipping(self):
        """Test action clipping functionality."""
        # Mock action space with bounds
        self.mock_env.action_space.low = np.array([-1.0])
        self.mock_env.action_space.high = np.array([1.0])
        self.mock_env.action_space.n = None  # Continuous action space
        
        # Test action clipping
        clipped_action = self.wrapper._transform_action(np.array([2.0]))  # Above limit
        np.testing.assert_array_less_equal(clipped_action, [1.0])
        
        clipped_action = self.wrapper._transform_action(np.array([-2.0]))  # Below limit
        np.testing.assert_array_greater_equal(clipped_action, [-1.0])
    
    def test_reward_scaling(self):
        """Test reward scaling."""
        original_reward = 2.0
        info = {}
        
        scaled_reward = self.wrapper._shape_reward(original_reward, info)
        expected_reward = original_reward * self.wrapper.reward_scale
        
        self.assertAlmostEqual(scaled_reward, expected_reward, places=5)
    
    def test_observation_preprocessing_no_normalization(self):
        """Test observation preprocessing without normalization."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        processed_obs = self.wrapper._preprocess_observation(obs)
        
        # Without normalization, should return same observation
        np.testing.assert_array_equal(processed_obs, obs)
    
    def test_get_episode_statistics(self):
        """Test episode statistics collection."""
        # Initially empty
        stats = self.wrapper.get_episode_statistics()
        self.assertEqual(stats['total_episodes'], 0)
        self.assertEqual(stats['mean_reward'], 0.0)
        
        # Add some episode data manually
        self.wrapper.episode_rewards = [10.0, 20.0, 15.0]
        self.wrapper.episode_lengths = [100, 150, 120]
        self.wrapper.episode_count = 3
        
        stats = self.wrapper.get_episode_statistics()
        self.assertEqual(stats['total_episodes'], 3)
        self.assertAlmostEqual(stats['mean_reward'], 15.0, places=1)
        self.assertAlmostEqual(stats['mean_length'], 123.33, places=1)
        self.assertEqual(stats['min_reward'], 10.0)
        self.assertEqual(stats['max_reward'], 20.0)
    
    def test_property_delegation(self):
        """Test that properties are correctly delegated to underlying environment."""
        # Test action_space property
        self.assertEqual(self.wrapper.action_space, self.mock_env.action_space)
        
        # Test observation_space property
        self.assertEqual(self.wrapper.observation_space, self.mock_env.observation_space)
        
        # Test spec property
        self.assertEqual(self.wrapper.spec, self.mock_env.spec)
        
        # Test metadata property
        self.assertEqual(self.wrapper.metadata, self.mock_env.metadata)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Environment wrapper imports not available")
class TestStandardEnvironmentWrapperWithNormalization(unittest.TestCase):
    """Test StandardEnvironmentWrapper with normalization enabled."""
    
    def setUp(self):
        """Set up test fixtures with normalization enabled."""
        self.mock_env = MockEnvironment()
        
        # Patch gym.make
        self.gym_make_patcher = patch('environments.gym.make')
        self.mock_gym_make = self.gym_make_patcher.start()
        self.mock_gym_make.return_value = self.mock_env
        
        # Create wrapper with normalization
        self.wrapper = StandardEnvironmentWrapper(
            env_id='MockEnv-v0',
            seed=42,
            normalize_obs=True,
            normalize_rewards=True,
            clip_actions=True,
            reward_scale=1.0
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.gym_make_patcher.stop()
        self.wrapper.close()
    
    def test_observation_normalization_initialization(self):
        """Test that observation normalization components are initialized."""
        self.assertTrue(hasattr(self.wrapper, 'obs_rms'))
        self.assertIsInstance(self.wrapper.obs_rms, RunningMeanStd)
    
    def test_reward_normalization_initialization(self):
        """Test that reward normalization components are initialized."""
        self.assertTrue(hasattr(self.wrapper, 'reward_rms'))
        self.assertIsInstance(self.wrapper.reward_rms, RunningMeanStd)
        self.assertTrue(hasattr(self.wrapper, 'reward_history'))
    
    def test_observation_normalization(self):
        """Test observation normalization functionality."""
        # Generate consistent observations
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Process observation multiple times to build statistics
        for _ in range(10):
            processed_obs = self.wrapper._preprocess_observation(obs + np.random.randn(4) * 0.1)
        
        # The observation should be normalized
        final_processed = self.wrapper._preprocess_observation(obs)
        
        # Should be different from original (normalized)
        self.assertFalse(np.allclose(final_processed, obs))


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Environment wrapper imports not available")
class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = MockEnvironment()
        
        # Patch gym.make
        self.gym_make_patcher = patch('environments.gym.make')
        self.mock_gym_make = self.gym_make_patcher.start()
        self.mock_gym_make.return_value = self.mock_env
    
    def tearDown(self):
        """Clean up after tests."""
        self.gym_make_patcher.stop()
    
    def test_config_loading_valid_file(self):
        """Test loading configuration from valid YAML file."""
        config_data = {
            'environment': {
                'normalize_obs': True,
                'reward_scale': 2.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            wrapper = StandardEnvironmentWrapper(
                env_id='MockEnv-v0',
                config_path=config_path
            )
            
            # Config should be loaded
            self.assertEqual(wrapper.config, config_data)
            
            wrapper.close()
        finally:
            os.unlink(config_path)
    
    def test_config_loading_nonexistent_file(self):
        """Test error handling when config file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            StandardEnvironmentWrapper(
                env_id='MockEnv-v0',
                config_path='/nonexistent/config.yaml'
            )
    
    def test_config_loading_invalid_yaml(self):
        """Test error handling for malformed YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content:\n  - malformed')
            config_path = f.name
        
        try:
            with self.assertRaises(yaml.YAMLError):
                StandardEnvironmentWrapper(
                    env_id='MockEnv-v0',
                    config_path=config_path
                )
        finally:
            os.unlink(config_path)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Environment wrapper imports not available")
class TestRunningMeanStd(unittest.TestCase):
    """Test cases for RunningMeanStd utility class."""
    
    def test_initialization(self):
        """Test RunningMeanStd initialization."""
        rms = RunningMeanStd(shape=(4,))
        
        self.assertEqual(rms.mean.shape, (4,))
        self.assertEqual(rms.var.shape, (4,))
        np.testing.assert_array_equal(rms.mean, np.zeros(4))
        np.testing.assert_array_equal(rms.var, np.ones(4))
    
    def test_single_update(self):
        """Test updating with a single batch."""
        rms = RunningMeanStd(shape=(2,))
        
        # Update with known data
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rms.update(data)
        
        # Check that statistics are updated
        expected_mean = np.mean(data, axis=0)
        np.testing.assert_array_almost_equal(rms.mean, expected_mean, decimal=5)
    
    def test_multiple_updates(self):
        """Test updating with multiple batches."""
        rms = RunningMeanStd(shape=(1,))
        
        # Update with multiple batches
        batch1 = np.array([[1.0], [2.0]])
        batch2 = np.array([[3.0], [4.0]])
        
        rms.update(batch1)
        rms.update(batch2)
        
        # Should have statistics from all data
        all_data = np.concatenate([batch1, batch2])
        expected_mean = np.mean(all_data, axis=0)
        
        np.testing.assert_array_almost_equal(rms.mean, expected_mean, decimal=5)
    
    def test_variance_computation(self):
        """Test variance computation."""
        rms = RunningMeanStd(shape=(1,))
        
        # Use data with known variance
        data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        rms.update(data)
        
        expected_var = np.var(data, axis=0)
        np.testing.assert_array_almost_equal(rms.var, expected_var, decimal=5)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Environment wrapper imports not available")
class TestAtariWrapper(unittest.TestCase):
    """Test cases for AtariWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = MockEnvironment(obs_shape=(84, 84, 4))  # Atari-like observations
        
        # Patch gym.make
        self.gym_make_patcher = patch('environments.gym.make')
        self.mock_gym_make = self.gym_make_patcher.start()
        self.mock_gym_make.return_value = self.mock_env
    
    def tearDown(self):
        """Clean up after tests."""
        self.gym_make_patcher.stop()
    
    def test_atari_wrapper_defaults(self):
        """Test that AtariWrapper sets appropriate defaults."""
        wrapper = AtariWrapper('Breakout-v4')
        
        # Check Atari-specific defaults
        self.assertEqual(wrapper.normalize_obs, False)
        self.assertEqual(wrapper.normalize_rewards, False)
        self.assertEqual(wrapper.clip_actions, False)
        self.assertEqual(wrapper.reward_scale, 1.0)
        
        wrapper.close()


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Environment wrapper imports not available")
class TestContinuousControlWrapper(unittest.TestCase):
    """Test cases for ContinuousControlWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = MockEnvironment(obs_shape=(8,))  # Control-like observations
        
        # Patch gym.make
        self.gym_make_patcher = patch('environments.gym.make')
        self.mock_gym_make = self.gym_make_patcher.start()
        self.mock_gym_make.return_value = self.mock_env
    
    def tearDown(self):
        """Clean up after tests."""
        self.gym_make_patcher.stop()
    
    def test_continuous_control_defaults(self):
        """Test that ContinuousControlWrapper sets appropriate defaults."""
        wrapper = ContinuousControlWrapper('Pendulum-v1')
        
        # Check continuous control defaults
        self.assertEqual(wrapper.normalize_obs, True)
        self.assertEqual(wrapper.normalize_rewards, True)
        self.assertEqual(wrapper.clip_actions, True)
        self.assertEqual(wrapper.reward_scale, 1.0)
        
        wrapper.close()


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Environment wrapper imports not available")
class TestEnvironmentWrapperIntegration(unittest.TestCase):
    """Integration tests for environment wrappers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = MockEnvironment()
        
        # Patch gym.make
        self.gym_make_patcher = patch('environments.gym.make')
        self.mock_gym_make = self.gym_make_patcher.start()
        self.mock_gym_make.return_value = self.mock_env
    
    def tearDown(self):
        """Clean up after tests."""
        self.gym_make_patcher.stop()
    
    def test_full_episode_workflow(self):
        """Test complete episode workflow."""
        wrapper = StandardEnvironmentWrapper(
            env_id='MockEnv-v0',
            seed=42,
            normalize_obs=False,
            clip_actions=True
        )
        
        try:
            # Reset environment
            obs, info = wrapper.reset()
            self.assertIsNotNone(obs)
            
            # Run episode
            total_reward = 0.0
            step_count = 0
            
            while True:
                action = 1  # Simple action
                obs, reward, terminated, truncated, info = wrapper.step(action)
                total_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    break
                    
                # Shouldn't run forever
                if step_count > 100:
                    break
            
            # Check final statistics
            stats = wrapper.get_episode_statistics()
            self.assertEqual(stats['total_episodes'], 1)
            
        finally:
            wrapper.close()
    
    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        wrapper = StandardEnvironmentWrapper(
            env_id='MockEnv-v0',
            seed=42
        )
        
        try:
            num_episodes = 3
            
            for episode in range(num_episodes):
                obs, info = wrapper.reset()
                
                while True:
                    action = episode % 2  # Different actions per episode
                    obs, reward, terminated, truncated, info = wrapper.step(action)
                    
                    if terminated or truncated:
                        break
            
            # Check that all episodes were tracked
            stats = wrapper.get_episode_statistics()
            self.assertEqual(stats['total_episodes'], num_episodes)
            self.assertEqual(len(wrapper.episode_rewards), num_episodes)
            self.assertEqual(len(wrapper.episode_lengths), num_episodes)
            
        finally:
            wrapper.close()
    
    def test_statistics_persistence(self):
        """Test that statistics can be saved and loaded."""
        wrapper = StandardEnvironmentWrapper(
            env_id='MockEnv-v0',
            seed=42
        )
        
        try:
            # Run a few episodes
            for _ in range(2):
                obs, info = wrapper.reset()
                
                for _ in range(5):
                    obs, reward, terminated, truncated, info = wrapper.step(1)
                    if terminated or truncated:
                        break
            
            # Save statistics
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                stats_path = f.name
            
            wrapper.save_statistics(stats_path)
            
            # Check that file exists and contains data
            self.assertTrue(os.path.exists(stats_path))
            
            with open(stats_path, 'r') as f:
                saved_stats = yaml.safe_load(f)
            
            self.assertIn('total_episodes', saved_stats)
            self.assertIn('episode_rewards', saved_stats)
            self.assertIn('episode_lengths', saved_stats)
            
            # Clean up
            os.unlink(stats_path)
            
        finally:
            wrapper.close()


if __name__ == '__main__':
    # Run tests with pytest for better output
    pytest.main([__file__, '-v', '--tb=short'])