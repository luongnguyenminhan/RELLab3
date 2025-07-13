"""
Unit Tests for Utils Module - Extended Version.

This module contains comprehensive unit tests for all utility components in the Modular DRL Framework,
including the BaseAgent abstract base class, Logger, HyperparameterManager, and Noise generators.

Detailed Description:
Tests in this module validate the correctness of all utility components including logging utilities,
hyperparameter management, noise generation, and base agent functionality. Each test targets specific
utility functions and covers edge cases, reproducibility, and integration scenarios.

Key Test Categories:
- BaseAgent: Abstract class validation and lifecycle testing
- Logger: Experiment tracking, metrics logging, and file I/O
- HyperparameterManager: Configuration management and validation
- Noise Generators: Gaussian and Ornstein-Uhlenbeck noise processes
- Integration: Cross-component functionality and workflows

Important Test Features:
- Reproducibility testing with seed management
- Error handling and edge case validation
- File I/O operations and persistence
- Configuration loading and validation
- Statistical properties verification

Dependencies:
- pytest, unittest, numpy, yaml, logging, tempfile, mock

Author: REL Project Team
Date: 2025-07-13
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pytest
import numpy as np
import tempfile
import yaml
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
import io
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from utils.base_agent import BaseAgent, create_agent
    BASE_AGENT_AVAILABLE = True
except ImportError:
    BASE_AGENT_AVAILABLE = False

try:
    from utils.logger import Logger
    from utils.hyperparameters import HyperparameterManager
    from utils.noise import (
        BaseNoise, GaussianNoise, OrnsteinUhlenbeckNoise, 
        AdaptiveNoise, create_noise_from_config
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Test configurations
TEST_HYPERPARAMETER_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'gamma': 0.99,
    'epsilon': {
        'start': 1.0,
        'end': 0.01,
        'decay': 0.995
    },
    'network': {
        'hidden_sizes': [64, 64],
        'activation': 'relu'
    }
}

TEST_LOGGER_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',
    'save_to_file': True,
    'metrics_to_track': ['reward', 'episode_length', 'loss'],
    'plot_interval': 100
}

# Concrete agent class for testing BaseAgent
class TestConcreteAgent(BaseAgent):
    """Test concrete implementation of BaseAgent."""
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'test_param': 'default_value',
            'batch_size': 32,
            'learning_rate': 0.001
        }
    
    def _validate_config(self) -> None:
        if 'test_param' not in self.config:
            raise ValueError("Missing required config parameter: test_param")
        if self.config.get('batch_size', 0) <= 0:
            raise ValueError("batch_size must be positive")
    
    def _setup_agent(self) -> None:
        self.state['setup_called'] = True
        self.state['processing_count'] = 0
    
    def process(self, data: str = "test_data") -> str:
        self.state['processing_count'] += 1
        self._step()
        return f"processed_{data}_{self.state['processing_count']}"
    
    def _reset_agent(self) -> None:
        self.state['processing_count'] = 0
        
    def _cleanup_agent(self) -> None:
        self.state['cleanup_called'] = True


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Utils imports not available")
class TestUtilsBase(unittest.TestCase):
    """Base test class for common utils testing utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.seed = 42
        
        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        
    def create_temp_config(self, config_dict: Dict[str, Any]) -> str:
        """Create a temporary configuration file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name
        
    def create_temp_json_config(self, config_dict: Dict[str, Any]) -> str:
        """Create a temporary JSON configuration file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name
        
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


@pytest.mark.skipif(not BASE_AGENT_AVAILABLE, reason="BaseAgent imports not available")
class TestBaseAgent(TestUtilsBase):
    """Test cases for BaseAgent abstract base class."""
    
    def test_abstract_class_enforcement(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseAgent("test_agent")
    
    def test_concrete_agent_initialization(self):
        """Test successful initialization of concrete agent."""
        agent = TestConcreteAgent(
            agent_name="test_agent",
            log_dir=self.temp_dir,
            seed=42
        )
        
        self.assertEqual(agent.agent_name, "test_agent")
        self.assertEqual(agent.seed, 42)
        self.assertTrue(agent.is_setup)
        self.assertIn('setup_called', agent.state)
        self.assertTrue(agent.state['setup_called'])
    
    def test_config_loading_and_merging(self):
        """Test configuration loading and merging with defaults."""
        config_dict = {
            'test_agent': {
                'test_param': 'yaml_value',
                'new_param': 'yaml_only'
            }
        }
        config_file = self.create_temp_config(config_dict)
        
        try:
            agent = TestConcreteAgent(
                agent_name="test_agent",
                config_path=config_file,
                log_dir=self.temp_dir
            )
            
            # Should merge defaults with YAML config
            self.assertEqual(agent.config['test_param'], 'yaml_value')  # Overridden
            self.assertEqual(agent.config['batch_size'], 32)  # Default
            self.assertEqual(agent.config['new_param'], 'yaml_only')  # From YAML
            
        finally:
            os.unlink(config_file)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        agent = TestConcreteAgent(
            agent_name="test_agent",
            log_dir=self.temp_dir
        )
        self.assertTrue(agent.is_setup)
        
        # Test invalid config by creating an agent with a bad config file
        bad_config = {
            'test_agent_invalid': {
                'batch_size': -1  # Invalid value
            }
        }
        config_file = self.create_temp_config(bad_config)
        
        try:
            with self.assertRaises(ValueError):
                TestConcreteAgent(
                    agent_name="test_agent_invalid",
                    config_path=config_file,
                    log_dir=self.temp_dir
                )
        finally:
            os.unlink(config_file)
    
    def test_metric_logging(self):
        """Test metric logging functionality."""
        agent = TestConcreteAgent(
            agent_name="test_agent",
            log_dir=self.temp_dir
        )
        
        # Log some metrics
        agent.log_metric('reward', 100.5)
        agent.log_metric('loss', 0.25)
        agent.log_metric('reward', 105.2)
        
        # Check metrics are stored
        self.assertIn('reward', agent.metrics)
        self.assertIn('loss', agent.metrics)
        self.assertEqual(len(agent.metrics['reward']), 2)
        self.assertEqual(len(agent.metrics['loss']), 1)
    
    def test_episode_logging(self):
        """Test episode data logging."""
        agent = TestConcreteAgent(
            agent_name="test_agent",
            log_dir=self.temp_dir
        )
        
        # Log episode data
        episode_data = {
            'total_reward': 150.0,
            'episode_length': 200,
            'success': True
        }
        agent.log_episode_data(episode_data)
        
        # Check episode counter incremented
        self.assertEqual(agent._episode_count, 1)
    
    def test_state_persistence(self):
        """Test agent state saving and loading."""
        agent = TestConcreteAgent(
            agent_name="test_agent",
            log_dir=self.temp_dir,
            seed=42
        )
        
        # Modify agent state
        agent.process("test_data")
        agent.log_metric('test_metric', 42.0)
        
        # Save state
        save_path = os.path.join(self.temp_dir, 'agent_state.json')
        agent.save_state(save_path)
        
        # Create new agent and load state
        new_agent = TestConcreteAgent(
            agent_name="loaded_agent",
            log_dir=self.temp_dir
        )
        new_agent.load_state(save_path)
        
        # Check state was loaded correctly
        self.assertEqual(new_agent.seed, 42)
        self.assertEqual(new_agent.state['processing_count'], 1)
        self.assertIn('test_metric', new_agent.metrics)


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Logger imports not available")
class TestLogger(TestUtilsBase):
    """Test cases for Logger utility."""
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = Logger(
            experiment_name="test_experiment",
            log_dir=self.temp_dir,
            log_level=logging.INFO
        )
        
        self.assertEqual(logger.experiment_name, "test_experiment")
        self.assertTrue(os.path.exists(logger.log_dir))
        self.assertIsInstance(logger.logger, logging.Logger)
        
    def test_log_scalar_metric(self):
        """Test logging scalar metrics."""
        logger = Logger(
            experiment_name="test_experiment",
            log_dir=self.temp_dir
        )
        
        # Log some metrics
        logger.log_scalar('reward', 100.5, step=1)
        logger.log_scalar('loss', 0.25, step=1)
        logger.log_scalar('reward', 105.2, step=2)
        
        # Check that metrics are stored
        self.assertIn('reward', logger.metrics)
        self.assertIn('loss', logger.metrics)
        self.assertEqual(len(logger.metrics['reward']), 2)
        self.assertEqual(len(logger.metrics['loss']), 1)
        
    def test_log_episode_data(self):
        """Test logging episode data."""
        logger = Logger(
            experiment_name="test_experiment",
            log_dir=self.temp_dir
        )
        
        # Log episode data
        episode_data = {
            'episode': 1,
            'total_reward': 150.0,
            'episode_length': 200,
            'success': True
        }
        
        logger.log_episode(episode_data)
        
        # Check that episode data is stored
        self.assertEqual(len(logger.episodes), 1)
        self.assertEqual(logger.episodes[0]['total_reward'], 150.0)
        
    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        logger = Logger(
            experiment_name="test_experiment",
            log_dir=self.temp_dir
        )
        
        # Log multiple metrics
        for i in range(10):
            logger.log_scalar('reward', i * 10.0, step=i)
            logger.log_scalar('loss', 1.0 / (i + 1), step=i)
            
        summary = logger.get_metrics_summary()
        
        # Check summary contents
        self.assertIn('reward', summary)
        self.assertIn('loss', summary)
        self.assertIn('mean', summary['reward'])
        self.assertIn('std', summary['reward'])
        self.assertIn('max', summary['reward'])
        self.assertIn('min', summary['reward'])
        
    def test_save_and_load_metrics(self):
        """Test saving and loading metrics to/from file."""
        logger = Logger(
            experiment_name="test_experiment",
            log_dir=self.temp_dir,
            save_to_file=True
        )
        
        # Log some data
        for i in range(5):
            logger.log_scalar('reward', i * 20.0, step=i)
            
        # Save metrics
        save_path = logger.save_metrics()
        self.assertTrue(os.path.exists(save_path))
        
        # Load metrics in new logger
        new_logger = Logger(
            experiment_name="test_experiment_2",
            log_dir=self.temp_dir
        )
        new_logger.load_metrics(save_path)
        
        # Check that metrics were loaded
        self.assertIn('reward', new_logger.metrics)
        self.assertEqual(len(new_logger.metrics['reward']), 5)
        
    def test_metrics_filtering(self):
        """Test filtering metrics by step range."""
        logger = Logger(
            experiment_name="test_experiment",
            log_dir=self.temp_dir
        )
        
        # Log metrics with different steps
        for i in range(20):
            logger.log_scalar('reward', i * 5.0, step=i)
            
        # Filter metrics
        filtered = logger.get_metrics_in_range('reward', min_step=5, max_step=15)
        
        # Should have 11 values (steps 5-15 inclusive)
        self.assertEqual(len(filtered), 11)
        self.assertEqual(filtered[0][1], 25.0)  # Step 5, value 25.0
        self.assertEqual(filtered[-1][1], 75.0)  # Step 15, value 75.0


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="HyperparameterManager imports not available")
class TestHyperparameterManager(TestUtilsBase):
    """Test cases for HyperparameterManager utility."""
    
    def test_hyperparameter_loading(self):
        """Test loading hyperparameters from config file."""
        config_file = self.create_temp_config(TEST_HYPERPARAMETER_CONFIG)
        
        try:
            manager = HyperparameterManager(config_path=config_file)
            
            self.assertEqual(manager.get('learning_rate'), 0.001)
            self.assertEqual(manager.get('batch_size'), 32)
            self.assertEqual(manager.get('epsilon.start'), 1.0)
            self.assertEqual(manager.get('network.hidden_sizes'), [64, 64])
            
        finally:
            os.unlink(config_file)
            
    def test_hyperparameter_validation(self):
        """Test hyperparameter validation."""
        config_file = self.create_temp_config(TEST_HYPERPARAMETER_CONFIG)
        
        try:
            manager = HyperparameterManager(config_path=config_file)
            
            # Test valid ranges
            self.assertTrue(manager.validate_range('learning_rate', min_val=0.0, max_val=1.0))
            self.assertTrue(manager.validate_range('gamma', min_val=0.0, max_val=1.0))
            
            # Test invalid ranges
            self.assertFalse(manager.validate_range('learning_rate', min_val=0.01, max_val=1.0))
            
        finally:
            os.unlink(config_file)
            
    def test_hyperparameter_update(self):
        """Test updating hyperparameters."""
        manager = HyperparameterManager()
        
        # Set initial parameters
        manager.set('learning_rate', 0.001)
        manager.set('batch_size', 32)
        
        # Update parameters
        manager.update({'learning_rate': 0.0005, 'gamma': 0.95})
        
        self.assertEqual(manager.get('learning_rate'), 0.0005)
        self.assertEqual(manager.get('batch_size'), 32)  # Should remain unchanged
        self.assertEqual(manager.get('gamma'), 0.95)
        
    def test_nested_parameter_access(self):
        """Test accessing nested parameters."""
        config_file = self.create_temp_config(TEST_HYPERPARAMETER_CONFIG)
        
        try:
            manager = HyperparameterManager(config_path=config_file)
            
            # Test nested access
            self.assertEqual(manager.get('epsilon.start'), 1.0)
            self.assertEqual(manager.get('epsilon.end'), 0.01)
            self.assertEqual(manager.get('network.activation'), 'relu')
            
            # Test setting nested parameters
            manager.set('epsilon.decay', 0.99)
            self.assertEqual(manager.get('epsilon.decay'), 0.99)
            
        finally:
            os.unlink(config_file)
            
    def test_parameter_search_space(self):
        """Test defining and sampling from parameter search spaces."""
        manager = HyperparameterManager()
        
        # Define search spaces
        search_space = {
            'learning_rate': {'type': 'uniform', 'low': 0.0001, 'high': 0.01},
            'batch_size': {'type': 'choice', 'choices': [16, 32, 64, 128]},
            'gamma': {'type': 'uniform', 'low': 0.9, 'high': 0.999}
        }
        
        manager.define_search_space(search_space)
        
        # Sample parameters
        sampled = manager.sample_parameters(seed=42)
        
        self.assertIn('learning_rate', sampled)
        self.assertIn('batch_size', sampled)
        self.assertIn('gamma', sampled)
        
        # Check ranges
        self.assertGreaterEqual(sampled['learning_rate'], 0.0001)
        self.assertLessEqual(sampled['learning_rate'], 0.01)
        self.assertIn(sampled['batch_size'], [16, 32, 64, 128])


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Noise imports not available")
class TestNoiseGenerators(TestUtilsBase):
    """Test cases for noise generation utilities."""
    
    def test_gaussian_noise_initialization(self):
        """Test Gaussian noise initialization."""
        noise = GaussianNoise(
            mean=0.0,
            std=0.1,
            size=2,
            seed=42
        )
        
        self.assertEqual(noise.mean, 0.0)
        self.assertEqual(noise.std, 0.1)
        self.assertEqual(noise.size, (2,))
        
    def test_gaussian_noise_generation(self):
        """Test Gaussian noise generation."""
        noise = GaussianNoise(
            mean=0.0,
            std=0.1,
            size=3,
            seed=42
        )
        
        # Generate noise
        noise_sample = noise.sample()
        
        self.assertEqual(len(noise_sample), 3)
        self.assertIsInstance(noise_sample, np.ndarray)
        
        # Test reproducibility
        noise.reset(seed=42)
        noise_sample2 = noise.sample()
        np.testing.assert_array_equal(noise_sample, noise_sample2)
        
    def test_gaussian_noise_statistics(self):
        """Test Gaussian noise statistical properties."""
        noise = GaussianNoise(
            mean=5.0,
            std=2.0,
            size=1,
            seed=42
        )
        
        # Generate many samples
        samples = [noise.sample() for _ in range(10000)]
        
        # Check statistical properties
        sample_mean = np.mean(samples)
        sample_std = np.std(samples)
        
        # Should be close to specified parameters (within tolerance)
        self.assertAlmostEqual(sample_mean, 5.0, delta=0.1)
        self.assertAlmostEqual(sample_std, 2.0, delta=0.1)
        
    def test_ornstein_uhlenbeck_noise_initialization(self):
        """Test Ornstein-Uhlenbeck noise initialization."""
        noise = OrnsteinUhlenbeckNoise(
            size=2,
            mu=0.0,
            theta=0.15,
            sigma=0.2,
            seed=42
        )
        
        self.assertEqual(noise.size, (2,))
        self.assertEqual(noise.mu, 0.0)
        self.assertEqual(noise.theta, 0.15)
        self.assertEqual(noise.sigma, 0.2)
        
    def test_ornstein_uhlenbeck_noise_generation(self):
        """Test Ornstein-Uhlenbeck noise generation."""
        noise = OrnsteinUhlenbeckNoise(
            size=2,
            mu=0.0,
            theta=0.15,
            sigma=0.2,
            seed=42
        )
        
        # Generate noise sequence
        samples = []
        for _ in range(100):
            sample = noise.sample()
            samples.append(sample)
            
        self.assertEqual(len(samples), 100)
        self.assertEqual(len(samples[0]), 2)
        
        # Check that noise evolves (not independent)
        # OU noise should have temporal correlation
        first_sample = samples[0]
        last_sample = samples[-1]
        self.assertFalse(np.array_equal(first_sample, last_sample))
        
    def test_ornstein_uhlenbeck_noise_reset(self):
        """Test Ornstein-Uhlenbeck noise reset functionality."""
        noise = OrnsteinUhlenbeckNoise(
            size=2,
            mu=0.0,
            theta=0.15,
            sigma=0.2,
            seed=42
        )
        
        # Generate some samples
        for _ in range(10):
            noise.sample()
            
        # Reset noise
        noise.reset()
        
        # State should be reset to initial values
        np.testing.assert_array_equal(noise.state, np.zeros(2))
        
    def test_noise_decay(self):
        """Test noise decay functionality."""
        noise = GaussianNoise(
            mean=0.0,
            std=1.0,
            size=1,
            decay_rate=0.99,
            min_std=0.01,
            seed=42
        )
        
        initial_std = noise.std
        
        # Apply decay multiple times
        for _ in range(100):
            noise.decay()
            
        # Standard deviation should have decreased
        self.assertLess(noise.std, initial_std)
        self.assertGreaterEqual(noise.std, 0.01)  # Should not go below min_std
        
    def test_adaptive_noise(self):
        """Test adaptive noise wrapper."""
        base_noise = GaussianNoise(mean=0.0, std=1.0, size=2, seed=42)
        adaptive_noise = AdaptiveNoise(
            base_noise=base_noise,
            schedule_type='linear',
            schedule_params={
                'initial_value': 1.0,
                'final_value': 0.1,
                'total_steps': 100
            }
        )
        
        # Test initial state
        initial_std = base_noise.std
        self.assertEqual(initial_std, 1.0)
        
        # Update schedule
        adaptive_noise.update_schedule(50)  # Halfway
        self.assertLess(base_noise.std, initial_std)
        self.assertGreater(base_noise.std, 0.1)
        
        # At end of schedule
        adaptive_noise.update_schedule(100)
        self.assertAlmostEqual(base_noise.std, 0.1, places=5)
        
    def test_create_noise_from_config(self):
        """Test creating noise from configuration."""
        # Test Gaussian noise config
        gaussian_config = {
            'type': 'gaussian',
            'mean': 0.0,
            'std': 0.2,
            'decay_rate': 0.995
        }
        
        noise = create_noise_from_config(gaussian_config, action_dim=4, seed=42)
        self.assertIsInstance(noise, GaussianNoise)
        self.assertEqual(noise.mean, 0.0)
        self.assertEqual(noise.std, 0.2)
        
        # Test OU noise config
        ou_config = {
            'type': 'ou',
            'mu': 0.0,
            'theta': 0.15,
            'sigma': 0.3
        }
        
        ou_noise = create_noise_from_config(ou_config, action_dim=2, seed=42)
        self.assertIsInstance(ou_noise, OrnsteinUhlenbeckNoise)
        self.assertEqual(ou_noise.mu, 0.0)
        self.assertEqual(ou_noise.theta, 0.15)
        self.assertEqual(ou_noise.sigma, 0.3)


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Utils imports not available")
class TestUtilsIntegration(TestUtilsBase):
    """Integration tests for utils components."""
    
    def test_logger_hyperparameter_integration(self):
        """Test integration between logger and hyperparameter manager."""
        config_file = self.create_temp_config(TEST_HYPERPARAMETER_CONFIG)
        
        try:
            # Initialize components
            manager = HyperparameterManager(config_path=config_file)
            logger = Logger(
                experiment_name="integration_test",
                log_dir=self.temp_dir
            )
            
            # Log hyperparameters
            logger.log_hyperparameters(manager.get_all())
            
            # Check that hyperparameters are logged
            self.assertIn('hyperparameters', logger.metadata)
            self.assertEqual(
                logger.metadata['hyperparameters']['learning_rate'],
                0.001
            )
            
        finally:
            os.unlink(config_file)
            
    def test_noise_logger_integration(self):
        """Test integration between noise generators and logger."""
        logger = Logger(
            experiment_name="noise_test",
            log_dir=self.temp_dir
        )
        
        noise = GaussianNoise(
            mean=0.0,
            std=1.0,
            size=2,
            seed=42
        )
        
        # Log noise parameters
        logger.log_dict({
            'noise_type': 'gaussian',
            'noise_std': noise.std,
            'noise_mean': noise.mean
        })
        
        # Generate and log noise samples
        for step in range(10):
            noise_sample = noise.sample()
            logger.log_scalar('noise_magnitude', np.linalg.norm(noise_sample), step)
            
        # Check logged data
        self.assertIn('noise_magnitude', logger.metrics)
        self.assertEqual(len(logger.metrics['noise_magnitude']), 10)
        
    def test_complete_experiment_workflow(self):
        """Test complete experiment workflow with all utils."""
        config_file = self.create_temp_config({
            **TEST_HYPERPARAMETER_CONFIG,
            'noise': {
                'type': 'ou',
                'theta': 0.15,
                'sigma': 0.2
            }
        })
        
        try:
            # Initialize all components
            manager = HyperparameterManager(config_path=config_file)
            logger = Logger(
                experiment_name="full_workflow",
                log_dir=self.temp_dir,
                config=manager.get_all()
            )
            
            # Create noise from config
            noise_config = manager.get('noise', {})
            noise = create_noise_from_config(noise_config, action_dim=2, seed=42)
            
            # Log initial setup
            logger.log_hyperparameters(manager.get_all())
            
            # Simulate training episodes
            for episode in range(10):
                episode_reward = 0.0
                
                for step in range(50):
                    # Generate noise for exploration
                    exploration_noise = noise.sample()
                    
                    # Simulate reward with noise influence
                    reward = np.random.randn() + 0.01 * np.sum(exploration_noise)
                    episode_reward += reward
                    
                    # Log step metrics
                    if step % 10 == 0:
                        logger.log_scalar('step_reward', reward, step=episode * 50 + step)
                        logger.log_scalar('noise_magnitude', np.linalg.norm(exploration_noise), step=episode * 50 + step)
                
                # Log episode data
                logger.log_episode({
                    'episode': episode,
                    'total_reward': episode_reward,
                    'episode_length': 50,
                    'avg_reward': episode_reward / 50
                })
            
            # Save final results
            metrics_path = logger.save_metrics()
            config_path = os.path.join(self.temp_dir, 'final_config.yaml')
            manager.save_config(config_path)
            
            # Verify files exist and contain expected data
            self.assertTrue(os.path.exists(metrics_path))
            self.assertTrue(os.path.exists(config_path))
            
            # Check data integrity
            summary = logger.get_metrics_summary()
            self.assertIn('step_reward', summary)
            self.assertIn('noise_magnitude', summary)
            self.assertEqual(len(logger.episodes), 10)
            
        finally:
            os.unlink(config_file)


@pytest.mark.skipif(not BASE_AGENT_AVAILABLE, reason="BaseAgent imports not available")
class TestAgentFactory(TestUtilsBase):
    """Test cases for agent factory function."""
    
    def test_factory_unknown_type(self):
        """Test factory with unknown agent type."""
        with self.assertRaises(ValueError) as context:
            create_agent(
                agent_type="unknown_agent",
                agent_name="test_agent"
            )
        
        self.assertIn("Unknown agent type", str(context.exception))
        self.assertIn("unknown_agent", str(context.exception))
    
    def test_factory_import_error(self):
        """Test factory with import error."""
        # This should raise ImportError since the module doesn't exist
        with self.assertRaises(ImportError):
            create_agent(
                agent_type="data_processor",
                agent_name="test_processor"
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)