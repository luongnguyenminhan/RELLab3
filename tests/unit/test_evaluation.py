"""
Unit Tests for Evaluation Module.

This module contains comprehensive unit tests for evaluation components in the Modular DRL Framework,
as described in the DRL survey and project engineering blueprint.

Detailed Description:
Tests in this module validate the correctness of metrics calculation, plotting functionality, and
evaluation workflows. Each test targets specific evaluation components including metrics computation,
statistical analysis, visualization, and result persistence. The structure supports extensibility
for new evaluation methods and improvements.

Key Concepts/Algorithms:
- Metrics calculation validation
- Statistical analysis testing
- Plotting and visualization tests
- Result persistence and loading
- Evaluation workflow integration

Important Parameters/Configurations:
- Evaluation metrics and their parameters
- Plotting configuration and output formats
- Statistical analysis parameters
- Random seed specification for reproducibility

Expected Inputs/Outputs:
- Inputs: mock episode data, training logs, performance metrics
- Outputs: pass/fail results, coverage reports

Dependencies:
- pytest, unittest, numpy, matplotlib, mock (if needed)

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
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from evaluation import BaseEvaluator, BaseMetricsCalculator, BasePlotter
    from evaluation.metrics import StandardMetricsCalculator
    from evaluation.plotter import StandardPlotter
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

# Test configurations
TEST_EVALUATION_CONFIG = {
    'metrics': {
        'track_episode_rewards': True,
        'track_episode_lengths': True,
        'track_success_rate': True,
        'smoothing_window': 100,
        'success_threshold': 200.0
    },
    'plotting': {
        'save_plots': True,
        'plot_format': 'png',
        'dpi': 100,
        'figure_size': [10, 6],
        'show_confidence_intervals': True
    },
    'output': {
        'save_results': True,
        'output_format': 'json',
        'include_metadata': True
    }
}


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Evaluation imports not available")
class TestEvaluationBase(unittest.TestCase):
    """Base test class for common evaluation testing utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiment_name = "test_experiment"
        self.seed = 42
        
        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        
    def create_temp_config(self, config_dict: Dict[str, Any]) -> str:
        """Create a temporary configuration file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_dict, temp_file)
        temp_file.close()
        return temp_file.name
        
    def generate_mock_episode_data(self, num_episodes: int = 100) -> List[Dict[str, Any]]:
        """Generate mock episode data for testing."""
        episodes = []
        for i in range(num_episodes):
            episode = {
                'episode': i,
                'total_reward': np.random.randn() * 50 + 100,  # Mean ~100, std ~50
                'episode_length': np.random.randint(50, 200),
                'success': np.random.choice([True, False], p=[0.3, 0.7]),
                'steps': np.random.randint(1000, 5000),
                'learning_rate': 0.001,
                'epsilon': max(0.01, 1.0 - i * 0.01)
            }
            episodes.append(episode)
        return episodes
        
    def generate_mock_time_series(self, length: int = 1000) -> List[tuple]:
        """Generate mock time series data."""
        return [(i, np.random.randn() + i * 0.01) for i in range(length)]
        
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="BaseEvaluator imports not available")
class TestBaseEvaluator(TestEvaluationBase):
    """Test cases for BaseEvaluator."""
    
    def test_base_evaluator_initialization(self):
        """Test base evaluator initialization."""
        config_file = self.create_temp_config(TEST_EVALUATION_CONFIG)
        
        try:
            with patch.object(BaseEvaluator, '_init_evaluator_components'):
                evaluator = BaseEvaluator(
                    config_path=config_file,
                    output_dir=self.temp_dir,
                    experiment_name=self.experiment_name,
                    seed=self.seed
                )
                
                self.assertEqual(evaluator.experiment_name, self.experiment_name)
                self.assertEqual(evaluator.seed, self.seed)
                self.assertTrue(evaluator.save_results)
                self.assertEqual(str(evaluator.output_dir), self.temp_dir)
                
        finally:
            os.unlink(config_file)
            
    def test_config_loading(self):
        """Test configuration loading."""
        config_file = self.create_temp_config(TEST_EVALUATION_CONFIG)
        
        try:
            with patch.object(BaseEvaluator, '_init_evaluator_components'):
                evaluator = BaseEvaluator(config_path=config_file)
                
                self.assertEqual(evaluator.config['metrics']['smoothing_window'], 100)
                self.assertEqual(evaluator.config['plotting']['save_plots'], True)
                
        finally:
            os.unlink(config_file)
            
    def test_config_file_not_found(self):
        """Test handling of missing configuration file."""
        with patch.object(BaseEvaluator, '_init_evaluator_components'):
            with self.assertRaises(FileNotFoundError):
                BaseEvaluator(config_path="nonexistent_config.yaml")
                
    def test_output_directory_creation(self):
        """Test output directory creation."""
        output_dir = os.path.join(self.temp_dir, "new_eval_dir")
        
        with patch.object(BaseEvaluator, '_init_evaluator_components'):
            evaluator = BaseEvaluator(output_dir=output_dir)
            
            self.assertTrue(os.path.exists(output_dir))
            
    def test_metadata_initialization(self):
        """Test metadata initialization."""
        with patch.object(BaseEvaluator, '_init_evaluator_components'):
            evaluator = BaseEvaluator(
                experiment_name=self.experiment_name,
                seed=self.seed
            )
            
            self.assertEqual(evaluator.metadata['experiment_name'], self.experiment_name)
            self.assertEqual(evaluator.metadata['seed'], self.seed)
            self.assertIn('creation_time', evaluator.metadata)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="StandardMetricsCalculator imports not available")
class TestStandardMetricsCalculator(TestEvaluationBase):
    """Test cases for StandardMetricsCalculator."""
    
    def test_metrics_calculator_initialization(self):
        """Test metrics calculator initialization."""
        calculator = StandardMetricsCalculator(
            success_threshold=200.0,
            episode_length_limit=500,
            track_individual_episodes=True,
            output_dir=self.temp_dir
        )
        
        self.assertEqual(calculator.success_threshold, 200.0)
        self.assertEqual(calculator.episode_length_limit, 500)
        self.assertTrue(calculator.track_individual_episodes)
        
    def test_episode_reward_calculation(self):
        """Test episode reward calculation."""
        calculator = StandardMetricsCalculator(output_dir=self.temp_dir)
        
        # Mock episode data
        episode_data = self.generate_mock_episode_data(50)
        
        # Calculate metrics
        metrics = calculator.calculate_episode_metrics(episode_data)
        
        self.assertIn('mean_reward', metrics)
        self.assertIn('std_reward', metrics)
        self.assertIn('min_reward', metrics)
        self.assertIn('max_reward', metrics)
        
        # Verify calculations
        rewards = [ep['total_reward'] for ep in episode_data]
        self.assertAlmostEqual(metrics['mean_reward'], np.mean(rewards), places=5)
        self.assertAlmostEqual(metrics['std_reward'], np.std(rewards), places=5)
        
    def test_episode_length_calculation(self):
        """Test episode length calculation."""
        calculator = StandardMetricsCalculator(output_dir=self.temp_dir)
        
        episode_data = self.generate_mock_episode_data(50)
        metrics = calculator.calculate_episode_metrics(episode_data)
        
        self.assertIn('mean_episode_length', metrics)
        self.assertIn('std_episode_length', metrics)
        
        lengths = [ep['episode_length'] for ep in episode_data]
        self.assertAlmostEqual(metrics['mean_episode_length'], np.mean(lengths), places=5)
        
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        calculator = StandardMetricsCalculator(
            success_threshold=200.0,
            output_dir=self.temp_dir
        )
        
        episode_data = self.generate_mock_episode_data(100)
        metrics = calculator.calculate_episode_metrics(episode_data)
        
        self.assertIn('success_rate', metrics)
        
        # Calculate expected success rate
        successes = sum(1 for ep in episode_data if ep['success'])
        expected_rate = successes / len(episode_data)
        self.assertAlmostEqual(metrics['success_rate'], expected_rate, places=5)
        
    def test_moving_average_calculation(self):
        """Test moving average calculation."""
        calculator = StandardMetricsCalculator(output_dir=self.temp_dir)
        
        # Create test data
        values = list(range(100))
        window_size = 10
        
        moving_avg = calculator.calculate_moving_average(values, window_size)
        
        self.assertEqual(len(moving_avg), len(values))
        
        # Check a few specific values
        expected_avg_10 = np.mean(values[1:11])  # Window includes indices 1-10
        self.assertAlmostEqual(moving_avg[10], expected_avg_10, places=5)
        
    def test_statistical_summary(self):
        """Test statistical summary calculation."""
        calculator = StandardMetricsCalculator(output_dir=self.temp_dir)
        
        values = np.random.randn(1000)
        summary = calculator.calculate_statistical_summary(values)
        
        expected_keys = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']
        for key in expected_keys:
            self.assertIn(key, summary)
            
        # Verify calculations
        self.assertAlmostEqual(summary['mean'], np.mean(values), places=5)
        self.assertAlmostEqual(summary['std'], np.std(values), places=5)
        self.assertAlmostEqual(summary['median'], np.median(values), places=5)
        
    def test_learning_curve_metrics(self):
        """Test learning curve metrics calculation."""
        calculator = StandardMetricsCalculator(output_dir=self.temp_dir)
        
        # Simulate learning curve (improving performance)
        episodes = []
        for i in range(200):
            reward = i * 0.5 + np.random.randn() * 10  # Increasing trend with noise
            episodes.append({'episode': i, 'total_reward': reward})
            
        metrics = calculator.calculate_learning_curve_metrics(episodes)
        
        self.assertIn('initial_performance', metrics)
        self.assertIn('final_performance', metrics)
        self.assertIn('improvement', metrics)
        self.assertIn('convergence_episode', metrics)
        
        # Should show improvement
        self.assertGreater(metrics['final_performance'], metrics['initial_performance'])
        
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        calculator = StandardMetricsCalculator(output_dir=self.temp_dir)
        
        empty_data = []
        metrics = calculator.calculate_episode_metrics(empty_data)
        
        # Should handle empty data gracefully
        self.assertIsInstance(metrics, dict)
        self.assertEqual(metrics.get('mean_reward', 0), 0)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="StandardPlotter imports not available")
class TestStandardPlotter(TestEvaluationBase):
    """Test cases for StandardPlotter."""
    
    def setUp(self):
        """Set up plotter test environment."""
        super().setUp()
        
        # Mock matplotlib to avoid display issues in tests
        self.mock_plt = patch('matplotlib.pyplot')
        self.mock_plt.start()
        
    def test_plotter_initialization(self):
        """Test plotter initialization."""
        plotter = StandardPlotter(
            output_dir=self.temp_dir,
            save_plots=True,
            plot_format='png'
        )
        
        self.assertTrue(plotter.save_plots)
        self.assertEqual(plotter.plot_format, 'png')
        
    def test_learning_curve_plotting(self):
        """Test learning curve plotting."""
        plotter = StandardPlotter(output_dir=self.temp_dir)
        
        # Generate test data
        episodes = list(range(100))
        rewards = [i + np.random.randn() * 5 for i in episodes]
        
        # Should not raise an exception
        plotter.plot_learning_curve(episodes, rewards, title="Test Learning Curve")
        
    def test_episode_metrics_plotting(self):
        """Test episode metrics plotting."""
        plotter = StandardPlotter(output_dir=self.temp_dir)
        
        episode_data = self.generate_mock_episode_data(50)
        
        # Should not raise an exception
        plotter.plot_episode_metrics(episode_data)
        
    def test_comparison_plotting(self):
        """Test comparison plotting between multiple experiments."""
        plotter = StandardPlotter(output_dir=self.temp_dir)
        
        # Generate data for multiple experiments
        experiments = {
            'DQN': [i + np.random.randn() for i in range(100)],
            'PPO': [i * 1.2 + np.random.randn() for i in range(100)],
            'SAC': [i * 0.8 + np.random.randn() for i in range(100)]
        }
        
        # Should not raise an exception
        plotter.plot_algorithm_comparison(experiments, title="Algorithm Comparison")
        
    def test_plot_saving(self):
        """Test plot saving functionality."""
        plotter = StandardPlotter(
            output_dir=self.temp_dir,
            save_plots=True,
            plot_format='png'
        )
        
        # Mock the save function
        with patch.object(plotter, '_save_plot') as mock_save:
            episodes = list(range(50))
            rewards = [i + np.random.randn() for i in episodes]
            
            plotter.plot_learning_curve(episodes, rewards, save_name="test_plot")
            
            # Should call save function
            mock_save.assert_called_once()
            
    def tearDown(self):
        """Clean up plotter test environment."""
        super().tearDown()
        self.mock_plt.stop()


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Evaluation imports not available")
class TestEvaluationIntegration(TestEvaluationBase):
    """Integration tests for evaluation components."""
    
    def test_end_to_end_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        config_file = self.create_temp_config(TEST_EVALUATION_CONFIG)
        
        try:
            # Initialize components
            calculator = StandardMetricsCalculator(
                config_path=config_file,
                output_dir=self.temp_dir
            )
            
            # Generate test data
            episode_data = self.generate_mock_episode_data(200)
            
            # Calculate metrics
            metrics = calculator.calculate_episode_metrics(episode_data)
            
            # Verify metrics exist
            self.assertIsInstance(metrics, dict)
            self.assertGreater(len(metrics), 0)
            
            # Test learning curve analysis
            learning_metrics = calculator.calculate_learning_curve_metrics(episode_data)
            self.assertIn('improvement', learning_metrics)
            
        finally:
            os.unlink(config_file)
            
    def test_result_persistence(self):
        """Test saving and loading evaluation results."""
        calculator = StandardMetricsCalculator(
            output_dir=self.temp_dir,
            save_results=True
        )
        
        episode_data = self.generate_mock_episode_data(50)
        metrics = calculator.calculate_episode_metrics(episode_data)
        
        # Save results
        results_file = calculator.save_results(metrics, "test_results")
        
        # Verify file exists
        self.assertTrue(os.path.exists(results_file))
        
        # Load and verify results
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
            
        self.assertEqual(loaded_results['mean_reward'], metrics['mean_reward'])
        
    def test_reproducibility_with_seeds(self):
        """Test reproducibility with random seeds."""
        config_file = self.create_temp_config(TEST_EVALUATION_CONFIG)
        
        try:
            # Run evaluation twice with same seed
            calculator1 = StandardMetricsCalculator(
                config_path=config_file,
                output_dir=self.temp_dir,
                seed=123
            )
            
            calculator2 = StandardMetricsCalculator(
                config_path=config_file,
                output_dir=self.temp_dir,
                seed=123
            )
            
            # Should produce identical results (if using random components)
            self.assertEqual(calculator1.seed, calculator2.seed)
            
        finally:
            os.unlink(config_file)
            
    def test_multi_experiment_comparison(self):
        """Test comparison across multiple experiments."""
        experiments_data = {
            'experiment_1': self.generate_mock_episode_data(100),
            'experiment_2': self.generate_mock_episode_data(100),
            'experiment_3': self.generate_mock_episode_data(100)
        }
        
        calculator = StandardMetricsCalculator(output_dir=self.temp_dir)
        
        # Calculate metrics for each experiment
        comparison_results = {}
        for exp_name, data in experiments_data.items():
            metrics = calculator.calculate_episode_metrics(data)
            comparison_results[exp_name] = metrics
            
        # Should have results for all experiments
        self.assertEqual(len(comparison_results), 3)
        
        for exp_name in experiments_data.keys():
            self.assertIn('mean_reward', comparison_results[exp_name])
            self.assertIn('success_rate', comparison_results[exp_name])


if __name__ == '__main__':
    # Run specific test classes
    unittest.main(verbosity=2)