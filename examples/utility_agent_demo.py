"""
Utility Agent Demonstration Script.

This script demonstrates the comprehensive usage of utility agents, logger, hyperparameter manager,
and noise generators in the Modular DRL Framework. It showcases how these components work together
to provide a robust foundation for DRL experiments.

Detailed Description:
The script creates concrete implementations of utility agents that inherit from the BaseAgent
abstract class. It demonstrates configuration management, logging, noise generation, and all
the integration patterns that would be common in real DRL experiments. This serves as both
a demonstration and a template for creating custom utility agents.

Key Concepts/Algorithms:
- BaseAgent abstract class inheritance and implementation
- Configuration-driven design with YAML support
- Comprehensive logging and metrics tracking
- Noise generation for exploration
- Integration patterns between utility components

Important Parameters/Configurations:
- All parameters configurable via YAML files
- Logging directory and experiment tracking
- Random seeds for reproducibility
- Agent-specific configuration sections

Expected Inputs/Outputs:
- Inputs: YAML configuration files, sample data
- Outputs: Log files, metrics, experiment results

Dependencies:
- numpy, PyYAML, matplotlib, logging
- src/utils/base_agent.py, src/utils/logger.py, src/utils/hyperparameters.py, src/utils/noise.py

Author: REL Project Team
Date: 2025-07-13
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import yaml
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.base_agent import BaseAgent, create_agent
from utils.logger import Logger
from utils.hyperparameters import HyperparameterManager
from utils.noise import GaussianNoise, OrnsteinUhlenbeckNoise, create_noise_from_config


class DataProcessorAgent(BaseAgent):
    """
    Example Data Processing Utility Agent.
    
    This agent demonstrates how to create a utility agent that processes data
    with configurable parameters and comprehensive logging.
    """
    
    def _get_default_config(self):
        return {
            'batch_size': 32,
            'processing_mode': 'normalize',
            'output_dim': 10,
            'use_noise': True,
            'noise': {
                'type': 'gaussian',
                'mean': 0.0,
                'std': 0.1
            }
        }
    
    def _validate_config(self):
        required_params = ['batch_size', 'processing_mode', 'output_dim']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")
        
        if self.config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.config['output_dim'] <= 0:
            raise ValueError("output_dim must be positive")
    
    def _setup_agent(self):
        # Initialize agent-specific components
        self.state['total_samples_processed'] = 0
        self.state['processing_history'] = []
        
        # Setup noise generator if enabled
        if self.config.get('use_noise', False):
            noise_config = self.config.get('noise', {})
            self.noise_generator = create_noise_from_config(
                noise_config, 
                action_dim=self.config['output_dim'], 
                seed=self.seed
            )
        else:
            self.noise_generator = None
        
        self.logger.info(f"DataProcessorAgent setup complete")
        self.logger.info(f"Configuration: {self.config}")
    
    def process(self, data):
        """
        Process input data according to configuration.
        
        Args:
            data: Input data to process (numpy array)
            
        Returns:
            Processed data with optional noise
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        batch_size = self.config['batch_size']
        processing_mode = self.config['processing_mode']
        output_dim = self.config['output_dim']
        
        # Process data based on mode
        if processing_mode == 'normalize':
            processed_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        elif processing_mode == 'standardize':
            processed_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        else:
            processed_data = data.copy()
        
        # Reshape to output dimension
        if len(processed_data) != output_dim:
            if len(processed_data) > output_dim:
                processed_data = processed_data[:output_dim]
            else:
                processed_data = np.pad(processed_data, (0, output_dim - len(processed_data)))
        
        # Add noise if enabled
        if self.noise_generator is not None:
            noise = self.noise_generator.sample()
            processed_data += noise
        
        # Update state and metrics
        self.state['total_samples_processed'] += len(data)
        self.state['processing_history'].append({
            'input_shape': data.shape,
            'output_shape': processed_data.shape,
            'processing_mode': processing_mode
        })
        
        # Log metrics
        self.log_metric('input_mean', np.mean(data))
        self.log_metric('input_std', np.std(data))
        self.log_metric('output_mean', np.mean(processed_data))
        self.log_metric('output_std', np.std(processed_data))
        
        self._step()
        
        return processed_data


class ExperimentTrackerAgent(BaseAgent):
    """
    Example Experiment Tracking Utility Agent.
    
    This agent demonstrates experiment tracking, metric aggregation,
    and result visualization capabilities.
    """
    
    def _get_default_config(self):
        return {
            'tracking_mode': 'full',
            'save_plots': True,
            'plot_interval': 100,
            'metrics_to_track': ['reward', 'loss', 'accuracy'],
            'aggregation_window': 50
        }
    
    def _validate_config(self):
        valid_modes = ['full', 'minimal', 'custom']
        if self.config.get('tracking_mode') not in valid_modes:
            raise ValueError(f"tracking_mode must be one of {valid_modes}")
        
        if self.config.get('plot_interval', 0) <= 0:
            raise ValueError("plot_interval must be positive")
    
    def _setup_agent(self):
        self.state['experiments'] = {}
        self.state['active_experiment'] = None
        self.state['plot_counter'] = 0
        
        # Create plots directory
        self.plots_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.logger.info("ExperimentTrackerAgent initialized")
    
    def process(self, experiment_data):
        """
        Process and track experiment data.
        
        Args:
            experiment_data: Dictionary containing experiment metrics
            
        Returns:
            Tracking summary and analysis
        """
        experiment_id = experiment_data.get('experiment_id', 'default')
        
        if experiment_id not in self.state['experiments']:
            self.state['experiments'][experiment_id] = {
                'metrics': {},
                'episodes': [],
                'start_time': self._step_count
            }
        
        experiment = self.state['experiments'][experiment_id]
        self.state['active_experiment'] = experiment_id
        
        # Track metrics
        for metric_name in self.config['metrics_to_track']:
            if metric_name in experiment_data:
                if metric_name not in experiment['metrics']:
                    experiment['metrics'][metric_name] = []
                experiment['metrics'][metric_name].append(experiment_data[metric_name])
                
                # Log to main logger
                self.log_metric(f"{experiment_id}_{metric_name}", experiment_data[metric_name])
        
        # Track episode data if provided
        if 'episode_data' in experiment_data:
            experiment['episodes'].append(experiment_data['episode_data'])
            self.log_episode_data(experiment_data['episode_data'])
        
        # Generate plots periodically
        if (self._step_count % self.config['plot_interval'] == 0 and 
            self.config.get('save_plots', True)):
            self._generate_plots(experiment_id)
        
        # Calculate moving averages
        summary = self._calculate_summary(experiment_id)
        
        self._step()
        
        return summary
    
    def _calculate_summary(self, experiment_id):
        """Calculate summary statistics for an experiment."""
        experiment = self.state['experiments'][experiment_id]
        summary = {'experiment_id': experiment_id}
        
        window = self.config['aggregation_window']
        
        for metric_name, values in experiment['metrics'].items():
            if values:
                summary[f'{metric_name}_latest'] = values[-1]
                summary[f'{metric_name}_mean'] = np.mean(values)
                
                if len(values) >= window:
                    recent_values = values[-window:]
                    summary[f'{metric_name}_recent_mean'] = np.mean(recent_values)
                    summary[f'{metric_name}_recent_std'] = np.std(recent_values)
        
        return summary
    
    def _generate_plots(self, experiment_id):
        """Generate visualization plots for an experiment."""
        experiment = self.state['experiments'][experiment_id]
        
        if not experiment['metrics']:
            return
        
        plt.figure(figsize=(12, 8))
        
        n_metrics = len(experiment['metrics'])
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        for i, (metric_name, values) in enumerate(experiment['metrics'].items()):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.plot(values)
            plt.title(f'{metric_name}')
            plt.xlabel('Step')
            plt.ylabel('Value')
            
            # Add moving average
            if len(values) > 10:
                window = min(50, len(values) // 4)
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(values)), moving_avg, 'r--', label=f'MA({window})')
                plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{experiment_id}_metrics_step_{self._step_count}.png"
        plot_path = os.path.join(self.plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.state['plot_counter'] += 1
        self.logger.info(f"Saved plot: {plot_path}")


def create_demo_config():
    """Create a demonstration configuration file."""
    config = {
        'data_processor': {
            'batch_size': 64,
            'processing_mode': 'normalize',
            'output_dim': 8,
            'use_noise': True,
            'noise': {
                'type': 'gaussian',
                'mean': 0.0,
                'std': 0.05,
                'decay_rate': 0.99,
                'min_std': 0.01
            }
        },
        'experiment_tracker': {
            'tracking_mode': 'full',
            'save_plots': True,
            'plot_interval': 25,
            'metrics_to_track': ['reward', 'loss', 'accuracy', 'episode_length'],
            'aggregation_window': 20
        },
        'logging': {
            'log_level': 'INFO',
            'save_to_file': True,
            'tensorboard_enabled': False
        }
    }
    return config


def main():
    """Main demonstration function."""
    print("=== Utility Agent Demonstration ===")
    
    # Create temporary directory for this demo
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create demo configuration
        config = create_demo_config()
        config_path = os.path.join(temp_dir, 'demo_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"Created configuration file: {config_path}")
        
        # Initialize hyperparameter manager
        print("\n1. Initializing HyperparameterManager...")
        hyperparams = HyperparameterManager(config_path=config_path)
        print(f"Loaded parameters: {list(hyperparams.get_all().keys())}")
        
        # Initialize main logger
        print("\n2. Initializing main Logger...")
        main_logger = Logger(
            experiment_name="utility_agent_demo",
            log_dir=temp_dir,
            save_to_file=True
        )
        main_logger.log_hyperparameters(hyperparams.get_all())
        
        # Initialize data processor agent
        print("\n3. Creating DataProcessorAgent...")
        data_processor = DataProcessorAgent(
            agent_name="data_processor",
            config_path=config_path,
            log_dir=temp_dir,
            seed=42
        )
        
        # Initialize experiment tracker agent
        print("\n4. Creating ExperimentTrackerAgent...")
        experiment_tracker = ExperimentTrackerAgent(
            agent_name="experiment_tracker", 
            config_path=config_path,
            log_dir=temp_dir,
            seed=42
        )
        
        # Demonstrate noise generation
        print("\n5. Demonstrating noise generators...")
        
        # Gaussian noise
        gaussian_noise = GaussianNoise(mean=0.0, std=0.2, size=5, seed=42)
        print(f"Gaussian noise sample: {gaussian_noise.sample()}")
        
        # Ornstein-Uhlenbeck noise
        ou_noise = OrnsteinUhlenbeckNoise(size=3, mu=0.0, theta=0.15, sigma=0.3, seed=42)
        print(f"OU noise samples:")
        for i in range(3):
            print(f"  Step {i}: {ou_noise.sample()}")
        
        # Simulate experiment workflow
        print("\n6. Running simulated experiment...")
        
        np.random.seed(42)
        
        for episode in range(100):
            # Generate synthetic data
            raw_data = np.random.randn(10) * 2 + 1
            
            # Process data
            processed_data = data_processor.process(raw_data)
            
            # Simulate experiment metrics
            reward = np.sum(processed_data) * 0.1 + np.random.randn() * 0.5
            loss = np.exp(-episode * 0.01) + np.random.randn() * 0.1
            accuracy = min(0.95, 0.5 + episode * 0.005 + np.random.randn() * 0.05)
            episode_length = 50 + int(np.random.randn() * 10)
            
            # Track experiment
            experiment_data = {
                'experiment_id': 'demo_experiment',
                'reward': reward,
                'loss': loss,
                'accuracy': accuracy,
                'episode_length': episode_length,
                'episode_data': {
                    'episode': episode,
                    'total_reward': reward,
                    'episode_length': episode_length,
                    'success': accuracy > 0.8
                }
            }
            
            summary = experiment_tracker.process(experiment_data)
            
            # Log to main logger
            main_logger.log_scalar('episode_reward', reward, step=episode)
            main_logger.log_scalar('episode_loss', loss, step=episode)
            main_logger.log_scalar('episode_accuracy', accuracy, step=episode)
            
            if episode % 20 == 0:
                print(f"Episode {episode}: reward={reward:.3f}, accuracy={accuracy:.3f}")
        
        # Demonstrate parameter search
        print("\n7. Demonstrating parameter search...")
        
        search_space = {
            'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-2},
            'batch_size': {'type': 'choice', 'choices': [16, 32, 64, 128]},
            'gamma': {'type': 'uniform', 'low': 0.9, 'high': 0.999}
        }
        
        hyperparams.define_search_space(search_space)
        
        for trial in range(5):
            sampled_params = hyperparams.sample_parameters(seed=42 + trial)
            print(f"Trial {trial}: {sampled_params}")
        
        # Generate final reports
        print("\n8. Generating final reports...")
        
        # Save metrics
        metrics_file = main_logger.save_metrics()
        print(f"Saved main metrics to: {metrics_file}")
        
        # Get summaries
        data_processor_info = data_processor.get_info()
        experiment_tracker_info = experiment_tracker.get_info()
        
        print(f"\nDataProcessor processed {data_processor_info['step_count']} samples")
        print(f"ExperimentTracker tracked {experiment_tracker_info['step_count']} datapoints")
        
        # Display metrics summary
        metrics_summary = main_logger.get_metrics_summary()
        print(f"\nMetrics Summary:")
        for metric_name, stats in metrics_summary.items():
            print(f"  {metric_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        # Cleanup
        print("\n9. Cleanup...")
        data_processor.cleanup()
        experiment_tracker.cleanup()
        main_logger.close()
        
        print("\n=== Demo completed successfully! ===")
        print(f"All files saved in: {temp_dir}")
        
        # List generated files
        print("\nGenerated files:")
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), temp_dir)
                print(f"  {rel_path}")


if __name__ == "__main__":
    main()