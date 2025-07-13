# Utility Agent Integration Guide

## Overview

This guide demonstrates how to integrate and use the utility agent base classes and supporting modules in the Modular DRL Framework. The utility system provides a robust foundation for building DRL experiments with proper logging, configuration management, and reproducibility.

## Core Components

### 1. BaseAgent Abstract Class

The `BaseAgent` class provides a unified interface for all utility agents in the framework.

#### Key Features:
- YAML-based configuration management
- Integrated logging and metrics tracking
- Hyperparameter management
- Deterministic seeding for reproducibility
- State persistence and loading
- Lifecycle management (setup, reset, cleanup)

#### Usage Example:

```python
from src.utils.base_agent import BaseAgent

class MyUtilityAgent(BaseAgent):
    def _get_default_config(self):
        return {
            'param1': 'default_value',
            'param2': 42
        }
    
    def _validate_config(self):
        if 'param1' not in self.config:
            raise ValueError("Missing required parameter: param1")
    
    def _setup_agent(self):
        # Initialize agent-specific components
        self.state['initialized'] = True
    
    def process(self, data):
        # Main processing logic
        result = self.my_processing_function(data)
        self._step()  # Increment step counter
        return result

# Initialize agent
agent = MyUtilityAgent(
    agent_name="my_agent",
    config_path="configs/my_agent_config.yaml",
    log_dir="./logs",
    seed=42
)

# Use agent
result = agent.process(input_data)

# Get agent information
info = agent.get_info()
print(f"Agent processed {info['step_count']} items")

# Save and load state
agent.save_state("checkpoints/agent_state.json")
agent.load_state("checkpoints/agent_state.json")
```

### 2. Logger Utility

The `Logger` class provides comprehensive experiment tracking and metrics logging.

#### Key Features:
- Multi-format logging (console, file, TensorBoard)
- Scalar metrics tracking with statistical analysis
- Episode-level data logging
- Hyperparameter persistence
- Metrics filtering and summarization

#### Usage Example:

```python
from src.utils.logger import Logger

# Initialize logger
logger = Logger(
    experiment_name="my_experiment",
    log_dir="./logs",
    save_to_file=True,
    tensorboard_enabled=True
)

# Log scalar metrics
logger.log_scalar('reward', 100.5, step=1)
logger.log_scalar('loss', 0.25, step=1)

# Log episode data
logger.log_episode({
    'episode': 1,
    'total_reward': 150.0,
    'episode_length': 200,
    'success': True
})

# Log hyperparameters
logger.log_hyperparameters({
    'learning_rate': 0.001,
    'batch_size': 32
})

# Get metrics summary
summary = logger.get_metrics_summary()

# Save metrics to file
logger.save_metrics()
```

### 3. HyperparameterManager

The `HyperparameterManager` class provides YAML-based configuration management with validation and search capabilities.

#### Key Features:
- YAML configuration loading and saving
- Hierarchical parameter access with dot notation
- Parameter validation and type checking
- Search space definition and sampling
- Dynamic parameter updates

#### Usage Example:

```python
from src.utils.hyperparameters import HyperparameterManager

# Initialize from config file
manager = HyperparameterManager('configs/experiment_config.yaml')

# Access parameters
lr = manager.get('learning_rate', 0.001)
batch_size = manager.get('training.batch_size', 32)

# Set parameters
manager.set('learning_rate', 0.0005)
manager.set('network.hidden_sizes', [64, 64])

# Validate parameters
is_valid = manager.validate_range('learning_rate', min_val=0.0, max_val=1.0)

# Define search space
search_space = {
    'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-2},
    'batch_size': {'type': 'choice', 'choices': [16, 32, 64, 128]}
}
manager.define_search_space(search_space)

# Sample parameters
sampled_params = manager.sample_parameters(seed=42)
manager.apply_sampled_parameters(sampled_params)
```

### 4. Noise Generators

The noise module provides various noise processes for exploration in continuous action spaces.

#### Key Features:
- Gaussian (Normal) noise for uncorrelated exploration
- Ornstein-Uhlenbeck noise for temporally correlated exploration
- Adaptive noise with scheduling capabilities
- Configuration-driven noise creation

#### Usage Example:

```python
from src.utils.noise import GaussianNoise, OrnsteinUhlenbeckNoise, create_noise_from_config

# Gaussian noise
gaussian_noise = GaussianNoise(
    mean=0.0,
    std=0.1,
    size=4,  # 4-dimensional action space
    decay_rate=0.995,
    seed=42
)

# Sample noise
noise_sample = gaussian_noise.sample()  # Shape: (4,)

# Apply decay
gaussian_noise.decay()

# Ornstein-Uhlenbeck noise
ou_noise = OrnsteinUhlenbeckNoise(
    size=2,
    mu=0.0,
    theta=0.15,
    sigma=0.2,
    seed=42
)

# Sample correlated noise
noise_sample = ou_noise.sample()

# Create noise from configuration
noise_config = {
    'type': 'ou',
    'theta': 0.15,
    'sigma': 0.2
}
noise = create_noise_from_config(noise_config, action_dim=4, seed=42)
```

## Integration Patterns

### 1. Basic Agent with Logging

```python
class DataProcessorAgent(BaseAgent):
    def _get_default_config(self):
        return {'batch_size': 32, 'normalize': True}
    
    def _validate_config(self):
        assert self.config['batch_size'] > 0
    
    def _setup_agent(self):
        self.state['total_processed'] = 0
    
    def process(self, data):
        # Process data
        if self.config['normalize']:
            data = (data - np.mean(data)) / np.std(data)
        
        # Log metrics
        self.log_metric('data_mean', np.mean(data))
        self.log_metric('data_std', np.std(data))
        
        self.state['total_processed'] += len(data)
        self._step()
        
        return data
```

### 2. Agent with Hyperparameter Integration

```python
class ExperimentAgent(BaseAgent):
    def _setup_agent(self):
        # Access hyperparameters
        self.lr = self.hyperparams.get('learning_rate', 0.001)
        self.batch_size = self.hyperparams.get('batch_size', 32)
        
        # Log hyperparameters
        self.logger.log_hyperparameters(self.hyperparams.get_all())
    
    def process(self, data):
        # Use hyperparameters in processing
        processed = self.process_with_lr(data, self.lr)
        return processed
```

### 3. Agent with Noise Integration

```python
class ExplorationAgent(BaseAgent):
    def _setup_agent(self):
        # Setup noise generator
        noise_config = self.config.get('noise', {})
        self.noise = create_noise_from_config(
            noise_config, 
            action_dim=self.config['action_dim'],
            seed=self.seed
        )
    
    def process(self, action):
        # Add exploration noise
        if self.noise is not None:
            noise = self.noise.sample()
            noisy_action = action + noise
            
            # Log noise statistics
            self.log_metric('noise_magnitude', np.linalg.norm(noise))
            
            return noisy_action
        return action
```

## Configuration Management

### YAML Configuration Structure

```yaml
# config/my_experiment.yaml
my_agent:
  batch_size: 64
  learning_rate: 0.001
  use_noise: true
  noise:
    type: "gaussian"
    mean: 0.0
    std: 0.1
    decay_rate: 0.995

logging:
  log_level: "INFO"
  save_to_file: true
  tensorboard_enabled: false

# Nested parameters
network:
  hidden_sizes: [64, 64]
  activation: "relu"
  dropout: 0.1
```

### Using Configurations

```python
# Load configuration
manager = HyperparameterManager('config/my_experiment.yaml')

# Create agent with config
agent = MyAgent(
    agent_name="my_agent",
    config_path="config/my_experiment.yaml",
    log_dir="./logs",
    seed=42
)

# Access nested parameters
hidden_sizes = agent.get_config('network.hidden_sizes', [32, 32])
dropout = agent.hyperparams.get('network.dropout', 0.0)
```

## Best Practices

### 1. Configuration Management
- Use YAML files for all experiment parameters
- Implement proper validation in `_validate_config()`
- Provide sensible defaults in `_get_default_config()`
- Use dot notation for nested parameter access

### 2. Logging and Metrics
- Log key metrics at each processing step
- Use episode-level logging for aggregated data
- Include hyperparameters in experiment logs
- Save metrics periodically for large experiments

### 3. Reproducibility
- Always set random seeds for reproducibility
- Save complete configuration with results
- Use state persistence for long-running experiments
- Version control your configuration files

### 4. Error Handling
- Implement robust validation in agents
- Use try-catch blocks for external dependencies
- Provide meaningful error messages
- Include fallback options where appropriate

### 5. Testing
- Write unit tests for all agent functionality
- Test configuration validation thoroughly
- Include integration tests for workflows
- Test reproducibility with fixed seeds

## Example Complete Workflow

```python
import numpy as np
from src.utils import *

# 1. Setup configuration
config_path = "configs/my_experiment.yaml"
hyperparams = HyperparameterManager(config_path)

# 2. Setup logging
logger = Logger("my_experiment", "./logs", save_to_file=True)
logger.log_hyperparameters(hyperparams.get_all())

# 3. Create agents
data_processor = DataProcessorAgent(
    agent_name="data_processor",
    config_path=config_path,
    log_dir="./logs",
    seed=42
)

experiment_tracker = ExperimentTrackerAgent(
    agent_name="experiment_tracker",
    config_path=config_path,
    log_dir="./logs",
    seed=42
)

# 4. Run experiment
for episode in range(100):
    # Generate data
    data = np.random.randn(50)
    
    # Process data
    processed_data = data_processor.process(data)
    
    # Track experiment
    experiment_data = {
        'episode': episode,
        'reward': np.sum(processed_data) * 0.1,
        'success': np.mean(processed_data) > 0
    }
    
    summary = experiment_tracker.process(experiment_data)
    
    # Log to main logger
    logger.log_scalar('episode_reward', experiment_data['reward'], episode)

# 5. Save results
logger.save_metrics()
data_processor.save_state("checkpoints/data_processor.json")
experiment_tracker.save_state("checkpoints/experiment_tracker.json")

# 6. Cleanup
data_processor.cleanup()
experiment_tracker.cleanup()
logger.close()
```

This integration guide provides a comprehensive overview of how to use the utility agent system effectively in DRL experiments.