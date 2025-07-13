# Environment Wrapper Implementation Guide

## Overview

This document provides a comprehensive guide for the BaseEnvironmentWrapper implementation in the Modular DRL Framework. The implementation follows the design patterns established in the project and adheres to the guidelines specified in the INSTRUCTION.md and project documentation.

## Architecture

### Base Class: `BaseEnvironmentWrapper`

The `BaseEnvironmentWrapper` is an abstract base class (ABC) that provides:

1. **Standardized Interface**: Compatible with Gymnasium environments
2. **Configuration Management**: YAML-based parameter loading
3. **Episode Tracking**: Comprehensive statistics collection
4. **Reproducibility**: Seeding and consistent interfaces
5. **Extensibility**: Abstract methods for customization

### Key Features

#### 1. Abstract Methods (Must be implemented by subclasses)
- `_init_wrapper_components()`: Initialize wrapper-specific components
- `_preprocess_observation()`: Preprocess observations before returning to agent
- `_transform_action()`: Transform actions before passing to environment
- `_shape_reward()`: Apply reward shaping or normalization

#### 2. Core Functionality
- Environment creation and management
- Episode statistics tracking
- Configuration loading from YAML files
- Gymnasium compatibility through property delegation

#### 3. Type Safety
- Comprehensive type aliases for better code readability
- Type hints throughout the implementation
- Union types for flexible input/output handling

## Implementation Details

### Class Hierarchy

```
BaseEnvironmentWrapper (ABC)
├── StandardEnvironmentWrapper
│   ├── AtariWrapper
│   └── ContinuousControlWrapper
└── [Custom implementations...]
```

### Configuration System

The wrapper supports YAML-based configuration through the `config_path` parameter:

```yaml
environment:
  env_id: "CartPole-v1"
  wrapper_type: "StandardEnvironmentWrapper"
  normalize_obs: false
  normalize_rewards: false
  clip_actions: true
  reward_scale: 1.0
```

### Episode Statistics

The base class automatically tracks:
- Episode rewards and lengths
- Total environment steps
- Episode count
- Current episode metrics

## Concrete Implementations

### 1. StandardEnvironmentWrapper

A general-purpose wrapper providing:
- Optional observation normalization
- Action clipping to action space bounds
- Reward scaling and normalization
- Running statistics tracking

### 2. AtariWrapper

Specialized for Atari environments with:
- Atari-specific preprocessing defaults
- Extensible for frame stacking, grayscale conversion, etc.

### 3. ContinuousControlWrapper

Optimized for continuous control tasks:
- Observation and reward normalization enabled by default
- Action clipping enabled
- Suitable for robotics and control environments

## Usage Examples

### Basic Usage

```python
from environments.env_wrapper import StandardEnvironmentWrapper

# Create wrapper
env = StandardEnvironmentWrapper(
    env_id="CartPole-v1",
    seed=42,
    normalize_obs=False,
    clip_actions=True
)

# Use like a standard Gymnasium environment
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Get episode statistics
stats = env.get_episode_statistics()
print(f"Mean reward: {stats['mean_reward']}")
```

### Configuration-Based Usage

```python
# Load wrapper configuration from YAML
env = StandardEnvironmentWrapper(
    env_id="CartPole-v1",
    config_path="configs/environment_config.yaml"
)
```

### Custom Wrapper Implementation

```python
from environments import BaseEnvironmentWrapper

class CustomWrapper(BaseEnvironmentWrapper):
    def _init_wrapper_components(self):
        # Initialize custom components
        pass
    
    def _preprocess_observation(self, observation):
        # Custom observation preprocessing
        return observation
    
    def _transform_action(self, action):
        # Custom action transformation
        return action
    
    def _shape_reward(self, reward, info):
        # Custom reward shaping
        return reward
```

## Design Principles

### 1. Modularity
- Clear separation of concerns between base class and implementations
- Abstract methods enable customization without breaking base functionality
- Pluggable components for different preprocessing needs

### 2. Consistency
- Follows the same patterns as `BaseBuffer` in the project
- Compatible with Gymnasium API
- Consistent error handling and logging

### 3. Configurability
- YAML-based configuration for all parameters
- Environment variables and runtime parameters supported
- No hardcoded values in the implementation

### 4. Reproducibility
- Comprehensive seeding support
- Configuration persistence
- Episode statistics saving and loading

### 5. Extensibility
- Easy to add new wrapper types
- Abstract base class enforces interface compliance
- Support for complex preprocessing pipelines

## Integration with DRL Framework

### Algorithm Compatibility

The wrappers are designed to work seamlessly with all DRL algorithms in the framework:

- **DQN**: Discrete action spaces, experience replay compatibility
- **DDPG**: Continuous action spaces, action noise support
- **PPO**: Both discrete and continuous environments
- **SAC**: Maximum entropy compatible reward shaping

### Buffer Integration

The wrapper outputs are compatible with the framework's buffer implementations:

```python
# Example integration with replay buffer
obs, _ = env.reset()
for step in range(1000):
    action = agent.select_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Store in buffer
    buffer.add(obs, action, reward, next_obs, terminated)
    
    obs = next_obs
    if terminated or truncated:
        obs, _ = env.reset()
```

### Configuration Integration

Environment configurations integrate with algorithm configurations:

```python
# Example: Loading environment config within algorithm config
algorithm_config = {
    'environment': {
        'env_id': 'CartPole-v1',
        'wrapper_type': 'StandardEnvironmentWrapper',
        'normalize_obs': False
    },
    'training': {
        'total_timesteps': 100000,
        'batch_size': 32
    }
}
```

## Testing and Validation

### Structure Validation

Run the structure test to validate the implementation:

```bash
python test_structure.py
```

### Functional Testing

For environments with dependencies installed:

```bash
python test_env_wrappers.py
```

## Best Practices

### 1. When to Create Custom Wrappers

- Environment-specific preprocessing needs
- Novel reward shaping techniques
- Multi-agent environment support
- Domain-specific observation transformations

### 2. Configuration Management

- Use YAML files for all configurable parameters
- Document all configuration options
- Provide sensible defaults for common use cases

### 3. Performance Considerations

- Minimize preprocessing overhead in tight loops
- Use vectorized operations where possible
- Cache expensive computations when appropriate

### 4. Error Handling

- Validate environment compatibility during initialization
- Provide informative error messages
- Graceful degradation for optional features

## Future Extensions

The base class design supports future enhancements:

1. **Multi-agent environments**: Extend for multiple agents
2. **Distributed training**: Support for distributed environment management
3. **Advanced preprocessing**: Integration with computer vision libraries
4. **Real-time environments**: Support for real-world robotic systems
5. **Environment benchmarking**: Automatic performance profiling

## Conclusion

The BaseEnvironmentWrapper implementation provides a robust, extensible foundation for environment management in the Modular DRL Framework. It follows established design patterns, supports comprehensive configuration, and enables easy customization while maintaining compatibility with the broader framework ecosystem.