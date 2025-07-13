# Environments Module

This module provides environment wrappers for the Modular Deep Reinforcement Learning Framework.

## Quick Start

```python
from environments.env_wrapper import StandardEnvironmentWrapper

# Create a wrapped environment
env = StandardEnvironmentWrapper(
    env_id="CartPole-v1",
    seed=42,
    normalize_obs=False,
    clip_actions=True
)

# Use like any Gymnasium environment
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Architecture

- **BaseEnvironmentWrapper**: Abstract base class defining the interface
- **StandardEnvironmentWrapper**: General-purpose wrapper with common preprocessing
- **AtariWrapper**: Specialized for Atari environments
- **ContinuousControlWrapper**: Optimized for continuous control tasks

## Key Features

- ✅ Gymnasium API compatibility
- ✅ YAML configuration support
- ✅ Episode statistics tracking
- ✅ Reproducible seeding
- ✅ Modular design for easy extension
- ✅ Type-safe implementation

## Configuration

Environment wrappers can be configured via YAML files:

```yaml
environment:
  env_id: "CartPole-v1"
  wrapper_type: "StandardEnvironmentWrapper"
  normalize_obs: false
  clip_actions: true
  reward_scale: 1.0
  seed: 42
```

## Available Wrappers

| Wrapper | Use Case | Key Features |
|---------|----------|--------------|
| `StandardEnvironmentWrapper` | General RL environments | Observation normalization, action clipping, reward scaling |
| `AtariWrapper` | Atari games | Atari-specific preprocessing defaults |
| `ContinuousControlWrapper` | Robotics/Control | Normalization enabled, continuous action support |

## Creating Custom Wrappers

```python
from environments import BaseEnvironmentWrapper

class MyCustomWrapper(BaseEnvironmentWrapper):
    def _init_wrapper_components(self):
        # Initialize custom components
        pass
    
    def _preprocess_observation(self, observation):
        # Custom preprocessing
        return observation
    
    def _transform_action(self, action):
        # Custom action transformation
        return action
    
    def _shape_reward(self, reward, info):
        # Custom reward shaping
        return reward
```

## Testing

Validate the module structure:

```bash
python test_structure.py
```

## Documentation

See `docs/environment_wrapper_guide.md` for comprehensive documentation.

## Dependencies

- `gymnasium`: Environment API
- `numpy`: Numerical operations
- `torch`: Tensor operations (optional)
- `PyYAML`: Configuration loading