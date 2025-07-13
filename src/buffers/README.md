# Buffer Implementations Documentation

## Overview

This module provides experience replay buffer implementations for Deep Reinforcement Learning (DRL) algorithms, following the modular architecture described in "Deep Reinforcement Learning: A Survey" by Wang et al. The implementation includes a base abstract class and two concrete implementations: standard uniform replay buffer and prioritized replay buffer.

## Architecture

### BaseBuffer (Abstract Base Class)

The `BaseBuffer` class provides the common interface and shared functionality for all buffer implementations:

```python
from src.buffers import BaseBuffer

class MyCustomBuffer(BaseBuffer):
    def _init_storage(self):
        # Initialize your storage structure
        pass
    
    def add(self, obs, action, reward, next_obs, done, **kwargs):
        # Add transition to buffer
        pass
    
    def sample(self, **kwargs):
        # Sample transitions from buffer
        pass
```

**Key Features:**
- Abstract base class ensuring consistent interface
- Common utility methods (`size()`, `can_sample()`, `clear()`)
- Device management for PyTorch tensors
- Reproducible sampling with seed support
- Type hints and comprehensive documentation

### ReplayBuffer (Uniform Sampling)

Standard experience replay buffer with uniform random sampling:

```python
from src.buffers import ReplayBuffer

# Create buffer
buffer = ReplayBuffer(
    buffer_size=10000,
    batch_size=32,
    device='cpu',
    seed=42
)

# Add transitions
buffer.add(obs, action, reward, next_obs, done)

# Sample batch
if buffer.can_sample():
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = buffer.sample()
```

**Features:**
- Circular buffer implementation for memory efficiency
- Dynamic storage initialization based on first transition
- Uniform random sampling
- Support for various data types (numpy arrays, PyTorch tensors)
- Buffer extension functionality

### PrioritizedReplayBuffer (Priority-Based Sampling)

Prioritized experience replay buffer using sum-tree for efficient sampling:

```python
from src.buffers import PrioritizedReplayBuffer

# Create prioritized buffer
buffer = PrioritizedReplayBuffer(
    buffer_size=10000,
    batch_size=32,
    alpha=0.6,       # Prioritization strength
    beta=0.4,        # Importance sampling correction
    epsilon=1e-6,    # Small value for numerical stability
    device='cpu'
)

# Add transitions with priorities
buffer.add(obs, action, reward, next_obs, done, priority=1.5)

# Sample batch with importance weights
if buffer.can_sample():
    obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, weights, indices = buffer.sample()
    
    # Update priorities based on new TD errors
    new_priorities = compute_td_errors(batch)
    buffer.update_priorities(indices, new_priorities)
```

**Features:**
- Sum-tree data structure for O(log n) sampling and updates
- Stochastic prioritization based on TD errors
- Importance sampling weights for bias correction
- Configurable alpha (prioritization) and beta (importance sampling) parameters
- Efficient priority updates

## Factory Function

Convenient factory function for creating buffers:

```python
from src.buffers import create_buffer

# Create replay buffer
replay_buffer = create_buffer('replay', buffer_size=1000, batch_size=32)

# Create prioritized buffer
prio_buffer = create_buffer('prioritized', buffer_size=1000, batch_size=32, alpha=0.6, beta=0.4)
```

## Configuration

Buffers are designed to work with YAML configuration files:

```yaml
# configs/dqn_config.yaml
buffer:
  type: "replay"
  buffer_size: 100000
  batch_size: 32
  device: "cpu"
  seed: 42

# configs/rainbow_dqn_config.yaml  
buffer:
  type: "prioritized"
  buffer_size: 100000
  batch_size: 32
  alpha: 0.6
  beta: 0.4
  epsilon: 1e-6
  device: "cuda"
```

## Usage in Algorithms

### DQN with Replay Buffer

```python
from src.buffers import ReplayBuffer

class DQNAgent:
    def __init__(self, config):
        self.replay_buffer = ReplayBuffer(
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            device=config['device']
        )
    
    def update(self):
        if self.replay_buffer.can_sample():
            batch = self.replay_buffer.sample()
            # Perform Q-learning update
```

### Rainbow DQN with Prioritized Replay

```python
from src.buffers import PrioritizedReplayBuffer

class RainbowDQNAgent:
    def __init__(self, config):
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            alpha=config['alpha'],
            beta=config['beta']
        )
    
    def update(self):
        if self.replay_buffer.can_sample():
            *batch, weights, indices = self.replay_buffer.sample()
            
            # Compute TD errors
            td_errors = self.compute_td_errors(batch)
            
            # Update priorities
            priorities = np.abs(td_errors) + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
            
            # Weighted loss computation
            loss = self.compute_loss(batch, weights)
```

## Testing

Run the test script to verify implementations:

```bash
cd RELLab3
python test_buffers.py
```

## Performance Considerations

1. **Memory Efficiency**: Buffers use numpy arrays for storage and convert to PyTorch tensors only when sampling
2. **Sum-tree Operations**: O(log n) complexity for priority updates and sampling in prioritized buffer
3. **Batch Processing**: Efficient batch sampling and tensor operations
4. **Device Management**: Automatic tensor placement on specified device (CPU/GPU)

## Extensibility

The modular design allows easy extension for new buffer types:

- **N-step Returns**: Implement N-step return computation
- **Hindsight Experience Replay (HER)**: Add goal-conditioned transitions
- **Reservoir Sampling**: Implement for streaming data scenarios
- **Compressed Replay**: Add state compression for image observations

## Dependencies

- `numpy`: Efficient array operations and storage
- `torch`: Tensor operations and device management
- `typing`: Type hints for better code documentation
- `abc`: Abstract base class functionality

## Integration with Algorithms

This buffer implementation integrates seamlessly with the algorithm implementations:

- **DQN**: Uses `ReplayBuffer` for experience replay
- **DDPG**: Uses `ReplayBuffer` for off-policy learning
- **SAC**: Uses `ReplayBuffer` with entropy regularization
- **Rainbow DQN**: Uses `PrioritizedReplayBuffer` for improved sample efficiency

The modular design ensures that algorithms can easily switch between buffer types based on configuration without code changes.