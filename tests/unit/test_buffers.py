"""
Unit Tests for Buffers.

This module contains comprehensive unit tests for experience replay buffer implementations in the 
Modular DRL Framework, as described in the DRL survey and project engineering blueprint.

Detailed Description:
Tests in this module validate the correctness, efficiency, and reproducibility of uniform and 
prioritized replay buffers. Each test targets buffer storage, sampling, and edge cases using mock 
transitions and controlled random seeds. The structure supports extensibility for new buffer types 
and improvements.

Key Concepts/Algorithms:
- Isolated testing of buffer storage and sampling
- Prioritized sampling logic validation
- Reproducibility checks via random seeds
- Edge case handling and error conditions

Important Parameters/Configurations:
- Buffer size, batch size, and prioritization parameters (from config)
- Random seed specification for reproducibility

Expected Inputs/Outputs:
- Inputs: mock transitions, buffer parameters
- Outputs: pass/fail results, coverage reports

Dependencies:
- pytest, unittest, numpy, torch, mock (if needed)

Author: REL Project Team
Date: 2025-07-13
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch
import pytest
import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from buffers import BaseBuffer, create_buffer
    from buffers.replay_buffer import ReplayBuffer
    from buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pytestmark = pytest.mark.skip("Buffer modules not available")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Buffer imports not available")
class TestBaseBuffer(unittest.TestCase):
    """Test cases for BaseBuffer abstract base class."""
    
    def test_base_buffer_is_abstract(self):
        """Test that BaseBuffer cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseBuffer(buffer_size=100, batch_size=32)
    
    def test_abstract_methods_defined(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = BaseBuffer.__abstractmethods__
        expected_methods = {'_init_storage', 'add', 'sample'}
        self.assertTrue(expected_methods.issubset(abstract_methods))


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Buffer imports not available")
class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer_size = 100
        self.batch_size = 32
        self.device = 'cpu'
        self.seed = 42
        self.buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed
        )
    
    def tearDown(self):
        """Clean up after tests."""
        del self.buffer
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(self.buffer.buffer_size, self.buffer_size)
        self.assertEqual(self.buffer.batch_size, self.batch_size)
        self.assertEqual(self.buffer.device, torch.device(self.device))
        self.assertEqual(self.buffer.pos, 0)
        self.assertFalse(self.buffer.full)
    
    def test_add_single_transition(self):
        """Test adding a single transition."""
        obs = np.random.randn(4)
        action = 1
        reward = 0.5
        next_obs = np.random.randn(4)
        done = False
        
        self.buffer.add(obs, action, reward, next_obs, done)
        
        self.assertEqual(self.buffer.size(), 1)
        self.assertEqual(self.buffer.pos, 1)
        self.assertFalse(self.buffer.full)
    
    def test_add_multiple_transitions(self):
        """Test adding multiple transitions."""
        num_transitions = 50
        
        for i in range(num_transitions):
            obs = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_obs = np.random.randn(4)
            done = np.random.choice([True, False])
            
            self.buffer.add(obs, action, reward, next_obs, done)
        
        self.assertEqual(self.buffer.size(), num_transitions)
        self.assertEqual(self.buffer.pos, num_transitions)
        self.assertFalse(self.buffer.full)
    
    def test_buffer_overflow(self):
        """Test buffer behavior when it overflows."""
        # Fill buffer beyond capacity
        for i in range(self.buffer_size + 10):
            obs = np.random.randn(4)
            action = i % 2
            reward = float(i)
            next_obs = np.random.randn(4)
            done = i == (self.buffer_size + 9)
            
            self.buffer.add(obs, action, reward, next_obs, done)
        
        self.assertEqual(self.buffer.size(), self.buffer_size)
        self.assertTrue(self.buffer.full)
        self.assertEqual(self.buffer.pos, 10)  # Wrapped around
    
    def test_can_sample(self):
        """Test can_sample method."""
        # Empty buffer cannot sample
        self.assertFalse(self.buffer.can_sample())
        
        # Add transitions less than batch size
        for i in range(self.batch_size - 1):
            self.buffer.add(np.random.randn(4), 0, 0.0, np.random.randn(4), False)
        self.assertFalse(self.buffer.can_sample())
        
        # Add one more to reach batch size
        self.buffer.add(np.random.randn(4), 0, 0.0, np.random.randn(4), False)
        self.assertTrue(self.buffer.can_sample())
    
    def test_sample_batch(self):
        """Test sampling a batch of transitions."""
        # Fill buffer with known data
        obs_dim = 4
        for i in range(self.batch_size * 2):
            obs = np.full(obs_dim, i, dtype=np.float32)
            action = i % 2
            reward = float(i)
            next_obs = np.full(obs_dim, i + 1, dtype=np.float32)
            done = (i % 10 == 9)
            
            self.buffer.add(obs, action, reward, next_obs, done)
        
        # Sample batch
        batch = self.buffer.sample()
        
        # Check batch structure
        self.assertEqual(len(batch), 5)  # obs, action, reward, next_obs, done
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch
        
        # Check batch sizes
        self.assertEqual(obs_batch.shape[0], self.batch_size)
        self.assertEqual(action_batch.shape[0], self.batch_size)
        self.assertEqual(reward_batch.shape[0], self.batch_size)
        self.assertEqual(next_obs_batch.shape[0], self.batch_size)
        self.assertEqual(done_batch.shape[0], self.batch_size)
        
        # Check data types
        self.assertTrue(isinstance(obs_batch, torch.Tensor))
        self.assertTrue(isinstance(action_batch, torch.Tensor))
        self.assertTrue(isinstance(reward_batch, torch.Tensor))
        self.assertTrue(isinstance(next_obs_batch, torch.Tensor))
        self.assertTrue(isinstance(done_batch, torch.Tensor))
    
    def test_sample_reproducibility(self):
        """Test that sampling is reproducible with same seed."""
        # Fill buffer
        for i in range(self.batch_size * 2):
            self.buffer.add(np.random.randn(4), i % 2, float(i), np.random.randn(4), False)
        
        # Sample twice with reset seed
        torch.manual_seed(42)
        batch1 = self.buffer.sample()
        
        torch.manual_seed(42)
        batch2 = self.buffer.sample()
        
        # Check that batches are identical
        for b1, b2 in zip(batch1, batch2):
            torch.testing.assert_close(b1, b2)
    
    def test_clear_buffer(self):
        """Test clearing the buffer."""
        # Add some transitions
        for i in range(10):
            self.buffer.add(np.random.randn(4), 0, 0.0, np.random.randn(4), False)
        
        self.assertEqual(self.buffer.size(), 10)
        
        # Clear buffer
        self.buffer.clear()
        
        self.assertEqual(self.buffer.size(), 0)
        self.assertEqual(self.buffer.pos, 0)
        self.assertFalse(self.buffer.full)
        self.assertFalse(self.buffer.can_sample())
    
    def test_different_observation_shapes(self):
        """Test buffer with different observation shapes."""
        # Test with image observations
        img_buffer = ReplayBuffer(100, 16, device='cpu', seed=42)
        
        img_obs = np.random.randn(84, 84, 3)
        action = 1
        reward = 1.0
        next_img_obs = np.random.randn(84, 84, 3)
        done = False
        
        img_buffer.add(img_obs, action, reward, next_img_obs, done)
        self.assertEqual(img_buffer.size(), 1)
    
    def test_sample_empty_buffer(self):
        """Test sampling from empty buffer raises appropriate error."""
        with self.assertRaises(AssertionError):
            self.buffer.sample()


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Buffer imports not available")
class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Test cases for PrioritizedReplayBuffer implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer_size = 100
        self.batch_size = 32
        self.alpha = 0.6
        self.beta = 0.4
        self.device = 'cpu'
        self.seed = 42
        self.buffer = PrioritizedReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            alpha=self.alpha,
            beta=self.beta,
            device=self.device,
            seed=self.seed
        )
    
    def tearDown(self):
        """Clean up after tests."""
        del self.buffer
    
    def test_initialization(self):
        """Test prioritized buffer initialization."""
        self.assertEqual(self.buffer.alpha, self.alpha)
        self.assertEqual(self.buffer.beta, self.beta)
        self.assertEqual(self.buffer.buffer_size, self.buffer_size)
        self.assertEqual(self.buffer.batch_size, self.batch_size)
    
    def test_add_with_priority(self):
        """Test adding transitions with explicit priority."""
        obs = np.random.randn(4)
        action = 1
        reward = 0.5
        next_obs = np.random.randn(4)
        done = False
        priority = 2.0
        
        self.buffer.add(obs, action, reward, next_obs, done, priority=priority)
        
        self.assertEqual(self.buffer.size(), 1)
    
    def test_add_without_priority(self):
        """Test adding transitions without explicit priority (uses max priority)."""
        obs = np.random.randn(4)
        action = 1
        reward = 0.5
        next_obs = np.random.randn(4)
        done = False
        
        self.buffer.add(obs, action, reward, next_obs, done)
        
        self.assertEqual(self.buffer.size(), 1)
    
    def test_sample_with_priorities(self):
        """Test sampling with importance weights."""
        # Add transitions with different priorities
        priorities = [1.0, 2.0, 0.5, 3.0]
        for i, priority in enumerate(priorities * 10):  # Repeat to fill buffer
            obs = np.random.randn(4)
            action = i % 2
            reward = float(i)
            next_obs = np.random.randn(4)
            done = False
            
            self.buffer.add(obs, action, reward, next_obs, done, priority=priority)
        
        # Sample batch
        *batch, weights, indices = self.buffer.sample()
        
        # Check batch structure
        self.assertEqual(len(batch), 5)  # obs, action, reward, next_obs, done
        
        # Check weights and indices
        self.assertEqual(weights.shape[0], self.batch_size)
        self.assertEqual(indices.shape[0], self.batch_size)
        self.assertTrue(isinstance(weights, torch.Tensor))
        self.assertTrue(isinstance(indices, np.ndarray))
        
        # Weights should be positive
        self.assertTrue(torch.all(weights > 0))
    
    def test_update_priorities(self):
        """Test updating priorities of sampled transitions."""
        # Add some transitions
        for i in range(self.batch_size * 2):
            obs = np.random.randn(4)
            action = i % 2
            reward = float(i)
            next_obs = np.random.randn(4)
            done = False
            priority = 1.0
            
            self.buffer.add(obs, action, reward, next_obs, done, priority=priority)
        
        # Sample batch
        *batch, weights, indices = self.buffer.sample()
        
        # Update priorities
        new_priorities = np.random.uniform(0.1, 2.0, size=len(indices))
        self.buffer.update_priorities(indices, new_priorities)
        
        # This should not raise any errors
        self.assertTrue(True)
    
    def test_beta_annealing(self):
        """Test beta annealing functionality."""
        initial_beta = self.buffer.beta
        
        # Sample multiple times (should increase beta if annealing is implemented)
        for _ in range(5):
            if self.buffer.can_sample():
                self.buffer.sample()
        
        # Beta should remain within valid range
        self.assertGreaterEqual(self.buffer.beta, initial_beta)
        self.assertLessEqual(self.buffer.beta, 1.0)
    
    def test_priority_sum_tree_properties(self):
        """Test that priority sum tree maintains correct properties."""
        # Add transitions with known priorities
        priorities = [1.0, 2.0, 3.0, 4.0]
        for i, priority in enumerate(priorities):
            obs = np.random.randn(4)
            action = i
            reward = float(i)
            next_obs = np.random.randn(4)
            done = False
            
            self.buffer.add(obs, action, reward, next_obs, done, priority=priority)
        
        # The sum tree should maintain the total sum correctly
        # This is an internal property test
        if hasattr(self.buffer, '_sum_tree'):
            total_priority = sum(priorities)
            tree_total = self.buffer._sum_tree.total_sum()
            self.assertAlmostEqual(tree_total, total_priority ** self.alpha, places=5)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Buffer imports not available")
class TestBufferFactory(unittest.TestCase):
    """Test cases for buffer factory function."""
    
    def test_create_replay_buffer(self):
        """Test creating ReplayBuffer via factory."""
        buffer = create_buffer('replay', buffer_size=100, batch_size=32, device='cpu')
        self.assertIsInstance(buffer, ReplayBuffer)
        self.assertEqual(buffer.buffer_size, 100)
        self.assertEqual(buffer.batch_size, 32)
    
    def test_create_prioritized_buffer(self):
        """Test creating PrioritizedReplayBuffer via factory."""
        buffer = create_buffer(
            'prioritized', 
            buffer_size=100, 
            batch_size=32, 
            alpha=0.6, 
            beta=0.4,
            device='cpu'
        )
        self.assertIsInstance(buffer, PrioritizedReplayBuffer)
        self.assertEqual(buffer.buffer_size, 100)
        self.assertEqual(buffer.batch_size, 32)
        self.assertEqual(buffer.alpha, 0.6)
        self.assertEqual(buffer.beta, 0.4)
    
    def test_create_invalid_buffer(self):
        """Test creating buffer with invalid type raises error."""
        with self.assertRaises(ValueError):
            create_buffer('invalid_type', buffer_size=100, batch_size=32)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Buffer imports not available")
class TestBufferIntegration(unittest.TestCase):
    """Integration tests for buffer implementations."""
    
    def test_replay_buffer_with_torch_tensors(self):
        """Test ReplayBuffer with PyTorch tensor inputs."""
        buffer = ReplayBuffer(100, 16, device='cpu', seed=42)
        
        obs = torch.randn(4)
        action = torch.tensor(1)
        reward = torch.tensor(0.5)
        next_obs = torch.randn(4)
        done = torch.tensor(False)
        
        buffer.add(obs, action, reward, next_obs, done)
        self.assertEqual(buffer.size(), 1)
        
        # Should be able to sample
        for _ in range(16):
            buffer.add(torch.randn(4), torch.randint(0, 2, (1,)), 
                      torch.randn(1), torch.randn(4), torch.tensor(False))
        
        batch = buffer.sample()
        self.assertEqual(len(batch), 5)
    
    def test_buffer_memory_efficiency(self):
        """Test that buffers don't consume excessive memory."""
        import gc
        
        # Create and fill buffer
        buffer = ReplayBuffer(1000, 32, device='cpu')
        
        for i in range(1000):
            obs = np.random.randn(4).astype(np.float32)
            action = i % 2
            reward = float(i % 100)
            next_obs = np.random.randn(4).astype(np.float32)
            done = (i % 100 == 99)
            
            buffer.add(obs, action, reward, next_obs, done)
        
        # Sample multiple batches
        for _ in range(10):
            batch = buffer.sample()
            del batch
        
        # Clear buffer and force garbage collection
        buffer.clear()
        gc.collect()
        
        # Buffer should be empty
        self.assertEqual(buffer.size(), 0)
    
    def test_buffer_thread_safety_basic(self):
        """Basic test for potential thread safety issues."""
        import threading
        import time
        
        buffer = ReplayBuffer(100, 32, device='cpu', seed=42)
        errors = []
        
        def add_transitions():
            try:
                for i in range(50):
                    obs = np.random.randn(4)
                    action = i % 2
                    reward = float(i)
                    next_obs = np.random.randn(4)
                    done = False
                    
                    buffer.add(obs, action, reward, next_obs, done)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Start two threads adding transitions
        thread1 = threading.Thread(target=add_transitions)
        thread2 = threading.Thread(target=add_transitions)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Check for errors
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(buffer.size(), 100)  # Should be full


if __name__ == '__main__':
    # Run tests with pytest for better output
    pytest.main([__file__, '-v', '--tb=short'])