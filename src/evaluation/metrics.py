"""
Metrics Evaluation Module.

This module provides standardized metrics calculation utilities for evaluating Deep Reinforcement Learning (DRL) algorithms in the Modular DRL Framework, as outlined in "Deep Reinforcement Learning: A Survey" by Wang et al. and the project engineering blueprint.

Detailed Description:
The metrics module enables quantitative assessment of agent performance, learning progress, and experiment reproducibility. It includes functions for computing cumulative rewards, episode lengths, moving averages, statistical summaries, and other key evaluation metrics. These metrics are essential for comparing algorithms, tuning hyperparameters, and reporting results in a reproducible manner. The design supports extensibility for custom metrics and integrates seamlessly with the framework's logging and plotting utilities.

Key Concepts/Algorithms:
- Cumulative reward and episode return calculation
- Episode length and success rate tracking
- Moving average and smoothing for learning curves
- Statistical summaries (mean, std, min, max)
- Extensible metric registration for custom evaluation

Important Parameters/Configurations:
- Metrics to compute (specified in config or script)
- Smoothing window size for moving averages
- Logging frequency and output format
- All parameters are configurable via YAML files in `configs/`

Expected Inputs/Outputs:
- Inputs: lists or arrays of rewards, episode results, agent-environment interaction data
- Outputs: computed metrics (floats, arrays, dicts), ready for logging or plotting

Dependencies:
- numpy, collections
- src/utils/logger.py (for logging results)

Author: REL Project Team
Date: 2025-07-13
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import deque
import warnings

from . import BaseMetricsCalculator, EpisodeData, MetricValue, MetricDict, TimeSeries


class StandardMetricsCalculator(BaseMetricsCalculator):
    """
    Standard implementation of metrics calculator for DRL algorithms.
    
    This class provides comprehensive metrics calculation for evaluating DRL agent
    performance including rewards, episode lengths, success rates, and learning
    progress indicators. It supports both episode-based and step-based metrics.
    
    Args:
        success_threshold (Optional[float]): Threshold for success rate calculation
        episode_length_limit (Optional[int]): Maximum episode length for truncation detection
        track_individual_episodes (bool): Whether to track detailed episode data
        **kwargs: Additional arguments passed to BaseMetricsCalculator
        
    Attributes:
        success_threshold (Optional[float]): Success threshold value
        episode_length_limit (Optional[int]): Episode length limit
        track_individual_episodes (bool): Individual episode tracking flag
        episode_data_buffer (deque): Buffer for recent episode data
    """
    
    def __init__(
        self,
        success_threshold: Optional[float] = None,
        episode_length_limit: Optional[int] = None,
        track_individual_episodes: bool = True,
        **kwargs
    ):
        self.success_threshold = success_threshold
        self.episode_length_limit = episode_length_limit
        self.track_individual_episodes = track_individual_episodes
        self.episode_data_buffer = deque(maxlen=1000)  # Keep last 1000 episodes
        
        super().__init__(**kwargs)
    
    def _init_evaluator_components(self) -> None:
        """Initialize standard metrics calculator components."""
        super()._init_evaluator_components()
        
        # Register additional standard metrics
        self.register_metric('reward_std', self._calculate_reward_std)
        self.register_metric('min_reward', self._calculate_min_reward)
        self.register_metric('max_reward', self._calculate_max_reward)
        self.register_metric('median_reward', self._calculate_median_reward)
        self.register_metric('reward_range', self._calculate_reward_range)
        self.register_metric('convergence_rate', self._calculate_convergence_rate)
        self.register_metric('stability_score', self._calculate_stability_score)
    
    def process_data(self, data: EpisodeData, **kwargs) -> Dict[str, Any]:
        """
        Process episode data and compute all metrics.
        
        Args:
            data: Episode data containing rewards, lengths, etc.
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary containing computed metrics and analysis
        """
        # Validate input data
        self._validate_episode_data(data)
        
        # Store episode data if tracking is enabled
        if self.track_individual_episodes:
            self.episode_data_buffer.append(data.copy())
        
        # Compute all registered metrics
        metrics = self.compute_all_metrics(data)
        
        # Compute smoothed versions if requested
        if 'compute_smoothed' in kwargs and kwargs['compute_smoothed']:
            smoothed_metrics = self._compute_smoothed_metrics(data)
            metrics.update(smoothed_metrics)
        
        # Compute confidence intervals if enabled
        if self.compute_confidence_intervals:
            confidence_intervals = self._compute_all_confidence_intervals(data)
            metrics['confidence_intervals'] = confidence_intervals
        
        # Add metadata
        results = {
            'metrics': metrics,
            'metadata': {
                'num_episodes': len(data.get('rewards', [])),
                'data_keys': list(data.keys()),
                'computation_time': self._get_timestamp(),
                'success_threshold': self.success_threshold,
                'smoothing_window': self.smoothing_window
            }
        }
        
        # Auto-save if enabled
        if self.save_results:
            self.save_results(results)
        
        return results
    
    def _validate_episode_data(self, data: EpisodeData) -> None:
        """
        Validate episode data format and content.
        
        Args:
            data: Episode data to validate
            
        Raises:
            ValueError: If data format is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Episode data must be a dictionary")
        
        required_keys = ['rewards']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        if not data['rewards']:
            warnings.warn("Empty rewards list provided")
    
    def _calculate_mean_reward(self, data: EpisodeData) -> float:
        """Calculate mean reward from episode data."""
        rewards = data.get('rewards', [])
        return float(np.mean(rewards)) if rewards else 0.0
    
    def _calculate_episode_length(self, data: EpisodeData) -> float:
        """Calculate mean episode length from episode data."""
        lengths = data.get('lengths', data.get('episode_lengths', []))
        if not lengths:
            # If lengths not provided, infer from rewards assuming 1 step per reward
            return 1.0 if data.get('rewards') else 0.0
        return float(np.mean(lengths))
    
    def _calculate_success_rate(self, data: EpisodeData) -> float:
        """Calculate success rate from episode data."""
        if self.success_threshold is None:
            # Use explicit success flags if available
            successes = data.get('successes', data.get('success_flags', []))
            if successes:
                return float(np.mean(successes))
            else:
                warnings.warn("No success threshold set and no success flags provided")
                return 0.0
        
        rewards = data.get('rewards', [])
        if not rewards:
            return 0.0
        
        successes = [1 if reward >= self.success_threshold else 0 for reward in rewards]
        return float(np.mean(successes))
    
    def _calculate_cumulative_reward(self, data: EpisodeData) -> List[float]:
        """Calculate cumulative rewards over time."""
        rewards = data.get('rewards', [])
        return np.cumsum(rewards).tolist() if rewards else []
    
    def _calculate_reward_std(self, data: EpisodeData) -> float:
        """Calculate standard deviation of rewards."""
        rewards = data.get('rewards', [])
        return float(np.std(rewards)) if rewards else 0.0
    
    def _calculate_min_reward(self, data: EpisodeData) -> float:
        """Calculate minimum reward."""
        rewards = data.get('rewards', [])
        return float(np.min(rewards)) if rewards else 0.0
    
    def _calculate_max_reward(self, data: EpisodeData) -> float:
        """Calculate maximum reward."""
        rewards = data.get('rewards', [])
        return float(np.max(rewards)) if rewards else 0.0
    
    def _calculate_median_reward(self, data: EpisodeData) -> float:
        """Calculate median reward."""
        rewards = data.get('rewards', [])
        return float(np.median(rewards)) if rewards else 0.0
    
    def _calculate_reward_range(self, data: EpisodeData) -> float:
        """Calculate reward range (max - min)."""
        rewards = data.get('rewards', [])
        if not rewards:
            return 0.0
        return float(np.max(rewards) - np.min(rewards))
    
    def _calculate_convergence_rate(self, data: EpisodeData) -> float:
        """
        Calculate convergence rate based on reward trend.
        
        Returns positive value for improving trend, negative for declining.
        """
        rewards = data.get('rewards', [])
        if len(rewards) < 2:
            return 0.0
        
        # Use linear regression to estimate trend
        x = np.arange(len(rewards))
        y = np.array(rewards)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _calculate_stability_score(self, data: EpisodeData) -> float:
        """
        Calculate stability score based on reward variance.
        
        Returns value between 0 and 1, where 1 is most stable.
        """
        rewards = data.get('rewards', [])
        if len(rewards) < 2:
            return 1.0
        
        # Calculate coefficient of variation (inverse stability)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        if mean_reward == 0:
            return 0.0 if std_reward > 0 else 1.0
        
        cv = std_reward / abs(mean_reward)
        
        # Convert to stability score (0-1 scale)
        stability = 1.0 / (1.0 + cv)
        return float(stability)
    
    def _compute_smoothed_metrics(self, data: EpisodeData) -> Dict[str, List[float]]:
        """Compute smoothed versions of time-series metrics."""
        smoothed = {}
        
        # Smooth rewards
        rewards = data.get('rewards', [])
        if rewards:
            smoothed['smoothed_rewards'] = self.smooth_timeseries(rewards)
        
        # Smooth episode lengths if available
        lengths = data.get('lengths', data.get('episode_lengths', []))
        if lengths:
            smoothed['smoothed_lengths'] = self.smooth_timeseries(lengths)
        
        # Smooth cumulative rewards
        cumulative_rewards = self._calculate_cumulative_reward(data)
        if cumulative_rewards:
            smoothed['smoothed_cumulative_rewards'] = self.smooth_timeseries(cumulative_rewards)
        
        return smoothed
    
    def _compute_all_confidence_intervals(self, data: EpisodeData) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for all applicable metrics."""
        intervals = {}
        
        # Confidence interval for rewards
        rewards = data.get('rewards', [])
        if len(rewards) > 1:
            intervals['reward_ci'] = self.compute_confidence_interval(rewards)
        
        # Confidence interval for episode lengths
        lengths = data.get('lengths', data.get('episode_lengths', []))
        if len(lengths) > 1:
            intervals['length_ci'] = self.compute_confidence_interval(lengths)
        
        return intervals
    
    def get_episode_statistics(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics from recent episodes.
        
        Args:
            window_size: Number of recent episodes to analyze (all if None)
            
        Returns:
            Dictionary of episode statistics
        """
        if not self.episode_data_buffer:
            return {'message': 'No episode data available'}
        
        # Get recent episodes
        if window_size is None:
            recent_episodes = list(self.episode_data_buffer)
        else:
            recent_episodes = list(self.episode_data_buffer)[-window_size:]
        
        # Aggregate data from recent episodes
        all_rewards = []
        all_lengths = []
        
        for episode in recent_episodes:
            all_rewards.extend(episode.get('rewards', []))
            all_lengths.extend(episode.get('lengths', episode.get('episode_lengths', [])))
        
        # Create aggregated data
        aggregated_data = {
            'rewards': all_rewards,
            'lengths': all_lengths
        }
        
        # Compute statistics
        stats = self.compute_all_metrics(aggregated_data)
        stats['num_episodes_analyzed'] = len(recent_episodes)
        stats['total_steps'] = sum(all_lengths) if all_lengths else len(all_rewards)
        
        return stats
    
    def compare_episodes(self, data1: EpisodeData, data2: EpisodeData) -> Dict[str, Any]:
        """
        Compare metrics between two sets of episode data.
        
        Args:
            data1: First episode data set
            data2: Second episode data set
            
        Returns:
            Dictionary containing comparison results
        """
        metrics1 = self.compute_all_metrics(data1)
        metrics2 = self.compute_all_metrics(data2)
        
        comparison = {
            'metrics_1': metrics1,
            'metrics_2': metrics2,
            'differences': {},
            'relative_changes': {}
        }
        
        # Calculate differences and relative changes
        for key in metrics1:
            if key in metrics2 and isinstance(metrics1[key], (int, float)):
                diff = metrics2[key] - metrics1[key]
                comparison['differences'][key] = diff
                
                if metrics1[key] != 0:
                    rel_change = (diff / metrics1[key]) * 100
                    comparison['relative_changes'][key] = rel_change
        
        return comparison


class PerformanceTracker(StandardMetricsCalculator):
    """
    Extended metrics calculator for tracking performance over training.
    
    This class adds capabilities for tracking learning progress, detecting
    plateaus, and monitoring training dynamics over extended periods.
    
    Args:
        plateau_patience (int): Steps to wait before declaring plateau
        plateau_threshold (float): Minimum improvement threshold
        track_gradients (bool): Whether to track performance gradients
        **kwargs: Additional arguments passed to StandardMetricsCalculator
    """
    
    def __init__(
        self,
        plateau_patience: int = 100,
        plateau_threshold: float = 0.01,
        track_gradients: bool = True,
        **kwargs
    ):
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold
        self.track_gradients = track_gradients
        
        # Performance tracking
        self.best_performance = float('-inf')
        self.steps_since_improvement = 0
        self.performance_history = deque(maxlen=1000)
        
        super().__init__(**kwargs)
    
    def _init_evaluator_components(self) -> None:
        """Initialize performance tracker components."""
        super()._init_evaluator_components()
        
        # Register performance tracking metrics
        self.register_metric('plateau_detected', self._detect_plateau)
        self.register_metric('improvement_rate', self._calculate_improvement_rate)
        self.register_metric('performance_gradient', self._calculate_performance_gradient)
    
    def update_performance(self, current_performance: float) -> Dict[str, Any]:
        """
        Update performance tracking with new performance value.
        
        Args:
            current_performance: Current performance metric
            
        Returns:
            Dictionary with tracking information
        """
        self.performance_history.append(current_performance)
        
        # Check for improvement
        if current_performance > self.best_performance + self.plateau_threshold:
            self.best_performance = current_performance
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
        
        return {
            'current_performance': current_performance,
            'best_performance': self.best_performance,
            'steps_since_improvement': self.steps_since_improvement,
            'plateau_detected': self._detect_plateau({}),  # Pass empty dict for compatibility
            'total_steps': len(self.performance_history)
        }
    
    def _detect_plateau(self, data: EpisodeData) -> bool:
        """Detect if performance has plateaued."""
        return self.steps_since_improvement >= self.plateau_patience
    
    def _calculate_improvement_rate(self, data: EpisodeData) -> float:
        """Calculate rate of improvement over recent history."""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent_performance = list(self.performance_history)[-10:]
        x = np.arange(len(recent_performance))
        slope = np.polyfit(x, recent_performance, 1)[0]
        
        return float(slope)
    
    def _calculate_performance_gradient(self, data: EpisodeData) -> float:
        """Calculate performance gradient (rate of change)."""
        if len(self.performance_history) < 2:
            return 0.0
        
        recent_values = list(self.performance_history)[-2:]
        gradient = recent_values[-1] - recent_values[-2]
        
        return float(gradient)


# Export classes
__all__ = [
    'StandardMetricsCalculator',
    'PerformanceTracker'
]