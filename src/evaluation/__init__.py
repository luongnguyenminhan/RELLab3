"""
Evaluation Module.

This package provides tools for evaluating and visualizing the performance of DRL algorithms in the Modular DRL Framework, as outlined in "Deep Reinforcement Learning: A Survey" by Wang et al. and the project engineering blueprint.

Detailed Description:
The evaluation module implements a modular hierarchy for performance assessment and visualization of DRL algorithms. It provides base classes for metrics calculation and plotting, enabling standardized evaluation workflows across different algorithms and environments. The design emphasizes reproducibility, extensibility, and integration with the framework's configuration system. All evaluation components support YAML-based configuration and provide consistent interfaces for data processing and visualization.

Key Concepts/Algorithms:
- Performance metrics calculation (rewards, episode lengths, success rates)
- Statistical analysis and smoothing (moving averages, confidence intervals)
- Training curve visualization and experiment comparison
- Publication-quality plotting with customizable styles
- Reproducible evaluation workflows and result persistence

Important Parameters/Configurations:
- Metrics to compute and their calculation parameters
- Plotting styles, output formats, and figure specifications
- Statistical analysis options (smoothing windows, confidence levels)
- Data persistence and loading configurations
- All parameters are configurable via YAML files in `configs/`

Expected Inputs/Outputs:
- Inputs: Training logs, episode data, algorithm performance metrics
- Outputs: Computed statistics, publication-ready plots, evaluation reports
- Supports various data formats (numpy arrays, pandas DataFrames, JSON logs)

Dependencies:
- numpy, matplotlib, seaborn (optional), pandas (optional)
- PyYAML for configuration management
- Compatible with framework logging and algorithm modules

Author: REL Project Team
Date: 2025-07-13
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import os
import yaml
from pathlib import Path
from collections import defaultdict, deque
import json
import warnings

# Type aliases for better code readability and consistency
MetricValue = Union[float, int, np.ndarray, List[float]]
MetricDict = Dict[str, MetricValue]
DataPoint = Tuple[int, float]  # (step, value)
TimeSeries = List[DataPoint]
EpisodeData = Dict[str, Union[List[float], List[int], float, int]]
EvaluationResults = Dict[str, Union[MetricDict, Dict[str, Any]]]


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluation components in the Modular DRL Framework.
    
    This class provides the common interface and shared functionality for evaluation components
    including metrics calculation and visualization. It ensures consistent configuration management,
    data handling, and result persistence across all evaluation implementations.
    
    Args:
        config_path (Optional[str]): Path to YAML configuration file
        output_dir (str): Directory for saving evaluation results and plots
        experiment_name (str): Name identifier for the experiment
        seed (Optional[int]): Random seed for reproducible evaluation
        save_results (bool): Whether to automatically save evaluation results
        
    Attributes:
        config (Dict[str, Any]): Loaded configuration parameters
        output_dir (Path): Output directory for results
        experiment_name (str): Experiment identifier
        seed (Optional[int]): Random seed for reproducibility
        save_results (bool): Auto-save flag
        results_history (Dict[str, List]): Historical evaluation results
        metadata (Dict[str, Any]): Evaluation metadata and settings
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: str = "evaluation_results",
        experiment_name: str = "experiment",
        seed: Optional[int] = None,
        save_results: bool = True,
    ):
        self.config = self._load_config(config_path) if config_path else {}
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.seed = seed
        self.save_results = save_results
        
        # Initialize tracking structures
        self.results_history: Dict[str, List] = defaultdict(list)
        self.metadata: Dict[str, Any] = {
            'experiment_name': experiment_name,
            'creation_time': self._get_timestamp(),
            'seed': seed,
            'config': self.config.copy()
        }
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize component-specific attributes
        self._init_evaluator_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file {config_path}: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    @abstractmethod
    def _init_evaluator_components(self) -> None:
        """
        Initialize evaluator-specific components.
        
        This method should be implemented by concrete evaluator classes to set up
        any specialized functionality, data structures, or processing pipelines.
        """
        pass
    
    @abstractmethod
    def process_data(self, data: Any, **kwargs) -> EvaluationResults:
        """
        Process input data and generate evaluation results.
        
        Args:
            data: Input data to process (format depends on concrete implementation)
            **kwargs: Additional processing parameters
            
        Returns:
            Evaluation results dictionary
        """
        pass
    
    def save_results(self, results: EvaluationResults, filename: Optional[str] = None) -> str:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{self.experiment_name}_results_{self._get_timestamp().replace(':', '-')}.json"
        
        filepath = self.output_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            return str(filepath)
        except Exception as e:
            warnings.warn(f"Failed to save results to {filepath}: {e}")
            return ""
    
    def load_results(self, filepath: str) -> EvaluationResults:
        """
        Load evaluation results from file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Loaded evaluation results
        """
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to load results from {filepath}: {e}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert numpy arrays and other non-serializable objects to JSON-compatible format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def add_result(self, key: str, value: MetricValue) -> None:
        """
        Add a result to the history tracking.
        
        Args:
            key: Result identifier
            value: Result value
        """
        self.results_history[key].append(value)
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for all tracked results.
        
        Returns:
            Dictionary of summary statistics for each tracked metric
        """
        summary = {}
        
        for key, values in self.results_history.items():
            if not values:
                continue
                
            numeric_values = [v for v in values if isinstance(v, (int, float, np.number))]
            if not numeric_values:
                continue
            
            summary[key] = {
                'mean': float(np.mean(numeric_values)),
                'std': float(np.std(numeric_values)),
                'min': float(np.min(numeric_values)),
                'max': float(np.max(numeric_values)),
                'count': len(numeric_values),
                'latest': float(numeric_values[-1]) if numeric_values else 0.0
            }
        
        return summary
    
    def reset(self) -> None:
        """Reset the evaluator state and clear results history."""
        self.results_history.clear()
        self.metadata['reset_time'] = self._get_timestamp()
    
    def update_metadata(self, **kwargs) -> None:
        """
        Update evaluation metadata.
        
        Args:
            **kwargs: Metadata key-value pairs to update
        """
        self.metadata.update(kwargs)
        self.metadata['last_updated'] = self._get_timestamp()


class BaseMetricsCalculator(BaseEvaluator):
    """
    Abstract base class for metrics calculation components.
    
    This class specializes the BaseEvaluator for computing performance metrics
    from training data, episode results, and algorithm outputs. It provides
    standardized interfaces for metric computation and statistical analysis.
    
    Args:
        smoothing_window (int): Window size for moving average smoothing
        compute_confidence_intervals (bool): Whether to compute confidence intervals
        confidence_level (float): Confidence level for interval computation
        **kwargs: Additional arguments passed to BaseEvaluator
        
    Attributes:
        smoothing_window (int): Smoothing window size
        compute_confidence_intervals (bool): CI computation flag
        confidence_level (float): Confidence level for intervals
        registered_metrics (Dict[str, Callable]): Registry of available metrics
    """
    
    def __init__(
        self,
        smoothing_window: int = 100,
        compute_confidence_intervals: bool = True,
        confidence_level: float = 0.95,
        **kwargs
    ):
        self.smoothing_window = smoothing_window
        self.compute_confidence_intervals = compute_confidence_intervals
        self.confidence_level = confidence_level
        self.registered_metrics: Dict[str, Callable] = {}
        
        super().__init__(**kwargs)
    
    def _init_evaluator_components(self) -> None:
        """Initialize metrics calculator specific components."""
        # Register default metrics
        self._register_default_metrics()
    
    def _register_default_metrics(self) -> None:
        """Register default metrics functions."""
        self.register_metric('mean_reward', self._calculate_mean_reward)
        self.register_metric('episode_length', self._calculate_episode_length)
        self.register_metric('success_rate', self._calculate_success_rate)
        self.register_metric('cumulative_reward', self._calculate_cumulative_reward)
    
    @abstractmethod
    def _calculate_mean_reward(self, data: EpisodeData) -> float:
        """Calculate mean reward from episode data."""
        pass
    
    @abstractmethod
    def _calculate_episode_length(self, data: EpisodeData) -> float:
        """Calculate mean episode length from episode data."""
        pass
    
    @abstractmethod
    def _calculate_success_rate(self, data: EpisodeData) -> float:
        """Calculate success rate from episode data."""
        pass
    
    @abstractmethod
    def _calculate_cumulative_reward(self, data: EpisodeData) -> List[float]:
        """Calculate cumulative rewards over time."""
        pass
    
    def register_metric(self, name: str, func: Callable) -> None:
        """
        Register a custom metric function.
        
        Args:
            name: Metric name identifier
            func: Function that computes the metric
        """
        self.registered_metrics[name] = func
    
    def compute_metric(self, name: str, data: EpisodeData) -> MetricValue:
        """
        Compute a specific metric.
        
        Args:
            name: Name of the metric to compute
            data: Episode data for computation
            
        Returns:
            Computed metric value
            
        Raises:
            KeyError: If metric is not registered
        """
        if name not in self.registered_metrics:
            raise KeyError(f"Metric '{name}' not registered. Available: {list(self.registered_metrics.keys())}")
        
        return self.registered_metrics[name](data)
    
    def compute_all_metrics(self, data: EpisodeData) -> MetricDict:
        """
        Compute all registered metrics.
        
        Args:
            data: Episode data for computation
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        for name, func in self.registered_metrics.items():
            try:
                metrics[name] = func(data)
            except Exception as e:
                warnings.warn(f"Failed to compute metric '{name}': {e}")
                metrics[name] = np.nan
        
        return metrics
    
    def smooth_timeseries(self, data: List[float], window: Optional[int] = None) -> List[float]:
        """
        Apply moving average smoothing to time series data.
        
        Args:
            data: Input time series data
            window: Smoothing window size (uses self.smoothing_window if None)
            
        Returns:
            Smoothed time series
        """
        if window is None:
            window = self.smoothing_window
        
        if len(data) < window:
            return data.copy()
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            smoothed.append(np.mean(data[start_idx:end_idx]))
        
        return smoothed
    
    def compute_confidence_interval(
        self, 
        data: List[float], 
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for data.
        
        Args:
            data: Input data
            confidence_level: Confidence level (uses self.confidence_level if None)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        sem = np.std(data_array) / np.sqrt(len(data_array))
        
        # Use t-distribution for small samples
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence_level) / 2, len(data_array) - 1)
        
        margin = t_value * sem
        return mean - margin, mean + margin


# Export classes and type aliases
__all__ = [
    'BaseEvaluator',
    'BaseMetricsCalculator', 
    'MetricValue',
    'MetricDict',
    'DataPoint',
    'TimeSeries',
    'EpisodeData',
    'EvaluationResults'
]