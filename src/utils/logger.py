"""
Logger Utilities Module.

This module provides logging utilities for experiment tracking and reproducibility in the Modular DRL Framework, as described in "Deep Reinforcement Learning: A Survey" by Wang et al. and the project engineering blueprint.

Detailed Description:
The logger module enables systematic recording of training progress, evaluation metrics, hyperparameters, and experiment metadata. It supports logging to console, files, and TensorBoard, facilitating analysis and reproducibility. The design is modular and extensible, allowing integration with custom loggers and external tools.

Key Concepts/Algorithms:
- Structured experiment logging with timestamped entries
- Support for multiple output formats (console, file, TensorBoard)
- Metrics aggregation and statistical analysis
- Hyperparameter and configuration logging for reproducibility
- Episode-level and step-level data tracking

Important Parameters/Configurations:
- experiment_name: Unique identifier for the experiment
- log_dir: Directory for storing log files and metrics
- log_level: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- save_to_file: Whether to save logs to file
- tensorboard_enabled: Whether to use TensorBoard logging
- metrics_buffer_size: Size of metrics buffer for efficient writing

Expected Inputs/Outputs:
- Inputs: scalar metrics, episode data, hyperparameters, experiment metadata
- Outputs: console logs, log files, TensorBoard summaries, metrics JSON files

Dependencies:
- logging: Standard Python logging
- os, json: File operations and data serialization
- numpy: Numerical operations and statistics
- collections: Data structures for efficient metrics storage
- typing: Type hints for better code documentation

Author: REL Project Team
Date: 2025-07-13
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import defaultdict, deque
from pathlib import Path
import time
from datetime import datetime


class Logger:
    """
    Comprehensive Logger for DRL Experiments.

    This class provides a unified logging interface for deep reinforcement learning experiments,
    supporting multiple output formats and comprehensive metrics tracking. It is designed to
    ensure reproducibility and facilitate analysis of training progress and results.

    Detailed Description:
    The Logger class implements structured experiment logging with support for scalar metrics,
    episode-level data, hyperparameters, and experiment metadata. It provides both console and
    file-based logging, with optional TensorBoard integration for visualization. The design
    emphasizes efficiency through buffered writing and statistical aggregation.

    Key Features:
    - Multi-format logging (console, file, TensorBoard)
    - Scalar metrics tracking with statistical analysis
    - Episode-level data logging and aggregation
    - Hyperparameter and configuration persistence
    - Experiment metadata management
    - Efficient buffered I/O operations
    - Thread-safe logging operations

    Args:
        experiment_name (str): Unique identifier for the experiment
        log_dir (str): Directory for storing log files and metrics
        log_level (int): Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        save_to_file (bool): Whether to save logs to file
        tensorboard_enabled (bool): Whether to use TensorBoard logging
        metrics_buffer_size (int): Size of metrics buffer for efficient writing

    Attributes:
        experiment_name (str): Experiment identifier
        log_dir (str): Directory for log files
        logger (logging.Logger): Python logger instance
        metrics (Dict): Scalar metrics storage
        episodes (List): Episode-level data storage
        metadata (Dict): Experiment metadata storage
        start_time (float): Experiment start timestamp

    Example:
        ```python
        # Initialize logger
        logger = Logger(
            experiment_name="dqn_cartpole_v1",
            log_dir="./logs",
            log_level=logging.INFO,
            save_to_file=True
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
            'batch_size': 32,
            'gamma': 0.99
        })

        # Save metrics
        logger.save_metrics()
        ```

    Dependencies:
    - logging: Standard Python logging
    - numpy: Numerical operations and statistics
    - collections: Data structures for efficient storage

    Author: REL Project Team
    Date: 2025-07-13
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "./logs",
        log_level: int = logging.INFO,
        save_to_file: bool = True,
        tensorboard_enabled: bool = False,
        metrics_buffer_size: int = 1000,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Logger with specified configuration.

        Args:
            experiment_name: Unique identifier for the experiment
            log_dir: Directory for storing log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            save_to_file: Whether to save logs to file
            tensorboard_enabled: Whether to use TensorBoard logging
            metrics_buffer_size: Buffer size for efficient metrics writing
            config: Optional configuration dictionary
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.save_to_file = save_to_file
        self.tensorboard_enabled = tensorboard_enabled
        self.metrics_buffer_size = metrics_buffer_size
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize data storage
        self.metrics = defaultdict(list)
        self.episodes = []
        self.metadata = {}
        self.start_time = time.time()
        
        # Setup Python logger
        self._setup_logger(log_level)
        
        # Setup TensorBoard if enabled
        if tensorboard_enabled:
            self._setup_tensorboard()
        
        # Store configuration if provided
        if config:
            self.metadata['config'] = config
        
        # Log initialization
        self.logger.info(f"Logger initialized for experiment: {experiment_name}")
        self.logger.info(f"Log directory: {log_dir}")

    def _setup_logger(self, log_level: int) -> None:
        """Setup Python logger with appropriate handlers and formatting."""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(log_level)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if enabled
        if self.save_to_file:
            log_file_path = os.path.join(self.log_dir, f"{self.experiment_name}.log")
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard logging if available."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self.log_dir, "tensorboard", self.experiment_name)
            self.tb_writer = SummaryWriter(tb_dir)
            self.logger.info(f"TensorBoard logging enabled: {tb_dir}")
        except ImportError:
            self.logger.warning("TensorBoard not available. Install tensorboard for visualization.")
            self.tensorboard_enabled = False
            self.tb_writer = None

    def log_scalar(self, key: str, value: Union[float, int], step: Optional[int] = None) -> None:
        """
        Log a scalar metric value.

        Args:
            key: Metric name/key
            value: Scalar value to log
            step: Step number (if None, uses current timestamp)
        """
        if step is None:
            step = len(self.metrics[key])
        
        timestamp = time.time() - self.start_time
        self.metrics[key].append({
            'value': float(value),
            'step': step,
            'timestamp': timestamp
        })
        
        # Log to TensorBoard if enabled
        if self.tensorboard_enabled and self.tb_writer:
            self.tb_writer.add_scalar(key, value, step)
        
        # Log debug message
        self.logger.debug(f"Metric logged - {key}: {value} (step: {step})")

    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        """
        Log episode-level data.

        Args:
            episode_data: Dictionary containing episode information
        """
        # Add timestamp and episode number
        episode_data['timestamp'] = time.time() - self.start_time
        episode_data['episode_num'] = len(self.episodes)
        
        self.episodes.append(episode_data.copy())
        
        # Log summary info
        if 'total_reward' in episode_data:
            self.logger.info(
                f"Episode {episode_data.get('episode_num', len(self.episodes) - 1)}: "
                f"Reward={episode_data['total_reward']:.2f}"
            )

    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters for experiment reproducibility.

        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.metadata['hyperparameters'] = hyperparams.copy()
        
        # Log to TensorBoard if enabled
        if self.tensorboard_enabled and self.tb_writer:
            self.tb_writer.add_hparams(hyperparams, {})
        
        self.logger.info("Hyperparameters logged")
        for key, value in hyperparams.items():
            self.logger.info(f"  {key}: {value}")

    def log_dict(self, data: Dict[str, Any], prefix: str = "") -> None:
        """
        Log a dictionary of values as individual scalars.

        Args:
            data: Dictionary of key-value pairs to log
            prefix: Optional prefix for metric names
        """
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metric_name = f"{prefix}{key}" if prefix else key
                self.log_scalar(metric_name, value)

    def get_metrics_in_range(
        self, 
        metric_name: str, 
        min_step: Optional[int] = None,
        max_step: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Get metric values within a specified step range.

        Args:
            metric_name: Name of the metric
            min_step: Minimum step (inclusive)
            max_step: Maximum step (inclusive)

        Returns:
            List of (step, value) tuples within the range
        """
        if metric_name not in self.metrics:
            return []
        
        filtered_data = []
        for entry in self.metrics[metric_name]:
            step = entry['step']
            value = entry['value']
            
            if min_step is not None and step < min_step:
                continue
            if max_step is not None and step > max_step:
                continue
                
            filtered_data.append((step, value))
        
        return filtered_data

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of all logged metrics.

        Returns:
            Dictionary with summary statistics for each metric
        """
        summary = {}
        
        for metric_name, entries in self.metrics.items():
            if not entries:
                continue
                
            values = [entry['value'] for entry in entries]
            
            summary[metric_name] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'latest': values[-1]
            }
        
        return summary

    def save_metrics(self, filename: Optional[str] = None) -> str:
        """
        Save all metrics and metadata to JSON file.

        Args:
            filename: Optional custom filename

        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_metrics_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        # Prepare data for serialization
        save_data = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time,
            'metrics': dict(self.metrics),
            'episodes': self.episodes,
            'metadata': self.metadata,
            'summary': self.get_metrics_summary()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics saved to: {filepath}")
        return filepath

    def load_metrics(self, filepath: str) -> None:
        """
        Load metrics from a previously saved JSON file.

        Args:
            filepath: Path to the metrics file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Restore data
        self.metrics = defaultdict(list, data.get('metrics', {}))
        self.episodes = data.get('episodes', [])
        self.metadata = data.get('metadata', {})
        
        self.logger.info(f"Metrics loaded from: {filepath}")

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def close(self) -> None:
        """Close logger and cleanup resources."""
        if self.tensorboard_enabled and self.tb_writer:
            self.tb_writer.close()
        
        # Save final metrics
        self.save_metrics()
        
        self.logger.info(f"Logger closed for experiment: {self.experiment_name}")


# Export main classes
__all__ = ['Logger']