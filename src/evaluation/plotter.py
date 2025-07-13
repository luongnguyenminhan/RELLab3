"""
Plotter Evaluation Module.

This module provides plotting and visualization utilities for analyzing the performance of DRL algorithms in the Modular DRL Framework, as described in "Deep Reinforcement Learning: A Survey" by Wang et al. and the project engineering blueprint.

Detailed Description:
The plotter module enables the creation of publication-quality visualizations for training curves, evaluation metrics, and experiment comparisons. It supports line plots, bar charts, and statistical overlays for cumulative rewards, losses, and other tracked metrics. The design emphasizes reproducibility, clarity, and extensibility, allowing users to customize plots for different experiments and reporting needs. Integration with the metrics and logger modules ensures seamless workflow from data collection to visualization.

Key Concepts/Algorithms:
- Training curve visualization (reward, loss, etc.)
- Comparison of multiple runs or algorithms
- Smoothing and moving average overlays
- Customizable plot styles and export options
- Support for reproducible figure generation

Important Parameters/Configurations:
- Metrics to plot (from config or script)
- Smoothing window size and plot style
- Output directory and file format (e.g., PNG, PDF)
- All parameters are configurable via YAML files in `configs/`

Expected Inputs/Outputs:
- Inputs: computed metrics (lists, arrays, dicts), log files
- Outputs: saved plot images, optionally displayed interactively

Dependencies:
- matplotlib, seaborn (optional), numpy
- src/evaluation/metrics.py (for metric data)

Author: REL Project Team
Date: 2025-07-13
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import warnings

# Import base evaluator
from . import BaseEvaluator, EvaluationResults, TimeSeries, MetricDict

# Try to import seaborn for enhanced styling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not available. Some advanced styling options will be limited.")

# Try to import pandas for data handling
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class BasePlotter(BaseEvaluator):
    """
    Abstract base class for plotting and visualization components.
    
    This class specializes the BaseEvaluator for creating publication-quality
    visualizations of DRL training and evaluation results. It provides standardized
    interfaces for plot generation, styling, and export functionality.
    
    Args:
        figure_size (Tuple[int, int]): Default figure size (width, height)
        dpi (int): Resolution for saved figures
        style (str): Matplotlib/seaborn style to use
        color_palette (Optional[str]): Color palette for plots
        save_format (str): Default format for saved plots
        show_plots (bool): Whether to display plots interactively
        **kwargs: Additional arguments passed to BaseEvaluator
        
    Attributes:
        figure_size (Tuple[int, int]): Default figure dimensions
        dpi (int): Figure resolution
        style (str): Plot style
        color_palette (Optional[str]): Color palette name
        save_format (str): File format for saved plots
        show_plots (bool): Interactive display flag
        plot_registry (Dict[str, Callable]): Registry of available plot types
    """
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (10, 6),
        dpi: int = 300,
        style: str = "seaborn-v0_8" if HAS_SEABORN else "default",
        color_palette: Optional[str] = None,
        save_format: str = "png",
        show_plots: bool = False,
        **kwargs
    ):
        self.figure_size = figure_size
        self.dpi = dpi
        self.style = style
        self.color_palette = color_palette
        self.save_format = save_format
        self.show_plots = show_plots
        self.plot_registry: Dict[str, Callable] = {}
        
        super().__init__(**kwargs)
    
    def _init_evaluator_components(self) -> None:
        """Initialize plotter-specific components."""
        # Set up matplotlib style
        self._setup_plotting_style()
        
        # Register default plot types
        self._register_default_plots()
    
    def _setup_plotting_style(self) -> None:
        """Configure matplotlib and seaborn styling."""
        try:
            plt.style.use(self.style)
        except Exception:
            warnings.warn(f"Style '{self.style}' not available, using default")
            plt.style.use('default')
        
        # Set up seaborn if available
        if HAS_SEABORN:
            sns.set_theme()
            if self.color_palette:
                sns.set_palette(self.color_palette)
        
        # Configure matplotlib defaults
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 10
    
    def _register_default_plots(self) -> None:
        """Register default plot types."""
        # This will be implemented by concrete classes
        pass
    
    def register_plot_type(self, name: str, func: Callable) -> None:
        """
        Register a custom plot type.
        
        Args:
            name: Plot type name identifier
            func: Function that creates the plot
        """
        self.plot_registry[name] = func
    
    def create_figure(
        self, 
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
        """
        Create a new figure with specified parameters.
        
        Args:
            figsize: Figure size override
            **kwargs: Additional arguments for plt.subplots
            
        Returns:
            Tuple of (figure, axes)
        """
        if figsize is None:
            figsize = self.figure_size
        
        # Set default subplot parameters
        default_kwargs = {
            'figsize': figsize,
            'dpi': self.dpi
        }
        default_kwargs.update(kwargs)
        
        return plt.subplots(**default_kwargs)
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        format_override: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save figure to file.
        
        Args:
            fig: Figure to save
            filename: Output filename (without extension)
            format_override: Override default format
            **kwargs: Additional arguments for savefig
            
        Returns:
            Path to saved file
        """
        save_format = format_override or self.save_format
        filepath = self.output_dir / f"{filename}.{save_format}"
        
        # Set default save parameters
        default_kwargs = {
            'dpi': self.dpi,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        default_kwargs.update(kwargs)
        
        try:
            fig.savefig(filepath, **default_kwargs)
            return str(filepath)
        except Exception as e:
            warnings.warn(f"Failed to save figure to {filepath}: {e}")
            return ""
    
    def show_figure(self, fig: plt.Figure) -> None:
        """
        Display figure if show_plots is enabled.
        
        Args:
            fig: Figure to display
        """
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    def process_data(self, data: Any, plot_type: str, **kwargs) -> EvaluationResults:
        """
        Process data and create specified plot.
        
        Args:
            data: Input data to plot
            plot_type: Type of plot to create
            **kwargs: Additional plotting parameters
            
        Returns:
            Results including plot information
        """
        if plot_type not in self.plot_registry:
            raise ValueError(f"Plot type '{plot_type}' not registered. Available: {list(self.plot_registry.keys())}")
        
        # Create plot
        plot_func = self.plot_registry[plot_type]
        fig, ax, plot_info = plot_func(data, **kwargs)
        
        # Save figure
        filename = kwargs.get('filename', f"{self.experiment_name}_{plot_type}")
        saved_path = self.save_figure(fig, filename)
        
        # Show if requested
        self.show_figure(fig)
        
        # Prepare results
        results = {
            'plot_info': plot_info,
            'saved_path': saved_path,
            'plot_type': plot_type,
            'metadata': {
                'figure_size': self.figure_size,
                'dpi': self.dpi,
                'style': self.style,
                'creation_time': self._get_timestamp()
            }
        }
        
        return results


class StandardPlotter(BasePlotter):
    """
    Standard implementation of plotter for DRL evaluation.
    
    This class provides comprehensive plotting capabilities for DRL training
    and evaluation including learning curves, performance comparisons, and
    statistical visualizations.
    
    Args:
        confidence_alpha (float): Alpha value for confidence interval shading
        smoothing_alpha (float): Alpha value for smoothed line overlay
        grid_enabled (bool): Whether to show grid on plots
        **kwargs: Additional arguments passed to BasePlotter
    """
    
    def __init__(
        self,
        confidence_alpha: float = 0.2,
        smoothing_alpha: float = 0.8,
        grid_enabled: bool = True,
        **kwargs
    ):
        self.confidence_alpha = confidence_alpha
        self.smoothing_alpha = smoothing_alpha
        self.grid_enabled = grid_enabled
        
        super().__init__(**kwargs)
    
    def _register_default_plots(self) -> None:
        """Register standard plot types."""
        self.register_plot_type('learning_curve', self._plot_learning_curve)
        self.register_plot_type('episode_rewards', self._plot_episode_rewards)
        self.register_plot_type('episode_lengths', self._plot_episode_lengths)
        self.register_plot_type('comparison', self._plot_comparison)
        self.register_plot_type('distribution', self._plot_distribution)
        self.register_plot_type('correlation_matrix', self._plot_correlation_matrix)
        self.register_plot_type('performance_over_time', self._plot_performance_over_time)
    
    def _plot_learning_curve(
        self, 
        data: Dict[str, List[float]], 
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
        """
        Plot learning curve with optional smoothing and confidence intervals.
        
        Args:
            data: Dictionary containing 'rewards' and optionally 'steps'
            **kwargs: Additional plotting parameters
            
        Returns:
            Tuple of (figure, axes, plot_info)
        """
        fig, ax = self.create_figure()
        
        rewards = data.get('rewards', [])
        steps = data.get('steps', list(range(len(rewards))))
        
        if not rewards:
            warnings.warn("No reward data provided for learning curve")
            return fig, ax, {'error': 'No data'}
        
        # Plot raw data
        ax.plot(steps, rewards, alpha=0.3, color='blue', label='Raw rewards')
        
        # Add smoothed line if enough data
        if len(rewards) > 10:
            # Simple moving average
            window = min(len(rewards) // 10, 50)
            smoothed = self._moving_average(rewards, window)
            smooth_steps = steps[window-1:]
            ax.plot(smooth_steps, smoothed, alpha=self.smoothing_alpha, 
                   color='red', linewidth=2, label=f'Smoothed (window={window})')
        
        # Add confidence intervals if provided
        if 'confidence_intervals' in data:
            ci_lower, ci_upper = data['confidence_intervals']
            ax.fill_between(steps, ci_lower, ci_upper, 
                           alpha=self.confidence_alpha, color='blue', 
                           label='Confidence interval')
        
        # Formatting
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Learning Curve')
        ax.legend()
        
        if self.grid_enabled:
            ax.grid(True, alpha=0.3)
        
        plot_info = {
            'num_episodes': len(rewards),
            'final_reward': rewards[-1] if rewards else 0,
            'max_reward': max(rewards) if rewards else 0,
            'min_reward': min(rewards) if rewards else 0
        }
        
        return fig, ax, plot_info
    
    def _plot_episode_rewards(
        self, 
        data: Dict[str, List[float]], 
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
        """Plot episode rewards over time."""
        fig, ax = self.create_figure()
        
        rewards = data.get('rewards', [])
        episodes = data.get('episodes', list(range(len(rewards))))
        
        ax.plot(episodes, rewards, marker='o', markersize=2, alpha=0.7)
        
        # Add running mean
        if len(rewards) > 10:
            running_mean = self._running_mean(rewards, window=20)
            ax.plot(episodes, running_mean, color='red', linewidth=2, 
                   label='Running mean (20 episodes)')
            ax.legend()
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        
        if self.grid_enabled:
            ax.grid(True, alpha=0.3)
        
        plot_info = {
            'total_episodes': len(rewards),
            'mean_reward': np.mean(rewards) if rewards else 0,
            'std_reward': np.std(rewards) if rewards else 0
        }
        
        return fig, ax, plot_info
    
    def _plot_episode_lengths(
        self, 
        data: Dict[str, List[int]], 
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
        """Plot episode lengths over time."""
        fig, ax = self.create_figure()
        
        lengths = data.get('lengths', data.get('episode_lengths', []))
        episodes = data.get('episodes', list(range(len(lengths))))
        
        ax.plot(episodes, lengths, marker='o', markersize=2, alpha=0.7)
        
        # Add mean line
        if lengths:
            mean_length = np.mean(lengths)
            ax.axhline(y=mean_length, color='red', linestyle='--', 
                      label=f'Mean length: {mean_length:.1f}')
            ax.legend()
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Lengths')
        
        if self.grid_enabled:
            ax.grid(True, alpha=0.3)
        
        plot_info = {
            'total_episodes': len(lengths),
            'mean_length': np.mean(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0
        }
        
        return fig, ax, plot_info
    
    def _plot_comparison(
        self, 
        data: Dict[str, Dict[str, List[float]]], 
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
        """Plot comparison between multiple experiments or algorithms."""
        fig, ax = self.create_figure()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
        
        for i, (name, experiment_data) in enumerate(data.items()):
            rewards = experiment_data.get('rewards', [])
            steps = experiment_data.get('steps', list(range(len(rewards))))
            
            if rewards:
                # Plot smoothed version for comparison
                if len(rewards) > 10:
                    window = min(len(rewards) // 10, 50)
                    smoothed = self._moving_average(rewards, window)
                    smooth_steps = steps[window-1:]
                    ax.plot(smooth_steps, smoothed, color=colors[i], 
                           linewidth=2, label=name)
                else:
                    ax.plot(steps, rewards, color=colors[i], 
                           linewidth=2, label=name)
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Algorithm Comparison')
        ax.legend()
        
        if self.grid_enabled:
            ax.grid(True, alpha=0.3)
        
        plot_info = {
            'num_experiments': len(data),
            'experiment_names': list(data.keys())
        }
        
        return fig, ax, plot_info
    
    def _plot_distribution(
        self, 
        data: Dict[str, List[float]], 
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
        """Plot distribution of rewards or other metrics."""
        fig, ax = self.create_figure()
        
        values = data.get('values', data.get('rewards', []))
        
        if not values:
            warnings.warn("No values provided for distribution plot")
            return fig, ax, {'error': 'No data'}
        
        # Create histogram
        n_bins = kwargs.get('bins', 30)
        ax.hist(values, bins=n_bins, alpha=0.7, density=True, 
               color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        ax.axvline(mean_val, color='red', linestyle='--', 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle='--', 
                  alpha=0.7, label=f'+1 STD: {mean_val + std_val:.2f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle='--', 
                  alpha=0.7, label=f'-1 STD: {mean_val - std_val:.2f}')
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Distribution')
        ax.legend()
        
        if self.grid_enabled:
            ax.grid(True, alpha=0.3)
        
        plot_info = {
            'num_values': len(values),
            'mean': mean_val,
            'std': std_val,
            'min': min(values),
            'max': max(values)
        }
        
        return fig, ax, plot_info
    
    def _plot_correlation_matrix(
        self, 
        data: Dict[str, List[float]], 
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
        """Plot correlation matrix of different metrics."""
        if not HAS_PANDAS or not HAS_SEABORN:
            warnings.warn("Pandas and Seaborn required for correlation matrix")
            fig, ax = self.create_figure()
            return fig, ax, {'error': 'Missing dependencies'}
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        fig, ax = self.create_figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax)
        
        ax.set_title('Metric Correlation Matrix')
        
        plot_info = {
            'num_metrics': len(corr_matrix.columns),
            'metric_names': list(corr_matrix.columns)
        }
        
        return fig, ax, plot_info
    
    def _plot_performance_over_time(
        self, 
        data: Dict[str, List[float]], 
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
        """Plot multiple performance metrics over time."""
        fig, ax = self.create_figure()
        
        # Plot multiple metrics on the same axis (normalized)
        metrics_plotted = 0
        
        for metric_name, values in data.items():
            if metric_name in ['steps', 'episodes']:
                continue  # Skip x-axis data
            
            if values and isinstance(values[0], (int, float)):
                # Normalize values to 0-1 range for comparison
                norm_values = self._normalize_values(values)
                steps = data.get('steps', list(range(len(values))))
                
                ax.plot(steps, norm_values, label=metric_name, linewidth=2)
                metrics_plotted += 1
        
        if metrics_plotted == 0:
            warnings.warn("No plottable metrics found")
            return fig, ax, {'error': 'No metrics'}
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Normalized Performance')
        ax.set_title('Performance Metrics Over Time')
        ax.legend()
        
        if self.grid_enabled:
            ax.grid(True, alpha=0.3)
        
        plot_info = {
            'metrics_plotted': metrics_plotted,
            'metric_names': [k for k in data.keys() if k not in ['steps', 'episodes']]
        }
        
        return fig, ax, plot_info
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average of data."""
        return [np.mean(data[max(0, i-window+1):i+1]) for i in range(window-1, len(data))]
    
    def _running_mean(self, data: List[float], window: int) -> List[float]:
        """Calculate running mean with specified window."""
        return [np.mean(data[max(0, i-window+1):i+1]) for i in range(len(data))]
    
    def _normalize_values(self, values: List[float]) -> List[float]:
        """Normalize values to 0-1 range."""
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return [0.5] * len(values)
        
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    def create_multi_plot_figure(
        self, 
        plot_configs: List[Dict[str, Any]]
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create figure with multiple subplots.
        
        Args:
            plot_configs: List of plot configuration dictionaries
            
        Returns:
            Tuple of (figure, list_of_axes)
        """
        n_plots = len(plot_configs)
        
        # Determine subplot layout
        if n_plots <= 2:
            nrows, ncols = 1, n_plots
        elif n_plots <= 4:
            nrows, ncols = 2, 2
        else:
            nrows = int(np.ceil(np.sqrt(n_plots)))
            ncols = int(np.ceil(n_plots / nrows))
        
        fig, axes = self.create_figure(
            figsize=(ncols * 6, nrows * 4),
            nrows=nrows, 
            ncols=ncols
        )
        
        # Ensure axes is always a list
        if n_plots == 1:
            axes = [axes]
        elif nrows == 1 or ncols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        return fig, axes


# Export classes
__all__ = [
    'BasePlotter',
    'StandardPlotter'
]