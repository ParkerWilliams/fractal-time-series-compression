import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns
from matplotlib.gridspec import GridSpec


class CompressionVisualizer:
    """Visualization utilities for fractal time series compression results."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer with plotting style.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self._setup_style()
    
    def _setup_style(self):
        """Setup plotting style and parameters."""
        try:
            plt.style.use(self.style)
        except:
            # Fallback to default style if seaborn is not available
            plt.style.use('default')
        
        # Set default parameters
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_compression_comparison(self, original_time: np.ndarray, original_values: np.ndarray,
                                  reconstructed_values: np.ndarray, method_name: str = "",
                                  metrics: Dict[str, float] = None, save_path: str = None) -> plt.Figure:
        """
        Plot comparison between original and reconstructed time series.
        
        Args:
            original_time: Time axis for original data
            original_values: Original time series values
            reconstructed_values: Reconstructed time series values
            method_name: Name of compression method
            metrics: Dictionary of evaluation metrics
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Compression Results: {method_name}', fontsize=16, fontweight='bold')
        
        # Create time axis for reconstructed data if needed
        if len(reconstructed_values) != len(original_time):
            reconstructed_time = np.linspace(original_time[0], original_time[-1], len(reconstructed_values))
        else:
            reconstructed_time = original_time
        
        # Plot 1: Overlay comparison
        axes[0, 0].plot(original_time, original_values, 'b-', label='Original', alpha=0.7, linewidth=1.5)
        axes[0, 0].plot(reconstructed_time, reconstructed_values, 'r--', label='Reconstructed', alpha=0.8, linewidth=1.5)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Original vs Reconstructed')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error signal
        if len(original_values) == len(reconstructed_values):
            error = original_values - reconstructed_values
            error_time = original_time
        else:
            # Interpolate for error calculation
            error = np.interp(original_time, reconstructed_time, reconstructed_values) - original_values
            error_time = original_time
        
        axes[0, 1].plot(error_time, error, 'g-', alpha=0.7, linewidth=1)
        axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].set_title('Reconstruction Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot (correlation)
        if len(original_values) == len(reconstructed_values):
            scatter_original = original_values
            scatter_reconstructed = reconstructed_values
        else:
            scatter_original = original_values
            scatter_reconstructed = np.interp(original_time, reconstructed_time, reconstructed_values)
        
        axes[1, 0].scatter(scatter_original, scatter_reconstructed, alpha=0.6, s=20)
        
        # Add perfect correlation line
        min_val = min(np.min(scatter_original), np.min(scatter_reconstructed))
        max_val = max(np.max(scatter_original), np.max(scatter_reconstructed))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        axes[1, 0].set_xlabel('Original Values')
        axes[1, 0].set_ylabel('Reconstructed Values')
        axes[1, 0].set_title('Correlation Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Metrics summary
        axes[1, 1].axis('off')
        if metrics:
            metrics_text = self._format_metrics_text(metrics)
            axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontfamily='monospace', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_frequency_analysis(self, original: np.ndarray, reconstructed: np.ndarray,
                               sampling_rate: float = 1.0, method_name: str = "",
                               save_path: str = None) -> plt.Figure:
        """
        Plot frequency domain analysis of compression results.
        
        Args:
            original: Original time series
            reconstructed: Reconstructed time series
            sampling_rate: Sampling rate of the data
            method_name: Name of compression method
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Frequency Domain Analysis: {method_name}', fontsize=16, fontweight='bold')
        
        # Ensure same length for comparison
        if len(original) != len(reconstructed):
            min_len = min(len(original), len(reconstructed))
            original = original[:min_len]
            reconstructed = reconstructed[:min_len]
        
        # Compute FFT
        fft_original = np.fft.fft(original)
        fft_reconstructed = np.fft.fft(reconstructed)
        freqs = np.fft.fftfreq(len(original), 1/sampling_rate)
        
        # Only plot positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        fft_original_pos = fft_original[:len(freqs)//2]
        fft_reconstructed_pos = fft_reconstructed[:len(freqs)//2]
        
        # Plot 1: Magnitude spectrum
        axes[0, 0].loglog(positive_freqs[1:], np.abs(fft_original_pos)[1:], 'b-', label='Original', alpha=0.7)
        axes[0, 0].loglog(positive_freqs[1:], np.abs(fft_reconstructed_pos)[1:], 'r--', label='Reconstructed', alpha=0.7)
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].set_ylabel('Magnitude')
        axes[0, 0].set_title('Magnitude Spectrum')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Phase spectrum
        phase_original = np.angle(fft_original_pos)
        phase_reconstructed = np.angle(fft_reconstructed_pos)
        
        axes[0, 1].plot(positive_freqs, phase_original, 'b-', label='Original', alpha=0.7)
        axes[0, 1].plot(positive_freqs, phase_reconstructed, 'r--', label='Reconstructed', alpha=0.7)
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Phase (radians)')
        axes[0, 1].set_title('Phase Spectrum')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Magnitude error
        mag_error = np.abs(np.abs(fft_original_pos) - np.abs(fft_reconstructed_pos))
        axes[1, 0].semilogy(positive_freqs, mag_error, 'g-', alpha=0.7)
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_ylabel('Magnitude Error')
        axes[1, 0].set_title('Frequency Domain Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Power spectral density
        from scipy.signal import welch
        
        try:
            freqs_psd, psd_original = welch(original, fs=sampling_rate, nperseg=min(256, len(original)//4))
            freqs_psd, psd_reconstructed = welch(reconstructed, fs=sampling_rate, nperseg=min(256, len(reconstructed)//4))
            
            axes[1, 1].loglog(freqs_psd, psd_original, 'b-', label='Original', alpha=0.7)
            axes[1, 1].loglog(freqs_psd, psd_reconstructed, 'r--', label='Reconstructed', alpha=0.7)
            axes[1, 1].set_xlabel('Frequency')
            axes[1, 1].set_ylabel('Power Spectral Density')
            axes[1, 1].set_title('Power Spectral Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        except:
            axes[1, 1].text(0.5, 0.5, 'PSD calculation failed', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_method_comparison(self, results: Dict[str, Dict[str, Any]], 
                              metric_names: List[str] = None, save_path: str = None) -> plt.Figure:
        """
        Compare multiple compression methods across different metrics.
        
        Args:
            results: Dictionary with method names as keys and evaluation results as values
            metric_names: List of metrics to compare (if None, use default selection)
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if metric_names is None:
            metric_names = ['compression_ratio', 'normalized_rmse', 'pearson_correlation', 'snr_db']
        
        n_metrics = len(metric_names)
        n_methods = len(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Compression Methods Comparison', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        method_names = list(results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, n_methods))
        
        for i, metric in enumerate(metric_names[:4]):  # Limit to 4 metrics for 2x2 layout
            if i >= len(axes):
                break
                
            values = []
            labels = []
            
            for method_name in method_names:
                if metric in results[method_name]:
                    values.append(results[method_name][metric])
                    labels.append(method_name)
            
            if values:
                bars = axes[i].bar(labels, values, color=colors[:len(values)], alpha=0.7)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_compression_performance(self, results: Dict[str, Dict[str, Any]], 
                                   save_path: str = None) -> plt.Figure:
        """
        Plot compression performance (ratio vs quality trade-off).
        
        Args:
            results: Dictionary with method names as keys and evaluation results as values
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        compression_ratios = []
        quality_scores = []
        method_names = []
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
        
        for i, (method_name, metrics) in enumerate(results.items()):
            if 'compression_ratio' in metrics and 'pearson_correlation' in metrics:
                compression_ratios.append(metrics['compression_ratio'])
                quality_scores.append(metrics['pearson_correlation'])
                method_names.append(method_name)
        
        if compression_ratios and quality_scores:
            scatter = ax.scatter(compression_ratios, quality_scores, 
                               c=colors[:len(compression_ratios)], s=100, alpha=0.7)
            
            # Add method name annotations
            for i, method in enumerate(method_names):
                ax.annotate(method, (compression_ratios[i], quality_scores[i]),
                          xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax.set_xlabel('Compression Ratio')
            ax.set_ylabel('Correlation with Original')
            ax.set_title('Compression Performance: Ratio vs Quality Trade-off')
            ax.grid(True, alpha=0.3)
            
            # Add ideal region
            ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='High Quality (>0.9)')
            ax.axvline(x=5.0, color='orange', linestyle='--', alpha=0.5, label='Good Compression (>5x)')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _format_metrics_text(self, metrics: Dict[str, float]) -> str:
        """Format metrics dictionary into readable text."""
        formatted_lines = []
        
        # Key metrics to display
        key_metrics = [
            ('Compression Ratio', 'compression_ratio'),
            ('Space Savings (%)', 'space_savings_percent'),
            ('RMSE', 'rmse'),
            ('Correlation', 'pearson_correlation'),
            ('SNR (dB)', 'snr_db'),
            ('PSNR (dB)', 'psnr_db'),
            ('SSIM', 'ssim'),
            ('Compression Time (s)', 'compression_time'),
            ('Decompression Time (s)', 'decompression_time')
        ]
        
        for display_name, metric_key in key_metrics:
            if metric_key in metrics:
                value = metrics[metric_key]
                if isinstance(value, float):
                    if abs(value) < 0.001 or abs(value) > 1000:
                        formatted_lines.append(f"{display_name}: {value:.2e}")
                    else:
                        formatted_lines.append(f"{display_name}: {value:.4f}")
                else:
                    formatted_lines.append(f"{display_name}: {value}")
        
        return '\n'.join(formatted_lines)
    
    def create_comprehensive_report(self, original_time: np.ndarray, original_values: np.ndarray,
                                   results: Dict[str, Tuple], save_dir: str = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualization report for multiple compression methods.
        
        Args:
            original_time: Original time axis
            original_values: Original time series values
            results: Dictionary with method names as keys and (reconstructed_values, metrics) tuples as values
            save_dir: Directory to save figures
            
        Returns:
            Dictionary of figure objects
        """
        figures = {}
        
        # Individual method comparisons
        for method_name, (reconstructed_values, metrics) in results.items():
            save_path = f"{save_dir}/{method_name}_comparison.png" if save_dir else None
            fig = self.plot_compression_comparison(
                original_time, original_values, reconstructed_values, 
                method_name, metrics, save_path
            )
            figures[f"{method_name}_comparison"] = fig
            
            # Frequency analysis
            save_path = f"{save_dir}/{method_name}_frequency.png" if save_dir else None
            fig = self.plot_frequency_analysis(
                original_values, reconstructed_values, 
                method_name=method_name, save_path=save_path
            )
            figures[f"{method_name}_frequency"] = fig
        
        # Method comparison
        metrics_dict = {name: metrics for name, (_, metrics) in results.items()}
        
        save_path = f"{save_dir}/methods_comparison.png" if save_dir else None
        fig = self.plot_method_comparison(metrics_dict, save_path=save_path)
        figures["methods_comparison"] = fig
        
        save_path = f"{save_dir}/performance_tradeoff.png" if save_dir else None
        fig = self.plot_compression_performance(metrics_dict, save_path=save_path)
        figures["performance_tradeoff"] = fig
        
        return figures