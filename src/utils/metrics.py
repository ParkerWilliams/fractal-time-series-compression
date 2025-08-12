import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.fft import fft, fftfreq
import warnings


class CompressionMetrics:
    """Utilities for evaluating compression quality and performance."""
    
    @staticmethod
    def compression_ratio(original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio (original/compressed)."""
        return original_size / compressed_size if compressed_size > 0 else float('inf')
    
    @staticmethod
    def space_savings_percentage(original_size: int, compressed_size: int) -> float:
        """Calculate space savings as percentage."""
        return (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    
    @staticmethod
    def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Mean Squared Error between original and reconstructed signals."""
        if len(original) != len(reconstructed):
            raise ValueError("Arrays must have the same length")
        return np.mean((original - reconstructed) ** 2)
    
    @staticmethod
    def root_mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(CompressionMetrics.mean_squared_error(original, reconstructed))
    
    @staticmethod
    def mean_absolute_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        if len(original) != len(reconstructed):
            raise ValueError("Arrays must have the same length")
        return np.mean(np.abs(original - reconstructed))
    
    @staticmethod
    def normalized_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Normalized Root Mean Squared Error."""
        rmse = CompressionMetrics.root_mean_squared_error(original, reconstructed)
        original_range = np.max(original) - np.min(original)
        return rmse / original_range if original_range > 0 else float('inf')
    
    @staticmethod
    def signal_to_noise_ratio(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB."""
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        
        if noise_power == 0:
            return float('inf')
        if signal_power == 0:
            return float('-inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    @staticmethod
    def peak_signal_to_noise_ratio(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio in dB."""
        mse = CompressionMetrics.mean_squared_error(original, reconstructed)
        if mse == 0:
            return float('inf')
        
        max_value = np.max(original)
        return 20 * np.log10(max_value / np.sqrt(mse))
    
    @staticmethod
    def pearson_correlation(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(original) != len(reconstructed):
            raise ValueError("Arrays must have the same length")
        
        correlation, _ = stats.pearsonr(original, reconstructed)
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def structural_similarity_index(original: np.ndarray, reconstructed: np.ndarray,
                                   dynamic_range: float = None) -> float:
        """Calculate Structural Similarity Index (SSIM) for 1D signals."""
        if len(original) != len(reconstructed):
            raise ValueError("Arrays must have the same length")
        
        if dynamic_range is None:
            dynamic_range = np.max(original) - np.min(original)
        
        # Constants for stability
        c1 = (0.01 * dynamic_range) ** 2
        c2 = (0.03 * dynamic_range) ** 2
        
        # Calculate means
        mu1 = np.mean(original)
        mu2 = np.mean(reconstructed)
        
        # Calculate variances and covariance
        var1 = np.var(original)
        var2 = np.var(reconstructed)
        covar = np.mean((original - mu1) * (reconstructed - mu2))
        
        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * covar + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (var1 + var2 + c2)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    @staticmethod
    def frequency_domain_error(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """Calculate frequency domain reconstruction errors."""
        if len(original) != len(reconstructed):
            raise ValueError("Arrays must have the same length")
        
        # Compute FFT
        fft_original = fft(original)
        fft_reconstructed = fft(reconstructed)
        
        # Calculate magnitude and phase
        mag_original = np.abs(fft_original)
        mag_reconstructed = np.abs(fft_reconstructed)
        
        phase_original = np.angle(fft_original)
        phase_reconstructed = np.angle(fft_reconstructed)
        
        # Calculate errors
        magnitude_error = np.mean(np.abs(mag_original - mag_reconstructed))
        phase_error = np.mean(np.abs(np.angle(np.exp(1j * (phase_original - phase_reconstructed)))))
        
        # Spectral correlation
        spectral_correlation = CompressionMetrics.pearson_correlation(mag_original, mag_reconstructed)
        
        return {
            'magnitude_error': magnitude_error,
            'phase_error': phase_error,
            'spectral_correlation': spectral_correlation
        }
    
    @staticmethod
    def fractal_dimension_preservation(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """Evaluate how well fractal properties are preserved."""
        def box_counting_dimension(signal, max_boxes=50):
            """Estimate fractal dimension using box counting method."""
            # Embed signal in 2D space
            points = np.column_stack([np.arange(len(signal)), signal])
            
            # Normalize to unit square
            points_norm = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0))
            
            scales = np.logspace(-2, 0, max_boxes)
            box_counts = []
            
            for scale in scales:
                # Count boxes containing at least one point
                grid_x = np.floor(points_norm[:, 0] / scale).astype(int)
                grid_y = np.floor(points_norm[:, 1] / scale).astype(int)
                
                # Count unique grid cells
                unique_boxes = len(set(zip(grid_x, grid_y)))
                box_counts.append(unique_boxes)
            
            # Fit log-log relationship
            log_scales = np.log(scales)
            log_counts = np.log(box_counts)
            
            # Remove invalid values
            valid_mask = np.isfinite(log_scales) & np.isfinite(log_counts)
            if np.sum(valid_mask) < 3:
                return 1.0  # Default dimension
            
            try:
                slope, _, _, _, _ = stats.linregress(log_scales[valid_mask], log_counts[valid_mask])
                return -slope  # Fractal dimension
            except:
                return 1.0
        
        def hurst_exponent(signal):
            """Estimate Hurst exponent using R/S analysis."""
            n = len(signal)
            if n < 10:
                return 0.5
            
            # Calculate cumulative departures from mean
            mean_signal = np.mean(signal)
            cumulative_deviations = np.cumsum(signal - mean_signal)
            
            # Calculate range and standard deviation for different time scales
            scales = np.unique(np.logspace(1, np.log10(n//4), 20).astype(int))
            rs_values = []
            
            for scale in scales:
                if scale >= n:
                    continue
                
                # Divide into segments
                n_segments = n // scale
                rs_segment = []
                
                for i in range(n_segments):
                    start = i * scale
                    end = start + scale
                    segment = cumulative_deviations[start:end]
                    
                    # Range
                    range_val = np.max(segment) - np.min(segment)
                    
                    # Standard deviation
                    std_val = np.std(signal[start:end])
                    
                    if std_val > 0:
                        rs_segment.append(range_val / std_val)
                
                if rs_segment:
                    rs_values.append(np.mean(rs_segment))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Fit log-log relationship
            try:
                log_scales = np.log(scales[:len(rs_values)])
                log_rs = np.log(rs_values)
                
                valid_mask = np.isfinite(log_scales) & np.isfinite(log_rs)
                if np.sum(valid_mask) < 3:
                    return 0.5
                
                slope, _, _, _, _ = stats.linregress(log_scales[valid_mask], log_rs[valid_mask])
                return slope
            except:
                return 0.5
        
        # Calculate fractal dimensions
        try:
            fd_original = box_counting_dimension(original)
            fd_reconstructed = box_counting_dimension(reconstructed)
            
            hurst_original = hurst_exponent(original)
            hurst_reconstructed = hurst_exponent(reconstructed)
            
            return {
                'fractal_dimension_original': fd_original,
                'fractal_dimension_reconstructed': fd_reconstructed,
                'fractal_dimension_error': abs(fd_original - fd_reconstructed),
                'hurst_exponent_original': hurst_original,
                'hurst_exponent_reconstructed': hurst_reconstructed,
                'hurst_exponent_error': abs(hurst_original - hurst_reconstructed)
            }
        except Exception as e:
            warnings.warn(f"Error in fractal dimension calculation: {e}")
            return {
                'fractal_dimension_original': 1.0,
                'fractal_dimension_reconstructed': 1.0,
                'fractal_dimension_error': 0.0,
                'hurst_exponent_original': 0.5,
                'hurst_exponent_reconstructed': 0.5,
                'hurst_exponent_error': 0.0
            }
    
    @staticmethod
    def comprehensive_evaluation(original: np.ndarray, reconstructed: np.ndarray,
                               original_size: int, compressed_size: int,
                               compression_time: float, decompression_time: float) -> Dict[str, Any]:
        """Perform comprehensive evaluation of compression method."""
        metrics = {}
        
        # Compression efficiency
        metrics['compression_ratio'] = CompressionMetrics.compression_ratio(original_size, compressed_size)
        metrics['space_savings_percent'] = CompressionMetrics.space_savings_percentage(original_size, compressed_size)
        metrics['compression_time'] = compression_time
        metrics['decompression_time'] = decompression_time
        metrics['total_time'] = compression_time + decompression_time
        
        # Reconstruction quality
        metrics['mse'] = CompressionMetrics.mean_squared_error(original, reconstructed)
        metrics['rmse'] = CompressionMetrics.root_mean_squared_error(original, reconstructed)
        metrics['mae'] = CompressionMetrics.mean_absolute_error(original, reconstructed)
        metrics['normalized_rmse'] = CompressionMetrics.normalized_rmse(original, reconstructed)
        
        # Signal quality
        metrics['snr_db'] = CompressionMetrics.signal_to_noise_ratio(original, reconstructed)
        metrics['psnr_db'] = CompressionMetrics.peak_signal_to_noise_ratio(original, reconstructed)
        metrics['pearson_correlation'] = CompressionMetrics.pearson_correlation(original, reconstructed)
        metrics['ssim'] = CompressionMetrics.structural_similarity_index(original, reconstructed)
        
        # Frequency domain analysis
        freq_metrics = CompressionMetrics.frequency_domain_error(original, reconstructed)
        metrics.update({f'freq_{k}': v for k, v in freq_metrics.items()})
        
        # Fractal properties
        fractal_metrics = CompressionMetrics.fractal_dimension_preservation(original, reconstructed)
        metrics.update({f'fractal_{k}': v for k, v in fractal_metrics.items()})
        
        return metrics