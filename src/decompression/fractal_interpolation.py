import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from scipy.interpolate import interp1d
from ..compression.base_compressor import DecompressionResult


class FractalInterpolationFunction:
    """Represents a Fractal Interpolation Function (FIF) for time series reconstruction."""
    
    def __init__(self, control_points: np.ndarray, vertical_scaling_factors: np.ndarray,
                 interpolation_method: str = 'linear'):
        """
        Initialize FIF with control points and vertical scaling factors.
        
        Args:
            control_points: Array of shape (n, 2) containing (x, y) control points
            vertical_scaling_factors: Array of scaling factors for each interval
            interpolation_method: Method for base interpolation ('linear', 'cubic', etc.)
        """
        self.control_points = control_points
        self.vertical_scaling_factors = vertical_scaling_factors
        self.interpolation_method = interpolation_method
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate FIF parameters."""
        if len(self.control_points) < 2:
            raise ValueError("At least 2 control points are required")
        
        if len(self.vertical_scaling_factors) != len(self.control_points) - 1:
            raise ValueError("Number of scaling factors must be n_points - 1")
        
        # Check that scaling factors are contractive
        if np.any(np.abs(self.vertical_scaling_factors) >= 1.0):
            raise ValueError("All vertical scaling factors must satisfy |d| < 1 for contractivity")
    
    def _base_interpolation(self, x: np.ndarray) -> np.ndarray:
        """Compute base interpolation through control points."""
        x_points = self.control_points[:, 0]
        y_points = self.control_points[:, 1]
        
        # Create interpolation function
        if self.interpolation_method == 'linear':
            interp_func = interp1d(x_points, y_points, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
        elif self.interpolation_method == 'cubic':
            if len(x_points) >= 4:
                interp_func = interp1d(x_points, y_points, kind='cubic',
                                     bounds_error=False, fill_value='extrapolate')
            else:
                interp_func = interp1d(x_points, y_points, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
        else:
            raise ValueError(f"Unsupported interpolation method: {self.interpolation_method}")
        
        return interp_func(x)
    
    def _fractal_perturbation(self, x: np.ndarray, base_values: np.ndarray) -> np.ndarray:
        """Apply fractal perturbation to base interpolation."""
        x_points = self.control_points[:, 0]
        y_points = self.control_points[:, 1]
        
        # Initialize perturbation
        perturbation = np.zeros_like(x)
        
        # Apply scaling for each interval
        for i in range(len(x_points) - 1):
            # Find points in current interval
            mask = (x >= x_points[i]) & (x <= x_points[i + 1])
            if not np.any(mask):
                continue
            
            x_interval = x[mask]
            
            # Map interval to [0, 1]
            t = (x_interval - x_points[i]) / (x_points[i + 1] - x_points[i])
            
            # Linear interpolation between interval endpoints
            y_interval_linear = y_points[i] + t * (y_points[i + 1] - y_points[i])
            
            # Get base interpolation values for this interval
            base_interval = base_values[mask]
            
            # Apply vertical scaling to the difference
            scaling_factor = self.vertical_scaling_factors[i]
            perturbation[mask] = scaling_factor * (base_interval - y_interval_linear)
        
        return perturbation
    
    def evaluate(self, x: np.ndarray, iterations: int = 10) -> np.ndarray:
        """
        Evaluate the Fractal Interpolation Function at given points.
        
        Args:
            x: Points where to evaluate the FIF
            iterations: Number of iterations for fractal convergence
            
        Returns:
            Array of FIF values at x points
        """
        # Start with base interpolation
        current_values = self._base_interpolation(x)
        
        # Iteratively apply fractal transformations
        for _ in range(iterations):
            perturbation = self._fractal_perturbation(x, current_values)
            new_values = self._base_interpolation(x) + perturbation
            
            # Check for convergence
            if np.max(np.abs(new_values - current_values)) < 1e-10:
                break
            
            current_values = new_values
        
        return current_values


class FractalInterpolationDecompressor:
    """Decompressor using Fractal Interpolation Functions."""
    
    def __init__(self, interpolation_method: str = 'linear', max_iterations: int = 50,
                 convergence_threshold: float = 1e-8):
        self.interpolation_method = interpolation_method
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def _estimate_control_points(self, compressed_data: Dict[str, Any], 
                                target_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate control points and scaling factors from compressed data."""
        
        # This method adapts different compression formats to FIF parameters
        if 'transformations' in compressed_data:
            # IFS compression format
            return self._extract_from_ifs_data(compressed_data, target_length)
        elif 'fractal_blocks' in compressed_data:
            # Fractal coding format
            return self._extract_from_fractal_coding_data(compressed_data, target_length)
        else:
            raise ValueError("Unsupported compressed data format for FIF decompression")
    
    def _extract_from_ifs_data(self, ifs_data: Dict[str, Any], 
                              target_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract control points from IFS compression data."""
        transformations = ifs_data['transformations']
        time_range = ifs_data['time_range']
        value_range = ifs_data['value_range']
        
        # Use IFS fixed points as control points
        n_controls = min(len(transformations) + 2, 10)  # Limit number of control points
        
        # Create evenly spaced time points
        time_points = np.linspace(time_range[0], time_range[1], n_controls)
        
        # Estimate values at control points using IFS attractors
        value_points = []
        for i, t in enumerate(time_points):
            # Use transformation parameters to estimate value
            if i < len(transformations):
                transform = transformations[i]
                # Simple mapping from transformation parameters to values
                estimated_value = transform[4] * (value_range[1] - value_range[0]) + value_range[0]
            else:
                # Interpolate for additional points
                estimated_value = value_range[0] + (value_range[1] - value_range[0]) * (i / (n_controls - 1))
            
            value_points.append(estimated_value)
        
        control_points = np.column_stack([time_points, value_points])
        
        # Extract scaling factors from IFS transformations (use 'a' coefficients)
        scaling_factors = []
        for transform in transformations[:n_controls-1]:
            scaling = abs(transform[0])  # 'a' coefficient
            scaling = min(scaling, 0.9)  # Ensure contractivity
            scaling_factors.append(scaling)
        
        # Pad if needed
        while len(scaling_factors) < n_controls - 1:
            scaling_factors.append(0.5)
        
        return control_points, np.array(scaling_factors[:n_controls-1])
    
    def _extract_from_fractal_coding_data(self, fc_data: Dict[str, Any], 
                                         target_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract control points from fractal coding data."""
        fractal_blocks = fc_data['fractal_blocks']
        time_range = fc_data['time_range']
        original_length = fc_data['original_length']
        stats = fc_data['value_statistics']
        
        # Use block boundaries as control points
        time_points = [time_range[0]]
        value_points = [stats['min']]
        scaling_factors = []
        
        # Add control points from fractal blocks
        for block_data in fractal_blocks:
            # Map range positions to time
            range_start_time = time_range[0] + (block_data['range_start'] / original_length) * (time_range[1] - time_range[0])
            range_end_time = time_range[0] + (block_data['range_end'] / original_length) * (time_range[1] - time_range[0])
            
            # Estimate value from block offset
            estimated_value = block_data['offset']
            
            # Add control point at block end
            if range_end_time not in time_points:
                time_points.append(range_end_time)
                value_points.append(estimated_value)
                
                # Use scaling factor from block
                scaling = abs(block_data['scaling_factor'])
                scaling = min(scaling, 0.9)  # Ensure contractivity
                scaling_factors.append(scaling)
        
        # Add final control point
        if time_range[1] not in time_points:
            time_points.append(time_range[1])
            value_points.append(stats['max'])
            scaling_factors.append(0.5)
        
        # Ensure we have at least 2 control points
        if len(time_points) < 2:
            time_points = [time_range[0], time_range[1]]
            value_points = [stats['min'], stats['max']]
            scaling_factors = [0.5]
        
        control_points = np.column_stack([time_points, value_points])
        
        # Adjust scaling factors length
        while len(scaling_factors) < len(time_points) - 1:
            scaling_factors.append(0.5)
        scaling_factors = scaling_factors[:len(time_points) - 1]
        
        return control_points, np.array(scaling_factors)
    
    def decompress_with_fif(self, compressed_data: Dict[str, Any], 
                           target_length: int, time_range: Tuple[float, float] = None) -> DecompressionResult:
        """
        Decompress using Fractal Interpolation Functions.
        
        Args:
            compressed_data: Compressed data from any fractal compression method
            target_length: Desired length of reconstructed series
            time_range: Time range for reconstruction (if None, use from compressed data)
            
        Returns:
            DecompressionResult with reconstructed time series
        """
        import time as time_module
        start_time = time_module.time()
        
        try:
            # Extract control points and scaling factors
            control_points, scaling_factors = self._estimate_control_points(compressed_data, target_length)
            
            # Determine time range
            if time_range is None:
                if 'time_range' in compressed_data:
                    time_range = compressed_data['time_range']
                else:
                    time_range = (0, 1)
            
            # Create evaluation points
            eval_times = np.linspace(time_range[0], time_range[1], target_length)
            
            # Create and evaluate FIF
            fif = FractalInterpolationFunction(
                control_points=control_points,
                vertical_scaling_factors=scaling_factors,
                interpolation_method=self.interpolation_method
            )
            
            reconstructed_values = fif.evaluate(eval_times, iterations=self.max_iterations)
            
            elapsed_time = time_module.time() - start_time
            
            # Calculate quality metrics
            quality_metrics = {
                'control_points_count': len(control_points),
                'mean_scaling_factor': np.mean(np.abs(scaling_factors)),
                'max_scaling_factor': np.max(np.abs(scaling_factors))
            }
            
            return DecompressionResult(
                reconstructed_data=reconstructed_values,
                decompression_time=elapsed_time,
                method_name="Fractal_Interpolation_Decompressor",
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            # Fallback to simple interpolation if FIF fails
            elapsed_time = time_module.time() - start_time
            
            if 'value_statistics' in compressed_data:
                stats = compressed_data['value_statistics']
                fallback_values = np.random.normal(stats['mean'], stats['std'] * 0.1, target_length)
            else:
                fallback_values = np.zeros(target_length)
            
            return DecompressionResult(
                reconstructed_data=fallback_values,
                decompression_time=elapsed_time,
                method_name="Fractal_Interpolation_Decompressor_Fallback",
                quality_metrics={'error': str(e)}
            )
    
    def create_adaptive_fif(self, data_points: np.ndarray, max_control_points: int = 20) -> FractalInterpolationFunction:
        """
        Create an adaptive FIF by automatically selecting control points.
        
        Args:
            data_points: Array of shape (n, 2) with (time, value) pairs
            max_control_points: Maximum number of control points to use
            
        Returns:
            Optimized FractalInterpolationFunction
        """
        if len(data_points) <= max_control_points:
            # Use all points as control points
            control_points = data_points
            scaling_factors = np.full(len(control_points) - 1, 0.5)
        else:
            # Adaptively select control points using curvature or significant changes
            control_indices = self._select_significant_points(data_points, max_control_points)
            control_points = data_points[control_indices]
            
            # Estimate scaling factors based on local variations
            scaling_factors = self._estimate_scaling_factors(data_points, control_indices)
        
        return FractalInterpolationFunction(
            control_points=control_points,
            vertical_scaling_factors=scaling_factors,
            interpolation_method=self.interpolation_method
        )
    
    def _select_significant_points(self, data_points: np.ndarray, max_points: int) -> List[int]:
        """Select significant points based on curvature and variation."""
        if len(data_points) <= max_points:
            return list(range(len(data_points)))
        
        # Always include first and last points
        selected = [0, len(data_points) - 1]
        
        # Calculate curvature approximation
        if len(data_points) >= 3:
            curvature = np.zeros(len(data_points))
            for i in range(1, len(data_points) - 1):
                # Simple curvature approximation using second differences
                y_prev, y_curr, y_next = data_points[i-1:i+2, 1]
                curvature[i] = abs(y_next - 2*y_curr + y_prev)
            
            # Select points with highest curvature
            remaining_points = max_points - 2
            if remaining_points > 0:
                curvature_indices = np.argsort(curvature)[-remaining_points:]
                selected.extend(curvature_indices)
        
        # Sort and return unique indices
        return sorted(list(set(selected)))
    
    def _estimate_scaling_factors(self, data_points: np.ndarray, control_indices: List[int]) -> np.ndarray:
        """Estimate vertical scaling factors for FIF intervals."""
        scaling_factors = []
        
        for i in range(len(control_indices) - 1):
            start_idx = control_indices[i]
            end_idx = control_indices[i + 1]
            
            if end_idx - start_idx > 1:
                # Calculate variation in the interval
                interval_data = data_points[start_idx:end_idx+1, 1]
                variation = np.std(interval_data)
                
                # Map variation to scaling factor (ensuring contractivity)
                max_variation = np.std(data_points[:, 1])
                if max_variation > 0:
                    scaling = min(0.9, variation / max_variation)
                else:
                    scaling = 0.5
            else:
                scaling = 0.5
            
            scaling_factors.append(scaling)
        
        return np.array(scaling_factors)