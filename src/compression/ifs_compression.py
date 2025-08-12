import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.optimize import differential_evolution, minimize
from .base_compressor import BaseCompressor, CompressionResult, DecompressionResult


class IFSTransformation:
    """Represents a single affine transformation in an IFS."""
    
    def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float):
        """
        Affine transformation: [x', y'] = [a b; c d] * [x; y] + [e; f]
        
        Args:
            a, b, c, d: Linear transformation matrix elements
            e, f: Translation vector elements
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
    
    def apply(self, x: float, y: float) -> Tuple[float, float]:
        """Apply transformation to a point."""
        x_new = self.a * x + self.b * y + self.e
        y_new = self.c * x + self.d * y + self.f
        return x_new, y_new
    
    def apply_batch(self, points: np.ndarray) -> np.ndarray:
        """Apply transformation to array of points."""
        if points.shape[1] != 2:
            raise ValueError("Points array must have shape (n, 2)")
        
        transform_matrix = np.array([[self.a, self.b], [self.c, self.d]])
        translation = np.array([self.e, self.f])
        
        return points @ transform_matrix.T + translation
    
    @property
    def contractivity_factor(self) -> float:
        """Calculate the contractivity factor (largest eigenvalue)."""
        matrix = np.array([[self.a, self.b], [self.c, self.d]])
        eigenvals = np.linalg.eigvals(matrix)
        return np.max(np.abs(eigenvals))
    
    def to_array(self) -> np.ndarray:
        """Convert transformation to parameter array."""
        return np.array([self.a, self.b, self.c, self.d, self.e, self.f])
    
    @classmethod
    def from_array(cls, params: np.ndarray):
        """Create transformation from parameter array."""
        return cls(*params)


class IFSCompressor(BaseCompressor):
    """Iterated Function Systems compression for time series data."""
    
    def __init__(self, n_transformations: int = 4, max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6, contractivity_bound: float = 0.9):
        super().__init__("IFS_Compressor")
        self.n_transformations = n_transformations
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.contractivity_bound = contractivity_bound
    
    def _time_series_to_points(self, time_data: np.ndarray, value_data: np.ndarray) -> np.ndarray:
        """Convert time series to 2D point cloud."""
        # Normalize both time and value data to [0, 1]
        time_norm = (time_data - np.min(time_data)) / (np.max(time_data) - np.min(time_data))
        value_norm = (value_data - np.min(value_data)) / (np.max(value_data) - np.min(value_data))
        
        return np.column_stack([time_norm, value_norm])
    
    def _points_to_time_series(self, points: np.ndarray, original_time: np.ndarray,
                              original_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 2D point cloud back to time series."""
        # Denormalize
        time_min, time_max = np.min(original_time), np.max(original_time)
        value_min, value_max = np.min(original_values), np.max(original_values)
        
        time_denorm = points[:, 0] * (time_max - time_min) + time_min
        value_denorm = points[:, 1] * (value_max - value_min) + value_min
        
        return time_denorm, value_denorm
    
    def _evaluate_ifs_fitness(self, params: np.ndarray, target_points: np.ndarray) -> float:
        """Evaluate fitness of IFS parameters against target point cloud."""
        n_params_per_transform = 6
        transformations = []
        
        # Extract transformations from parameter vector
        for i in range(self.n_transformations):
            start_idx = i * n_params_per_transform
            end_idx = start_idx + n_params_per_transform
            transform_params = params[start_idx:end_idx]
            
            # Ensure contractivity
            a, b, c, d = transform_params[:4]
            matrix = np.array([[a, b], [c, d]])
            eigenvals = np.linalg.eigvals(matrix)
            max_eigenval = np.max(np.abs(eigenvals))
            
            if max_eigenval >= self.contractivity_bound:
                return 1e6  # Penalty for non-contractive transformations
            
            transformations.append(IFSTransformation.from_array(transform_params))
        
        # Generate attractor using chaos game
        try:
            generated_points = self._generate_attractor(transformations, len(target_points))
            
            # Calculate Hausdorff-like distance between generated and target points
            distance = self._point_cloud_distance(generated_points, target_points)
            return distance
            
        except Exception:
            return 1e6  # Penalty for invalid transformations
    
    def _generate_attractor(self, transformations: List[IFSTransformation], 
                           n_points: int = 10000) -> np.ndarray:
        """Generate attractor points using the chaos game algorithm."""
        if not transformations:
            raise ValueError("At least one transformation is required")
        
        points = []
        # Start with a random point in [0, 1] x [0, 1]
        current_point = np.random.rand(2)
        
        # Skip initial transient iterations
        for _ in range(100):
            transform = np.random.choice(transformations)
            current_point = transform.apply_batch(current_point.reshape(1, -1))[0]
        
        # Collect attractor points
        for _ in range(n_points):
            transform = np.random.choice(transformations)
            current_point = transform.apply_batch(current_point.reshape(1, -1))[0]
            points.append(current_point.copy())
        
        return np.array(points)
    
    def _point_cloud_distance(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """Calculate distance between two point clouds."""
        # Use a simplified version of Hausdorff distance
        # For each point in points1, find closest point in points2
        distances1 = []
        for p1 in points1:
            dists = np.linalg.norm(points2 - p1, axis=1)
            distances1.append(np.min(dists))
        
        # For each point in points2, find closest point in points1
        distances2 = []
        for p2 in points2:
            dists = np.linalg.norm(points1 - p2, axis=1)
            distances2.append(np.min(dists))
        
        # Return maximum of average distances
        return max(np.mean(distances1), np.mean(distances2))
    
    def _optimize_ifs_parameters(self, target_points: np.ndarray) -> List[IFSTransformation]:
        """Optimize IFS parameters to approximate target point cloud."""
        n_params = self.n_transformations * 6
        
        # Define parameter bounds to ensure contractivity
        bounds = []
        for _ in range(self.n_transformations):
            # Bounds for a, b, c, d (ensure contractivity)
            bounds.extend([(-0.8, 0.8)] * 4)
            # Bounds for e, f (translation)
            bounds.extend([(-0.5, 1.5)] * 2)
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            lambda params: self._evaluate_ifs_fitness(params, target_points),
            bounds,
            maxiter=100,
            popsize=15,
            seed=42
        )
        
        # Extract optimized transformations
        transformations = []
        for i in range(self.n_transformations):
            start_idx = i * 6
            end_idx = start_idx + 6
            transform_params = result.x[start_idx:end_idx]
            transformations.append(IFSTransformation.from_array(transform_params))
        
        return transformations
    
    def compress(self, time_data: np.ndarray, value_data: np.ndarray, **kwargs) -> CompressionResult:
        """Compress time series using IFS approximation."""
        self._validate_input_data(time_data, value_data)
        
        with self._time_operation() as timer:
            # Convert time series to 2D point cloud
            points = self._time_series_to_points(time_data, value_data)
            
            # Optimize IFS parameters
            transformations = self._optimize_ifs_parameters(points)
            
            # Store normalization parameters for decompression
            compression_data = {
                'transformations': [t.to_array() for t in transformations],
                'time_range': (np.min(time_data), np.max(time_data)),
                'value_range': (np.min(value_data), np.max(value_data)),
                'original_length': len(time_data)
            }
        
        # Calculate sizes
        original_size = self._calculate_data_size((time_data, value_data))
        compressed_size = self._calculate_data_size(compression_data)
        
        return CompressionResult(
            compressed_data=compression_data,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_time=timer.elapsed_time,
            method_name=self.name,
            parameters={'n_transformations': self.n_transformations}
        )
    
    def decompress(self, compression_result: CompressionResult, 
                  target_length: Optional[int] = None, **kwargs) -> DecompressionResult:
        """Decompress time series from IFS parameters."""
        if compression_result.method_name != self.name:
            raise ValueError(f"Compression result is not from {self.name}")
        
        with self._time_operation() as timer:
            data = compression_result.compressed_data
            
            # Reconstruct transformations
            transformations = [IFSTransformation.from_array(params) 
                             for params in data['transformations']]
            
            # Determine target length
            if target_length is None:
                target_length = data['original_length']
            
            # Generate attractor points
            generated_points = self._generate_attractor(transformations, target_length)
            
            # Sort points by time coordinate for proper time series reconstruction
            sorted_indices = np.argsort(generated_points[:, 0])
            sorted_points = generated_points[sorted_indices]
            
            # Take first target_length points
            if len(sorted_points) > target_length:
                sorted_points = sorted_points[:target_length]
            
            # Convert back to time series
            time_reconstructed, value_reconstructed = self._points_to_time_series(
                sorted_points, 
                np.linspace(data['time_range'][0], data['time_range'][1], target_length),
                np.array([data['value_range'][0], data['value_range'][1]])
            )
        
        return DecompressionResult(
            reconstructed_data=value_reconstructed,
            decompression_time=timer.elapsed_time,
            method_name=self.name
        )