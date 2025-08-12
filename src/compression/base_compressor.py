from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, Optional
import pickle
import time


class CompressionResult:
    """Container for compression results and metadata."""
    
    def __init__(self, compressed_data: Any, original_size: int, 
                 compressed_size: int, compression_time: float,
                 method_name: str, parameters: Dict[str, Any] = None):
        self.compressed_data = compressed_data
        self.original_size = original_size
        self.compressed_size = compressed_size
        self.compression_time = compression_time
        self.method_name = method_name
        self.parameters = parameters or {}
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (original_size / compressed_size)."""
        return self.original_size / self.compressed_size if self.compressed_size > 0 else float('inf')
    
    @property
    def space_savings(self) -> float:
        """Calculate space savings as percentage."""
        return (1 - self.compressed_size / self.original_size) * 100 if self.original_size > 0 else 0
    
    def __str__(self) -> str:
        return (f"CompressionResult(method={self.method_name}, "
                f"ratio={self.compression_ratio:.2f}, "
                f"savings={self.space_savings:.1f}%, "
                f"time={self.compression_time:.3f}s)")


class DecompressionResult:
    """Container for decompression results and metadata."""
    
    def __init__(self, reconstructed_data: np.ndarray, decompression_time: float,
                 method_name: str, quality_metrics: Dict[str, float] = None):
        self.reconstructed_data = reconstructed_data
        self.decompression_time = decompression_time
        self.method_name = method_name
        self.quality_metrics = quality_metrics or {}
    
    def __str__(self) -> str:
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in self.quality_metrics.items()])
        return (f"DecompressionResult(method={self.method_name}, "
                f"time={self.decompression_time:.3f}s, "
                f"metrics=[{metrics_str}])")


class BaseCompressor(ABC):
    """Abstract base class for fractal time series compression methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.fitted_parameters = {}
    
    @abstractmethod
    def compress(self, time_data: np.ndarray, value_data: np.ndarray, 
                **kwargs) -> CompressionResult:
        """
        Compress time series data.
        
        Args:
            time_data: Array of time points
            value_data: Array of time series values
            **kwargs: Method-specific parameters
            
        Returns:
            CompressionResult object with compressed data and metadata
        """
        pass
    
    @abstractmethod
    def decompress(self, compression_result: CompressionResult, 
                  target_length: Optional[int] = None, **kwargs) -> DecompressionResult:
        """
        Decompress time series data.
        
        Args:
            compression_result: Result from compression
            target_length: Desired length of reconstructed series (if different from original)
            **kwargs: Method-specific parameters
            
        Returns:
            DecompressionResult object with reconstructed data and metadata
        """
        pass
    
    def _calculate_data_size(self, data: Any) -> int:
        """Calculate size of data in bytes using pickle serialization."""
        return len(pickle.dumps(data))
    
    def _validate_input_data(self, time_data: np.ndarray, value_data: np.ndarray) -> None:
        """Validate input data format and consistency."""
        if not isinstance(time_data, np.ndarray) or not isinstance(value_data, np.ndarray):
            raise TypeError("Input data must be numpy arrays")
        
        if len(time_data) != len(value_data):
            raise ValueError("Time and value arrays must have the same length")
        
        if len(time_data) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if np.any(np.isnan(value_data)) or np.any(np.isinf(value_data)):
            raise ValueError("Value data contains NaN or infinite values")
    
    def _time_operation(self, operation):
        """Context manager for timing operations."""
        class TimingContext:
            def __init__(self):
                self.start_time = None
                self.elapsed_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.elapsed_time = time.time() - self.start_time
        
        return TimingContext()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current compression parameters."""
        return self.fitted_parameters.copy()
    
    def set_parameters(self, **params) -> None:
        """Set compression parameters."""
        self.fitted_parameters.update(params)
    
    def save_model(self, filepath: str) -> None:
        """Save fitted compression model to file."""
        model_data = {
            'name': self.name,
            'parameters': self.fitted_parameters,
            'class_name': self.__class__.__name__
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load fitted compression model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance and restore parameters
        instance = cls(model_data['name'])
        instance.fitted_parameters = model_data['parameters']
        return instance