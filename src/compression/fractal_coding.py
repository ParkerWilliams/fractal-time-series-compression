import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.optimize import minimize_scalar
from .base_compressor import BaseCompressor, CompressionResult, DecompressionResult


class FractalBlock:
    """Represents a block in fractal coding with its transformation parameters."""
    
    def __init__(self, domain_start: int, domain_end: int, range_start: int, range_end: int,
                 scaling_factor: float, offset: float, reflection: bool = False):
        self.domain_start = domain_start
        self.domain_end = domain_end
        self.range_start = range_start
        self.range_end = range_end
        self.scaling_factor = scaling_factor
        self.offset = offset
        self.reflection = reflection
    
    def domain_size(self) -> int:
        return self.domain_end - self.domain_start
    
    def range_size(self) -> int:
        return self.range_end - self.range_start
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'domain_start': self.domain_start,
            'domain_end': self.domain_end,
            'range_start': self.range_start,
            'range_end': self.range_end,
            'scaling_factor': self.scaling_factor,
            'offset': self.offset,
            'reflection': self.reflection
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)


class FractalCodingCompressor(BaseCompressor):
    """Fractal coding compression for time series data."""
    
    def __init__(self, range_block_size: int = 8, domain_block_size: int = 16,
                 overlap_factor: float = 0.5, max_scaling: float = 0.9,
                 search_tolerance: float = 1e-6):
        super().__init__("Fractal_Coding_Compressor")
        self.range_block_size = range_block_size
        self.domain_block_size = domain_block_size
        self.overlap_factor = overlap_factor
        self.max_scaling = max_scaling
        self.search_tolerance = search_tolerance
    
    def _partition_into_blocks(self, data: np.ndarray, block_size: int, 
                              overlap_factor: float = 0.0) -> List[Tuple[int, int, np.ndarray]]:
        """Partition data into overlapping blocks."""
        blocks = []
        step_size = max(1, int(block_size * (1 - overlap_factor)))
        
        for start in range(0, len(data) - block_size + 1, step_size):
            end = start + block_size
            block_data = data[start:end]
            blocks.append((start, end, block_data))
        
        return blocks
    
    def _downsample_block(self, block: np.ndarray, target_size: int) -> np.ndarray:
        """Downsample a block to target size using linear interpolation."""
        if len(block) == target_size:
            return block
        
        # Create indices for interpolation
        old_indices = np.linspace(0, len(block) - 1, len(block))
        new_indices = np.linspace(0, len(block) - 1, target_size)
        
        # Interpolate
        downsampled = np.interp(new_indices, old_indices, block)
        return downsampled
    
    def _find_best_domain_match(self, range_block: np.ndarray, 
                               domain_blocks: List[Tuple[int, int, np.ndarray]]) -> FractalBlock:
        """Find the best matching domain block for a given range block."""
        best_error = float('inf')
        best_match = None
        
        for domain_start, domain_end, domain_block in domain_blocks:
            # Downsample domain block to match range block size
            downsampled_domain = self._downsample_block(domain_block, len(range_block))
            
            # Try both normal and reflected domain blocks
            for reflection in [False, True]:
                test_domain = downsampled_domain[::-1] if reflection else downsampled_domain
                
                # Find optimal scaling factor and offset using least squares
                def error_function(params):
                    scaling, offset = params
                    if abs(scaling) > self.max_scaling:
                        return 1e6
                    
                    transformed = scaling * test_domain + offset
                    return np.mean((range_block - transformed) ** 2)
                
                # Optimize scaling and offset
                try:
                    # Initial guess for scaling and offset
                    scaling_guess = np.std(range_block) / np.std(test_domain) if np.std(test_domain) > 0 else 0.5
                    scaling_guess = np.clip(scaling_guess, -self.max_scaling, self.max_scaling)
                    offset_guess = np.mean(range_block) - scaling_guess * np.mean(test_domain)
                    
                    # Simple grid search for optimization (more robust than scipy.optimize for this case)
                    best_params = (scaling_guess, offset_guess)
                    best_local_error = error_function(best_params)
                    
                    # Search around the initial guess
                    for scaling in np.linspace(-self.max_scaling, self.max_scaling, 20):
                        for offset in np.linspace(offset_guess - 2, offset_guess + 2, 20):
                            error = error_function((scaling, offset))
                            if error < best_local_error:
                                best_local_error = error
                                best_params = (scaling, offset)
                    
                    if best_local_error < best_error:
                        best_error = best_local_error
                        scaling, offset = best_params
                        best_match = FractalBlock(
                            domain_start=domain_start,
                            domain_end=domain_end,
                            range_start=0,  # Will be set by caller
                            range_end=0,    # Will be set by caller
                            scaling_factor=scaling,
                            offset=offset,
                            reflection=reflection
                        )
                
                except Exception:
                    continue
        
        return best_match
    
    def _encode_fractal_blocks(self, data: np.ndarray) -> List[FractalBlock]:
        """Encode time series data using fractal coding."""
        # Partition data into range and domain blocks
        range_blocks = self._partition_into_blocks(data, self.range_block_size, 0)
        domain_blocks = self._partition_into_blocks(data, self.domain_block_size, self.overlap_factor)
        
        fractal_blocks = []
        
        for range_start, range_end, range_block in range_blocks:
            # Find best matching domain block
            best_match = self._find_best_domain_match(range_block, domain_blocks)
            
            if best_match is not None:
                # Update range block indices
                best_match.range_start = range_start
                best_match.range_end = range_end
                fractal_blocks.append(best_match)
            else:
                # Fallback: create a simple block with zero scaling
                fallback_block = FractalBlock(
                    domain_start=0,
                    domain_end=min(self.domain_block_size, len(data)),
                    range_start=range_start,
                    range_end=range_end,
                    scaling_factor=0.0,
                    offset=np.mean(range_block),
                    reflection=False
                )
                fractal_blocks.append(fallback_block)
        
        return fractal_blocks
    
    def _decode_fractal_blocks(self, fractal_blocks: List[FractalBlock], 
                              original_data: np.ndarray, iterations: int = 10) -> np.ndarray:
        """Decode fractal blocks to reconstruct time series."""
        # Initialize with original data (or zeros)
        reconstructed = original_data.copy()
        
        # Iteratively apply fractal transformations
        for iteration in range(iterations):
            new_reconstructed = reconstructed.copy()
            
            for block in fractal_blocks:
                # Extract domain block
                domain_data = reconstructed[block.domain_start:block.domain_end]
                
                # Downsample to range block size
                range_size = block.range_end - block.range_start
                domain_downsampled = self._downsample_block(domain_data, range_size)
                
                # Apply reflection if needed
                if block.reflection:
                    domain_downsampled = domain_downsampled[::-1]
                
                # Apply affine transformation
                transformed = block.scaling_factor * domain_downsampled + block.offset
                
                # Update range block
                new_reconstructed[block.range_start:block.range_end] = transformed
            
            # Check for convergence
            if np.mean(np.abs(new_reconstructed - reconstructed)) < self.search_tolerance:
                break
                
            reconstructed = new_reconstructed
        
        return reconstructed
    
    def compress(self, time_data: np.ndarray, value_data: np.ndarray, **kwargs) -> CompressionResult:
        """Compress time series using fractal coding."""
        self._validate_input_data(time_data, value_data)
        
        with self._time_operation() as timer:
            # Encode using fractal blocks
            fractal_blocks = self._encode_fractal_blocks(value_data)
            
            # Prepare compression data
            compression_data = {
                'fractal_blocks': [block.to_dict() for block in fractal_blocks],
                'original_length': len(value_data),
                'time_range': (np.min(time_data), np.max(time_data)),
                'value_statistics': {
                    'mean': np.mean(value_data),
                    'std': np.std(value_data),
                    'min': np.min(value_data),
                    'max': np.max(value_data)
                }
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
            parameters={
                'range_block_size': self.range_block_size,
                'domain_block_size': self.domain_block_size,
                'overlap_factor': self.overlap_factor
            }
        )
    
    def decompress(self, compression_result: CompressionResult, 
                  target_length: Optional[int] = None, **kwargs) -> DecompressionResult:
        """Decompress time series from fractal coding."""
        if compression_result.method_name != self.name:
            raise ValueError(f"Compression result is not from {self.name}")
        
        with self._time_operation() as timer:
            data = compression_result.compressed_data
            
            # Reconstruct fractal blocks
            fractal_blocks = [FractalBlock.from_dict(block_data) 
                             for block_data in data['fractal_blocks']]
            
            # Determine target length
            if target_length is None:
                target_length = data['original_length']
            
            # Initialize reconstruction with noise or mean value
            stats = data['value_statistics']
            initial_data = np.random.normal(stats['mean'], stats['std'] * 0.1, target_length)
            
            # Decode fractal blocks
            reconstructed = self._decode_fractal_blocks(fractal_blocks, initial_data)
            
            # Trim or pad to target length
            if len(reconstructed) > target_length:
                reconstructed = reconstructed[:target_length]
            elif len(reconstructed) < target_length:
                # Pad with last value
                padding = np.full(target_length - len(reconstructed), reconstructed[-1])
                reconstructed = np.concatenate([reconstructed, padding])
        
        return DecompressionResult(
            reconstructed_data=reconstructed,
            decompression_time=timer.elapsed_time,
            method_name=self.name
        )