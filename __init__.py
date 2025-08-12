"""
Fractal Time Series Compression

A Python package for compressing time series data using fractal-based methods:
- Iterated Function Systems (IFS)
- Fractal Coding
- Fractal Interpolation

Author: Parker Williams
Email: parker.williams@gmail.com
Repository: https://github.com/ParkerWilliams/fractal-time-series-compression
"""

__version__ = "1.0.0"
__author__ = "Parker Williams"
__email__ = "parker.williams@gmail.com"

# Main imports for easy access
from src.compression.ifs_compression import IFSCompressor
from src.compression.fractal_coding import FractalCodingCompressor
from src.decompression.fractal_interpolation import FractalInterpolationDecompressor
from src.data.generator import TimeSeriesGenerator
from src.data.loader import TimeSeriesLoader
from src.utils.metrics import CompressionMetrics
from src.utils.plotting import CompressionVisualizer

__all__ = [
    "IFSCompressor",
    "FractalCodingCompressor", 
    "FractalInterpolationDecompressor",
    "TimeSeriesGenerator",
    "TimeSeriesLoader",
    "CompressionMetrics",
    "CompressionVisualizer",
]