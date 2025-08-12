# Fractal Time Series Compression

A proof-of-concept implementation of fractal-based compression methods for time series data, featuring three distinct approaches:

1. **Iterated Function Systems (IFS)** - Uses contractive transformations to represent time series as attractors
2. **Fractal Coding** - Block-based compression exploiting self-similarity in the data  
3. **Fractal Interpolation** - Decompression using Fractal Interpolation Functions (FIFs)

## Overview

This project demonstrates the application of fractal mathematics to time series compression, exploring how self-similarity and scale invariance properties in temporal data can be exploited for efficient storage and reconstruction.

### Key Features

- **Multiple Compression Methods**: Three different fractal-based approaches
- **Comprehensive Evaluation**: Extensive metrics for compression ratio and reconstruction quality
- **Fractal Analysis**: Preservation of fractal properties (Hurst exponent, fractal dimension)
- **Flexible Data Generation**: Support for various synthetic time series types
- **Visualization Tools**: Rich plotting capabilities for analysis and comparison
- **Modular Design**: Clean, extensible architecture with abstract base classes

## Installation

### Prerequisites

- Python 3.7+
- Required packages (see `requirements.txt`)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd fractal-time-series-compression
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.data.generator import TimeSeriesGenerator
from src.compression.ifs_compression import IFSCompressor
from src.utils.metrics import CompressionMetrics

# Generate test data
time_data, value_data = TimeSeriesGenerator.sine_wave(n_points=1000, frequency=2.0)

# Compress using IFS
compressor = IFSCompressor(n_transformations=4)
result = compressor.compress(time_data, value_data)

# Decompress
decompressed = compressor.decompress(result)

# Evaluate quality
correlation = CompressionMetrics.pearson_correlation(
    value_data, decompressed.reconstructed_data
)
print(f"Compression ratio: {result.compression_ratio:.2f}")
print(f"Correlation: {correlation:.4f}")
```

### Running the Demo

The comprehensive demo script showcases all compression methods:

```bash
cd src/examples
python demo.py --data-type multi_sine --length 1000 --save-plots --output-dir results
```

#### Demo Options

- `--data-type`: Choose from `sine`, `multi_sine`, `fractal_brownian`, `random_walk`, `stock`, `sensor`
- `--length`: Number of data points (default: 1000)
- `--save-plots`: Save visualization plots
- `--output-dir`: Output directory for results (default: "results")
- `--sensitivity`: Run parameter sensitivity analysis

### Examples

#### Different Data Types

```python
# Fractal Brownian motion with specific Hurst exponent
t, y = TimeSeriesGenerator.fractal_brownian_motion(n_points=500, hurst=0.7)

# Multi-component sine wave
components = [
    {'type': 'sine', 'frequency': 1.0, 'amplitude': 1.0},
    {'type': 'sine', 'frequency': 3.0, 'amplitude': 0.5},
    {'type': 'sine', 'frequency': 7.0, 'amplitude': 0.3}
]
t, y = TimeSeriesGenerator.multi_component_series(n_points=1000, components=components)

# Stock price simulation
t, y = TimeSeriesGenerator.stock_price_simulation(
    n_points=252, initial_price=100.0, volatility=0.2
)
```

#### Compression Method Comparison

```python
from src.compression.fractal_coding import FractalCodingCompressor
from src.decompression.fractal_interpolation import FractalInterpolationDecompressor

# Compare different methods
compressors = {
    'IFS': IFSCompressor(n_transformations=4),
    'Fractal_Coding': FractalCodingCompressor(range_block_size=8)
}

results = {}
for name, compressor in compressors.items():
    comp_result = compressor.compress(time_data, value_data)
    decomp_result = compressor.decompress(comp_result)
    
    metrics = CompressionMetrics.comprehensive_evaluation(
        value_data, decomp_result.reconstructed_data,
        comp_result.original_size, comp_result.compressed_size,
        comp_result.compression_time, decomp_result.decompression_time
    )
    
    results[name] = metrics
    print(f"{name}: Ratio={metrics['compression_ratio']:.2f}, "
          f"Correlation={metrics['pearson_correlation']:.4f}")
```

## Architecture

### Project Structure

```
fractal-time-series-compression/
├── src/
│   ├── data/                    # Data generation and loading
│   │   ├── generator.py         # Synthetic time series generation
│   │   └── loader.py            # Data loading and preprocessing
│   ├── compression/             # Compression methods
│   │   ├── base_compressor.py   # Abstract base class
│   │   ├── ifs_compression.py   # IFS compression
│   │   └── fractal_coding.py    # Fractal coding compression
│   ├── decompression/           # Decompression methods
│   │   └── fractal_interpolation.py  # FIF decompression
│   ├── utils/                   # Utilities
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── plotting.py          # Visualization tools
│   └── examples/
│       └── demo.py              # Demonstration script
├── tests/
│   └── test_compression.py      # Unit tests
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

### Compression Methods

#### 1. Iterated Function Systems (IFS)

IFS compression represents time series as attractors of sets of contractive affine transformations:

```
T_i(x,y) = [a_i b_i; c_i d_i] * [x; y] + [e_i; f_i]
```

**Key Features:**
- Global optimization using differential evolution
- Contractivity constraints ensure convergence
- Chaos game algorithm for attractor generation
- Excellent for self-similar signals

**Parameters:**
- `n_transformations`: Number of transformations (default: 4)
- `max_iterations`: Optimization iterations (default: 1000)
- `contractivity_bound`: Maximum eigenvalue bound (default: 0.9)

#### 2. Fractal Coding

Block-based compression exploiting self-similarity through domain-range block matching:

**Key Features:**
- Partitions signal into range and domain blocks
- Finds optimal affine transformations between blocks
- Iterative reconstruction process
- Good for signals with local self-similarity

**Parameters:**
- `range_block_size`: Size of range blocks (default: 8)
- `domain_block_size`: Size of domain blocks (default: 16)
- `overlap_factor`: Domain block overlap (default: 0.5)
- `max_scaling`: Maximum scaling factor (default: 0.9)

#### 3. Fractal Interpolation

Decompression using Fractal Interpolation Functions with vertical scaling:

**Key Features:**
- Constructs FIFs from compressed parameters
- Adaptive control point selection
- Compatible with both IFS and fractal coding
- Preserves fractal properties during reconstruction

### Evaluation Metrics

The framework provides comprehensive evaluation including:

#### Compression Efficiency
- Compression ratio (original_size / compressed_size)
- Space savings percentage
- Compression and decompression time

#### Reconstruction Quality
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Normalized RMSE
- Pearson correlation coefficient
- Structural Similarity Index (SSIM)

#### Signal Quality
- Signal-to-Noise Ratio (SNR)
- Peak Signal-to-Noise Ratio (PSNR)

#### Frequency Domain Analysis
- Magnitude spectrum error
- Phase spectrum error
- Spectral correlation

#### Fractal Properties
- Fractal dimension preservation (box counting)
- Hurst exponent preservation (R/S analysis)

## Testing

Run the comprehensive test suite:

```bash
python tests/test_compression.py
```

The tests cover:
- Data generation and preprocessing
- Individual compression method functionality
- Metric calculations
- Integration scenarios
- Error handling and edge cases

## Performance Characteristics

### Typical Results

Based on testing with various signal types:

| Method | Compression Ratio | Correlation | Best Use Case |
|--------|------------------|-------------|---------------|
| IFS | 5-20x | 0.7-0.9 | Self-similar, smooth signals |
| Fractal Coding | 3-10x | 0.5-0.8 | Locally self-similar signals |
| Combined w/ FIF | 5-15x | 0.6-0.85 | Mixed signal types |

### Computational Complexity

- **IFS**: O(n × m × i) where n=data length, m=transformations, i=iterations
- **Fractal Coding**: O(n × b²) where b=block size
- **FIF Decompression**: O(n × k) where k=iterations

## Limitations and Considerations

1. **Compression Ratio**: Generally lower than specialized algorithms (GZIP, LZ77)
2. **Computational Cost**: Higher than traditional methods due to optimization
3. **Signal Suitability**: Best for signals with fractal or self-similar properties
4. **Parameter Sensitivity**: Performance depends on proper parameter tuning
5. **Reconstruction Quality**: Trade-off between compression ratio and fidelity

## Use Cases

This implementation is particularly suitable for:

- **Research and Education**: Understanding fractal compression principles
- **Signal Analysis**: Studying self-similarity in time series
- **Proof of Concept**: Demonstrating fractal-based compression feasibility
- **Benchmarking**: Comparing with traditional compression methods
- **Specialized Applications**: Signals with known fractal properties

## Contributing

Contributions are welcome! Areas for improvement:

1. **Optimization**: Faster algorithms for parameter estimation
2. **Methods**: Additional fractal compression techniques
3. **Evaluation**: More comprehensive quality metrics
4. **Applications**: Real-world dataset testing
5. **Performance**: Algorithmic improvements and parallelization

## Technical References

1. **IFS Theory**: Barnsley, M. F. "Fractals Everywhere" (1988)
2. **Fractal Coding**: Jacquin, A. E. "Image coding based on a fractal theory of iterated contractive image transformations" (1992)
3. **Fractal Interpolation**: Barnsley, M. F. "Fractal functions and interpolation" (1986)
4. **Time Series Analysis**: Box, G. E. P. "Time Series Analysis: Forecasting and Control" (2015)

## License

This project is provided for educational and research purposes. Please cite appropriately if used in academic work.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fractal_time_series_compression,
  title={Fractal Time Series Compression: A Proof of Concept},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[username]/fractal-time-series-compression}
}
```

---

For questions, issues, or contributions, please use the GitHub issue tracker.