#!/usr/bin/env python3
"""
Google Colab Setup Script for Fractal Time Series Compression

Run this script in Google Colab to install and test the fractal compression package.

Usage in Colab:
    !pip install git+https://github.com/ParkerWilliams/fractal-time-series-compression
    
    # Then run this setup script
    exec(open('colab_setup.py').read())
"""

import sys
import subprocess
import importlib

def install_and_import(package_name, import_name=None):
    """Install a package and import it."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} already installed")
    except ImportError:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")

def setup_colab_environment():
    """Set up the Google Colab environment for fractal compression."""
    
    print("🚀 Setting up Fractal Time Series Compression in Google Colab")
    print("=" * 60)
    
    # Install required packages if not already installed
    required_packages = [
        ("numpy>=1.24.0", "numpy"),
        ("scipy>=1.10.0", "scipy"), 
        ("matplotlib>=3.7.0", "matplotlib"),
        ("pandas>=2.0.0", "pandas"),
        ("scikit-learn>=1.3.0", "sklearn"),
        ("seaborn>=0.12.0", "seaborn"),
        ("jupyter", "jupyter"),
    ]
    
    for package, import_name in required_packages:
        install_and_import(package, import_name)
    
    print("\n📊 Testing fractal compression import...")
    
    try:
        # Test imports
        from src.compression.ifs_compression import IFSCompressor
        from src.compression.fractal_coding import FractalCodingCompressor
        from src.decompression.fractal_interpolation import FractalInterpolationDecompressor
        from src.data.generator import TimeSeriesGenerator
        from src.utils.metrics import CompressionMetrics
        from src.utils.plotting import CompressionVisualizer
        
        print("✅ All fractal compression modules imported successfully!")
        
        # Quick test
        print("\n🧪 Running quick test...")
        
        # Generate test data
        generator = TimeSeriesGenerator()
        t, y = generator.sine_wave(n_points=100, frequency=1.0)
        
        # Test IFS compression
        compressor = IFSCompressor(n_transformations=2, max_iterations=10)
        result = compressor.compress(t, y)
        decompressed = compressor.decompress(result)
        
        # Calculate metrics
        metrics = CompressionMetrics()
        correlation = metrics.pearson_correlation(y, decompressed.reconstructed_data)
        
        print(f"✅ Test completed successfully!")
        print(f"   Compression ratio: {result.compression_ratio:.2f}x")
        print(f"   Reconstruction correlation: {correlation:.4f}")
        
        print("\n🎉 Setup complete! You can now use fractal compression in Colab!")
        print("\nExample usage:")
        print("```python")
        print("from src.data.generator import TimeSeriesGenerator")
        print("from src.compression.ifs_compression import IFSCompressor")
        print("")
        print("# Generate data")
        print("generator = TimeSeriesGenerator()")
        print("t, y = generator.multi_component_series(n_points=500)")
        print("")
        print("# Compress")
        print("compressor = IFSCompressor()")
        print("result = compressor.compress(t, y)")
        print("print(f'Compression ratio: {result.compression_ratio:.2f}x')")
        print("```")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you installed with: !pip install git+https://github.com/ParkerWilliams/fractal-time-series-compression")
        print("2. Try restarting the Colab runtime")
        print("3. Check that all dependencies are installed")

if __name__ == "__main__":
    setup_colab_environment()