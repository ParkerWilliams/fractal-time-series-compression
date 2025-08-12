#!/usr/bin/env python3
"""
Quick demonstration of fractal time series compression.
This is a simplified version for quick testing.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.generator import TimeSeriesGenerator
from compression.ifs_compression import IFSCompressor
from compression.fractal_coding import FractalCodingCompressor
from utils.metrics import CompressionMetrics


def quick_demo():
    """Run a quick demonstration of fractal compression."""
    print("🌀 Fractal Time Series Compression - Quick Demo")
    print("=" * 50)
    
    # Generate test data
    print("\n📊 Generating test data...")
    time_data, value_data = TimeSeriesGenerator.multi_component_series(
        n_points=500,
        components=[
            {'type': 'sine', 'frequency': 1.0, 'amplitude': 1.0},
            {'type': 'sine', 'frequency': 3.0, 'amplitude': 0.5},
            {'type': 'sine', 'frequency': 7.0, 'amplitude': 0.3}
        ]
    )
    
    # Normalize data
    value_data = (value_data - np.min(value_data)) / (np.max(value_data) - np.min(value_data))
    
    print(f"   Data length: {len(value_data)} points")
    print(f"   Data range: [{np.min(value_data):.3f}, {np.max(value_data):.3f}]")
    
    # Test IFS compression
    print("\n🔄 Testing IFS Compression...")
    ifs_compressor = IFSCompressor(n_transformations=3, max_iterations=50)
    
    try:
        ifs_result = ifs_compressor.compress(time_data, value_data)
        ifs_decompressed = ifs_compressor.decompress(ifs_result)
        
        ifs_correlation = CompressionMetrics.pearson_correlation(
            value_data, ifs_decompressed.reconstructed_data
        )
        
        print(f"   ✅ Compression ratio: {ifs_result.compression_ratio:.2f}x")
        print(f"   ✅ Correlation: {ifs_correlation:.4f}")
        print(f"   ⏱️  Time: {ifs_result.compression_time:.3f}s compression + {ifs_decompressed.decompression_time:.3f}s decompression")
        
    except Exception as e:
        print(f"   ❌ IFS failed: {e}")
        ifs_result = None
        ifs_decompressed = None
    
    # Test Fractal Coding compression
    print("\n🔄 Testing Fractal Coding Compression...")
    fc_compressor = FractalCodingCompressor(range_block_size=8, domain_block_size=16)
    
    try:
        fc_result = fc_compressor.compress(time_data, value_data)
        fc_decompressed = fc_compressor.decompress(fc_result)
        
        fc_correlation = CompressionMetrics.pearson_correlation(
            value_data, fc_decompressed.reconstructed_data
        )
        
        print(f"   ✅ Compression ratio: {fc_result.compression_ratio:.2f}x")
        print(f"   ✅ Correlation: {fc_correlation:.4f}")
        print(f"   ⏱️  Time: {fc_result.compression_time:.3f}s compression + {fc_decompressed.decompression_time:.3f}s decompression")
        
    except Exception as e:
        print(f"   ❌ Fractal Coding failed: {e}")
        fc_result = None
        fc_decompressed = None
    
    # Create a simple plot
    print("\n📈 Creating visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original data
    axes[0].plot(time_data, value_data, 'b-', label='Original', linewidth=2)
    axes[0].set_title('Original Time Series')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot reconstructions
    axes[1].plot(time_data, value_data, 'b-', label='Original', alpha=0.7, linewidth=2)
    
    if ifs_decompressed is not None:
        axes[1].plot(time_data, ifs_decompressed.reconstructed_data, 'r--', 
                    label=f'IFS (r={ifs_correlation:.3f})', linewidth=2)
    
    if fc_decompressed is not None:
        axes[1].plot(time_data, fc_decompressed.reconstructed_data, 'g:', 
                    label=f'Fractal Coding (r={fc_correlation:.3f})', linewidth=2)
    
    axes[1].set_title('Compression Results Comparison')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    # Try to save plot
    try:
        os.makedirs('quick_demo_results', exist_ok=True)
        plt.savefig('quick_demo_results/compression_comparison.png', dpi=150, bbox_inches='tight')
        print("   ✅ Plot saved to 'quick_demo_results/compression_comparison.png'")
    except Exception as e:
        print(f"   ⚠️  Could not save plot: {e}")
    
    # Show plot if possible
    try:
        plt.show()
    except Exception:
        print("   ℹ️  Plot display not available (no GUI)")
    
    plt.close()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    if ifs_result and fc_result:
        print(f"{'Method':<15} {'Ratio':<8} {'Quality':<8} {'Time(s)':<8}")
        print("-" * 45)
        print(f"{'IFS':<15} {ifs_result.compression_ratio:<8.2f} {ifs_correlation:<8.4f} {ifs_result.compression_time + ifs_decompressed.decompression_time:<8.3f}")
        print(f"{'Fractal Coding':<15} {fc_result.compression_ratio:<8.2f} {fc_correlation:<8.4f} {fc_result.compression_time + fc_decompressed.decompression_time:<8.3f}")
        
        # Determine winner
        if ifs_correlation > fc_correlation:
            print(f"\n🏆 Best quality: IFS (correlation = {ifs_correlation:.4f})")
        else:
            print(f"\n🏆 Best quality: Fractal Coding (correlation = {fc_correlation:.4f})")
            
        if ifs_result.compression_ratio > fc_result.compression_ratio:
            print(f"🏆 Best compression: IFS ({ifs_result.compression_ratio:.2f}x)")
        else:
            print(f"🏆 Best compression: Fractal Coding ({fc_result.compression_ratio:.2f}x)")
    
    print("\n✨ Quick demo completed!")
    print("\nFor more detailed analysis, run:")
    print("   python3 demo.py --save-plots --sensitivity")


if __name__ == "__main__":
    quick_demo()