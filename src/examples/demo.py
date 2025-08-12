#!/usr/bin/env python3
"""
Fractal Time Series Compression Demo

This script demonstrates the three fractal compression methods:
1. Iterated Function Systems (IFS)
2. Fractal Coding
3. Fractal Interpolation for decompression

Usage:
    python demo.py [--data-type TYPE] [--length N] [--save-plots] [--output-dir DIR]
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.generator import TimeSeriesGenerator
from data.loader import TimeSeriesLoader
from compression.ifs_compression import IFSCompressor
from compression.fractal_coding import FractalCodingCompressor
from decompression.fractal_interpolation import FractalInterpolationDecompressor
from utils.metrics import CompressionMetrics
from utils.plotting import CompressionVisualizer


class FractalCompressionDemo:
    """Demonstration class for fractal time series compression methods."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.visualizer = CompressionVisualizer()
        self.metrics_calculator = CompressionMetrics()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize compressors
        self.compressors = {
            'IFS': IFSCompressor(n_transformations=4, max_iterations=100),
            'Fractal_Coding': FractalCodingCompressor(range_block_size=8, domain_block_size=16),
        }
        
        # Initialize decompressor
        self.fif_decompressor = FractalInterpolationDecompressor()
    
    def generate_test_data(self, data_type: str, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test time series data."""
        print(f"Generating {data_type} time series with {length} points...")
        
        if data_type == 'sine':
            t, y = TimeSeriesGenerator.sine_wave(
                n_points=length, frequency=2.0, amplitude=1.0, noise_level=0.1
            )
        elif data_type == 'multi_sine':
            t, y = TimeSeriesGenerator.multi_component_series(
                n_points=length,
                components=[
                    {'type': 'sine', 'frequency': 1.0, 'amplitude': 1.0},
                    {'type': 'sine', 'frequency': 3.0, 'amplitude': 0.5},
                    {'type': 'sine', 'frequency': 7.0, 'amplitude': 0.3}
                ]
            )
        elif data_type == 'fractal_brownian':
            t, y = TimeSeriesGenerator.fractal_brownian_motion(
                n_points=length, hurst=0.7, scale=1.0
            )
        elif data_type == 'random_walk':
            t, y = TimeSeriesGenerator.random_walk(
                n_points=length, step_size=0.1, drift=0.01
            )
        elif data_type == 'stock':
            t, y = TimeSeriesGenerator.stock_price_simulation(
                n_points=length, initial_price=100.0, volatility=0.2
            )
        elif data_type == 'sensor':
            t, y = TimeSeriesLoader.load_sensor_data_sample()
            if len(y) != length:
                # Resample to desired length
                t = np.linspace(t[0], t[-1], length)
                y = np.interp(t, np.linspace(t[0], t[-1], len(y)), y)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        return t, y
    
    def run_compression_test(self, method_name: str, time_data: np.ndarray, 
                           value_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run compression test for a specific method."""
        print(f"\nTesting {method_name} compression...")
        
        compressor = self.compressors[method_name]
        
        try:
            # Compress data
            compression_result = compressor.compress(time_data, value_data)
            print(f"  Compression completed in {compression_result.compression_time:.3f}s")
            print(f"  Compression ratio: {compression_result.compression_ratio:.2f}")
            print(f"  Space savings: {compression_result.space_savings:.1f}%")
            
            # Decompress using the method's own decompressor
            decompression_result = compressor.decompress(compression_result)
            reconstructed_values = decompression_result.reconstructed_data
            print(f"  Decompression completed in {decompression_result.decompression_time:.3f}s")
            
            # Alternative: Decompress using fractal interpolation
            fif_result = self.fif_decompressor.decompress_with_fif(
                compression_result.compressed_data, 
                len(value_data),
                (time_data[0], time_data[-1])
            )
            reconstructed_fif = fif_result.reconstructed_data
            
            # Calculate comprehensive metrics for both decompression methods
            metrics_native = self.metrics_calculator.comprehensive_evaluation(
                value_data, reconstructed_values,
                compression_result.original_size, compression_result.compressed_size,
                compression_result.compression_time, decompression_result.decompression_time
            )
            
            metrics_fif = self.metrics_calculator.comprehensive_evaluation(
                value_data, reconstructed_fif,
                compression_result.original_size, compression_result.compressed_size,
                compression_result.compression_time, fif_result.decompression_time
            )
            
            # Add method identification
            metrics_native['decompression_method'] = 'native'
            metrics_fif['decompression_method'] = 'fractal_interpolation'
            
            print(f"  Native decompression - RMSE: {metrics_native['rmse']:.4f}, Correlation: {metrics_native['pearson_correlation']:.4f}")
            print(f"  FIF decompression - RMSE: {metrics_fif['rmse']:.4f}, Correlation: {metrics_fif['pearson_correlation']:.4f}")
            
            # Return the better reconstruction
            if metrics_native['pearson_correlation'] >= metrics_fif['pearson_correlation']:
                return reconstructed_values, metrics_native
            else:
                return reconstructed_fif, metrics_fif
                
        except Exception as e:
            print(f"  Error in {method_name}: {e}")
            return np.zeros_like(value_data), {'error': str(e)}
    
    def run_comprehensive_demo(self, data_type: str = 'multi_sine', length: int = 1000,
                             save_plots: bool = True) -> Dict[str, Any]:
        """Run comprehensive demonstration of all compression methods."""
        print("=" * 60)
        print("FRACTAL TIME SERIES COMPRESSION DEMONSTRATION")
        print("=" * 60)
        
        # Generate test data
        time_data, value_data = self.generate_test_data(data_type, length)
        
        # Normalize data for better compression performance
        time_data, value_data = TimeSeriesLoader.preprocess_data(
            time_data, value_data, normalize=True, remove_trend=False
        )
        
        print(f"\nData characteristics:")
        print(f"  Length: {len(value_data)}")
        print(f"  Mean: {np.mean(value_data):.4f}")
        print(f"  Std: {np.std(value_data):.4f}")
        print(f"  Range: [{np.min(value_data):.4f}, {np.max(value_data):.4f}]")
        
        # Run compression tests
        results = {}
        reconstruction_data = {}
        
        for method_name in self.compressors.keys():
            try:
                reconstructed, metrics = self.run_compression_test(method_name, time_data, value_data)
                results[method_name] = metrics
                reconstruction_data[method_name] = reconstructed
            except Exception as e:
                print(f"Failed to test {method_name}: {e}")
                results[method_name] = {'error': str(e)}
                reconstruction_data[method_name] = np.zeros_like(value_data)
        
        # Create visualizations
        if save_plots:
            self._create_visualizations(time_data, value_data, reconstruction_data, results, data_type)
        
        # Print summary
        self._print_summary(results)
        
        return {
            'original_data': (time_data, value_data),
            'reconstructions': reconstruction_data,
            'metrics': results,
            'data_type': data_type
        }
    
    def _create_visualizations(self, time_data: np.ndarray, value_data: np.ndarray,
                             reconstructions: Dict[str, np.ndarray], 
                             results: Dict[str, Dict], data_type: str):
        """Create and save visualization plots."""
        print("\nCreating visualizations...")
        
        # Individual method plots
        for method_name, reconstructed in reconstructions.items():
            if 'error' not in results[method_name]:
                # Compression comparison plot
                fig = self.visualizer.plot_compression_comparison(
                    time_data, value_data, reconstructed, method_name, 
                    results[method_name],
                    save_path=os.path.join(self.output_dir, f"{method_name}_{data_type}_comparison.png")
                )
                plt.close(fig)
                
                # Frequency analysis plot
                fig = self.visualizer.plot_frequency_analysis(
                    value_data, reconstructed, method_name=method_name,
                    save_path=os.path.join(self.output_dir, f"{method_name}_{data_type}_frequency.png")
                )
                plt.close(fig)
        
        # Method comparison plots
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if len(valid_results) > 1:
            fig = self.visualizer.plot_method_comparison(
                valid_results,
                save_path=os.path.join(self.output_dir, f"{data_type}_methods_comparison.png")
            )
            plt.close(fig)
            
            fig = self.visualizer.plot_compression_performance(
                valid_results,
                save_path=os.path.join(self.output_dir, f"{data_type}_performance_tradeoff.png")
            )
            plt.close(fig)
        
        print(f"  Plots saved to {self.output_dir}/")
    
    def _print_summary(self, results: Dict[str, Dict]):
        """Print summary of compression results."""
        print("\n" + "=" * 60)
        print("COMPRESSION RESULTS SUMMARY")
        print("=" * 60)
        
        # Prepare data for summary table
        methods = []
        compression_ratios = []
        correlations = []
        rmses = []
        times = []
        
        for method_name, metrics in results.items():
            if 'error' not in metrics:
                methods.append(method_name)
                compression_ratios.append(metrics.get('compression_ratio', 0))
                correlations.append(metrics.get('pearson_correlation', 0))
                rmses.append(metrics.get('rmse', float('inf')))
                times.append(metrics.get('total_time', 0))
        
        if methods:
            print(f"{'Method':<20} {'Ratio':<8} {'Corr':<8} {'RMSE':<10} {'Time(s)':<8}")
            print("-" * 60)
            
            for i, method in enumerate(methods):
                print(f"{method:<20} {compression_ratios[i]:<8.2f} {correlations[i]:<8.4f} "
                      f"{rmses[i]:<10.4f} {times[i]:<8.3f}")
            
            # Find best methods
            if compression_ratios:
                best_ratio_idx = np.argmax(compression_ratios)
                best_quality_idx = np.argmax(correlations)
                fastest_idx = np.argmin(times)
                
                print("\nBest performers:")
                print(f"  Highest compression ratio: {methods[best_ratio_idx]} ({compression_ratios[best_ratio_idx]:.2f})")
                print(f"  Best reconstruction quality: {methods[best_quality_idx]} (r={correlations[best_quality_idx]:.4f})")
                print(f"  Fastest processing: {methods[fastest_idx]} ({times[fastest_idx]:.3f}s)")
        
        print("\n" + "=" * 60)
    
    def run_parameter_sensitivity_analysis(self, data_type: str = 'multi_sine', 
                                         length: int = 500, save_plots: bool = True):
        """Run sensitivity analysis for compression parameters."""
        print("\nRunning parameter sensitivity analysis...")
        
        time_data, value_data = self.generate_test_data(data_type, length)
        time_data, value_data = TimeSeriesLoader.preprocess_data(time_data, value_data)
        
        # IFS parameter sensitivity
        ifs_results = {}
        for n_transforms in [2, 3, 4, 5, 6]:
            compressor = IFSCompressor(n_transformations=n_transforms, max_iterations=50)
            try:
                comp_result = compressor.compress(time_data, value_data)
                decomp_result = compressor.decompress(comp_result)
                
                metrics = self.metrics_calculator.comprehensive_evaluation(
                    value_data, decomp_result.reconstructed_data,
                    comp_result.original_size, comp_result.compressed_size,
                    comp_result.compression_time, decomp_result.decompression_time
                )
                ifs_results[f'IFS_{n_transforms}_transforms'] = metrics
                print(f"  IFS with {n_transforms} transforms: ratio={metrics['compression_ratio']:.2f}, corr={metrics['pearson_correlation']:.4f}")
            except Exception as e:
                print(f"  IFS with {n_transforms} transforms failed: {e}")
        
        # Fractal Coding parameter sensitivity
        fc_results = {}
        for block_size in [4, 8, 12, 16]:
            compressor = FractalCodingCompressor(range_block_size=block_size, domain_block_size=block_size*2)
            try:
                comp_result = compressor.compress(time_data, value_data)
                decomp_result = compressor.decompress(comp_result)
                
                metrics = self.metrics_calculator.comprehensive_evaluation(
                    value_data, decomp_result.reconstructed_data,
                    comp_result.original_size, comp_result.compressed_size,
                    comp_result.compression_time, decomp_result.decompression_time
                )
                fc_results[f'FC_block_{block_size}'] = metrics
                print(f"  FC with block size {block_size}: ratio={metrics['compression_ratio']:.2f}, corr={metrics['pearson_correlation']:.4f}")
            except Exception as e:
                print(f"  FC with block size {block_size} failed: {e}")
        
        # Create parameter sensitivity plots
        if save_plots:
            all_results = {**ifs_results, **fc_results}
            if all_results:
                fig = self.visualizer.plot_method_comparison(
                    all_results,
                    save_path=os.path.join(self.output_dir, f"{data_type}_parameter_sensitivity.png")
                )
                plt.close(fig)
        
        return {'IFS': ifs_results, 'Fractal_Coding': fc_results}


def main():
    """Main function for running the demo."""
    parser = argparse.ArgumentParser(description='Fractal Time Series Compression Demo')
    parser.add_argument('--data-type', choices=['sine', 'multi_sine', 'fractal_brownian', 'random_walk', 'stock', 'sensor'],
                       default='multi_sine', help='Type of time series data to generate')
    parser.add_argument('--length', type=int, default=1000, help='Length of time series')
    parser.add_argument('--save-plots', action='store_true', help='Save visualization plots')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--sensitivity', action='store_true', help='Run parameter sensitivity analysis')
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = FractalCompressionDemo(output_dir=args.output_dir)
    
    try:
        # Run main demonstration
        results = demo.run_comprehensive_demo(
            data_type=args.data_type,
            length=args.length,
            save_plots=args.save_plots
        )
        
        # Run sensitivity analysis if requested
        if args.sensitivity:
            sensitivity_results = demo.run_parameter_sensitivity_analysis(
                data_type=args.data_type,
                length=min(args.length, 500),  # Limit length for sensitivity analysis
                save_plots=args.save_plots
            )
        
        print(f"\nDemo completed successfully! Results saved to '{args.output_dir}/'")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())