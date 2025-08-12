#!/usr/bin/env python3
"""
Unit tests for fractal time series compression methods.
"""

import unittest
import numpy as np
import os
import sys
from typing import Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.generator import TimeSeriesGenerator
from data.loader import TimeSeriesLoader
from compression.base_compressor import BaseCompressor, CompressionResult, DecompressionResult
from compression.ifs_compression import IFSCompressor, IFSTransformation
from compression.fractal_coding import FractalCodingCompressor, FractalBlock
from decompression.fractal_interpolation import FractalInterpolationFunction, FractalInterpolationDecompressor
from utils.metrics import CompressionMetrics


class TestTimeSeriesGenerator(unittest.TestCase):
    """Test time series data generation."""
    
    def test_sine_wave_generation(self):
        """Test sine wave generation."""
        t, y = TimeSeriesGenerator.sine_wave(n_points=100, frequency=1.0, amplitude=2.0)
        
        self.assertEqual(len(t), 100)
        self.assertEqual(len(y), 100)
        self.assertAlmostEqual(np.max(y), 2.0, places=1)
        self.assertAlmostEqual(np.min(y), -2.0, places=1)
    
    def test_random_walk_generation(self):
        """Test random walk generation."""
        t, y = TimeSeriesGenerator.random_walk(n_points=100, step_size=1.0)
        
        self.assertEqual(len(t), 100)
        self.assertEqual(len(y), 100)
        self.assertTrue(np.all(t == np.arange(100)))
    
    def test_fractal_brownian_motion(self):
        """Test fractional Brownian motion generation."""
        t, y = TimeSeriesGenerator.fractal_brownian_motion(n_points=100, hurst=0.7)
        
        self.assertEqual(len(t), 100)
        self.assertEqual(len(y), 100)
        self.assertIsInstance(y[0], (float, np.floating))
    
    def test_multi_component_series(self):
        """Test multi-component series generation."""
        components = [
            {'type': 'sine', 'frequency': 1.0, 'amplitude': 1.0},
            {'type': 'sine', 'frequency': 2.0, 'amplitude': 0.5}
        ]
        t, y = TimeSeriesGenerator.multi_component_series(n_points=100, components=components)
        
        self.assertEqual(len(t), 100)
        self.assertEqual(len(y), 100)


class TestTimeSeriesLoader(unittest.TestCase):
    """Test time series data loading and preprocessing."""
    
    def test_normalize_data(self):
        """Test data normalization methods."""
        data = np.array([1, 2, 3, 4, 5])
        
        # Min-max normalization
        normalized = TimeSeriesLoader.normalize_data(data, method='minmax')
        self.assertAlmostEqual(np.min(normalized), 0.0)
        self.assertAlmostEqual(np.max(normalized), 1.0)
        
        # Z-score normalization
        normalized = TimeSeriesLoader.normalize_data(data, method='zscore')
        self.assertAlmostEqual(np.mean(normalized), 0.0, places=10)
        self.assertAlmostEqual(np.std(normalized), 1.0, places=10)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        t = np.linspace(0, 10, 100)
        y = np.sin(t) + 0.1 * t  # Sine wave with linear trend
        
        t_processed, y_processed = TimeSeriesLoader.preprocess_data(
            t, y, normalize=True, remove_trend=True
        )
        
        self.assertEqual(len(t_processed), len(t))
        self.assertEqual(len(y_processed), len(y))
        self.assertAlmostEqual(np.min(y_processed), 0.0, places=1)
        self.assertAlmostEqual(np.max(y_processed), 1.0, places=1)


class TestIFSTransformation(unittest.TestCase):
    """Test IFS transformation functionality."""
    
    def test_transformation_creation(self):
        """Test IFS transformation creation."""
        transform = IFSTransformation(0.5, 0.0, 0.0, 0.5, 0.25, 0.25)
        
        self.assertEqual(transform.a, 0.5)
        self.assertEqual(transform.b, 0.0)
        self.assertEqual(transform.c, 0.0)
        self.assertEqual(transform.d, 0.5)
        self.assertEqual(transform.e, 0.25)
        self.assertEqual(transform.f, 0.25)
    
    def test_transformation_application(self):
        """Test applying transformation to points."""
        transform = IFSTransformation(0.5, 0.0, 0.0, 0.5, 0.25, 0.25)
        
        x, y = transform.apply(1.0, 1.0)
        self.assertAlmostEqual(x, 0.75)
        self.assertAlmostEqual(y, 0.75)
    
    def test_contractivity_factor(self):
        """Test contractivity factor calculation."""
        transform = IFSTransformation(0.3, 0.0, 0.0, 0.3, 0.0, 0.0)
        contractivity = transform.contractivity_factor
        
        self.assertLess(contractivity, 1.0)
        self.assertAlmostEqual(contractivity, 0.3, places=5)
    
    def test_array_conversion(self):
        """Test conversion to/from array representation."""
        original = IFSTransformation(0.5, 0.1, 0.2, 0.4, 0.3, 0.6)
        array = original.to_array()
        reconstructed = IFSTransformation.from_array(array)
        
        self.assertAlmostEqual(original.a, reconstructed.a)
        self.assertAlmostEqual(original.b, reconstructed.b)
        self.assertAlmostEqual(original.c, reconstructed.c)
        self.assertAlmostEqual(original.d, reconstructed.d)
        self.assertAlmostEqual(original.e, reconstructed.e)
        self.assertAlmostEqual(original.f, reconstructed.f)


class TestFractalBlock(unittest.TestCase):
    """Test fractal coding block functionality."""
    
    def test_block_creation(self):
        """Test fractal block creation."""
        block = FractalBlock(
            domain_start=0, domain_end=16,
            range_start=8, range_end=16,
            scaling_factor=0.5, offset=0.1,
            reflection=True
        )
        
        self.assertEqual(block.domain_size(), 16)
        self.assertEqual(block.range_size(), 8)
        self.assertTrue(block.reflection)
    
    def test_block_serialization(self):
        """Test block to/from dictionary conversion."""
        original = FractalBlock(0, 16, 8, 16, 0.5, 0.1, True)
        data = original.to_dict()
        reconstructed = FractalBlock.from_dict(data)
        
        self.assertEqual(original.domain_start, reconstructed.domain_start)
        self.assertEqual(original.domain_end, reconstructed.domain_end)
        self.assertEqual(original.range_start, reconstructed.range_start)
        self.assertEqual(original.range_end, reconstructed.range_end)
        self.assertAlmostEqual(original.scaling_factor, reconstructed.scaling_factor)
        self.assertAlmostEqual(original.offset, reconstructed.offset)
        self.assertEqual(original.reflection, reconstructed.reflection)


class TestCompressionMethods(unittest.TestCase):
    """Test compression and decompression methods."""
    
    def setUp(self):
        """Set up test data."""
        # Generate simple test data
        np.random.seed(42)  # For reproducibility
        self.time_data = np.linspace(0, 2*np.pi, 100)
        self.value_data = np.sin(self.time_data) + 0.1 * np.sin(5 * self.time_data)
        
        # Normalize data
        self.time_data, self.value_data = TimeSeriesLoader.preprocess_data(
            self.time_data, self.value_data, normalize=True
        )
    
    def test_ifs_compression_basic(self):
        """Test basic IFS compression functionality."""
        compressor = IFSCompressor(n_transformations=3, max_iterations=50)
        
        # Test compression
        result = compressor.compress(self.time_data, self.value_data)
        
        self.assertIsInstance(result, CompressionResult)
        self.assertEqual(result.method_name, "IFS_Compressor")
        self.assertGreater(result.compression_ratio, 0)
        self.assertIsNotNone(result.compressed_data)
        
        # Test decompression
        decompressed = compressor.decompress(result)
        
        self.assertIsInstance(decompressed, DecompressionResult)
        self.assertEqual(len(decompressed.reconstructed_data), len(self.value_data))
    
    def test_fractal_coding_basic(self):
        """Test basic fractal coding functionality."""
        compressor = FractalCodingCompressor(range_block_size=4, domain_block_size=8)
        
        # Test compression
        result = compressor.compress(self.time_data, self.value_data)
        
        self.assertIsInstance(result, CompressionResult)
        self.assertEqual(result.method_name, "Fractal_Coding_Compressor")
        self.assertGreater(result.compression_ratio, 0)
        self.assertIsNotNone(result.compressed_data)
        
        # Test decompression
        decompressed = compressor.decompress(result)
        
        self.assertIsInstance(decompressed, DecompressionResult)
        self.assertEqual(len(decompressed.reconstructed_data), len(self.value_data))
    
    def test_compression_with_invalid_data(self):
        """Test compression with invalid input data."""
        compressor = IFSCompressor()
        
        # Test with mismatched array lengths
        with self.assertRaises(ValueError):
            compressor.compress(self.time_data, self.value_data[:-1])
        
        # Test with empty arrays
        with self.assertRaises(ValueError):
            compressor.compress(np.array([]), np.array([]))
        
        # Test with NaN values
        bad_data = self.value_data.copy()
        bad_data[0] = np.nan
        with self.assertRaises(ValueError):
            compressor.compress(self.time_data, bad_data)


class TestFractalInterpolation(unittest.TestCase):
    """Test fractal interpolation functionality."""
    
    def test_fif_creation(self):
        """Test Fractal Interpolation Function creation."""
        control_points = np.array([[0, 0], [1, 1], [2, 0]])
        scaling_factors = np.array([0.5, 0.3])
        
        fif = FractalInterpolationFunction(control_points, scaling_factors)
        
        self.assertEqual(len(fif.control_points), 3)
        self.assertEqual(len(fif.vertical_scaling_factors), 2)
    
    def test_fif_evaluation(self):
        """Test FIF evaluation."""
        control_points = np.array([[0, 0], [1, 1], [2, 0]])
        scaling_factors = np.array([0.3, 0.3])
        
        fif = FractalInterpolationFunction(control_points, scaling_factors)
        
        # Evaluate at control points
        x_eval = np.array([0, 1, 2])
        y_eval = fif.evaluate(x_eval, iterations=5)
        
        self.assertEqual(len(y_eval), 3)
        # Values at control points should be close to control point values
        self.assertLess(abs(y_eval[0] - 0), 0.1)
        self.assertLess(abs(y_eval[2] - 0), 0.1)
    
    def test_fif_decompressor(self):
        """Test FIF decompressor with different compression formats."""
        decompressor = FractalInterpolationDecompressor()
        
        # Create mock IFS compression data
        ifs_data = {
            'transformations': [[0.5, 0.0, 0.0, 0.5, 0.25, 0.25]],
            'time_range': (0, 1),
            'value_range': (0, 1),
            'original_length': 100
        }
        
        result = decompressor.decompress_with_fif(ifs_data, 100)
        
        self.assertIsInstance(result, DecompressionResult)
        self.assertEqual(len(result.reconstructed_data), 100)


class TestCompressionMetrics(unittest.TestCase):
    """Test compression evaluation metrics."""
    
    def setUp(self):
        """Set up test data for metrics."""
        np.random.seed(42)
        self.original = np.sin(np.linspace(0, 4*np.pi, 100))
        self.reconstructed = self.original + 0.01 * np.random.randn(100)
    
    def test_basic_metrics(self):
        """Test basic compression metrics."""
        # Test MSE
        mse = CompressionMetrics.mean_squared_error(self.original, self.reconstructed)
        self.assertGreaterEqual(mse, 0)
        
        # Test RMSE
        rmse = CompressionMetrics.root_mean_squared_error(self.original, self.reconstructed)
        self.assertAlmostEqual(rmse, np.sqrt(mse))
        
        # Test MAE
        mae = CompressionMetrics.mean_absolute_error(self.original, self.reconstructed)
        self.assertGreaterEqual(mae, 0)
        
        # Test correlation
        corr = CompressionMetrics.pearson_correlation(self.original, self.reconstructed)
        self.assertGreaterEqual(corr, -1)
        self.assertLessEqual(corr, 1)
    
    def test_signal_quality_metrics(self):
        """Test signal quality metrics."""
        # Test SNR
        snr = CompressionMetrics.signal_to_noise_ratio(self.original, self.reconstructed)
        self.assertIsInstance(snr, (float, np.floating))
        
        # Test PSNR
        psnr = CompressionMetrics.peak_signal_to_noise_ratio(self.original, self.reconstructed)
        self.assertIsInstance(psnr, (float, np.floating))
        
        # Test SSIM
        ssim = CompressionMetrics.structural_similarity_index(self.original, self.reconstructed)
        self.assertGreaterEqual(ssim, -1)
        self.assertLessEqual(ssim, 1)
    
    def test_frequency_domain_metrics(self):
        """Test frequency domain analysis."""
        freq_metrics = CompressionMetrics.frequency_domain_error(self.original, self.reconstructed)
        
        self.assertIn('magnitude_error', freq_metrics)
        self.assertIn('phase_error', freq_metrics)
        self.assertIn('spectral_correlation', freq_metrics)
        
        self.assertGreaterEqual(freq_metrics['magnitude_error'], 0)
        self.assertGreaterEqual(freq_metrics['phase_error'], 0)
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation function."""
        metrics = CompressionMetrics.comprehensive_evaluation(
            self.original, self.reconstructed,
            original_size=800, compressed_size=200,
            compression_time=0.1, decompression_time=0.05
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'compression_ratio', 'space_savings_percent', 'mse', 'rmse', 'mae',
            'snr_db', 'psnr_db', 'pearson_correlation', 'ssim'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check compression ratio calculation
        self.assertAlmostEqual(metrics['compression_ratio'], 4.0)
        self.assertAlmostEqual(metrics['space_savings_percent'], 75.0)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete compression workflows."""
    
    def test_end_to_end_workflow(self):
        """Test complete compression-decompression workflow."""
        # Generate test data
        time_data, value_data = TimeSeriesGenerator.sine_wave(n_points=200, noise_level=0.05)
        time_data, value_data = TimeSeriesLoader.preprocess_data(time_data, value_data)
        
        # Test IFS workflow
        ifs_compressor = IFSCompressor(n_transformations=3, max_iterations=20)
        ifs_result = ifs_compressor.compress(time_data, value_data)
        ifs_decompressed = ifs_compressor.decompress(ifs_result)
        
        # Verify reconstruction quality
        correlation = CompressionMetrics.pearson_correlation(
            value_data, ifs_decompressed.reconstructed_data
        )
        self.assertGreater(correlation, 0.5)  # Reasonable correlation threshold
        
        # Test Fractal Coding workflow
        fc_compressor = FractalCodingCompressor(range_block_size=8, domain_block_size=16)
        fc_result = fc_compressor.compress(time_data, value_data)
        fc_decompressed = fc_compressor.decompress(fc_result)
        
        # Verify reconstruction
        fc_correlation = CompressionMetrics.pearson_correlation(
            value_data, fc_decompressed.reconstructed_data
        )
        self.assertGreater(fc_correlation, 0.3)  # Lower threshold due to block-based nature
    
    def test_different_data_types(self):
        """Test compression on different types of time series."""
        data_generators = [
            lambda: TimeSeriesGenerator.sine_wave(n_points=100),
            lambda: TimeSeriesGenerator.random_walk(n_points=100),
            lambda: TimeSeriesGenerator.fractal_brownian_motion(n_points=100)
        ]
        
        compressor = IFSCompressor(n_transformations=2, max_iterations=20)
        
        for generator in data_generators:
            time_data, value_data = generator()
            time_data, value_data = TimeSeriesLoader.preprocess_data(time_data, value_data)
            
            # Should not raise exceptions
            result = compressor.compress(time_data, value_data)
            decompressed = compressor.decompress(result)
            
            self.assertEqual(len(decompressed.reconstructed_data), len(value_data))
            self.assertIsInstance(result.compression_ratio, (float, int))


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTimeSeriesGenerator,
        TestTimeSeriesLoader, 
        TestIFSTransformation,
        TestFractalBlock,
        TestCompressionMethods,
        TestFractalInterpolation,
        TestCompressionMetrics,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\nAll tests passed! ✅")
    else:
        print("\nSome tests failed! ❌")
        sys.exit(1)