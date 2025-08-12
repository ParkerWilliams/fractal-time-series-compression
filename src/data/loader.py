import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
import urllib.request
import json


class TimeSeriesLoader:
    """Load real-world time series datasets for testing compression algorithms."""
    
    @staticmethod
    def load_csv(file_path: str, time_col: str = 'time', 
                value_col: str = 'value', delimiter: str = ',') -> Tuple[np.ndarray, np.ndarray]:
        """Load time series data from CSV file."""
        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            if time_col in df.columns:
                time_data = df[time_col].values
            else:
                time_data = np.arange(len(df))
                
            if value_col in df.columns:
                value_data = df[value_col].values
            else:
                # Use first numeric column if value_col not found
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    value_data = df[numeric_cols[0]].values
                else:
                    raise ValueError("No numeric columns found in CSV")
            
            return time_data, value_data
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
    
    @staticmethod
    def load_stock_data_sample() -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample stock-like data for demonstration."""
        # Since we can't easily load real stock data without API keys,
        # we'll generate realistic sample data
        np.random.seed(42)  # For reproducibility
        n_points = 500
        
        # Generate realistic stock price movement
        initial_price = 100.0
        returns = np.random.normal(0.0005, 0.02, n_points)  # Daily returns
        log_returns = np.cumsum(returns)
        prices = initial_price * np.exp(log_returns)
        
        time_data = np.arange(n_points)
        return time_data, prices
    
    @staticmethod
    def load_sensor_data_sample() -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample sensor data (temperature-like) for demonstration."""
        np.random.seed(123)
        n_points = 1000
        
        # Simulate daily temperature pattern with seasonal variation
        time_data = np.arange(n_points)
        
        # Base temperature with daily and seasonal cycles
        daily_cycle = 5 * np.sin(2 * np.pi * time_data / 24)  # Daily variation
        seasonal_cycle = 10 * np.sin(2 * np.pi * time_data / 365)  # Seasonal variation
        noise = np.random.normal(0, 1, n_points)  # Random noise
        
        temperature = 20 + seasonal_cycle + daily_cycle + noise
        
        return time_data, temperature
    
    @staticmethod
    def load_eeg_sample() -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample EEG-like data for demonstration."""
        np.random.seed(456)
        n_points = 2000
        sampling_rate = 250  # Hz
        
        time_data = np.arange(n_points) / sampling_rate
        
        # Simulate EEG with alpha (8-12 Hz) and beta (13-30 Hz) waves
        alpha_wave = 10 * np.sin(2 * np.pi * 10 * time_data)
        beta_wave = 5 * np.sin(2 * np.pi * 20 * time_data)
        noise = np.random.normal(0, 2, n_points)
        
        # Add some artifacts and varying amplitude
        amplitude_modulation = 1 + 0.3 * np.sin(2 * np.pi * 0.1 * time_data)
        eeg_signal = amplitude_modulation * (alpha_wave + beta_wave) + noise
        
        return time_data, eeg_signal
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize time series data."""
        if method == 'minmax':
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        elif method == 'zscore':
            return (data - np.mean(data)) / np.std(data)
        elif method == 'robust':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            return (data - median) / (1.4826 * mad) if mad > 0 else data - median
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def preprocess_data(time_data: np.ndarray, value_data: np.ndarray,
                       normalize: bool = True, remove_trend: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess time series data for compression."""
        processed_values = value_data.copy()
        
        # Remove linear trend if requested
        if remove_trend:
            coeffs = np.polyfit(time_data, processed_values, 1)
            trend = np.polyval(coeffs, time_data)
            processed_values = processed_values - trend
        
        # Normalize if requested
        if normalize:
            processed_values = TimeSeriesLoader.normalize_data(processed_values)
        
        return time_data, processed_values