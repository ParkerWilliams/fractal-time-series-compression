import numpy as np
from typing import Tuple, List, Optional


class TimeSeriesGenerator:
    """Generate synthetic time series data for testing compression algorithms."""
    
    @staticmethod
    def sine_wave(n_points: int = 1000, frequency: float = 1.0, 
                  amplitude: float = 1.0, phase: float = 0.0, 
                  noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a sine wave with optional noise."""
        t = np.linspace(0, 2 * np.pi * frequency, n_points)
        y = amplitude * np.sin(t + phase)
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, n_points)
            y += noise
            
        return t, y
    
    @staticmethod
    def random_walk(n_points: int = 1000, step_size: float = 1.0, 
                   drift: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a random walk time series."""
        t = np.arange(n_points)
        steps = np.random.normal(drift, step_size, n_points)
        y = np.cumsum(steps)
        return t, y
    
    @staticmethod
    def fractal_brownian_motion(n_points: int = 1000, hurst: float = 0.5, 
                               scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate fractional Brownian motion with specified Hurst parameter."""
        t = np.arange(n_points)
        
        # Generate white noise
        white_noise = np.random.normal(0, 1, n_points)
        
        # Convert to fractional Brownian motion using FFT method
        fft_white = np.fft.fft(white_noise)
        freqs = np.fft.fftfreq(n_points)
        freqs[0] = 1  # Avoid division by zero
        
        # Apply fractional filter
        filter_coeff = np.abs(freqs) ** (-hurst - 0.5)
        filter_coeff[0] = 0
        
        fft_filtered = fft_white * filter_coeff
        fbm = np.real(np.fft.ifft(fft_filtered))
        
        return t, fbm * scale
    
    @staticmethod
    def multi_component_series(n_points: int = 1000, 
                             components: List[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a multi-component time series."""
        if components is None:
            components = [
                {'type': 'sine', 'frequency': 1.0, 'amplitude': 1.0},
                {'type': 'sine', 'frequency': 3.0, 'amplitude': 0.5},
                {'type': 'sine', 'frequency': 5.0, 'amplitude': 0.3}
            ]
        
        t = np.linspace(0, 2 * np.pi, n_points)
        y = np.zeros(n_points)
        
        for comp in components:
            if comp['type'] == 'sine':
                freq = comp.get('frequency', 1.0)
                amp = comp.get('amplitude', 1.0)
                phase = comp.get('phase', 0.0)
                y += amp * np.sin(freq * t + phase)
            elif comp['type'] == 'cosine':
                freq = comp.get('frequency', 1.0)
                amp = comp.get('amplitude', 1.0)
                phase = comp.get('phase', 0.0)
                y += amp * np.cos(freq * t + phase)
        
        return t, y
    
    @staticmethod
    def stock_price_simulation(n_points: int = 1000, initial_price: float = 100.0,
                             volatility: float = 0.2, drift: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate stock price using geometric Brownian motion."""
        dt = 1/252  # Daily time step (252 trading days per year)
        t = np.arange(n_points) * dt
        
        # Generate random returns
        returns = np.random.normal((drift - 0.5 * volatility**2) * dt,
                                 volatility * np.sqrt(dt), n_points)
        
        # Calculate price path
        log_returns = np.cumsum(returns)
        prices = initial_price * np.exp(log_returns)
        
        return t, prices