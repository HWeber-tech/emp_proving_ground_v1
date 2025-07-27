"""
Vectorized Technical Indicators
==============================

Ultra-fast technical indicator calculations using NumPy vectorization.
Optimized for sub-millisecond performance on large datasets.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class VectorizedIndicators:
    """High-performance technical indicator calculations."""
    
    @staticmethod
    def sma(data: np.ndarray, window: int) -> np.ndarray:
        """Simple Moving Average - vectorized implementation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')
    
    @staticmethod
    def ema(data: np.ndarray, window: int, alpha: Optional[float] = None) -> np.ndarray:
        """Exponential Moving Average - vectorized implementation."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        if alpha is None:
            alpha = 2 / (window + 1)
        
        ema_values = np.zeros_like(data)
        ema_values[0] = data[0]
        
        for i in range(1, len(data)):
            ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
        
        return ema_values
    
    @staticmethod
    def rsi(data: np.ndarray, window: int = 14) -> np.ndarray:
        """Relative Strength Index - vectorized implementation."""
        if len(data) < window + 1:
            return np.full(len(data), np.nan)
        
        deltas = np.diff(data)
        seed = deltas[:window+1]
        
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        
        rs = up / down if down != 0 else 100
        rsi_values = np.zeros(len(data))
        rsi_values[:window] = np.nan
        rsi_values[window] = 100 - 100 / (1 + rs)
        
        for i in range(window + 1, len(data)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            
            rs = up / down if down != 0 else 100
            rsi_values[i] = 100 - 100 / (1 + rs)
        
        return rsi_values
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands - vectorized implementation."""
        if len(data) < window:
            return (np.full(len(data), np.nan), 
                   np.full(len(data), np.nan), 
                   np.full(len(data), np.nan))
        
        sma = VectorizedIndicators.sma(data, window)
        std = np.array([np.std(data[i:i+window]) for i in range(len(data) - window + 1)])
        
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        
        # Pad arrays to match original length
        padding = len(data) - len(sma)
        sma = np.concatenate([np.full(padding, np.nan), sma])
        upper_band = np.concatenate([np.full(padding, np.nan), upper_band])
        lower_band = np.concatenate([np.full(padding, np.nan), lower_band])
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD (Moving Average Convergence Divergence) - vectorized."""
        if len(data) < slow:
            return (np.full(len(data), np.nan), 
                   np.full(len(data), np.nan), 
                   np.full(len(data), np.nan))
        
        ema_fast = VectorizedIndicators.ema(data, fast)
        ema_slow = VectorizedIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = VectorizedIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
        """Average True Range - vectorized implementation."""
        if len(high) < 2:
            return np.full(len(high), np.nan)
        
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]  # First TR is just high-low
        
        return VectorizedIndicators.sma(tr, window)
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   k_window: int = 14, d_window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator - vectorized implementation."""
        if len(high) < k_window:
            return (np.full(len(high), np.nan), np.full(len(high), np.nan))
        
        lowest_low = np.array([np.min(low[i:i+k_window]) for i in range(len(low) - k_window + 1)])
        highest_high = np.array([np.max(high[i:i+k_window]) for i in range(len(high) - k_window + 1)])
        
        k_values = 100 * ((close[k_window-1:] - lowest_low) / (highest_high - lowest_low))
        d_values = VectorizedIndicators.sma(k_values, d_window)
        
        # Pad arrays to match original length
        padding = len(high) - len(k_values)
        k_values = np.concatenate([np.full(padding, np.nan), k_values])
        d_values = np.concatenate([np.full(padding + k_window - d_window, np.nan), d_values])
        
        return k_values, d_values
    
    @staticmethod
    def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
        """Williams %R - vectorized implementation."""
        if len(high) < window:
            return np.full(len(high), np.nan)
        
        highest_high = np.array([np.max(high[i:i+window]) for i in range(len(high) - window + 1)])
        lowest_low = np.array([np.min(low[i:i+window]) for i in range(len(low) - window + 1)])
        
        wr = -100 * ((highest_high - close[window-1:]) / (highest_high - lowest_low))
        
        # Pad array to match original length
        padding = len(high) - len(wr)
        return np.concatenate([np.full(padding, np.nan), wr])
    
    @staticmethod
    def calculate_all_indicators(data: Dict[str, np.ndarray], 
                               indicators: List[str] = None) -> Dict[str, np.ndarray]:
        """Calculate multiple indicators efficiently."""
        if indicators is None:
            indicators = ['sma', 'ema', 'rsi', 'bb', 'macd', 'atr', 'stoch', 'willr']
        
        results = {}
        
        # Ensure we have required data
        close = data.get('close', np.array([]))
        high = data.get('high', close)
        low = data.get('low', close)
        
        if len(close) == 0:
            return results
        
        # Calculate requested indicators
        if 'sma' in indicators:
            results['sma_20'] = VectorizedIndicators.sma(close, 20)
            results['sma_50'] = VectorizedIndicators.sma(close, 50)
        
        if 'ema' in indicators:
            results['ema_12'] = VectorizedIndicators.ema(close, 12)
            results['ema_26'] = VectorizedIndicators.ema(close, 26)
        
        if 'rsi' in indicators:
            results['rsi_14'] = VectorizedIndicators.rsi(close, 14)
        
        if 'bb' in indicators:
            bb_upper, bb_middle, bb_lower = VectorizedIndicators.bollinger_bands(close)
            results['bb_upper'] = bb_upper
            results['bb_middle'] = bb_middle
            results['bb_lower'] = bb_lower
        
        if 'macd' in indicators:
            macd_line, signal_line, histogram = VectorizedIndicators.macd(close)
            results['macd_line'] = macd_line
            results['macd_signal'] = signal_line
            results['macd_histogram'] = histogram
        
        if 'atr' in indicators:
            results['atr_14'] = VectorizedIndicators.atr(high, low, close, 14)
        
        if 'stoch' in indicators:
            stoch_k, stoch_d = VectorizedIndicators.stochastic(high, low, close)
            results['stoch_k'] = stoch_k
            results['stoch_d'] = stoch_d
        
        if 'willr' in indicators:
            results['willr_14'] = VectorizedIndicators.williams_r(high, low, close, 14)
        
        return results
    
    @staticmethod
    def benchmark_performance(data_size: int = 10000, iterations: int = 100) -> Dict[str, float]:
        """Benchmark indicator performance."""
        import time
        
        # Generate test data
        np.random.seed(42)
        close = np.cumsum(np.random.randn(data_size) * 0.001) + 1.0
        high = close + np.abs(np.random.randn(data_size) * 0.0005)
        low = close - np.abs(np.random.randn(data_size) * 0.0005)
        
        data = {'close': close, 'high': high, 'low': low}
        
        # Benchmark all indicators
        start_time = time.time()
        for _ in range(iterations):
            VectorizedIndicators.calculate_all_indicators(data)
        total_time = (time.time() - start_time) / iterations * 1000
        
        # Individual benchmarks
        benchmarks = {'total_all_indicators_ms': total_time}
        
        # SMA benchmark
        start_time = time.time()
        for _ in range(iterations):
            VectorizedIndicators.sma(close, 20)
        benchmarks['sma_20_ms'] = (time.time() - start_time) / iterations * 1000
        
        # RSI benchmark
        start_time = time.time()
        for _ in range(iterations):
            VectorizedIndicators.rsi(close, 14)
        benchmarks['rsi_14_ms'] = (time.time() - start_time) / iterations * 1000
        
        # MACD benchmark
        start_time = time.time()
        for _ in range(iterations):
            VectorizedIndicators.macd(close)
        benchmarks['macd_ms'] = (time.time() - start_time) / iterations * 1000
        
        return benchmarks


if __name__ == "__main__":
    # Quick performance test
    results = VectorizedIndicators.benchmark_performance(data_size=1000, iterations=10)
    
    print("Vectorized Indicators Performance:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value:.3f}ms")
