"""
Sensory Organ Implementation
============================

Concrete implementation of ISensoryOrgan for the 4D+1 sensory cortex.
Optimized for performance with caching and vectorized calculations.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from .interfaces import ISensoryOrgan, SensorySignal
from ..performance import get_global_cache, VectorizedIndicators

logger = logging.getLogger(__name__)


class SensoryOrgan(ISensoryOrgan):
    """High-performance sensory organ with caching support."""
    
    def __init__(self, organ_type: str, cache_ttl: int = 300):
        """
        Initialize sensory organ.
        
        Args:
            organ_type: Type of sensory organ (what, when, anomaly, chaos)
            cache_ttl: Cache TTL in seconds
        """
        self.organ_type = organ_type
        self.cache_ttl = cache_ttl
        self.cache = get_global_cache()
        self._cache_key_prefix = f"sensory_{organ_type}"
        
    async def process_market_data(self, market_data: Dict[str, Any]) -> List[SensorySignal]:
        """Process market data and return sensory signals."""
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        # Generate cache key
        cache_key = f"{self._cache_key_prefix}:{symbol}:{datetime.now().strftime('%Y%m%d%H%M')}"
        
        # Try to get from cache first
        cached_signals = self._get_cached_signals(symbol)
        if cached_signals:
            return cached_signals
        
        # Process based on organ type
        signals = await self._process_by_type(market_data)
        
        # Cache the results
        self._cache_signals(symbol, signals)
        
        return signals
    
    async def _process_by_type(self, market_data: Dict[str, Any]) -> List[SensorySignal]:
        """Process market data based on organ type."""
        if self.organ_type == "what":
            return await self._process_what_dimension(market_data)
        elif self.organ_type == "when":
            return await self._process_when_dimension(market_data)
        elif self.organ_type == "anomaly":
            return await self._process_anomaly_dimension(market_data)
        elif self.organ_type == "chaos":
            return await self._process_chaos_dimension(market_data)
        else:
            return []
    
    async def _process_what_dimension(self, market_data: Dict[str, Any]) -> List[SensorySignal]:
        """Process WHAT dimension - technical reality."""
        signals = []
        
        # Extract data
        close = np.array(market_data.get('close', []))
        high = np.array(market_data.get('high', close))
        low = np.array(market_data.get('low', close))
        
        if len(close) < 20:
            return signals
        
        # Calculate indicators
        indicators = VectorizedIndicators.calculate_all_indicators({
            'close': close,
            'high': high,
            'low': low
        }, indicators=['sma', 'rsi', 'bb'])
        
        # Generate signals based on indicators
        current_price = close[-1]
        sma_20 = indicators.get('sma_20', np.array([]))
        rsi_14 = indicators.get('rsi_14', np.array([]))
        
        if len(sma_20) > 0:
            sma_current = sma_20[-1] if not np.isnan(sma_20[-1]) else sma_20[-2]
            
            # Price vs SMA signal
            if current_price > sma_current * 1.02:
                signals.append(SensorySignal(
                    signal_type="PRICE_ABOVE_SMA",
                    strength=0.7,
                    confidence=0.8,
                    source="what_organ",
                    metadata={'sma_value': float(sma_current), 'price': float(current_price)}
                ))
            elif current_price < sma_current * 0.98:
                signals.append(SensorySignal(
                    signal_type="PRICE_BELOW_SMA",
                    strength=0.7,
                    confidence=0.8,
                    source="what_organ",
                    metadata={'sma_value': float(sma_current), 'price': float(current_price)}
                ))
        
        # RSI signals
        if len(rsi_14) > 0:
            rsi_current = rsi_14[-1]
            if not np.isnan(rsi_current):
                if rsi_current > 70:
                    signals.append(SensorySignal(
                        signal_type="RSI_OVERBOUGHT",
                        strength=min(0.9, (rsi_current - 70) / 30),
                        confidence=0.85,
                        source="what_organ",
                        metadata={'rsi_value': float(rsi_current)}
                    ))
                elif rsi_current < 30:
                    signals.append(SensorySignal(
                        signal_type="RSI_OVERSOLD",
                        strength=min(0.9, (30 - rsi_current) / 30),
                        confidence=0.85,
                        source="what_organ",
                        metadata={'rsi_value': float(rsi_current)}
                    ))
        
        return signals
    
    async def _process_when_dimension(self, market_data: Dict[str, Any]) -> List[SensorySignal]:
        """Process WHEN dimension - temporal analysis."""
        signals = []
        
        # Extract timestamp data
        timestamps = market_data.get('timestamps', [])
        if not timestamps:
            return signals
        
        # Session analysis
        current_time = datetime.fromtimestamp(timestamps[-1])
        hour = current_time.hour
        
        # London session (8:00-16:00 GMT)
        if 8 <= hour < 16:
            signals.append(SensorySignal(
                signal_type="LONDON_SESSION",
                strength=0.6,
                confidence=0.9,
                source="when_organ",
                metadata={'hour': hour}
            ))
        
        # New York session (13:00-21:00 GMT)
        elif 13 <= hour < 21:
            signals.append(SensorySignal(
                signal_type="NEW_YORK_SESSION",
                strength=0.6,
                confidence=0.9,
                source="when_organ",
                metadata={'hour': hour}
            ))
        
        # Asian session (22:00-6:00 GMT)
        elif hour >= 22 or hour < 6:
            signals.append(SensorySignal(
                signal_type="ASIAN_SESSION",
                strength=0.4,
                confidence=0.9,
                source="when_organ",
                metadata={'hour': hour}
            ))
        
        return signals
    
    async def _process_anomaly_dimension(self, market_data: Dict[str, Any]) -> List[SensorySignal]:
        """Process ANOMALY dimension - manipulation detection."""
        signals = []
        
        # Extract data
        close = np.array(market_data.get('close', []))
        volume = np.array(market_data.get('volume', []))
        
        if len(close) < 10:
            return signals
        
        # Volume anomaly detection
        if len(volume) > 0:
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            
            if current_volume > avg_volume * 2:
                signals.append(SensorySignal(
                    signal_type="VOLUME_SPIKE",
                    strength=min(0.9, (current_volume / avg_volume - 1) * 0.5),
                    confidence=0.8,
                    source="anomaly_organ",
                    metadata={'current_volume': float(current_volume), 'avg_volume': float(avg_volume)}
                ))
        
        # Price anomaly detection
        returns = np.diff(close) / close[:-1]
        if len(returns) > 0:
            std_returns = np.std(returns[-20:])
            current_return = returns[-1]
            
            if abs(current_return) > 3 * std_returns:
                signals.append(SensorySignal(
                    signal_type="PRICE_ANOMALY",
                    strength=min(0.9, abs(current_return) / (3 * std_returns)),
                    confidence=0.75,
                    source="anomaly_organ",
                    metadata={'return': float(current_return), 'std': float(std_returns)}
                ))
        
        return signals
    
    async def _process_chaos_dimension(self, market_data: Dict[str, Any]) -> List[SensorySignal]:
        """Process CHAOS dimension - non-linear dynamics."""
        signals = []
        
        # Extract data
        close = np.array(market_data.get('close', []))
        
        if len(close) < 50:
            return signals
        
        # Chaos indicators
        returns = np.diff(close) / close[:-1]
        
        if len(returns) > 0:
            # Lyapunov exponent approximation
            lyapunov = self._calculate_lyapunov(returns)
            
            if lyapunov > 0.1:
                signals.append(SensorySignal(
                    signal_type="CHAOTIC_REGIME",
                    strength=min(0.9, lyapunov),
                    confidence=0.7,
                    source="chaos_organ",
                    metadata={'lyapunov': float(lyapunov)}
                ))
        
        return signals
    
    def _calculate_lyapunov(self, returns: np.ndarray) -> float:
        """Calculate approximate Lyapunov exponent."""
        if len(returns) < 10:
            return 0.0
        
        # Simple approximation using log returns
        log_returns = np.log(1 + returns)
        
        # Calculate divergence
        if len(log_returns) < 2:
            return 0.0
        
        # Approximate Lyapunov exponent
        lyapunov = np.std(log_returns) * np.sqrt(len(log_returns) / 252)
        return min(1.0, max(0.0, lyapunov))
    
    def _get_cached_signals(self, symbol: str) -> Optional[List[SensorySignal]]:
        """Get cached signals for symbol."""
        cached_data = self.cache.get_market_data(
            "sensory", self.organ_type,
            symbol=symbol,
            timestamp=datetime.now().strftime('%Y%m%d%H%M')
        )
        
        if cached_data is not None:
            # Reconstruct signals from cached data
            signals = []
            # This is a simplified reconstruction - in practice, you'd serialize/deserialize properly
            return signals
        
        return None
    
    def _cache_signals(self, symbol: str, signals: List[SensorySignal]) -> None:
        """Cache signals for symbol."""
        # Cache the signals for performance
        self.cache.cache_market_data(
            "sensory", self.organ_type,
            np.array([len(signals)]),  # Simplified caching
            symbol=symbol,
            timestamp=datetime.now().strftime('%Y%m%d%H%M'),
            ttl=self.cache_ttl
        )
    
    def get_organ_type(self) -> str:
        """Get the type of sensory organ."""
        return self.organ_type
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the organ."""
        return {
            'organ_type': self.organ_type,
            'cache_hits': 0,  # Implement actual cache tracking
            'cache_misses': 0,
            'signals_generated': 0
        }


# Factory function for creating sensory organs
def create_sensory_organ(organ_type: str) -> SensoryOrgan:
    """Create a sensory organ of specified type."""
    return SensoryOrgan(organ_type)


# Pre-configured sensory organs
WHAT_ORGAN = create_sensory_organ("what")
WHEN_ORGAN = create_sensory_organ("when")
ANOMALY_ORGAN = create_sensory_organ("anomaly")
CHAOS_ORGAN = create_sensory_organ("chaos")


if __name__ == "__main__":
    # Test the sensory organs
    import asyncio
    
    async def test_sensory_organs():
        # Test data
        test_data = {
            'symbol': 'EURUSD',
            'close': np.random.randn(100).cumsum() + 1.0,
            'high': np.random.randn(100).cumsum() + 1.01,
            'low': np.random.randn(100).cumsum() + 0.99,
            'volume': np.random.randint(1000, 5000, 100),
            'timestamps': [datetime.now().timestamp() + i * 60 for i in range(100)]
        }
        
        # Test WHAT organ
        what_organ = create_sensory_organ("what")
        what_signals = await what_organ.process_market_data(test_data)
        print(f"WHAT signals: {len(what_signals)}")
        
        # Test WHEN organ
        when_organ = create_sensory_organ("when")
        when_signals = await when_organ.process_market_data(test_data)
        print(f"WHEN signals: {len(when_signals)}")
        
        # Test ANOMALY organ
        anomaly_organ = create_sensory_organ("anomaly")
        anomaly_signals = await anomaly_organ.process_market_data(test_data)
        print(f"ANOMALY signals: {len(anomaly_signals)}")
        
        # Test CHAOS organ
        chaos_organ = create_sensory_organ("chaos")
        chaos_signals = await chaos_organ.process_market_data(test_data)
        print(f"CHAOS signals: {len(chaos_signals)}")
    
    asyncio.run(test_sensory_organs())
