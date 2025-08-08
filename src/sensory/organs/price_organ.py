"""
EMP Price Sensory Organ v1.1

Processes raw price data and extracts price-related sensory signals
including trends, momentum, volatility, and price action patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging

from src.sensory.core.base import SensoryOrgan, MarketData
from src.sensory.signals import SensorSignal as SensorySignal
from src.core.exceptions import SensoryException

logger = logging.getLogger(__name__)


@dataclass
class PriceSignal:
    """Price-related sensory signal."""
    timestamp: datetime
    signal_type: str
    value: float
    confidence: float
    metadata: Dict[str, Any]


class PriceOrgan(SensoryOrgan):
    """Sensory organ for processing price data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.calibrated = False
        self._price_history: List[float] = []
        self._max_history = self.config.get('max_history', 1000)
        
    def perceive(self, data: MarketData) -> SensorySignal:
        """Process raw price data into sensory signals."""
        try:
            # Add to price history
            self._price_history.append(data.close)
            if len(self._price_history) > self._max_history:
                self._price_history.pop(0)
                
            # Calculate price signals
            signals = []
            
            # Trend signal
            if len(self._price_history) >= 20:
                trend_signal = self._calculate_trend_signal()
                signals.append(trend_signal)
                
            # Momentum signal
            if len(self._price_history) >= 14:
                momentum_signal = self._calculate_momentum_signal()
                signals.append(momentum_signal)
                
            # Volatility signal
            if len(self._price_history) >= 20:
                volatility_signal = self._calculate_volatility_signal()
                signals.append(volatility_signal)
                
            # Price action signal
            if len(self._price_history) >= 5:
                price_action_signal = self._calculate_price_action_signal(data)
                signals.append(price_action_signal)
                
            # Combine signals into a single sensory reading
            combined_signal = self._combine_signals(signals)
            
            return SensorySignal(
                timestamp=data.timestamp,
                signal_type="price_composite",
                value=combined_signal.value,
                confidence=combined_signal.confidence,
                metadata={
                    'signals': [s.__dict__ for s in signals],
                    'price_history_length': len(self._price_history),
                    'organ_id': 'price_organ'
                }
            )
            
        except Exception as e:
            raise SensoryException(f"Error in price perception: {e}")
            
    def calibrate(self) -> bool:
        """Calibrate the price organ."""
        try:
            # Reset calibration state
            self.calibrated = False
            
            # Perform calibration checks
            if len(self._price_history) < 20:
                logger.warning("Insufficient price history for calibration")
                return False
                
            # Calculate calibration metrics
            price_array = np.array(self._price_history)
            
            # Check for reasonable price values
            if np.any(price_array <= 0):
                raise SensoryException("Invalid price values detected")
                
            # Check for reasonable volatility
            volatility = np.std(price_array)
            if volatility == 0:
                raise SensoryException("Zero volatility detected")
                
            # Mark as calibrated
            self.calibrated = True
            logger.info("Price organ calibrated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Price organ calibration failed: {e}")
            return False
            
    def _calculate_trend_signal(self) -> PriceSignal:
        """Calculate trend signal using moving averages."""
        prices = np.array(self._price_history)
        
        # Calculate short and long moving averages
        short_ma = np.mean(prices[-10:])  # 10-period MA
        long_ma = np.mean(prices[-20:])   # 20-period MA
        
        # Calculate trend strength
        trend_strength = (short_ma - long_ma) / long_ma
        
        # Determine trend direction
        if trend_strength > 0.001:
            trend_direction = 1.0  # Bullish
        elif trend_strength < -0.001:
            trend_direction = -1.0  # Bearish
        else:
            trend_direction = 0.0  # Neutral
            
        # Calculate confidence based on trend strength
        confidence = min(abs(trend_strength) * 100, 1.0)
        
        return PriceSignal(
            timestamp=datetime.now(),
            signal_type="trend",
            value=trend_direction,
            confidence=confidence,
            metadata={
                'short_ma': short_ma,
                'long_ma': long_ma,
                'trend_strength': trend_strength
            }
        )
        
    def _calculate_momentum_signal(self) -> PriceSignal:
        """Calculate momentum signal using rate of change."""
        prices = np.array(self._price_history)
        
        # Calculate rate of change over 14 periods
        roc = (prices[-1] - prices[-14]) / prices[-14]
        
        # Normalize to [-1, 1] range
        momentum = np.tanh(roc * 10)  # Scale and bound
        
        # Calculate confidence based on magnitude
        confidence = min(abs(roc) * 5, 1.0)
        
        return PriceSignal(
            timestamp=datetime.now(),
            signal_type="momentum",
            value=momentum,
            confidence=confidence,
            metadata={
                'roc': roc,
                'period': 14
            }
        )
        
    def _calculate_volatility_signal(self) -> PriceSignal:
        """Calculate volatility signal using rolling standard deviation."""
        prices = np.array(self._price_history)
        
        # Calculate rolling volatility over 20 periods
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-20:])
        
        # Normalize volatility (typical forex volatility is 0.01-0.02)
        normalized_vol = min(volatility * 100, 1.0)
        
        # Calculate confidence based on data quality
        confidence = min(len(returns) / 20, 1.0)
        
        return PriceSignal(
            timestamp=datetime.now(),
            signal_type="volatility",
            value=normalized_vol,
            confidence=confidence,
            metadata={
                'volatility': volatility,
                'period': 20
            }
        )
        
    def _calculate_price_action_signal(self, data: MarketData) -> PriceSignal:
        """Calculate price action signal based on current bar."""
        if len(self._price_history) < 2:
            return PriceSignal(
                timestamp=data.timestamp,
                signal_type="price_action",
                value=0.0,
                confidence=0.0,
                metadata={}
            )
            
        prev_close = self._price_history[-2]
        current_close = data.close
        
        # Calculate price change
        price_change = (current_close - prev_close) / prev_close
        
        # Determine if this is a strong move
        if abs(price_change) > 0.001:  # 0.1% move
            if price_change > 0:
                action_signal = 1.0  # Strong up move
            else:
                action_signal = -1.0  # Strong down move
        else:
            action_signal = 0.0  # Weak move
            
        # Calculate confidence based on move strength
        confidence = min(abs(price_change) * 100, 1.0)
        
        return PriceSignal(
            timestamp=data.timestamp,
            signal_type="price_action",
            value=action_signal,
            confidence=confidence,
            metadata={
                'price_change': price_change,
                'prev_close': prev_close,
                'current_close': current_close
            }
        )
        
    def _combine_signals(self, signals: List[PriceSignal]) -> PriceSignal:
        """Combine multiple price signals into a composite signal."""
        if not signals:
            return PriceSignal(
                timestamp=datetime.now(),
                signal_type="price_composite",
                value=0.0,
                confidence=0.0,
                metadata={}
            )
            
        # Weighted average of signal values
        total_weight = 0
        weighted_sum = 0
        
        for signal in signals:
            weight = signal.confidence
            total_weight += weight
            weighted_sum += signal.value * weight
            
        if total_weight > 0:
            composite_value = weighted_sum / total_weight
            composite_confidence = np.mean([s.confidence for s in signals])
        else:
            composite_value = 0.0
            composite_confidence = 0.0
            
        return PriceSignal(
            timestamp=datetime.now(),
            signal_type="price_composite",
            value=composite_value,
            confidence=composite_confidence,
            metadata={
                'signal_count': len(signals),
                'individual_signals': [s.signal_type for s in signals]
            }
        )
