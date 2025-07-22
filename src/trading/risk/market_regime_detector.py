#!/usr/bin/env python3
"""
Market Regime Detector
=====================

This module implements the 8 market regime detection system:
1. TRENDING_UP - Bullish trending market
2. TRENDING_DOWN - Bearish trending market
3. RANGING - Sideways/ranging market
4. VOLATILE - High volatility regime
5. CRISIS - Market crisis conditions
6. RECOVERY - Post-crisis recovery
7. LOW_VOLATILITY - Low volatility environment
8. HIGH_VOLATILITY - High volatility environment

Uses volatility + trend analysis to detect regime changes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """8 market regime classifications"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"
    RECOVERY = "RECOVERY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


@dataclass
class RegimeCharacteristics:
    """Characteristics of the detected regime."""
    volatility_level: float
    trend_strength: float
    momentum: float
    volume_anomaly: float
    correlation_strength: float


@dataclass
class RegimeDetectionResult:
    """Result of market regime detection with characteristics."""
    regime: MarketRegime
    confidence: float
    characteristics: RegimeCharacteristics
    timestamp: datetime


class MarketRegimeDetector:
    """Advanced market regime detection system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration parameters
        self.lookback_period = self.config.get('lookback_period', 20)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.02)
        self.trend_threshold = self.config.get('trend_threshold', 0.01)
        self.crisis_threshold = self.config.get('crisis_threshold', -0.05)
        self.recovery_threshold = self.config.get('recovery_threshold', 0.03)
        
        # State tracking
        self.regime_history = []
        self.volatility_history = []
        self.trend_history = []
        
        logger.info("MarketRegimeDetector initialized")
    
    def detect_current_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime from price data"""
        try:
            if len(market_data) < self.lookback_period:
                return MarketRegime.RANGING
            
            # Calculate key metrics
            volatility = self._calculate_volatility(market_data)
            trend = self._calculate_trend(market_data)
            price_change = self._calculate_price_change(market_data)
            
            # Determine regime based on metrics
            regime = self._classify_regime(volatility, trend, price_change)
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.RANGING
    
    async def detect_regime(self, market_data: pd.DataFrame) -> RegimeDetectionResult:
        """Async version of regime detection with full characteristics."""
        try:
            if len(market_data) < self.lookback_period:
                regime = MarketRegime.RANGING
                confidence = 0.8
            else:
                # Calculate key metrics
                volatility = self._calculate_volatility(market_data)
                trend = self._calculate_trend(market_data)
                price_change = self._calculate_price_change(market_data)
                
                # Determine regime based on metrics
                regime = self._classify_regime(volatility, trend, price_change)
                
                # Calculate confidence based on metric clarity
                confidence = min(1.0, abs(trend) * 50 + volatility * 10)
            
            # Create characteristics
            characteristics = RegimeCharacteristics(
                volatility_level=volatility,
                trend_strength=abs(trend),
                momentum=trend * 100,
                volume_anomaly=0.0,  # Placeholder
                correlation_strength=0.5  # Placeholder
            )
            
            return RegimeDetectionResult(
                regime=regime,
                confidence=confidence,
                characteristics=characteristics,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return RegimeDetectionResult(
                regime=MarketRegime.RANGING,
                confidence=0.5,
                characteristics=RegimeCharacteristics(
                    volatility_level=0.02,
                    trend_strength=0.0,
                    momentum=0.0,
                    volume_anomaly=0.0,
                    correlation_strength=0.5
                ),
                timestamp=datetime.now()
            )
    
    def detect_regime_sequence(self, market_data: pd.DataFrame) -> List[MarketRegime]:
        """Detect regime sequence over time"""
        regimes = []
        
        # Process data in rolling windows
        for i in range(self.lookback_period, len(market_data)):
            window_data = market_data.iloc[i-self.lookback_period:i]
            regime = self.detect_current_regime(window_data)
            regimes.append(regime)
        
        return regimes
    
    def get_regime_probabilities(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Get probability distribution over all regimes"""
        try:
            volatility = self._calculate_volatility(market_data)
            trend = self._calculate_trend(market_data)
            price_change = self._calculate_price_change(market_data)
            
            # Calculate probabilities for each regime
            probabilities = {
                'TRENDING_UP': 0.0,
                'TRENDING_DOWN': 0.0,
                'RANGING': 0.0,
                'VOLATILE': 0.0,
                'CRISIS': 0.0,
                'RECOVERY': 0.0,
                'LOW_VOLATILITY': 0.0,
                'HIGH_VOLATILITY': 0.0
            }
            
            # Trend-based probabilities
            if abs(trend) > self.trend_threshold:
                if trend > 0:
                    probabilities['TRENDING_UP'] = min(abs(trend) * 10, 0.8)
                    probabilities['TRENDING_DOWN'] = 0.0
                else:
                    probabilities['TRENDING_UP'] = 0.0
                    probabilities['TRENDING_DOWN'] = min(abs(trend) * 10, 0.8)
            else:
                probabilities['RANGING'] = 0.7
            
            # Volatility-based probabilities
            if volatility > self.volatility_threshold * 2:
                probabilities['HIGH_VOLATILITY'] = min(volatility * 20, 0.8)
                probabilities['LOW_VOLATILITY'] = 0.0
                probabilities['VOLATILE'] = min(volatility * 15, 0.6)
            elif volatility < self.volatility_threshold * 0.5:
                probabilities['HIGH_VOLATILITY'] = 0.0
                probabilities['LOW_VOLATILITY'] = min(1.0 - volatility * 50, 0.8)
                probabilities['VOLATILE'] = 0.0
            else:
                probabilities['VOLATILE'] = min(volatility * 10, 0.5)
            
            # Crisis/recovery probabilities
            if price_change < self.crisis_threshold:
                probabilities['CRISIS'] = min(abs(price_change) * 10, 0.9)
                probabilities['RECOVERY'] = 0.0
            elif price_change > self.recovery_threshold:
                probabilities['CRISIS'] = 0.0
                probabilities['RECOVERY'] = min(price_change * 15, 0.8)
            
            # Normalize probabilities
            total = sum(probabilities.values())
            if total > 0:
                for key in probabilities:
                    probabilities[key] /= total
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating regime probabilities: {e}")
            return {'RANGING': 1.0}
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate rolling volatility"""
        try:
            if 'close' not in market_data.columns:
                return 0.02
            
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < 2:
                return 0.02
            
            volatility = returns.std()
            return max(volatility, 0.001)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.02
    
    def _calculate_trend(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength and direction"""
        try:
            if 'close' not in market_data.columns:
                return 0.0
            
            prices = market_data['close'].values
            if len(prices) < 2:
                return 0.0
            
            # Linear regression slope
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            
            # Normalize by average price
            avg_price = np.mean(prices)
            normalized_slope = slope / avg_price if avg_price > 0 else 0.0
            
            return normalized_slope
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 0.0
    
    def _calculate_price_change(self, market_data: pd.DataFrame) -> float:
        """Calculate overall price change"""
        try:
            if 'close' not in market_data.columns:
                return 0.0
            
            prices = market_data['close'].values
            if len(prices) < 2:
                return 0.0
            
            return (prices[-1] - prices[0]) / prices[0]
            
        except Exception as e:
            logger.error(f"Error calculating price change: {e}")
            return 0.0
    
    def _classify_regime(self, volatility: float, trend: float, price_change: float) -> MarketRegime:
        """Classify market regime based on metrics"""
        try:
            # Crisis detection
            if price_change < self.crisis_threshold:
                return MarketRegime.CRISIS
            
            # Recovery detection
            if price_change > self.recovery_threshold:
                return MarketRegime.RECOVERY
            
            # Volatility-based classification
            if volatility > self.volatility_threshold * 2:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < self.volatility_threshold * 0.5:
                return MarketRegime.LOW_VOLATILITY
            
            # Trend-based classification
            if abs(trend) > self.trend_threshold:
                if trend > 0:
                    return MarketRegime.TRENDING_UP
                else:
                    return MarketRegime.TRENDING_DOWN
            
            # Volatile regime
            if volatility > self.volatility_threshold:
                return MarketRegime.VOLATILE
            
            # Default to ranging
            return MarketRegime.RANGING
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return MarketRegime.RANGING


if __name__ == "__main__":
    """Test the market regime detector"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    test_data = pd.DataFrame({
        'close': [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 
                  1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20]
    })
    
    # Create detector
    detector = MarketRegimeDetector()
    
    # Test regime detection
    regime = detector.detect_current_regime(test_data)
    print(f"Current regime: {regime.value}")
    
    # Test probabilities
    probabilities = detector.get_regime_probabilities(test_data)
    print("Regime probabilities:")
    for regime_name, probability in probabilities.items():
        print(f"  {regime_name}: {probability:.2f}")
