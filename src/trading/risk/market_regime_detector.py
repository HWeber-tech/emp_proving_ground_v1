#!/usr/bin/env python3
"""
Market Regime Detection System - Phase 2B Implementation

This module implements sophisticated market regime detection using
statistical analysis, rule-based detection, and momentum analysis.

Author: EMP Development Team
Phase: 2B - Adaptive Risk Management
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
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
    """Characteristics of a market regime"""
    volatility_level: float  # 0-1 scale
    trend_strength: float  # -1 to 1
    momentum: float  # -1 to 1
    volume_ratio: float  # vs average
    correlation_strength: float  # 0-1
    duration_days: int
    confidence: float  # 0-1
    indicators: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegimeDetectionResult:
    """Result of regime detection"""
    regime: MarketRegime
    confidence: float
    regime_scores: Dict[MarketRegime, float]
    characteristics: RegimeCharacteristics
    detection_details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class MarketRegimeDetector:
    """
    Advanced market regime detection system using multiple detection engines.
    
    Combines statistical analysis, rule-based detection, and momentum analysis
    to identify current market conditions with high accuracy.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.lookback_period = self.config.get('lookback_period', 252)  # 1 year
        self.min_regime_duration = self.config.get('min_regime_duration', 5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Detection parameters
        self.volatility_window = self.config.get('volatility_window', 20)
        self.trend_window = self.config.get('trend_window', 50)
        self.momentum_window = self.config.get('momentum_window', 14)
        
        # Historical regime tracking
        self.regime_history: List[RegimeDetectionResult] = []
        
        logger.info("MarketRegimeDetector initialized")
    
    async def detect_regime(self, market_data: pd.DataFrame) -> RegimeDetectionResult:
        """
        Detect current market regime using multiple detection engines.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            RegimeDetectionResult with detected regime and confidence
        """
        try:
            # Ensure we have enough data
            if len(market_data) < self.lookback_period:
                logger.warning("Insufficient data for regime detection")
                return self._get_default_result()
            
            # Run detection engines
            statistical_result = await self._statistical_detection(market_data)
            rule_based_result = await self._rule_based_detection(market_data)
            momentum_result = await self._momentum_detection(market_data)
            
            # Combine results using weighted voting
            combined_result = await self._combine_detection_results(
                statistical_result, rule_based_result, momentum_result
            )
            
            # Store in history
            self.regime_history.append(combined_result)
            
            # Clean old history
            self._cleanup_history()
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return self._get_default_result()
    
    async def _statistical_detection(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Statistical regime detection using volatility and trend analysis."""
        try:
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            
            # Volatility analysis
            volatility = returns.rolling(window=self.volatility_window).std()
            avg_volatility = volatility.mean()
            current_volatility = volatility.iloc[-1]
            
            # Trend analysis using linear regression
            prices = market_data['close'].values
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            trend_strength = abs(slope) / np.std(prices)
            
            # Regime probability calculation
            regime_probabilities = {}
            
            # Volatility-based regime assignment
            vol_ratio = current_volatility / avg_volatility if avg_volatility > 0 else 1.0
            
            if vol_ratio > 2.0:
                regime_probabilities[MarketRegime.HIGH_VOLATILITY] = 0.8
                regime_probabilities[MarketRegime.CRISIS] = 0.6
            elif vol_ratio < 0.5:
                regime_probabilities[MarketRegime.LOW_VOLATILITY] = 0.8
            else:
                regime_probabilities[MarketRegime.VOLATILE] = 0.5
            
            # Trend-based regime assignment
            if slope > 0:
                regime_probabilities[MarketRegime.TRENDING_UP] = min(0.9, trend_strength * 10)
            elif slope < 0:
                regime_probabilities[MarketRegime.TRENDING_DOWN] = min(0.9, trend_strength * 10)
            else:
                regime_probabilities[MarketRegime.RANGING] = 0.7
            
            # Normalize probabilities
            total_prob = sum(regime_probabilities.values())
            if total_prob > 0:
                regime_probabilities = {k: v/total_prob for k, v in regime_probabilities.items()}
            
            return {
                'regime_probabilities': regime_probabilities,
                'confidence': 0.8,
                'characteristics': {
                    'volatility_level': min(1.0, current_volatility / avg_volatility) if avg_volatility > 0 else 0.5,
                    'trend_strength': min(1.0, trend_strength),
                    'momentum': 0.5,
                    'volume_ratio': 1.0,
                    'correlation_strength': 0.5,
                    'duration_days': len(market_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in statistical detection: {e}")
            return self._get_empty_detection_result()
    
    async def _rule_based_detection(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Rule-based regime detection using technical indicators."""
        try:
            close_prices = market_data['close']
            high_prices = market_data['high']
            low_prices = market_data['low']
            volume = market_data['volume']
            
            # Calculate technical indicators
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            sma_200 = close_prices.rolling(window=200).mean()
            
            # RSI calculation
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # ATR calculation
            tr = pd.DataFrame({
                'h-l': high_prices - low_prices,
                'h-pc': abs(high_prices - close_prices.shift()),
                'l-pc': abs(low_prices - close_prices.shift())
            }).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            # Current values
            current_price = close_prices.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_atr = atr.iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            current_sma_200 = sma_200.iloc[-1]
            
            # Regime rules
            regime_probabilities = {}
            
            # Trend rules
            if current_price > current_sma_20 > current_sma_50 > current_sma_200:
                regime_probabilities[MarketRegime.TRENDING_UP] = 0.9
            elif current_price < current_sma_20 < current_sma_50 < current_sma_200:
                regime_probabilities[MarketRegime.TRENDING_DOWN] = 0.9
            
            # Volatility rules
            atr_ratio = current_atr / current_price if current_price > 0 else 0
            if atr_ratio > 0.05:  # High volatility
                regime_probabilities[MarketRegime.HIGH_VOLATILITY] = 0.8
            elif atr_ratio < 0.01:  # Low volatility
                regime_probabilities[MarketRegime.LOW_VOLATILITY] = 0.8
            
            # RSI rules
            if 30 <= current_rsi <= 70:
                regime_probabilities[MarketRegime.RANGING] = 0.7
            
            # Crisis rules
            if current_rsi < 20 or current_rsi > 80:
                regime_probabilities[MarketRegime.CRISIS] = 0.6
            
            # Normalize probabilities
            total_prob = sum(regime_probabilities.values())
            if total_prob > 0:
                regime_probabilities = {k: v/total_prob for k, v in regime_probabilities.items()}
            
            return {
                'regime_probabilities': regime_probabilities,
                'confidence': 0.85,
                'characteristics': {
                    'volatility_level': min(1.0, atr_ratio * 20),
                    'trend_strength': 0.5,
                    'momentum': 0.5,
                    'volume_ratio': 1.0,
                    'correlation_strength': 0.5,
                    'duration_days': len(market_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based detection: {e}")
            return self._get_empty_detection_result()
    
    async def _momentum_detection(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Momentum-based regime detection using price momentum indicators."""
        try:
            close_prices = market_data['close']
            
            # Calculate momentum indicators
            # Rate of Change (ROC)
            roc = (close_prices - close_prices.shift(10)) / close_prices.shift(10) * 100
            
            # Moving Average Convergence Divergence (MACD)
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            
            # Current values
            current_roc = roc.iloc[-1]
            current_macd = macd.iloc[-1]
            current_macd_signal = macd_signal.iloc[-1]
            
            # Momentum calculation
            momentum_score = 0.5  # Neutral default
            
            # ROC momentum
            if current_roc > 2:
                momentum_score += 0.3
            elif current_roc < -2:
                momentum_score -= 0.3
            
            # MACD momentum
            if current_macd > current_macd_signal:
                momentum_score += 0.2
            elif current_macd < current_macd_signal:
                momentum_score -= 0.2
            
            # Regime probability calculation
            regime_probabilities = {}
            
            if momentum_score > 0.7:
                regime_probabilities[MarketRegime.TRENDING_UP] = 0.8
            elif momentum_score < 0.3:
                regime_probabilities[MarketRegime.TRENDING_DOWN] = 0.8
            else:
                regime_probabilities[MarketRegime.RANGING] = 0.7
            
            return {
                'regime_probabilities': regime_probabilities,
                'confidence': 0.75,
                'characteristics': {
                    'volatility_level': 0.5,
                    'trend_strength': abs(momentum_score - 0.5) * 2,
                    'momentum': momentum_score,
                    'volume_ratio': 1.0,
                    'correlation_strength': 0.5,
                    'duration_days': len(market_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in momentum detection: {e}")
            return self._get_empty_detection_result()
    
    async def _combine_detection_results(self, stat_result: Dict[str, Any], 
                                     rule_result: Dict[str, Any], 
                                     momentum_result: Dict[str, Any]) -> RegimeDetectionResult:
        """Combine results from multiple detection engines using weighted voting."""
        try:
            # Define weights for each detection engine
            weights = {
                'statistical': 0.4,
                'rule_based': 0.35,
                'momentum': 0.25
            }
            
            # Combine regime probabilities
            combined_probabilities = {}
            
            for regime in MarketRegime:
                prob = 0.0
                
                # Add weighted probabilities from each engine
                stat_probs = stat_result.get('regime_probabilities', {})
                rule_probs = rule_result.get('regime_probabilities', {})
                mom_probs = momentum_result.get('regime_probabilities', {})
                
                prob += stat_probs.get(regime, 0) * weights['statistical']
                prob += rule_probs.get(regime, 0) * weights['rule_based']
                prob += mom_probs.get(regime, 0) * weights['momentum']
                
                combined_probabilities[regime] = prob
            
            # Select regime with highest combined probability
            detected_regime = max(combined_probabilities, key=combined_probabilities.get)
            confidence = combined_probabilities[detected_regime]
            
            # Create characteristics
            characteristics = RegimeCharacteristics(
                volatility_level=np.mean([
                    stat_result['characteristics']['volatility_level'],
                    rule_result['characteristics']['volatility_level'],
                    momentum_result['characteristics']['volatility_level']
                ]),
                trend_strength=np.mean([
                    stat_result['characteristics']['trend_strength'],
                    rule_result['characteristics']['trend_strength'],
                    momentum_result['characteristics']['trend_strength']
                ]),
                momentum=np.mean([
                    stat_result['characteristics']['momentum'],
                    rule_result['characteristics']['momentum'],
                    momentum_result['characteristics']['momentum']
                ]),
                volume_ratio=1.0,
                correlation_strength=0.5,
                duration_days=len(market_data),
                confidence=confidence
            )
            
            return RegimeDetectionResult(
                regime=detected_regime,
                confidence=confidence,
                regime_scores=combined_probabilities,
                characteristics=characteristics,
                detection_details={
                    'statistical': stat_result,
                    'rule_based': rule_result,
                    'momentum': momentum_result
                }
            )
            
        except Exception as e:
            logger.error(f"Error combining detection results: {e}")
            return self._get_default_result()
    
    def _get_default_result(self) -> RegimeDetectionResult:
        """Get default regime detection result."""
        return RegimeDetectionResult(
            regime=MarketRegime.RANGING,
            confidence=0.5,
            regime_scores={regime: 0.125 for regime in MarketRegime},
            characteristics=RegimeCharacteristics(
                volatility_level=0.5,
                trend_strength=0.0,
                momentum=0.5,
                volume_ratio=1.0,
                correlation_strength=0.5,
                duration_days=0,
                confidence=0.5
            )
        )
    
    def _get_empty_detection_result(self) -> Dict[str, Any]:
        """Get empty detection result for error cases."""
        return {
            'regime_probabilities': {regime: 0.125 for regime in MarketRegime},
            'confidence': 0.5,
            'characteristics': {
                'volatility_level': 0.5,
                'trend_strength': 0.0,
                'momentum': 0.5,
                'volume_ratio': 1.0,
                'correlation_strength': 0.5,
                'duration_days': 0
            }
        }
    
    def _cleanup_history(self):
        """Clean old regime history."""
        cutoff_date = datetime.now() - timedelta(days=365)
        self.regime_history = [
            r for r in self.regime_history 
            if r.timestamp > cutoff_date
        ]
    
    def get_current_regime(self) -> Optional[RegimeDetectionResult]:
        """Get the most recent regime detection."""
        if self.regime_history:
            return self.regime_history[-1]
        return None
    
    def get_regime_history(self, days: int = 30) -> List[RegimeDetectionResult]:
        """Get regime history for specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            r for r in self.regime_history 
            if r.timestamp > cutoff_date
        ]
    
    def get_regime_statistics(self, regime: MarketRegime, days: int = 30) -> Dict[str, Any]:
        """Get statistics for a specific regime."""
        recent_regimes = self.get_regime_history(days)
        regime_occurrences = [
            r for r in recent_regimes 
            if r.regime == regime
        ]
        
        if not regime_occurrences:
            return {
                'count': 0,
                'average_duration': 0,
                'average_confidence': 0.0,
                'frequency': 0.0
            }
        
        return {
            'count': len(regime_occurrences),
            'average_duration': np.mean([r.characteristics.duration_days for r in regime_occurrences]),
            'average_confidence': np.mean([r.confidence for r in regime_occurrences]),
            'frequency': len(regime_occurrences) / len(recent_regimes) if recent_regimes else 0.0
        }


async def main():
    """Test the market regime detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test market regime detector")
    parser.add_argument("--test-detection", action="store_true", help="Test regime detection")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create detector
    detector = MarketRegimeDetector()
    
    if args.test_detection:
        print("Testing regime detection...")
        
        # Create test market data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Simulate trending up market
        trend = np.linspace(1.0, 1.3, 252)
        noise = np.random.normal(0, 0.02, 252)
        prices = 100 * trend * (1 + noise)
        
        market_data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 5000, 252)
        }, index=dates)
        
        # Detect regime
        result = await detector.detect_regime(market_data)
        
        print(f"Detected regime: {result.regime.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Characteristics: {result.characteristics}")
        
        for regime, score in result.regime_scores.items():
            print(f"  {regime.value}: {score:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
