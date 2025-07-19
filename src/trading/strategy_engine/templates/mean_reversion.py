"""
Mean Reversion Strategy Template

Specialized mean reversion strategy implementation using sensory layer indicators.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.sensory.core.base import MarketData
from src.sensory.dimensions.how.indicators import TechnicalIndicators
from src.trading.strategy_engine.base_strategy import BaseStrategy, StrategyType, StrategySignal, SignalType

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy Template
    
    Implements mean reversion using sensory layer indicators:
    - Bollinger Bands (from sensory layer)
    - RSI Divergence (from sensory layer)
    - Statistical Arbitrage
    - Z-Score Analysis
    - Volume Confirmation
    """
    
    def __init__(self, strategy_id: str, parameters: Dict[str, Any], symbols: List[str]):
        super().__init__(strategy_id, StrategyType.MEAN_REVERSION, parameters, symbols)
        
        # Strategy-specific parameters
        self.lookback_period = parameters.get('lookback_period', 20)
        self.bollinger_period = parameters.get('bollinger_period', 20)
        self.rsi_period = parameters.get('rsi_period', 14)
        self.rsi_oversold = parameters.get('rsi_oversold', 30)
        self.rsi_overbought = parameters.get('rsi_overbought', 70)
        self.z_score_threshold = parameters.get('z_score_threshold', 2.0)
        self.volume_threshold = parameters.get('volume_threshold', 1.2)
        
        # Initialize sensory layer indicators
        self.indicators = TechnicalIndicators()
        
        logger.info(f"MeanReversionStrategy {strategy_id} initialized")
    
    async def generate_signal(self, market_data: List[MarketData], symbol: str) -> Optional[StrategySignal]:
        """Generate mean reversion signal using sensory layer indicators"""
        if len(market_data) < self.lookback_period:
            return None
        
        # Convert market data to DataFrame for sensory layer
        df = self._market_data_to_dataframe(market_data)
        
        # Get indicators from sensory layer
        indicators = self.indicators.calculate_indicators(df, ['bollinger_bands', 'rsi'])
        
        # Extract indicator values
        bb_upper = indicators.get('bb_upper', None)
        bb_middle = indicators.get('bb_middle', None)
        bb_lower = indicators.get('bb_lower', None)
        rsi = indicators.get('rsi', None)
        
        if not all([bb_upper is not None, bb_middle is not None, bb_lower is not None, rsi is not None]):
            return None
        
        # Get current values
        current_bb_upper = bb_upper.iloc[-1] if bb_upper is not None else 0.0
        current_bb_middle = bb_middle.iloc[-1] if bb_middle is not None else 0.0
        current_bb_lower = bb_lower.iloc[-1] if bb_lower is not None else 0.0
        current_rsi = rsi.iloc[-1] if rsi is not None else 50.0
        
        # Z-Score calculation
        z_score = self._calculate_z_score(market_data)
        
        # Volume analysis
        volume_ratio = self._analyze_volume(market_data)
        
        # Mean reversion strength
        mean_reversion_strength = self._calculate_mean_reversion_strength(market_data)
        
        current_price = market_data[-1].close
        
        # Generate signal based on multiple confirmations
        signal = self._evaluate_signals(
            current_price, current_bb_upper, current_bb_middle, current_bb_lower, current_rsi, 
            z_score, volume_ratio, mean_reversion_strength
        )
        
        return signal
    
    def _market_data_to_dataframe(self, market_data: List[MarketData]):
        """Convert market data to DataFrame for sensory layer"""
        import pandas as pd
        
        data = {
            'open': [md.open for md in market_data],
            'high': [md.high for md in market_data],
            'low': [md.low for md in market_data],
            'close': [md.close for md in market_data],
            'volume': [md.volume for md in market_data]
        }
        
        return pd.DataFrame(data)
    
    def _calculate_z_score(self, market_data: List[MarketData]) -> float:
        """Calculate Z-Score for mean reversion"""
        if len(market_data) < self.lookback_period:
            return 0.0
        
        prices = [md.close for md in market_data]
        recent_prices = prices[-self.lookback_period:]
        
        mean_price = sum(recent_prices) / len(recent_prices)
        variance = sum((p - mean_price) ** 2 for p in recent_prices) / len(recent_prices)
        std_price = variance ** 0.5
        
        if std_price == 0:
            return 0.0
        
        current_price = prices[-1]
        z_score = (current_price - mean_price) / std_price
        
        return z_score
    
    def _analyze_volume(self, market_data: List[MarketData]) -> float:
        """Analyze volume relative to average"""
        if len(market_data) < 20:
            return 1.0
        
        volumes = [md.volume for md in market_data]
        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _calculate_mean_reversion_strength(self, market_data: List[MarketData]) -> float:
        """Calculate mean reversion strength using autocorrelation"""
        if len(market_data) < 10:
            return 0.0
        
        # Calculate price changes
        prices = [md.close for md in market_data[-10:]]
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Calculate autocorrelation
        if len(price_changes) > 1:
            # Simple autocorrelation calculation
            mean_change = sum(price_changes) / len(price_changes)
            variance = sum((c - mean_change) ** 2 for c in price_changes) / len(price_changes)
            
            if variance > 0:
                autocorr = sum((price_changes[i] - mean_change) * (price_changes[i-1] - mean_change) 
                              for i in range(1, len(price_changes))) / (len(price_changes) - 1) / variance
                return -autocorr  # Negative autocorrelation indicates mean reversion
            else:
                return 0.0
        else:
            return 0.0
    
    def _evaluate_signals(self, current_price: float, bb_upper: float, bb_middle: float, 
                         bb_lower: float, rsi: float, z_score: float,
                         volume_ratio: float, mean_reversion_strength: float) -> Optional[StrategySignal]:
        """Evaluate multiple signals to generate trading decision"""
        
        # Buy conditions (oversold)
        buy_signals = 0
        total_buy_signals = 0
        
        # Bollinger Band oversold
        if current_price <= bb_lower:
            buy_signals += 1
        total_buy_signals += 1
        
        # RSI oversold
        if rsi <= self.rsi_oversold:
            buy_signals += 1
        total_buy_signals += 1
        
        # Z-Score oversold
        if z_score <= -self.z_score_threshold:
            buy_signals += 1
        total_buy_signals += 1
        
        # Volume confirmation
        if volume_ratio > self.volume_threshold:
            buy_signals += 1
        total_buy_signals += 1
        
        # Mean reversion strength
        if mean_reversion_strength > 0.1:
            buy_signals += 1
        total_buy_signals += 1
        
        # Sell conditions (overbought)
        sell_signals = 0
        total_sell_signals = 0
        
        # Bollinger Band overbought
        if current_price >= bb_upper:
            sell_signals += 1
        total_sell_signals += 1
        
        # RSI overbought
        if rsi >= self.rsi_overbought:
            sell_signals += 1
        total_sell_signals += 1
        
        # Z-Score overbought
        if z_score >= self.z_score_threshold:
            sell_signals += 1
        total_sell_signals += 1
        
        # Volume confirmation
        if volume_ratio > self.volume_threshold:
            sell_signals += 1
        total_sell_signals += 1
        
        # Mean reversion strength
        if mean_reversion_strength > 0.1:
            sell_signals += 1
        total_sell_signals += 1
        
        # Generate signal based on signal strength
        buy_confidence = buy_signals / total_buy_signals if total_buy_signals > 0 else 0
        sell_confidence = sell_signals / total_sell_signals if total_sell_signals > 0 else 0
        
        if buy_confidence > 0.6:  # 60% of signals are bullish
            return StrategySignal(
                strategy_id=self.strategy_id,
                signal_type=SignalType.BUY,
                symbol=self.symbols[0],
                timestamp=datetime.utcnow(),
                price=current_price,
                quantity=1.0,
                confidence=buy_confidence,
                metadata={
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'rsi': rsi,
                    'z_score': z_score,
                    'volume_ratio': volume_ratio,
                    'mean_reversion_strength': mean_reversion_strength
                }
            )
        elif sell_confidence > 0.6:  # 60% of signals are bearish
            return StrategySignal(
                strategy_id=self.strategy_id,
                signal_type=SignalType.SELL,
                symbol=self.symbols[0],
                timestamp=datetime.utcnow(),
                price=current_price,
                quantity=1.0,
                confidence=sell_confidence,
                metadata={
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'rsi': rsi,
                    'z_score': z_score,
                    'volume_ratio': volume_ratio,
                    'mean_reversion_strength': mean_reversion_strength
                }
            )
        
        return None
    
    def calculate_position_size(self, signal: StrategySignal, available_capital: float) -> float:
        """Calculate position size based on mean reversion strength and confidence"""
        base_size = available_capital * self.parameters.get('risk_per_trade', 0.02)
        
        # Adjust based on signal confidence
        confidence_multiplier = signal.confidence
        
        # Adjust based on mean reversion strength
        mean_reversion_strength = signal.metadata.get('mean_reversion_strength', 0.0)
        strength_multiplier = min(abs(mean_reversion_strength) * 10, 2.0)
        
        # Adjust based on Z-Score (stronger signal for extreme values)
        z_score = abs(signal.metadata.get('z_score', 0.0))
        z_score_multiplier = min(z_score / self.z_score_threshold, 2.0)
        
        position_size = base_size * confidence_multiplier * strength_multiplier * z_score_multiplier
        
        # Apply maximum position size limit
        max_size = available_capital * self.parameters.get('max_position_size', 0.1)
        return min(position_size, max_size)
    
    def should_exit_position(self, position: Dict[str, Any], market_data: List[MarketData]) -> bool:
        """Determine if position should be closed based on mean reversion completion"""
        if not market_data:
            return False
        
        current_price = market_data[-1].close
        entry_price = position['entry_price']
        
        # Check stop loss
        stop_loss = self.parameters.get('stop_loss', 0.02)
        if position['quantity'] > 0:  # Long position
            if current_price <= entry_price * (1 - stop_loss):
                return True
        else:  # Short position
            if current_price >= entry_price * (1 + stop_loss):
                return True
        
        # Check mean reversion completion using sensory layer
        df = self._market_data_to_dataframe(market_data[-20:])
        indicators = self.indicators.calculate_indicators(df, ['bollinger_bands', 'rsi'])
        
        bb_middle = indicators.get('bb_middle', None)
        rsi = indicators.get('rsi', None)
        
        if bb_middle is not None and rsi is not None:
            current_bb_middle = bb_middle.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Exit long position when price approaches middle band or RSI becomes overbought
            if position['quantity'] > 0:
                if current_price >= current_bb_middle or current_rsi >= self.rsi_overbought:
                    return True
            
            # Exit short position when price approaches middle band or RSI becomes oversold
            elif position['quantity'] < 0:
                if current_price <= current_bb_middle or current_rsi <= self.rsi_oversold:
                    return True
        
        return False 