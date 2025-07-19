"""
Momentum Strategy Template

Specialized momentum strategy implementation using rate of change,
relative strength, and momentum indicators.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.sensory.core.base import MarketData
from src.sensory.dimensions.how.indicators import TechnicalIndicators
from src.trading.strategy_engine.base_strategy import BaseStrategy, StrategyType, StrategySignal, SignalType

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy Template
    
    Implements momentum trading using multiple techniques:
    - Rate of Change (ROC)
    - Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Momentum Oscillators
    - Volume Momentum
    """
    
    def __init__(self, strategy_id: str, parameters: Dict[str, Any], symbols: List[str]):
        super().__init__(strategy_id, StrategyType.MOMENTUM, parameters, symbols)
        
        # Strategy-specific parameters
        self.roc_period = parameters.get('roc_period', 10)
        self.rsi_period = parameters.get('rsi_period', 14)
        self.macd_fast = parameters.get('macd_fast', 12)
        self.macd_slow = parameters.get('macd_slow', 26)
        self.macd_signal = parameters.get('macd_signal', 9)
        self.momentum_period = parameters.get('momentum_period', 10)
        self.volume_momentum_period = parameters.get('volume_momentum_period', 5)
        self.momentum_threshold = parameters.get('momentum_threshold', 0.02)
        
        logger.info(f"MomentumStrategy {strategy_id} initialized")
    
    async def generate_signal(self, market_data: List[MarketData], symbol: str) -> Optional[StrategySignal]:
        """Generate momentum signal based on multiple indicators"""
        if len(market_data) < max(self.roc_period, self.macd_slow):
            return None
        
        # Calculate technical indicators
        prices = [md.close for md in market_data]
        volumes = [md.volume for md in market_data]
        
        # Rate of Change
        roc = self._calculate_roc(prices)
        
        # RSI
        rsi = self._calculate_rsi(prices, self.rsi_period)
        
        # MACD
        macd_line, macd_signal = self._calculate_macd(prices)
        
        # Momentum oscillator
        momentum = self._calculate_momentum(prices)
        
        # Volume momentum
        volume_momentum = self._calculate_volume_momentum(volumes)
        
        # Momentum strength
        momentum_strength = self._calculate_momentum_strength(prices)
        
        current_price = market_data[-1].close
        
        # Generate signal based on multiple confirmations
        signal = self._evaluate_signals(
            current_price, roc, rsi, macd_line, macd_signal,
            momentum, volume_momentum, momentum_strength
        )
        
        return signal
    
    def _calculate_roc(self, prices: List[float]) -> float:
        """Calculate Rate of Change"""
        if len(prices) < self.roc_period + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-self.roc_period - 1]
        
        if past_price == 0:
            return 0.0
        
        roc = ((current_price - past_price) / past_price) * 100
        return roc
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: List[float]) -> tuple:
        """Calculate MACD line and signal"""
        if len(prices) < self.macd_slow:
            return 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, self.macd_fast)
        ema_slow = self._calculate_ema(prices, self.macd_slow)
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        macd_values = [macd_line]  # Simplified for this example
        macd_signal = self._calculate_ema(macd_values, self.macd_signal)
        
        return macd_line, macd_signal
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1]
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate momentum oscillator"""
        if len(prices) < self.momentum_period + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-self.momentum_period - 1]
        
        return current_price - past_price
    
    def _calculate_volume_momentum(self, volumes: List[float]) -> float:
        """Calculate volume momentum"""
        if len(volumes) < self.volume_momentum_period + 1:
            return 0.0
        
        current_volume = volumes[-1]
        past_volume = volumes[-self.volume_momentum_period - 1]
        
        if past_volume == 0:
            return 0.0
        
        return (current_volume - past_volume) / past_volume
    
    def _calculate_momentum_strength(self, prices: List[float]) -> float:
        """Calculate momentum strength using multiple timeframes"""
        if len(prices) < 20:
            return 0.0
        
        # Calculate momentum across different periods
        momentum_5 = self._calculate_momentum(prices[-5:]) if len(prices) >= 5 else 0
        momentum_10 = self._calculate_momentum(prices[-10:]) if len(prices) >= 10 else 0
        momentum_20 = self._calculate_momentum(prices[-20:]) if len(prices) >= 20 else 0
        
        # Weighted average of momentum across timeframes
        momentum_strength = (momentum_5 * 0.5 + momentum_10 * 0.3 + momentum_20 * 0.2)
        
        # Normalize by average price
        avg_price = np.mean(prices[-20:])
        if avg_price > 0:
            momentum_strength = momentum_strength / avg_price
        
        return momentum_strength
    
    def _evaluate_signals(self, current_price: float, roc: float, rsi: float,
                         macd_line: float, macd_signal: float, momentum: float,
                         volume_momentum: float, momentum_strength: float) -> Optional[StrategySignal]:
        """Evaluate multiple signals to generate trading decision"""
        
        # Buy conditions (positive momentum)
        buy_signals = 0
        total_buy_signals = 0
        
        # Rate of Change positive
        if roc > self.momentum_threshold:
            buy_signals += 1
        total_buy_signals += 1
        
        # RSI not overbought
        if 30 < rsi < 70:
            buy_signals += 1
        total_buy_signals += 1
        
        # MACD bullish
        if macd_line > macd_signal:
            buy_signals += 1
        total_buy_signals += 1
        
        # Momentum positive
        if momentum > 0:
            buy_signals += 1
        total_buy_signals += 1
        
        # Volume momentum positive
        if volume_momentum > 0:
            buy_signals += 1
        total_buy_signals += 1
        
        # Momentum strength positive
        if momentum_strength > 0:
            buy_signals += 1
        total_buy_signals += 1
        
        # Sell conditions (negative momentum)
        sell_signals = 0
        total_sell_signals = 0
        
        # Rate of Change negative
        if roc < -self.momentum_threshold:
            sell_signals += 1
        total_sell_signals += 1
        
        # RSI not oversold
        if 30 < rsi < 70:
            sell_signals += 1
        total_sell_signals += 1
        
        # MACD bearish
        if macd_line < macd_signal:
            sell_signals += 1
        total_sell_signals += 1
        
        # Momentum negative
        if momentum < 0:
            sell_signals += 1
        total_sell_signals += 1
        
        # Volume momentum negative
        if volume_momentum < 0:
            sell_signals += 1
        total_sell_signals += 1
        
        # Momentum strength negative
        if momentum_strength < 0:
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
                    'roc': roc,
                    'rsi': rsi,
                    'macd_line': macd_line,
                    'macd_signal': macd_signal,
                    'momentum': momentum,
                    'volume_momentum': volume_momentum,
                    'momentum_strength': momentum_strength
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
                    'roc': roc,
                    'rsi': rsi,
                    'macd_line': macd_line,
                    'macd_signal': macd_signal,
                    'momentum': momentum,
                    'volume_momentum': volume_momentum,
                    'momentum_strength': momentum_strength
                }
            )
        
        return None
    
    def calculate_position_size(self, signal: StrategySignal, available_capital: float) -> float:
        """Calculate position size based on momentum strength and confidence"""
        base_size = available_capital * self.parameters.get('risk_per_trade', 0.02)
        
        # Adjust based on signal confidence
        confidence_multiplier = signal.confidence
        
        # Adjust based on momentum strength
        momentum_strength = abs(signal.metadata.get('momentum_strength', 0.0))
        strength_multiplier = min(momentum_strength * 100, 2.0)
        
        # Adjust based on ROC
        roc = abs(signal.metadata.get('roc', 0.0))
        roc_multiplier = min(roc / 10, 2.0)  # Normalize ROC
        
        position_size = base_size * confidence_multiplier * strength_multiplier * roc_multiplier
        
        # Apply maximum position size limit
        max_size = available_capital * self.parameters.get('max_position_size', 0.1)
        return min(position_size, max_size)
    
    def should_exit_position(self, position: Dict[str, Any], market_data: List[MarketData]) -> bool:
        """Determine if position should be closed based on momentum reversal"""
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
        
        # Check momentum reversal
        prices = [md.close for md in market_data[-20:]]
        if len(prices) >= 10:
            roc = self._calculate_roc(prices)
            rsi = self._calculate_rsi(prices, self.rsi_period)
            macd_line, macd_signal = self._calculate_macd(prices)
            
            # Exit long position when momentum weakens
            if position['quantity'] > 0:
                if roc < 0 or rsi > 70 or macd_line < macd_signal:
                    return True
            
            # Exit short position when momentum weakens
            elif position['quantity'] < 0:
                if roc > 0 or rsi < 30 or macd_line > macd_signal:
                    return True
        
        return False 