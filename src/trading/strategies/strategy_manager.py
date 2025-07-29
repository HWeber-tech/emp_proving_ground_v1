#!/usr/bin/env python3
"""
Strategy Manager - Integration between evolved strategies and live trading

This module manages the connection between evolved strategies from the genetic engine
and the live trading executor, enabling real-time strategy evaluation and selection.
"""

import logging
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from src.core.interfaces import DecisionGenome
from src.sensory.orchestration.master_orchestrator import MasterOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Performance tracking for individual strategies."""
    strategy_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_from_trade(self, trade_pnl: float):
        """Update performance metrics from a single trade."""
        self.total_trades += 1
        self.net_profit += trade_pnl
        
        if trade_pnl > 0:
            self.winning_trades += 1
            self.total_profit += trade_pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(trade_pnl)
        
        # Update derived metrics
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        self.avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
        self.avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0
        
        self.last_updated = datetime.now()


@dataclass
class StrategySignal:
    """Trading signal from an evolved strategy."""
    strategy_id: str
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: float = 0.01
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyManager:
    """
    Manages evolved strategies and their integration with live trading.
    
    This class handles:
    - Loading evolved strategies from the genetic engine
    - Real-time strategy evaluation
    - Dynamic strategy selection
    - Performance tracking
    - Strategy versioning and management
    """
    
    def __init__(self, strategies_dir: str = "strategies", max_strategies: int = 10):
        """
        Initialize the strategy manager.
        
        Args:
            strategies_dir: Directory to store strategy files
            max_strategies: Maximum number of strategies to maintain
        """
        self.strategies_dir = Path(strategies_dir)
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_strategies = max_strategies
        self.strategies: Dict[str, DecisionGenome] = {}
        self.performance: Dict[str, StrategyPerformance] = {}
        self.sensory_cortex: Optional[MasterOrchestrator] = None
        
        # Strategy selection parameters
        self.selection_method = 'performance_ranked'  # 'performance_ranked', 'ensemble', 'regime_based'
        self.min_confidence_threshold = 0.6
        self.performance_lookback_days = 30
        
        # Load existing strategies
        self._load_strategies()
        
        logger.info(f"Strategy manager initialized with {len(self.strategies)} strategies")
    
    def set_sensory_cortex(self, sensory_cortex: MasterOrchestrator):
        """Set the sensory cortex for strategy evaluation."""
        self.sensory_cortex = sensory_cortex
        logger.info("Sensory cortex set for strategy evaluation")
    
    def add_strategy(self, genome: DecisionGenome) -> bool:
        """
        Add a new evolved strategy.
        
        Args:
            genome: DecisionGenome from the evolution engine
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            strategy_id = genome.genome_id
            
            # Check if we're at capacity
            if len(self.strategies) >= self.max_strategies:
                # Remove worst performing strategy
                self._remove_worst_strategy()
            
            # Add the strategy
            self.strategies[strategy_id] = genome
            self.performance[strategy_id] = StrategyPerformance(strategy_id=strategy_id)
            
            # Save to disk
            self._save_strategy(genome)
            
            logger.info(f"Added strategy {strategy_id} (generation {genome.generation})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding strategy: {e}")
            return False
    
    def evaluate_strategies(self, symbol: str, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """
        Evaluate all strategies and generate signals.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            
        Returns:
            List of strategy signals
        """
        signals = []
        
        if not self.sensory_cortex:
            logger.warning("No sensory cortex available for strategy evaluation")
            return signals
        
        try:
            # Get sensory reading
            sensory_reading = self._get_sensory_reading(market_data)
            
            # Evaluate each strategy
            for strategy_id, genome in self.strategies.items():
                try:
                    signal = self._evaluate_strategy(genome, symbol, sensory_reading, market_data)
                    if signal and signal.confidence >= self.min_confidence_threshold:
                        signals.append(signal)
                        
                except Exception as e:
                    logger.error(f"Error evaluating strategy {strategy_id}: {e}")
                    continue
            
            # Sort signals by confidence
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.debug(f"Generated {len(signals)} strategy signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error evaluating strategies: {e}")
            return signals
    
    def select_best_strategy(self, symbol: str, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Select the best strategy for the current market conditions.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data
            
        Returns:
            Best strategy signal or None
        """
        signals = self.evaluate_strategies(symbol, market_data)
        
        if not signals:
            return None
        
        # Apply selection method
        if self.selection_method == 'performance_ranked':
            return self._select_by_performance_ranking(signals)
        elif self.selection_method == 'ensemble':
            return self._select_by_ensemble(signals)
        elif self.selection_method == 'regime_based':
            return self._select_by_regime(signals, market_data)
        else:
            # Default to highest confidence
            return signals[0] if signals else None
    
    def update_strategy_performance(self, strategy_id: str, trade_pnl: float):
        """Update performance metrics for a strategy."""
        if strategy_id in self.performance:
            self.performance[strategy_id].update_from_trade(trade_pnl)
            logger.debug(f"Updated performance for strategy {strategy_id}: P&L {trade_pnl:.2f}")
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[StrategyPerformance]:
        """Get performance metrics for a strategy."""
        return self.performance.get(strategy_id)
    
    def get_top_strategies(self, count: int = 5) -> List[Tuple[str, StrategyPerformance]]:
        """Get top performing strategies."""
        if not self.performance:
            return []
        
        # Sort by net profit
        sorted_strategies = sorted(
            self.performance.items(),
            key=lambda x: x[1].net_profit,
            reverse=True
        )
        
        return sorted_strategies[:count]
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of all strategies."""
        if not self.strategies:
            return {"total_strategies": 0}
        
        total_strategies = len(self.strategies)
        active_strategies = sum(1 for perf in self.performance.values() if perf.total_trades > 0)
        
        # Calculate aggregate performance
        total_profit = sum(perf.net_profit for perf in self.performance.values())
        avg_win_rate = np.mean([perf.win_rate for perf in self.performance.values() if perf.total_trades > 0])
        
        return {
            "total_strategies": total_strategies,
            "active_strategies": active_strategies,
            "total_profit": total_profit,
            "average_win_rate": avg_win_rate,
            "selection_method": self.selection_method,
            "min_confidence_threshold": self.min_confidence_threshold
        }
    
    def _evaluate_strategy(self, genome: DecisionGenome, symbol: str, 
                          sensory_reading, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Evaluate a single strategy and generate a signal.
        
        Args:
            genome: DecisionGenome to evaluate
            symbol: Trading symbol
            sensory_reading: Sensory reading from cortex
            market_data: Current market data
            
        Returns:
            Strategy signal or None
        """
        try:
            # Evaluate the decision tree
            decision = genome._evaluate_decision_tree(sensory_reading)
            
            if decision == 'HOLD':
                return None
            
            # Get current price
            current_price = market_data.get('close', market_data.get('bid', 1.1000))
            
            # Calculate confidence based on sensory reading strength
            confidence = self._calculate_confidence(sensory_reading, genome)
            
            # Generate signal
            signal = StrategySignal(
                strategy_id=genome.genome_id,
                symbol=symbol,
                action='buy' if decision == 'BUY' else 'sell',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=self._calculate_stop_loss(current_price, decision, genome),
                take_profit=self._calculate_take_profit(current_price, decision, genome),
                volume=self._calculate_position_size(genome),
                metadata={
                    'generation': genome.generation,
                    'fitness_score': genome.fitness_score,
                    'robustness_score': genome.robustness_score,
                    'decision': decision
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error evaluating strategy {genome.genome_id}: {e}")
            return None
    
    def _calculate_confidence(self, sensory_reading, genome: DecisionGenome) -> float:
        """Calculate confidence level for a strategy signal."""
        try:
            # Base confidence from genome fitness
            base_confidence = min(0.9, genome.fitness_score * 0.5 + 0.3)
            
            # Adjust based on sensory reading strength
            strength_multiplier = 1.0
            
            if hasattr(sensory_reading, 'macro_strength'):
                strength_multiplier *= (0.5 + sensory_reading.macro_strength * 0.5)
            
            if hasattr(sensory_reading, 'institutional_confidence'):
                strength_multiplier *= (0.5 + sensory_reading.institutional_confidence * 0.5)
            
            if hasattr(sensory_reading, 'overall_sentiment'):
                strength_multiplier *= (0.5 + sensory_reading.overall_sentiment * 0.5)
            
            # Cap confidence
            confidence = min(0.95, base_confidence * strength_multiplier)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_stop_loss(self, entry_price: float, decision: str, genome: DecisionGenome) -> float:
        """Calculate stop loss price."""
        try:
            # Get stop loss parameters from genome
            params = genome.decision_tree.get('parameters', {})
            stop_loss_pct = params.get('stop_loss_pct', 0.01)  # Default 1%
            
            if decision == 'BUY':
                return entry_price * (1 - stop_loss_pct)
            else:
                return entry_price * (1 + stop_loss_pct)
                
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            # Default stop loss
            return entry_price * 0.99 if decision == 'BUY' else entry_price * 1.01
    
    def _calculate_take_profit(self, entry_price: float, decision: str, genome: DecisionGenome) -> float:
        """Calculate take profit price."""
        try:
            # Get take profit parameters from genome
            params = genome.decision_tree.get('parameters', {})
            take_profit_pct = params.get('take_profit_pct', 0.02)  # Default 2%
            
            if decision == 'BUY':
                return entry_price * (1 + take_profit_pct)
            else:
                return entry_price * (1 - take_profit_pct)
                
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            # Default take profit
            return entry_price * 1.02 if decision == 'BUY' else entry_price * 0.98
    
    def _calculate_position_size(self, genome: DecisionGenome) -> float:
        """Calculate position size based on genome parameters."""
        try:
            params = genome.decision_tree.get('parameters', {})
            base_size = params.get('position_size', 0.01)  # Default 0.01 lots
            
            # Adjust based on fitness score
            fitness_multiplier = 0.5 + genome.fitness_score * 0.5
            
            return base_size * fitness_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01
    
    def _get_sensory_reading(self, market_data: Dict[str, Any]):
        """Get sensory reading from the sensory cortex."""
        try:
            if not self.sensory_cortex:
                return self._create_simple_reading(market_data)
            
            # Create MarketData object for sensory cortex
            from src.sensory.core.base import MarketData
            
            sensory_data = MarketData(
                symbol=market_data.get('symbol', 'EURUSD'),
                timestamp=market_data.get('timestamp', datetime.now()),
                open=market_data.get('open', market_data.get('close', 1.1000)),
                high=market_data.get('high', market_data.get('close', 1.1000)),
                low=market_data.get('low', market_data.get('close', 1.1000)),
                close=market_data.get('close', 1.1000),
                volume=market_data.get('volume', 1000),
                bid=market_data.get('bid', market_data.get('low', market_data.get('close', 1.1000))),
                ask=market_data.get('ask', market_data.get('high', market_data.get('close', 1.1002))),
                source="live",
                latency_ms=0.0
            )
            
            # Get sensory reading (this would be async in real usage)
            # For now, create a simple reading
            return self._create_simple_reading(market_data)
            
        except Exception as e:
            logger.error(f"Error getting sensory reading: {e}")
            return self._create_simple_reading(market_data)
    
    def _create_simple_reading(self, market_data: Dict[str, Any]):
        """Create a simple sensory reading when cortex is not available."""
        class SimpleReading:
            def __init__(self):
                self.macro_trend = 'NEUTRAL'
                self.macro_strength = 0.5
                self.technical_signal = 'HOLD'
                self.technical_strength = 0.5
                self.institutional_flow = 0.0
                self.institutional_confidence = 0.5
                self.momentum_score = 0.0
                self.overall_sentiment = 0.5
                self.manipulation_probability = 0.1
        
        reading = SimpleReading()
        
        # Simple trend detection based on price movement
        if 'close' in market_data and hasattr(self, '_prev_close'):
            change = market_data['close'] - self._prev_close
            if change > 0.0001:
                reading.macro_trend = 'BULLISH'
                reading.macro_strength = min(0.8, abs(change) * 1000)
                reading.technical_signal = 'BUY'
            elif change < -0.0001:
                reading.macro_trend = 'BEARISH'
                reading.macro_strength = min(0.8, abs(change) * 1000)
                reading.technical_signal = 'SELL'
            
            reading.momentum_score = change * 1000
        
        self._prev_close = market_data.get('close', 1.1000)
        return reading
    
    def _select_by_performance_ranking(self, signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """Select strategy based on performance ranking."""
        if not signals:
            return None
        
        # Get performance data for signals
        scored_signals = []
        for signal in signals:
            perf = self.performance.get(signal.strategy_id)
            if perf and perf.total_trades > 0:
                # Score based on win rate and profit factor
                score = perf.win_rate * 0.4 + min(perf.profit_factor / 10, 1.0) * 0.6
                scored_signals.append((signal, score))
            else:
                # New strategy, use fitness score
                genome = self.strategies.get(signal.strategy_id)
                score = genome.fitness_score if genome else 0.5
                scored_signals.append((signal, score))
        
        # Sort by score and return best
        scored_signals.sort(key=lambda x: x[1], reverse=True)
        return scored_signals[0][0] if scored_signals else signals[0]
    
    def _select_by_ensemble(self, signals: List[StrategySignal]) -> Optional[StrategySignal]:
        """Select strategy using ensemble method."""
        if not signals:
            return None
        
        # Group signals by action
        buy_signals = [s for s in signals if s.action == 'buy']
        sell_signals = [s for s in signals if s.action == 'sell']
        
        if not buy_signals and not sell_signals:
            return None
        
        # Calculate ensemble confidence
        if buy_signals and sell_signals:
            buy_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
            sell_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
            
            if buy_confidence > sell_confidence:
                return buy_signals[0]
            else:
                return sell_signals[0]
        elif buy_signals:
            return buy_signals[0]
        else:
            return sell_signals[0]
    
    def _select_by_regime(self, signals: List[StrategySignal], market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Select strategy based on market regime."""
        # For now, use performance ranking
        # In the future, this could use regime detection from sensory cortex
        return self._select_by_performance_ranking(signals)
    
    def _load_strategies(self):
        """Load strategies from disk."""
        try:
            for strategy_file in self.strategies_dir.glob("*.pkl"):
                try:
                    with open(strategy_file, 'rb') as f:
                        genome = pickle.load(f)
                    
                    strategy_id = genome.genome_id
                    self.strategies[strategy_id] = genome
                    self.performance[strategy_id] = StrategyPerformance(strategy_id=strategy_id)
                    
                except Exception as e:
                    logger.error(f"Error loading strategy from {strategy_file}: {e}")
                    continue
            
            logger.info(f"Loaded {len(self.strategies)} strategies from disk")
            
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
    
    def _save_strategy(self, genome: DecisionGenome):
        """Save strategy to disk."""
        try:
            strategy_file = self.strategies_dir / f"{genome.genome_id}.pkl"
            with open(strategy_file, 'wb') as f:
                pickle.dump(genome, f)
            
        except Exception as e:
            logger.error(f"Error saving strategy {genome.genome_id}: {e}")
    
    def _remove_worst_strategy(self):
        """Remove the worst performing strategy."""
        if not self.performance:
            return
        
        # Find worst performing strategy
        worst_strategy = min(
            self.performance.items(),
            key=lambda x: x[1].net_profit
        )
        
        strategy_id = worst_strategy[0]
        
        # Remove from memory and disk
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
        
        if strategy_id in self.performance:
            del self.performance[strategy_id]
        
        # Remove file
        strategy_file = self.strategies_dir / f"{strategy_id}.pkl"
        if strategy_file.exists():
            strategy_file.unlink()
        
        logger.info(f"Removed worst performing strategy: {strategy_id}")


def main():
    """Test the strategy manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test strategy manager")
    parser.add_argument("--test-evaluation", action="store_true", help="Test strategy evaluation")
    parser.add_argument("--test-selection", action="store_true", help="Test strategy selection")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create strategy manager
    manager = StrategyManager()
    
    if args.test_evaluation:
        print("Testing strategy evaluation...")
        
        # Create mock market data
        market_data = {
            'symbol': 'EURUSD',
            'timestamp': datetime.now(),
            'open': 1.1000,
            'high': 1.1010,
            'low': 1.0990,
            'close': 1.1005,
            'volume': 1000,
            'bid': 1.1004,
            'ask': 1.1006
        }
        
        # Test evaluation
        signals = manager.evaluate_strategies('EURUSD', market_data)
        print(f"Generated {len(signals)} signals")
        
        for signal in signals:
            print(f"Signal: {signal.action} {signal.symbol} (confidence: {signal.confidence:.2%})")
    
    if args.test_selection:
        print("Testing strategy selection...")
        
        # Create mock market data
        market_data = {
            'symbol': 'EURUSD',
            'timestamp': datetime.now(),
            'open': 1.1000,
            'high': 1.1010,
            'low': 1.0990,
            'close': 1.1005,
            'volume': 1000,
            'bid': 1.1004,
            'ask': 1.1006
        }
        
        # Test selection
        best_signal = manager.select_best_strategy('EURUSD', market_data)
        
        if best_signal:
            print(f"Best signal: {best_signal.action} {best_signal.symbol} (confidence: {best_signal.confidence:.2%})")
        else:
            print("No signals generated")


if __name__ == "__main__":
    main() 