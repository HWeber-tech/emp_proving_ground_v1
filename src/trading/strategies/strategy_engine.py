"""
Advanced Trading Strategy Engine for EMP

This module implements sophisticated trading strategy development including
strategy templates, parameter optimization, multi-strategy framework, and
real-time strategy management.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import random
from concurrent.futures import ThreadPoolExecutor

from src.sensory.core.base import MarketData, DimensionalReading
from src.data_integration.real_time_streaming import StreamEvent

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types of trading strategies"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    CUSTOM = "custom"


class StrategyStatus(Enum):
    """Strategy execution status"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class StrategySignal:
    """Trading signal from strategy"""
    strategy_id: str
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyParameters:
    """Strategy parameters for optimization"""
    lookback_period: int = 20
    threshold: float = 0.5
    stop_loss: float = 0.02
    take_profit: float = 0.04
    max_position_size: float = 0.1
    risk_per_trade: float = 0.02
    volatility_lookback: int = 30
    correlation_threshold: float = 0.7
    momentum_period: int = 14
    mean_reversion_period: int = 50
    breakout_threshold: float = 0.02
    scalping_threshold: float = 0.001


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    All strategies must inherit from this class and implement required methods.
    """
    
    def __init__(self, strategy_id: str, strategy_type: StrategyType, 
                 parameters: StrategyParameters, symbols: List[str]):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.parameters = parameters
        self.symbols = symbols
        self.status = StrategyStatus.INACTIVE
        self.performance = StrategyPerformance()
        
        # Strategy state
        self.positions: Dict[str, float] = {}
        self.signals: List[StrategySignal] = []
        self.market_data: Dict[str, List[MarketData]] = {}
        self.last_update = datetime.utcnow()
        
        # Performance tracking
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        
        logger.info(f"Strategy {strategy_id} initialized: {strategy_type.value}")
    
    @abstractmethod
    async def generate_signal(self, market_data: List[MarketData], 
                            symbol: str) -> Optional[StrategySignal]:
        """Generate trading signal based on market data"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: StrategySignal, 
                              available_capital: float) -> float:
        """Calculate position size for the signal"""
        pass
    
    @abstractmethod
    def should_exit_position(self, position: Dict[str, Any], 
                           market_data: List[MarketData]) -> bool:
        """Determine if position should be closed"""
        pass
    
    async def update(self, market_data: List[MarketData], symbol: str) -> None:
        """Update strategy with new market data"""
        if self.status != StrategyStatus.ACTIVE:
            return
        
        # Store market data
        if symbol not in self.market_data:
            self.market_data[symbol] = []
        self.market_data[symbol].extend(market_data)
        
        # Keep only recent data
        max_data_points = max(self.parameters.lookback_period * 2, 1000)
        if len(self.market_data[symbol]) > max_data_points:
            self.market_data[symbol] = self.market_data[symbol][-max_data_points:]
        
        # Generate signal
        signal = await self.generate_signal(market_data, symbol)
        if signal:
            self.signals.append(signal)
            logger.info(f"Strategy {self.strategy_id} generated {signal.signal_type.value} signal for {symbol}")
        
        # Check existing positions
        await self._check_positions(market_data, symbol)
        
        self.last_update = datetime.utcnow()
    
    async def _check_positions(self, market_data: List[MarketData], symbol: str) -> None:
        """Check existing positions for exit conditions"""
        if symbol in self.positions:
            position = {
                'symbol': symbol,
                'quantity': self.positions[symbol],
                'entry_price': self._get_entry_price(symbol),
                'entry_time': self._get_entry_time(symbol)
            }
            
            if self.should_exit_position(position, market_data):
                exit_signal = StrategySignal(
                    strategy_id=self.strategy_id,
                    signal_type=SignalType.CLOSE,
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    price=market_data[-1].close if market_data else 0.0,
                    quantity=abs(self.positions[symbol]),
                    confidence=1.0
                )
                self.signals.append(exit_signal)
                logger.info(f"Strategy {self.strategy_id} closing position for {symbol}")
    
    def _get_entry_price(self, symbol: str) -> float:
        """Get entry price for position"""
        # This would be implemented based on trade history
        return 0.0
    
    def _get_entry_time(self, symbol: str) -> datetime:
        """Get entry time for position"""
        # This would be implemented based on trade history
        return datetime.utcnow()
    
    def start(self) -> None:
        """Start strategy execution"""
        self.status = StrategyStatus.ACTIVE
        logger.info(f"Strategy {self.strategy_id} started")
    
    def stop(self) -> None:
        """Stop strategy execution"""
        self.status = StrategyStatus.STOPPED
        logger.info(f"Strategy {self.strategy_id} stopped")
    
    def pause(self) -> None:
        """Pause strategy execution"""
        self.status = StrategyStatus.PAUSED
        logger.info(f"Strategy {self.strategy_id} paused")
    
    def get_performance(self) -> StrategyPerformance:
        """Get current performance metrics"""
        return self.performance
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """Update performance metrics with trade result"""
        # This would implement performance calculation logic
        pass


class TrendFollowingStrategy(BaseStrategy):
    """Trend following strategy using moving averages"""
    
    def __init__(self, strategy_id: str, parameters: StrategyParameters, symbols: List[str]):
        super().__init__(strategy_id, StrategyType.TREND_FOLLOWING, parameters, symbols)
    
    async def generate_signal(self, market_data: List[MarketData], symbol: str) -> Optional[StrategySignal]:
        if len(market_data) < self.parameters.lookback_period:
            return None
        
        # Calculate moving averages
        prices = [md.close for md in market_data]
        short_ma = np.mean(prices[-self.parameters.lookback_period//2:])
        long_ma = np.mean(prices[-self.parameters.lookback_period:])
        
        current_price = market_data[-1].close
        
        # Generate signal based on moving average crossover
        if short_ma > long_ma and current_price > short_ma:
            return StrategySignal(
                strategy_id=self.strategy_id,
                signal_type=SignalType.BUY,
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=current_price,
                quantity=1.0,
                confidence=min((short_ma - long_ma) / long_ma, 1.0)
            )
        elif short_ma < long_ma and current_price < short_ma:
            return StrategySignal(
                strategy_id=self.strategy_id,
                signal_type=SignalType.SELL,
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=current_price,
                quantity=1.0,
                confidence=min((long_ma - short_ma) / long_ma, 1.0)
            )
        
        return None
    
    def calculate_position_size(self, signal: StrategySignal, available_capital: float) -> float:
        risk_amount = available_capital * self.parameters.risk_per_trade
        position_size = risk_amount / (signal.price * self.parameters.stop_loss)
        return min(position_size, available_capital * self.parameters.max_position_size)
    
    def should_exit_position(self, position: Dict[str, Any], market_data: List[MarketData]) -> bool:
        if not market_data:
            return False
        
        current_price = market_data[-1].close
        entry_price = position['entry_price']
        
        # Check stop loss
        if position['quantity'] > 0:  # Long position
            if current_price <= entry_price * (1 - self.parameters.stop_loss):
                return True
        else:  # Short position
            if current_price >= entry_price * (1 + self.parameters.stop_loss):
                return True
        
        return False


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands"""
    
    def __init__(self, strategy_id: str, parameters: StrategyParameters, symbols: List[str]):
        super().__init__(strategy_id, StrategyType.MEAN_REVERSION, parameters, symbols)
    
    async def generate_signal(self, market_data: List[MarketData], symbol: str) -> Optional[StrategySignal]:
        if len(market_data) < self.parameters.mean_reversion_period:
            return None
        
        prices = [md.close for md in market_data]
        prices_array = np.array(prices)
        
        # Calculate Bollinger Bands
        mean = np.mean(prices_array[-self.parameters.mean_reversion_period:])
        std = np.std(prices_array[-self.parameters.mean_reversion_period:])
        upper_band = mean + 2 * std
        lower_band = mean - 2 * std
        
        current_price = market_data[-1].close
        
        # Generate signal based on price position relative to bands
        if current_price <= lower_band:
            return StrategySignal(
                strategy_id=self.strategy_id,
                signal_type=SignalType.BUY,
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=current_price,
                quantity=1.0,
                confidence=min((mean - current_price) / std, 1.0)
            )
        elif current_price >= upper_band:
            return StrategySignal(
                strategy_id=self.strategy_id,
                signal_type=SignalType.SELL,
                symbol=symbol,
                timestamp=datetime.utcnow(),
                price=current_price,
                quantity=1.0,
                confidence=min((current_price - mean) / std, 1.0)
            )
        
        return None
    
    def calculate_position_size(self, signal: StrategySignal, available_capital: float) -> float:
        risk_amount = available_capital * self.parameters.risk_per_trade
        position_size = risk_amount / (signal.price * self.parameters.stop_loss)
        return min(position_size, available_capital * self.parameters.max_position_size)
    
    def should_exit_position(self, position: Dict[str, Any], market_data: List[MarketData]) -> bool:
        if not market_data:
            return False
        
        current_price = market_data[-1].close
        entry_price = position['entry_price']
        
        # Check stop loss and take profit
        if position['quantity'] > 0:  # Long position
            if (current_price <= entry_price * (1 - self.parameters.stop_loss) or
                current_price >= entry_price * (1 + self.parameters.take_profit)):
                return True
        else:  # Short position
            if (current_price >= entry_price * (1 + self.parameters.stop_loss) or
                current_price <= entry_price * (1 - self.parameters.take_profit)):
                return True
        
        return False


class StrategyTemplate:
    """Template for creating new strategies"""
    
    @staticmethod
    def create_strategy(strategy_type: StrategyType, strategy_id: str, 
                       parameters: StrategyParameters, symbols: List[str]) -> BaseStrategy:
        """Create strategy instance based on type"""
        if strategy_type == StrategyType.TREND_FOLLOWING:
            return TrendFollowingStrategy(strategy_id, parameters, symbols)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            return MeanReversionStrategy(strategy_id, parameters, symbols)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")


class GeneticOptimizer:
    """Genetic algorithm for strategy parameter optimization"""
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        logger.info("GeneticOptimizer initialized")
    
    def optimize_parameters(self, strategy_class: type, symbols: List[str], 
                          historical_data: Dict[str, List[MarketData]], 
                          fitness_function: Callable) -> StrategyParameters:
        """Optimize strategy parameters using genetic algorithm"""
        
        # Initialize population
        population = self._initialize_population()
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                parameters = self._individual_to_parameters(individual)
                strategy = strategy_class(f"temp_{generation}", parameters, symbols)
                fitness = fitness_function(strategy, historical_data)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self._crossover(selected[i], selected[i + 1])
                    child1 = self._mutation(child1)
                    child2 = self._mutation(child2)
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])
            
            population = new_population[:self.population_size]
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {best_fitness}")
        
        return self._individual_to_parameters(best_individual)
    
    def _initialize_population(self) -> List[List[float]]:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = [
                random.randint(10, 100),  # lookback_period
                random.uniform(0.1, 2.0),  # threshold
                random.uniform(0.01, 0.05),  # stop_loss
                random.uniform(0.02, 0.10),  # take_profit
                random.uniform(0.05, 0.20),  # max_position_size
                random.uniform(0.01, 0.05),  # risk_per_trade
            ]
            population.append(individual)
        return population
    
    def _individual_to_parameters(self, individual: List[float]) -> StrategyParameters:
        """Convert individual to strategy parameters"""
        return StrategyParameters(
            lookback_period=int(individual[0]),
            threshold=individual[1],
            stop_loss=individual[2],
            take_profit=individual[3],
            max_position_size=individual[4],
            risk_per_trade=individual[5]
        )
    
    def _selection(self, population: List[List[float]], 
                  fitness_scores: List[float]) -> List[List[float]]:
        """Tournament selection"""
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(enumerate(population)), 3)
            winner = max(tournament, key=lambda x: fitness_scores[x[0]])
            selected.append(winner[1])
        return selected
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> tuple:
        """Single-point crossover"""
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2
    
    def _mutation(self, individual: List[float]) -> List[float]:
        """Random mutation"""
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(individual) - 1)
            if idx == 0:  # lookback_period
                individual[idx] = random.randint(10, 100)
            else:  # other parameters
                individual[idx] *= random.uniform(0.8, 1.2)
        return individual


class MultiStrategyFramework:
    """Framework for managing multiple concurrent strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_allocations: Dict[str, float] = {}
        self.performance_monitor = {}
        self.risk_balancer = {}
        
        logger.info("MultiStrategyFramework initialized")
    
    def add_strategy(self, strategy: BaseStrategy, allocation: float = 1.0) -> None:
        """Add strategy to framework"""
        self.strategies[strategy.strategy_id] = strategy
        self.strategy_allocations[strategy.strategy_id] = allocation
        logger.info(f"Strategy {strategy.strategy_id} added with allocation {allocation}")
    
    def remove_strategy(self, strategy_id: str) -> None:
        """Remove strategy from framework"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            del self.strategy_allocations[strategy_id]
            logger.info(f"Strategy {strategy_id} removed")
    
    async def update_all_strategies(self, market_data: Dict[str, List[MarketData]]) -> List[StrategySignal]:
        """Update all strategies with market data"""
        all_signals = []
        
        for strategy_id, strategy in self.strategies.items():
            if strategy.status == StrategyStatus.ACTIVE:
                for symbol, data in market_data.items():
                    if symbol in strategy.symbols:
                        await strategy.update(data, symbol)
                        all_signals.extend(strategy.signals[-1:])  # Get latest signal
        
        return all_signals
    
    def get_total_performance(self) -> StrategyPerformance:
        """Get combined performance of all strategies"""
        total_performance = StrategyPerformance()
        
        for strategy in self.strategies.values():
            perf = strategy.get_performance()
            allocation = self.strategy_allocations[strategy.strategy_id]
            
            total_performance.total_return += perf.total_return * allocation
            total_performance.total_trades += perf.total_trades
            total_performance.winning_trades += perf.winning_trades
            total_performance.losing_trades += perf.losing_trades
        
        if total_performance.total_trades > 0:
            total_performance.win_rate = total_performance.winning_trades / total_performance.total_trades
        
        return total_performance
    
    def rebalance_allocations(self, performance_weights: Dict[str, float]) -> None:
        """Rebalance strategy allocations based on performance"""
        total_weight = sum(performance_weights.values())
        if total_weight > 0:
            for strategy_id, weight in performance_weights.items():
                if strategy_id in self.strategy_allocations:
                    self.strategy_allocations[strategy_id] = weight / total_weight
            logger.info("Strategy allocations rebalanced")


class StrategyEngine:
    """Main strategy engine orchestrating all strategy components"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.multi_strategy_framework = MultiStrategyFramework()
        self.genetic_optimizer = GeneticOptimizer()
        self.strategy_templates = StrategyTemplate()
        
        # Performance tracking
        self.performance_history: List[StrategyPerformance] = []
        self.signal_history: List[StrategySignal] = []
        
        logger.info("StrategyEngine initialized")
    
    def create_strategy(self, strategy_type: StrategyType, strategy_id: str,
                       parameters: StrategyParameters, symbols: List[str]) -> BaseStrategy:
        """Create new strategy"""
        strategy = self.strategy_templates.create_strategy(strategy_type, strategy_id, parameters, symbols)
        self.strategies[strategy_id] = strategy
        return strategy
    
    def add_strategy(self, strategy: BaseStrategy, allocation: float = 1.0) -> None:
        """Add strategy to engine"""
        self.strategies[strategy.strategy_id] = strategy
        self.multi_strategy_framework.add_strategy(strategy, allocation)
    
    def remove_strategy(self, strategy_id: str) -> None:
        """Remove strategy from engine"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.multi_strategy_framework.remove_strategy(strategy_id)
    
    async def update_strategies(self, market_data: Dict[str, List[MarketData]]) -> List[StrategySignal]:
        """Update all strategies with market data"""
        signals = await self.multi_strategy_framework.update_all_strategies(market_data)
        self.signal_history.extend(signals)
        return signals
    
    def optimize_strategy(self, strategy_id: str, historical_data: Dict[str, List[MarketData]]) -> StrategyParameters:
        """Optimize strategy parameters"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        strategy = self.strategies[strategy_id]
        
        def fitness_function(strategy_instance: BaseStrategy, data: Dict[str, List[MarketData]]) -> float:
            # Simple fitness function - can be enhanced
            return strategy_instance.get_performance().sharpe_ratio
        
        optimized_params = self.genetic_optimizer.optimize_parameters(
            type(strategy), strategy.symbols, historical_data, fitness_function
        )
        
        # Update strategy with optimized parameters
        strategy.parameters = optimized_params
        logger.info(f"Strategy {strategy_id} optimized")
        
        return optimized_params
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[StrategyPerformance]:
        """Get performance of specific strategy"""
        if strategy_id in self.strategies:
            return self.strategies[strategy_id].get_performance()
        return None
    
    def get_total_performance(self) -> StrategyPerformance:
        """Get total performance of all strategies"""
        return self.multi_strategy_framework.get_total_performance()
    
    def start_strategy(self, strategy_id: str) -> None:
        """Start specific strategy"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id].start()
    
    def stop_strategy(self, strategy_id: str) -> None:
        """Stop specific strategy"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id].stop()
    
    def start_all_strategies(self) -> None:
        """Start all strategies"""
        for strategy in self.strategies.values():
            strategy.start()
    
    def stop_all_strategies(self) -> None:
        """Stop all strategies"""
        for strategy in self.strategies.values():
            strategy.stop()


# Example usage
async def main():
    """Example usage of the strategy engine"""
    
    # Create strategy engine
    engine = StrategyEngine()
    
    # Create strategy parameters
    params = StrategyParameters(
        lookback_period=20,
        threshold=0.5,
        stop_loss=0.02,
        take_profit=0.04,
        max_position_size=0.1,
        risk_per_trade=0.02
    )
    
    # Create trend following strategy
    strategy = engine.create_strategy(
        StrategyType.TREND_FOLLOWING,
        "trend_strategy_1",
        params,
        ["EURUSD", "GBPUSD"]
    )
    
    # Add to engine
    engine.add_strategy(strategy, allocation=1.0)
    
    # Start strategy
    engine.start_strategy("trend_strategy_1")
    
    print("Strategy engine initialized and ready")


if __name__ == "__main__":
    asyncio.run(main()) 
