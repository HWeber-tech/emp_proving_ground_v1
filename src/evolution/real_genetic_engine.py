#!/usr/bin/env python3
"""
Real Genetic Programming Engine - True Evolutionary Trading

This module implements a real genetic programming system that evolves
actual trading strategies with real market data and performance evaluation.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import random
import copy
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal with timing and direction."""
    timestamp: datetime
    symbol: str
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    price: float
    volume: float
    strategy_id: str


@dataclass
class TradingStrategy:
    """Represents a trading strategy with parameters and logic."""
    id: str
    name: str
    parameters: Dict[str, float]
    indicators: List[str]
    entry_rules: List[str]
    exit_rules: List[str]
    risk_management: Dict[str, float]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'parameters': self.parameters,
            'indicators': self.indicators,
            'entry_rules': self.entry_rules,
            'exit_rules': self.exit_rules,
            'risk_management': self.risk_management,
            'performance_metrics': self.performance_metrics,
            'fitness_score': self.fitness_score,
            'generation': self.generation,
            'parent_ids': self.parent_ids
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingStrategy':
        """Create strategy from dictionary."""
        return cls(**data)


class TechnicalIndicators:
    """Technical analysis indicators for strategy evaluation."""
    
    @staticmethod
    def sma(data: pd.DataFrame, period: int) -> pd.Series:
        """Simple Moving Average."""
        return data['close'].rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.DataFrame, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return data['close'].ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator."""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()


class StrategyEvaluator:
    """Evaluates trading strategies on real market data."""
    
    def __init__(self, data_source):
        """
        Initialize strategy evaluator.
        
        Args:
            data_source: Data source for market data
        """
        self.data_source = data_source
        self.indicators = TechnicalIndicators()
        
    def evaluate_strategy(self, strategy: TradingStrategy, symbol: str, 
                         start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """
        Evaluate a trading strategy on real market data.
        
        Args:
            strategy: Trading strategy to evaluate
            symbol: Trading symbol
            start_date: Start date for evaluation
            end_date: End date for evaluation
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get real market data
            data = self.data_source.get_data_range(symbol, start_date, end_date)
            
            if data.empty:
                logger.warning(f"No data available for {symbol} evaluation")
                return self._empty_metrics()
            
            # Calculate technical indicators
            data_with_indicators = self._calculate_indicators(data, strategy.indicators, strategy.parameters)
            
            # Generate trading signals
            signals = self._generate_signals(data_with_indicators, strategy)
            
            # Execute backtest
            performance = self._backtest_strategy(data_with_indicators, signals, strategy)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating strategy {strategy.id}: {e}")
            return self._empty_metrics()
    
    def _calculate_indicators(self, data: pd.DataFrame, indicators: List[str], 
                            parameters: Dict[str, float]) -> pd.DataFrame:
        """Calculate technical indicators for the data."""
        data = data.copy()
        
        for indicator in indicators:
            if indicator == 'SMA':
                period = int(parameters.get('sma_period', 20))
                data[f'sma_{period}'] = self.indicators.sma(data, period)
            
            elif indicator == 'EMA':
                period = int(parameters.get('ema_period', 20))
                data[f'ema_{period}'] = self.indicators.ema(data, period)
            
            elif indicator == 'RSI':
                period = int(parameters.get('rsi_period', 14))
                data['rsi'] = self.indicators.rsi(data, period)
            
            elif indicator == 'MACD':
                fast = int(parameters.get('macd_fast', 12))
                slow = int(parameters.get('macd_slow', 26))
                signal = int(parameters.get('macd_signal', 9))
                macd_line, signal_line, histogram = self.indicators.macd(data, fast, slow, signal)
                data['macd'] = macd_line
                data['macd_signal'] = signal_line
                data['macd_histogram'] = histogram
            
            elif indicator == 'BOLLINGER':
                period = int(parameters.get('bb_period', 20))
                std_dev = parameters.get('bb_std', 2.0)
                upper, middle, lower = self.indicators.bollinger_bands(data, period, std_dev)
                data['bb_upper'] = upper
                data['bb_middle'] = middle
                data['bb_lower'] = lower
            
            elif indicator == 'ATR':
                period = int(parameters.get('atr_period', 14))
                data['atr'] = self.indicators.atr(data, period)
        
        return data
    
    def _generate_signals(self, data: pd.DataFrame, strategy: TradingStrategy) -> List[TradingSignal]:
        """Generate trading signals based on strategy rules."""
        signals = []
        
        for i in range(len(data)):
            if i < 50:  # Skip first 50 bars for indicator calculation
                continue
            
            row = data.iloc[i]
            timestamp = data.index[i]
            
            # Evaluate entry rules
            entry_signal = self._evaluate_entry_rules(row, strategy.entry_rules, strategy.parameters)
            
            if entry_signal:
                signal = TradingSignal(
                    timestamp=timestamp,
                    symbol=data.get('symbol', 'UNKNOWN').iloc[i] if hasattr(data, 'symbol') else 'UNKNOWN',
                    direction=entry_signal['direction'],
                    confidence=entry_signal['confidence'],
                    price=row['close'],
                    volume=row.get('volume', 1000),
                    strategy_id=strategy.id
                )
                signals.append(signal)
        
        return signals
    
    def _evaluate_entry_rules(self, row: pd.Series, rules: List[str], parameters: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Evaluate entry rules for a given data row."""
        for rule in rules:
            signal = self._evaluate_rule(row, rule, parameters)
            if signal:
                return signal
        return None
    
    def _evaluate_rule(self, row: pd.Series, rule: str, parameters: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Evaluate a single trading rule."""
        try:
            if rule == 'SMA_CROSSOVER':
                sma_fast = parameters.get('sma_fast', 10)
                sma_slow = parameters.get('sma_slow', 20)
                
                if f'sma_{sma_fast}' in row and f'sma_{sma_slow}' in row:
                    if row[f'sma_{sma_fast}'] > row[f'sma_{sma_slow}']:
                        return {'direction': 'BUY', 'confidence': 0.7}
                    elif row[f'sma_{sma_fast}'] < row[f'sma_{sma_slow}']:
                        return {'direction': 'SELL', 'confidence': 0.7}
            
            elif rule == 'RSI_OVERSOLD':
                rsi_threshold = parameters.get('rsi_oversold', 30)
                if 'rsi' in row and row['rsi'] < rsi_threshold:
                    return {'direction': 'BUY', 'confidence': 0.8}
            
            elif rule == 'RSI_OVERBOUGHT':
                rsi_threshold = parameters.get('rsi_overbought', 70)
                if 'rsi' in row and row['rsi'] > rsi_threshold:
                    return {'direction': 'SELL', 'confidence': 0.8}
            
            elif rule == 'MACD_CROSSOVER':
                if 'macd' in row and 'macd_signal' in row:
                    if row['macd'] > row['macd_signal']:
                        return {'direction': 'BUY', 'confidence': 0.6}
                    elif row['macd'] < row['macd_signal']:
                        return {'direction': 'SELL', 'confidence': 0.6}
            
            elif rule == 'BOLLINGER_BOUNCE':
                if 'bb_lower' in row and 'bb_upper' in row:
                    if row['close'] <= row['bb_lower']:
                        return {'direction': 'BUY', 'confidence': 0.7}
                    elif row['close'] >= row['bb_upper']:
                        return {'direction': 'SELL', 'confidence': 0.7}
            
        except Exception as e:
            logger.debug(f"Error evaluating rule {rule}: {e}")
        
        return None
    
    def _backtest_strategy(self, data: pd.DataFrame, signals: List[TradingSignal], 
                          strategy: TradingStrategy) -> Dict[str, float]:
        """Backtest the strategy and calculate performance metrics."""
        if not signals:
            return self._empty_metrics()
        
        # Initialize backtest variables
        initial_capital = 100000
        capital = initial_capital
        position = 0
        trades = []
        
        # Risk management parameters
        max_position_size = strategy.risk_management.get('max_position_size', 0.1)
        stop_loss = strategy.risk_management.get('stop_loss', 0.02)
        take_profit = strategy.risk_management.get('take_profit', 0.04)
        
        for signal in signals:
            if signal.direction == 'BUY' and position <= 0:
                # Open long position
                position_size = capital * max_position_size
                shares = position_size / signal.price
                position = shares
                entry_price = signal.price
                entry_time = signal.timestamp
                
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'shares': shares,
                    'type': 'LONG'
                })
            
            elif signal.direction == 'SELL' and position >= 0:
                # Close long position
                if position > 0:
                    exit_price = signal.price
                    pnl = (exit_price - entry_price) * position
                    capital += pnl
                    
                    trades[-1].update({
                        'exit_time': signal.timestamp,
                        'exit_price': exit_price,
                        'pnl': pnl
                    })
                    
                    position = 0
        
        # Calculate performance metrics
        total_return = (capital - initial_capital) / initial_capital
        num_trades = len([t for t in trades if 'exit_time' in t])
        
        if num_trades > 0:
            winning_trades = len([t for t in trades if 'pnl' in t and t['pnl'] > 0])
            win_rate = winning_trades / num_trades
            
            if winning_trades > 0:
                avg_win = np.mean([t['pnl'] for t in trades if 'pnl' in t and t['pnl'] > 0])
            else:
                avg_win = 0
            
            if num_trades - winning_trades > 0:
                avg_loss = np.mean([t['pnl'] for t in trades if 'pnl' in t and t['pnl'] < 0])
            else:
                avg_loss = 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for trade in trades:
            if 'pnl' in trade:
                returns.append(trade['pnl'] / initial_capital)
        
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self._calculate_max_drawdown(trades, initial_capital),
            'final_capital': capital
        }
    
    def _calculate_max_drawdown(self, trades: List[Dict], initial_capital: float) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0
        
        capital = initial_capital
        peak_capital = initial_capital
        max_drawdown = 0
        
        for trade in trades:
            if 'pnl' in trade:
                capital += trade['pnl']
                if capital > peak_capital:
                    peak_capital = capital
                
                drawdown = (peak_capital - capital) / peak_capital
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        return max_drawdown
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty performance metrics."""
        return {
            'total_return': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'final_capital': 100000.0
        }


class RealGeneticEngine:
    """
    Real genetic programming engine for evolving trading strategies.
    
    This replaces the mock evolution system with actual genetic programming
    that evolves real trading strategies on real market data.
    """
    
    def __init__(self, data_source, population_size: int = 50, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        """
        Initialize the genetic engine.
        
        Args:
            data_source: Data source for market data
            population_size: Size of the population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.data_source = data_source
        self.evaluator = StrategyEvaluator(data_source)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population: List[TradingStrategy] = []
        self.generation = 0
        self.best_strategy: Optional[TradingStrategy] = None
        
        # Strategy templates
        self.strategy_templates = self._create_strategy_templates()
        
        logger.info(f"Real genetic engine initialized with population size {population_size}")
    
    def _create_strategy_templates(self) -> List[Dict[str, Any]]:
        """Create templates for different strategy types."""
        return [
            {
                'name': 'SMA_Crossover',
                'indicators': ['SMA'],
                'entry_rules': ['SMA_CROSSOVER'],
                'exit_rules': ['SMA_CROSSOVER'],
                'parameters': {
                    'sma_fast': 10,
                    'sma_slow': 20,
                    'sma_period': 20
                }
            },
            {
                'name': 'RSI_Strategy',
                'indicators': ['RSI'],
                'entry_rules': ['RSI_OVERSOLD', 'RSI_OVERBOUGHT'],
                'exit_rules': ['RSI_OVERBOUGHT', 'RSI_OVERSOLD'],
                'parameters': {
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70
                }
            },
            {
                'name': 'MACD_Strategy',
                'indicators': ['MACD'],
                'entry_rules': ['MACD_CROSSOVER'],
                'exit_rules': ['MACD_CROSSOVER'],
                'parameters': {
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9
                }
            },
            {
                'name': 'Bollinger_Strategy',
                'indicators': ['BOLLINGER'],
                'entry_rules': ['BOLLINGER_BOUNCE'],
                'exit_rules': ['BOLLINGER_BOUNCE'],
                'parameters': {
                    'bb_period': 20,
                    'bb_std': 2.0
                }
            }
        ]
    
    def initialize_population(self, symbol: str, start_date: datetime, end_date: datetime):
        """Initialize the population with random strategies."""
        logger.info("Initializing population with random strategies...")
        
        self.population = []
        
        for i in range(self.population_size):
            strategy = self._create_random_strategy(f"Strategy_{i}")
            self.population.append(strategy)
        
        # Evaluate initial population
        self._evaluate_population(symbol, start_date, end_date)
        
        logger.info(f"Population initialized with {len(self.population)} strategies")
    
    def _create_random_strategy(self, name: str) -> TradingStrategy:
        """Create a random trading strategy."""
        template = random.choice(self.strategy_templates)
        
        # Randomize parameters
        parameters = template['parameters'].copy()
        for key, value in parameters.items():
            if 'period' in key:
                parameters[key] = random.randint(5, 50)
            elif 'fast' in key:
                parameters[key] = random.randint(5, 20)
            elif 'slow' in key:
                parameters[key] = random.randint(20, 50)
            elif 'signal' in key:
                parameters[key] = random.randint(5, 15)
            elif 'oversold' in key:
                parameters[key] = random.uniform(20, 40)
            elif 'overbought' in key:
                parameters[key] = random.uniform(60, 80)
            elif 'std' in key:
                parameters[key] = random.uniform(1.5, 3.0)
        
        # Randomize risk management
        risk_management = {
            'max_position_size': random.uniform(0.05, 0.2),
            'stop_loss': random.uniform(0.01, 0.05),
            'take_profit': random.uniform(0.02, 0.08)
        }
        
        return TradingStrategy(
            id=f"{name}_{random.randint(1000, 9999)}",
            name=name,
            parameters=parameters,
            indicators=template['indicators'],
            entry_rules=template['entry_rules'],
            exit_rules=template['exit_rules'],
            risk_management=risk_management,
            generation=0
        )
    
    def _evaluate_population(self, symbol: str, start_date: datetime, end_date: datetime):
        """Evaluate all strategies in the population."""
        logger.info(f"Evaluating population of {len(self.population)} strategies...")
        
        for strategy in self.population:
            performance = self.evaluator.evaluate_strategy(strategy, symbol, start_date, end_date)
            strategy.performance_metrics = performance
            
            # Calculate fitness score (weighted combination of metrics)
            fitness = (
                performance['total_return'] * 0.4 +
                performance['sharpe_ratio'] * 0.3 +
                performance['win_rate'] * 0.2 +
                (1 - performance['max_drawdown']) * 0.1
            )
            
            strategy.fitness_score = fitness
        
        # Sort by fitness
        self.population.sort(key=lambda s: s.fitness_score, reverse=True)
        
        # Update best strategy
        if self.population and (self.best_strategy is None or 
                               self.population[0].fitness_score > self.best_strategy.fitness_score):
            self.best_strategy = copy.deepcopy(self.population[0])
        
        logger.info(f"Population evaluated. Best fitness: {self.population[0].fitness_score if self.population else 0}")
    
    def evolve(self, symbol: str, start_date: datetime, end_date: datetime, 
               generations: int = 10) -> List[TradingStrategy]:
        """
        Evolve the population for multiple generations.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for evaluation
            end_date: End date for evaluation
            generations: Number of generations to evolve
            
        Returns:
            List of evolved strategies
        """
        logger.info(f"Starting evolution for {generations} generations...")
        
        evolution_history = []
        
        for gen in range(generations):
            logger.info(f"Generation {gen + 1}/{generations}")
            
            # Create new population through selection, crossover, and mutation
            new_population = self._create_new_generation()
            
            # Replace old population
            self.population = new_population
            self.generation = gen + 1
            
            # Evaluate new population
            self._evaluate_population(symbol, start_date, end_date)
            
            # Record best strategy
            if self.best_strategy:
                evolution_history.append(copy.deepcopy(self.best_strategy))
            
            logger.info(f"Generation {gen + 1} complete. Best fitness: {self.population[0].fitness_score if self.population else 0}")
        
        return evolution_history
    
    def _create_new_generation(self) -> List[TradingStrategy]:
        """Create a new generation through genetic operations."""
        new_population = []
        
        # Elitism: Keep top 10% of strategies
        elite_size = max(1, int(self.population_size * 0.1))
        new_population.extend(self.population[:elite_size])
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child = self._crossover(parent1, parent2)
            else:
                # Mutation
                parent = self._select_parent()
                child = self._mutate(parent)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _select_parent(self) -> TradingStrategy:
        """Select a parent using tournament selection."""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda s: s.fitness_score)
    
    def _crossover(self, parent1: TradingStrategy, parent2: TradingStrategy) -> TradingStrategy:
        """Perform crossover between two parents."""
        child = copy.deepcopy(parent1)
        child.id = f"Child_{random.randint(1000, 9999)}"
        child.generation = self.generation
        child.parent_ids = [parent1.id, parent2.id]
        
        # Crossover parameters
        for key in child.parameters:
            if random.random() < 0.5:
                child.parameters[key] = parent2.parameters.get(key, child.parameters[key])
        
        # Crossover risk management
        for key in child.risk_management:
            if random.random() < 0.5:
                child.risk_management[key] = parent2.risk_management.get(key, child.risk_management[key])
        
        return child
    
    def _mutate(self, parent: TradingStrategy) -> TradingStrategy:
        """Perform mutation on a parent."""
        child = copy.deepcopy(parent)
        child.id = f"Mutant_{random.randint(1000, 9999)}"
        child.generation = self.generation
        child.parent_ids = [parent.id]
        
        # Mutate parameters
        for key in child.parameters:
            if random.random() < self.mutation_rate:
                if 'period' in key:
                    child.parameters[key] = max(5, child.parameters[key] + random.randint(-5, 5))
                elif 'fast' in key:
                    child.parameters[key] = max(5, min(20, child.parameters[key] + random.randint(-3, 3)))
                elif 'slow' in key:
                    child.parameters[key] = max(20, min(50, child.parameters[key] + random.randint(-5, 5)))
                elif 'signal' in key:
                    child.parameters[key] = max(5, min(15, child.parameters[key] + random.randint(-2, 2)))
                elif 'oversold' in key:
                    child.parameters[key] = max(20, min(40, child.parameters[key] + random.uniform(-5, 5)))
                elif 'overbought' in key:
                    child.parameters[key] = max(60, min(80, child.parameters[key] + random.uniform(-5, 5)))
                elif 'std' in key:
                    child.parameters[key] = max(1.0, min(4.0, child.parameters[key] + random.uniform(-0.5, 0.5)))
        
        # Mutate risk management
        for key in child.risk_management:
            if random.random() < self.mutation_rate:
                if 'max_position_size' in key:
                    child.risk_management[key] = max(0.01, min(0.3, child.risk_management[key] + random.uniform(-0.05, 0.05)))
                elif 'stop_loss' in key:
                    child.risk_management[key] = max(0.005, min(0.1, child.risk_management[key] + random.uniform(-0.01, 0.01)))
                elif 'take_profit' in key:
                    child.risk_management[key] = max(0.01, min(0.15, child.risk_management[key] + random.uniform(-0.02, 0.02)))
        
        return child
    
    def get_best_strategies(self, count: int = 5) -> List[TradingStrategy]:
        """Get the best strategies from the population."""
        return self.population[:count]
    
    def save_strategies(self, filename: str):
        """Save strategies to a JSON file."""
        strategies_data = [strategy.to_dict() for strategy in self.population]
        
        with open(filename, 'w') as f:
            json.dump(strategies_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(strategies_data)} strategies to {filename}")
    
    def load_strategies(self, filename: str):
        """Load strategies from a JSON file."""
        with open(filename, 'r') as f:
            strategies_data = json.load(f)
        
        self.population = [TradingStrategy.from_dict(data) for data in strategies_data]
        logger.info(f"Loaded {len(self.population)} strategies from {filename}")


def main():
    """Test the real genetic engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test real genetic programming engine")
    parser.add_argument("symbol", help="Trading symbol (e.g., EURUSD)")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--population", type=int, default=20, help="Population size")
    
    args = parser.parse_args()
    
    # Initialize data source (you'll need to implement this)
    from src.data.real_data_ingestor import RealDataIngestor
    data_source = RealDataIngestor()
    
    # Initialize genetic engine
    engine = RealGeneticEngine(data_source, population_size=args.population)
    
    # Set up evaluation period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Initialize and evolve population
    engine.initialize_population(args.symbol, start_date, end_date)
    evolution_history = engine.evolve(args.symbol, start_date, end_date, args.generations)
    
    # Show results
    print(f"\n🎉 Evolution complete!")
    print(f"Best strategy: {engine.best_strategy.name if engine.best_strategy else 'None'}")
    if engine.best_strategy:
        print(f"Fitness score: {engine.best_strategy.fitness_score:.4f}")
        print(f"Total return: {engine.best_strategy.performance_metrics.get('total_return', 0):.2%}")
        print(f"Sharpe ratio: {engine.best_strategy.performance_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Win rate: {engine.best_strategy.performance_metrics.get('win_rate', 0):.2%}")


if __name__ == "__main__":
    main() 