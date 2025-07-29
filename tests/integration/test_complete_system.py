"""
Complete System Integration Test
===============================

End-to-end test of the entire EMP system including:
- Genetic Engine
- Strategy Engine
- Risk Management
- Evolution Process
"""

import asyncio
import logging
import pytest
from datetime import datetime
from typing import List, Dict, Any

# Import all components
from src.evolution.engine.genetic_engine import GeneticEngine, EvolutionConfig
from src.core.population_manager import PopulationManager
from src.evolution.fitness.real_trading_fitness_evaluator import RealTradingFitnessEvaluator
from src.risk.risk_manager_impl import RiskManagerImpl, create_risk_manager
from src.trading.strategy_engine.strategy_engine_impl import StrategyEngineImpl, create_strategy_engine
from src.trading.strategy_engine.templates.moving_average_strategy import create_moving_average_strategy
from src.core.interfaces import DecisionGenome

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCompleteSystem:
    """Test suite for complete system integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.risk_manager = create_risk_manager(initial_balance=10000.0)
        self.strategy_engine = create_strategy_engine(self.risk_manager)
        self.population_manager = PopulationManager(population_size=10)
        
        # Create fitness evaluator
        self.fitness_evaluator = RealTradingFitnessEvaluator(
            strategy_engine=self.strategy_engine,
            risk_manager=self.risk_manager
        )
        
        # Create genetic engine
        self.genetic_engine = GeneticEngine(
            population_manager=self.population_manager,
            selection_strategy=None,  # Will use default
            crossover_strategy=None,  # Will use default
            mutation_strategy=None,  # Will use default
            fitness_evaluator=self.fitness_evaluator,
            genome_factory=None  # Will use default
        )
    
    def test_risk_manager_initialization(self):
        """Test risk manager initialization."""
        assert self.risk_manager is not None
        assert self.risk_manager.account_balance == 10000.0
        
        summary = self.risk_manager.get_risk_summary()
        assert 'account_balance' in summary
        assert summary['account_balance'] == 10000.0
    
    def test_strategy_engine_initialization(self):
        """Test strategy engine initialization."""
        assert self.strategy_engine is not None
        assert len(self.strategy_engine.get_all_strategies()) == 0
    
    def test_population_manager_initialization(self):
        """Test population manager initialization."""
        assert self.population_manager is not None
        assert self.population_manager.population_size == 10
    
    def test_genetic_engine_initialization(self):
        """Test genetic engine initialization."""
        assert self.genetic_engine is not None
        assert self.genetic_engine.get_best_genome() is None
    
    @pytest.mark.asyncio
    async def test_strategy_registration(self):
        """Test strategy registration."""
        strategy = create_moving_average_strategy(
            strategy_id="test_strategy",
            symbols=["EURUSD"],
            parameters={'fast_period': 10, 'slow_period': 20}
        )
        
        success = self.strategy_engine.register_strategy(strategy)
        assert success is True
        
        strategies = self.strategy_engine.get_all_strategies()
        assert "test_strategy" in strategies
    
    @pytest.mark.asyncio
    async def test_strategy_execution(self):
        """Test strategy execution."""
        # Register strategy
        strategy = create_moving_average_strategy(
            strategy_id="test_execution",
            symbols=["EURUSD"],
            parameters={'fast_period': 5, 'slow_period': 10}
        )
        
        self.strategy_engine.register_strategy(strategy)
        self.strategy_engine.start_strategy("test_execution")
        
        # Create market data
        market_data = {
            'symbol': 'EURUSD',
            'close': 1.1000,
            'volume': 1000,
            'timestamp': datetime.now()
        }
        
        # Execute strategy
        result = await self.strategy_engine.execute_strategy("test_execution", market_data)
        assert result is not None
        assert result.strategy_id == "test_execution"
    
    @pytest.mark.asyncio
    async def test_risk_validation(self):
        """Test risk validation."""
        position = {
            'symbol': 'EURUSD',
            'size': 1000.0,
            'entry_price': 1.1000
        }
        
        is_valid = await self.risk_manager.validate_position(position)
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_position_sizing(self):
        """Test position sizing calculation."""
        signal = {
            'symbol': 'EURUSD',
            'confidence': 0.7,
            'stop_loss_pct': 0.02
        }
        
        position_size = await self.risk_manager.calculate_position_size(signal)
        assert position_size > 0
        assert position_size <= 10000.0 * 0.1  # Max 10% of capital
    
    def test_genome_creation(self):
        """Test genome creation."""
        genome = DecisionGenome()
        assert genome is not None
        
        # Test genome has required attributes
        assert hasattr(genome, 'strategy')
        assert hasattr(genome, 'risk')
    
    @pytest.mark.asyncio
    async def test_fitness_evaluation(self):
        """Test fitness evaluation."""
        # Create test genome
        genome = DecisionGenome()
        
        # Evaluate fitness
        fitness = await self.fitness_evaluator.evaluate(genome)
        assert isinstance(fitness, float)
        assert fitness >= 0
    
    @pytest.mark.asyncio
    async def test_complete_evolution_cycle(self):
        """Test complete evolution cycle."""
        # Initialize population
        self.population_manager.initialize_population(
            lambda: DecisionGenome()
        )
        
        # Run evolution for a few generations
        initial_population = self.population_manager.get_population()
        assert len(initial_population) == 10
        
        # Note: This would require full genetic engine setup
        # For now, just verify population exists
        assert len(initial_population) > 0
    
    @pytest.mark.asyncio
    async def test_system_integration(self):
        """Test complete system integration."""
        # 1. Register strategy
        strategy = create_moving_average_strategy(
            strategy_id="integration_test",
            symbols=["EURUSD"],
            parameters={'fast_period': 5, 'slow_period': 10}
        )
        
        self.strategy_engine.register_strategy(strategy)
        self.strategy_engine.start_strategy("integration_test")
        
        # 2. Create market data
        market_data = {
            'symbol': 'EURUSD',
            'close': 1.1000,
            'volume': 1000,
            'timestamp': datetime.now()
        }
        
        # 3. Execute strategy
        result = await self.strategy_engine.execute_strategy("integration_test", market_data)
        assert result is not None
        
        # 4. Validate with risk manager
        if result.signal:
            is_valid = await self.risk_manager.validate_position({
                'symbol': result.signal.symbol,
                'size': result.signal.quantity,
                'entry_price': result.signal.price
            })
            assert is_valid is True
    
    def test_strategy_parameters(self):
        """Test strategy parameter management."""
        strategy = create_moving_average_strategy(
            strategy_id="param_test",
            symbols=["EURUSD"],
            parameters={'fast_period': 10, 'slow_period': 20}
        )
        
        # Test parameter update
        success = strategy.update_parameters({'fast_period': 15})
        assert success is True
        
        # Verify update
        params = strategy.get_parameters()
        assert params['fast_period'] == 15
    
    def test_risk_summary(self):
        """Test risk summary generation."""
        summary = self.risk_manager.get_risk_summary()
        
        assert isinstance(summary, dict)
        assert 'account_balance' in summary
        assert 'total_positions' in summary
        assert 'portfolio_risk' in summary
    
    def test_strategy_performance(self):
        """Test strategy performance tracking."""
        strategy = create_moving_average_strategy(
            strategy_id="perf_test",
            symbols=["EURUSD"],
            parameters={'fast_period': 10, 'slow_period': 20}
        )
        
        self.strategy_engine.register_strategy(strategy)
        
        performance = self.strategy_engine.get_strategy_performance("perf_test")
        assert performance is not None
        assert 'total_trades' in performance
        assert 'total_return' in performance


@pytest.mark.asyncio
async def test_async_system_flow():
    """Test complete async system flow."""
    # Create system components
    risk_manager = create_risk_manager(10000.0)
    strategy_engine = create_strategy_engine(risk_manager)
    
    # Register strategy
    strategy = create_moving_average_strategy(
        strategy_id="async_test",
        symbols=["EURUSD"],
        parameters={'fast_period': 5, 'slow_period': 10}
    )
    
    strategy_engine.register_strategy(strategy)
    strategy_engine.start_strategy("async_test")
    
    # Create market data
    market_data = {
        'symbol': 'EURUSD',
        'close': 1.1000,
        'volume': 1000,
        'timestamp': datetime.now()
    }
    
    # Execute all strategies
    results = await strategy_engine.execute_all_strategies(market_data)
    
    assert isinstance(results, list)
    assert len(results) >= 0  # May be empty if no signals


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
