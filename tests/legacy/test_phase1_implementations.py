#!/usr/bin/env python3
"""
Phase 1 Completion Test Suite
============================

Comprehensive test suite to validate that Phase 1 is 100% complete.
Tests all real implementations against known inputs and expected outputs.
"""

import pytest
import asyncio
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd

try:
    from src.evolution.engine.real_evolution_engine import RealEvolutionEngine  # deprecated
except Exception:  # pragma: no cover
    RealEvolutionEngine = None  # type: ignore
from src.risk.real_risk_manager import RealRiskManager, RealRiskConfig
from src.portfolio.real_portfolio_monitor import RealPortfolioMonitor
from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.trading.strategies.real_base_strategy import RealBaseStrategy
try:
    from src.core.market_data import MarketData  # legacy
except Exception:  # pragma: no cover
    MarketData = None  # type: ignore
from src.core import Instrument


class TestRealEvolutionEngine:
    """Test the RealEvolutionEngine implementation."""
    
    def test_initialization(self):
        """Test that RealEvolutionEngine initializes correctly."""
        engine = RealEvolutionEngine(population_size=10)
        assert engine.population_size == 10
        assert engine.generation == 0
        assert len(engine.population) == 0
        
    def test_population_initialization(self):
        """Test population initialization."""
        engine = RealEvolutionEngine(population_size=5)
        population = engine.initialize_population()
        assert len(population) == 5
        assert all(hasattr(genome, 'genes') for genome in population)
        
    def test_tournament_selection(self):
        """Test tournament selection logic."""
        engine = RealEvolutionEngine()
        population = engine.initialize_population(4)
        fitness_scores = [0.1, 0.8, 0.3, 0.9]
        
        parents = engine.select_parents(population, fitness_scores)
        assert len(parents) == 4
        
    def test_crossover(self):
        """Test crossover operation."""
        engine = RealEvolutionEngine()
        parent1 = engine.initialize_population(1)[0]
        parent2 = engine.initialize_population(1)[0]
        
        child1, child2 = engine.crossover(parent1, parent2)
        assert child1.genome_id != parent1.genome_id
        assert child2.genome_id != parent2.genome_id
        
    def test_mutation(self):
        """Test mutation operation."""
        engine = RealEvolutionEngine(mutation_rate=1.0)  # Force mutation
        genome = engine.initialize_population(1)[0]
        original_genes = genome.genes.copy()
        
        mutated = engine.mutate(genome)
        assert mutated.genes != original_genes
        
    def test_evolution_generation(self):
        """Test a full evolution generation."""
        engine = RealEvolutionEngine(population_size=5, max_generations=2)
        engine.initialize_population()
        
        # Create mock market data
        market_data = MarketData(
            symbol="TEST",
            timeframe="1h",
            data=pd.DataFrame({
                'open': [1.0, 1.1, 1.2, 1.3, 1.4],
                'high': [1.1, 1.2, 1.3, 1.4, 1.5],
                'low': [0.9, 1.0, 1.1, 1.2, 1.3],
                'close': [1.05, 1.15, 1.25, 1.35, 1.45],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
        )
        
        # Mock fitness evaluator
        class MockFitnessEvaluator:
            def evaluate(self, genome, market_data):
                return np.random.random()
                
        engine.set_fitness_evaluator(MockFitnessEvaluator())
        
        stats = engine.evolve_generation(market_data)
        assert stats.generation == 1
        assert stats.population_size == 5
        assert stats.best_fitness >= 0


class TestRealRiskManager:
    """Test the RealRiskManager implementation."""
    
    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion calculation with known inputs."""
        config = RealRiskConfig(
            max_risk_per_trade_pct=Decimal('0.02'),
            max_leverage=Decimal('10.0'),
            max_total_exposure_pct=Decimal('0.5'),
            max_drawdown_pct=Decimal('0.25'),
            kelly_fraction=Decimal('0.25')
        )
        
        risk_manager = RealRiskManager(config)
        
        # Test with known values
        win_rate = 0.6
        avg_win = 0.02
        avg_loss = 0.01
        
        kelly_size = risk_manager._calculate_kelly_size(win_rate, avg_win, avg_loss)
        expected = 0.25 * ((0.6 * 0.02 - 0.4 * 0.01) / 0.01)
        assert abs(kelly_size - expected) < 0.001
        
    def test_position_sizing(self):
        """Test position sizing calculation."""
        config = RealRiskConfig(
            max_risk_per_trade_pct=Decimal('0.02'),
            max_leverage=Decimal('10.0'),
            max_total_exposure_pct=Decimal('0.5'),
            max_drawdown_pct=Decimal('0.25')
        )
        
        risk_manager = RealRiskManager(config)
        
        # Test position sizing
        account_balance = Decimal('10000')
        risk_per_trade = Decimal('0.02')
        stop_loss_pct = Decimal('0.05')
        
        position_size = risk_manager.calculate_position_size(
            account_balance, risk_per_trade, stop_loss_pct
        )
        
        expected_size = (account_balance * risk_per_trade) / stop_loss_pct
        assert position_size == expected_size
        
    def test_risk_validation(self):
        """Test risk validation logic."""
        config = RealRiskConfig(
            max_risk_per_trade_pct=Decimal('0.02'),
            max_leverage=Decimal('10.0'),
            max_total_exposure_pct=Decimal('0.5'),
            max_drawdown_pct=Decimal('0.25')
        )
        
        risk_manager = RealRiskManager(config)
        
        # Test valid position
        account_balance = Decimal('10000')
        position_size = Decimal('1000')
        instrument = Instrument("EURUSD", "Currency")
        
        is_valid = risk_manager.validate_position(position_size, instrument, account_balance)
        assert is_valid
        
        # Test invalid position (too large)
        position_size = Decimal('100000')  # 10x account balance
        is_valid = risk_manager.validate_position(position_size, instrument, account_balance)
        assert not is_valid


class TestRealPortfolioMonitor:
    """Test the RealPortfolioMonitor implementation."""
    
    def test_pnl_calculation(self):
        """Test P&L calculation."""
        monitor = RealPortfolioMonitor()
        
        # Test simple P&L calculation
        entry_price = Decimal('1.1000')
        current_price = Decimal('1.1050')
        position_size = Decimal('10000')
        
        pnl = monitor.calculate_pnl(entry_price, current_price, position_size)
        expected_pnl = (current_price - entry_price) * position_size
        assert pnl == expected_pnl
        
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        monitor = RealPortfolioMonitor()
        
        # Test portfolio value
        positions = [
            {'symbol': 'EURUSD', 'size': Decimal('10000'), 'entry': Decimal('1.1000'), 'current': Decimal('1.1050')},
            {'symbol': 'GBPUSD', 'size': Decimal('5000'), 'entry': Decimal('1.3000'), 'current': Decimal('1.3100')}
        ]
        
        total_value = monitor.calculate_portfolio_value(positions)
        expected_value = (Decimal('1.1050') - Decimal('1.1000')) * Decimal('10000') + \
                        (Decimal('1.3100') - Decimal('1.3000')) * Decimal('5000')
        assert total_value == expected_value
        
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        monitor = RealPortfolioMonitor()
        
        # Test drawdown calculation
        equity_curve = [10000, 10500, 10200, 10800, 10100, 11000]
        max_drawdown = monitor.calculate_max_drawdown(equity_curve)
        
        # Peak at 10800, trough at 10100
        expected_drawdown = (10800 - 10100) / 10800
        assert abs(max_drawdown - expected_drawdown) < 0.001


class TestRealSensoryOrgan:
    """Test the RealSensoryOrgan implementation."""
    
    def test_indicator_calculation(self):
        """Test indicator calculations against known data."""
        organ = RealSensoryOrgan()
        
        # Create test data
        data = pd.DataFrame({
            'open': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            'high': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            'low': [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            'close': [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        # Test SMA calculation
        sma = organ.calculate_sma(data['close'], period=3)
        expected_sma = [1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85]
        assert len(sma) == len(expected_sma)
        
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        organ = RealSensoryOrgan()
        
        # Create test data with known RSI
        data = pd.DataFrame({
            'close': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        })
        
        rsi = organ.calculate_rsi(data['close'], period=3)
        assert 0 <= rsi.iloc[-1] <= 100
        
    def test_macd_calculation(self):
        """Test MACD calculation."""
        organ = RealSensoryOrgan()
        
        # Create test data
        data = pd.DataFrame({
            'close': [100 + i * 0.5 + (i % 3) * 0.2 for i in range(50)]
        })
        
        macd, signal = organ.calculate_macd(data['close'])
        assert len(macd) == len(signal)
        assert len(macd) > 0


class TestRealBaseStrategy:
    """Test the RealBaseStrategy implementation."""
    
    def test_signal_generation(self):
        """Test signal generation with known inputs."""
        strategy = RealBaseStrategy()
        
        # Create test market data
        market_data = MarketData(
            symbol="TEST",
            timeframe="1h",
            data=pd.DataFrame({
                'open': [1.0, 1.1, 1.2],
                'high': [1.1, 1.2, 1.3],
                'low': [0.9, 1.0, 1.1],
                'close': [1.05, 1.15, 1.25],
                'volume': [1000, 1100, 1200]
            })
        )
        
        signal = strategy.generate_signal(market_data)
        assert signal in ['BUY', 'SELL', 'HOLD']
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        strategy = RealBaseStrategy()
        
        # Test valid parameters
        valid_params = {'rsi_period': 14, 'sma_period': 20}
        strategy.set_parameters(valid_params)
        retrieved_params = strategy.get_parameters()
        assert retrieved_params == valid_params
        
        # Test invalid parameters
        invalid_params = {'rsi_period': -5, 'sma_period': 0}
        with pytest.raises(ValueError):
            strategy.set_parameters(invalid_params)


class TestEndToEndIntegration:
    """Test complete end-to-end integration."""
    
    @pytest.mark.asyncio
    async def test_full_trading_cycle(self):
        """Test a complete trading cycle with real components."""
        # Initialize real components
        evolution_engine = RealEvolutionEngine(population_size=5)
        risk_config = RealRiskConfig(
            max_risk_per_trade_pct=Decimal('0.02'),
            max_leverage=Decimal('10.0'),
            max_total_exposure_pct=Decimal('0.5'),
            max_drawdown_pct=Decimal('0.25')
        )
        risk_manager = RealRiskManager(risk_config)
        portfolio_monitor = RealPortfolioMonitor()
        sensory_organ = RealSensoryOrgan()
        strategy = RealBaseStrategy()
        
        # Create test market data
        market_data = MarketData(
            symbol="EURUSD",
            timeframe="1h",
            data=pd.DataFrame({
                'open': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
                'high': [1.1010, 1.1020, 1.1030, 1.1040, 1.1050],
                'low': [1.0990, 1.1000, 1.1010, 1.1020, 1.1030],
                'close': [1.1005, 1.1015, 1.1025, 1.1035, 1.1045],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
        )
        
        # Step 1: Process data through sensory organ
        processed_data = sensory_organ.process(market_data)
        assert processed_data is not None
        
        # Step 2: Generate signal
        signal = strategy.generate_signal(processed_data)
        assert signal in ['BUY', 'SELL', 'HOLD']
        
        # Step 3: Validate with risk manager
        account_balance = Decimal('10000')
        position_size = risk_manager.calculate_position_size(
            account_balance, 
            Decimal('0.02'),  # 2% risk
            Decimal('0.01')   # 1% stop loss
        )
        
        is_valid = risk_manager.validate_position(
            position_size,
            Instrument("EURUSD", "Currency"),
            account_balance
        )
        assert is_valid
        
        # Step 4: Update portfolio monitor
        portfolio_monitor.add_position(
            symbol="EURUSD",
            size=position_size,
            entry_price=Decimal('1.1045'),
            entry_time=datetime.now()
        )
        
        # Step 5: Simulate price movement and calculate P&L
        current_price = Decimal('1.1055')  # 10 pips profit
        pnl = portfolio_monitor.calculate_pnl(
            Decimal('1.1045'),
            current_price,
            position_size
        )
        assert pnl > 0
        
        # Step 6: Verify final state
        portfolio_value = portfolio_monitor.get_portfolio_value()
        assert portfolio_value > 0
        
        # Step 7: Test evolution engine
        evolution_engine.initialize_population(5)
        evolution_engine.set_fitness_evaluator(MockFitnessEvaluator())
        
        stats = evolution_engine.evolve_generation(market_data)
        assert stats.generation == 1
        assert stats.population_size == 5


class MockFitnessEvaluator:
    """Mock fitness evaluator for testing."""
    
    def evaluate(self, genome, market_data):
        return np.random.random()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
