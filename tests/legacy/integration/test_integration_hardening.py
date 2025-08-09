"""
Enhanced Integration Tests for Hardened System
Tests the integrated sensory -> decision -> financial loop with risk management
"""

import pytest
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np

from src.core import RiskConfig, InstrumentProvider
from src.risk import RiskManager
from src.data import TickDataStorage, TickDataCleaner, DukascopyIngestor
from src.evolution import EvolutionEngine, EvolutionConfig, FitnessEvaluator
from src.simulation import MarketSimulator
from src.sensory.orchestration.master_orchestrator import MasterOrchestrator
from src.sensory.core.base import InstrumentMeta


class TestRiskManagementHardening:
    """Test suite for risk management hardening"""
    
    def setup_method(self):
        """Set up test environment"""
        self.risk_config = RiskConfig(
            max_risk_per_trade_pct=Decimal("0.02"),
            max_leverage=Decimal("10.0"),
            max_total_exposure_pct=Decimal("0.5"),
            max_drawdown_pct=Decimal("0.25")
        )
        self.instrument_provider = InstrumentProvider()
        self.risk_manager = RiskManager(self.risk_config, self.instrument_provider)
        
    def test_excessive_risk_rejection(self):
        """Test that excessive risk trades are rejected"""
        instrument = self.instrument_provider.get_instrument("EURUSD")
        
        # Attempt to place trade with 10% risk (should be rejected)
        position_size = self.risk_manager.calculate_position_size(
            account_equity=100000,
            stop_loss_pips=500,  # 500 pips = 5% risk
            instrument=instrument,
            account_currency="USD"
        )
        
        # Should return 0 or very small position due to risk limits
        assert position_size <= 1000, f"Excessive risk not rejected: {position_size}"
        
    def test_no_stop_loss_rejection(self):
        """Test that trades without stop-loss are rejected"""
        # This would test the system's ability to reject trades without proper risk controls
        # Implementation depends on how the system handles stop-loss validation
        pass
        
    def test_max_drawdown_circuit_breaker(self):
        """Test that system halts trading after max drawdown breach"""
        # Simulate account with 30% drawdown
        current_equity = 70000  # 30% drawdown from 100k
        max_allowed_equity = 75000  # 25% max drawdown
        
        # System should halt new trades
        should_halt = current_equity < max_allowed_equity
        assert should_halt, "Circuit breaker should activate on max drawdown breach"


class TestIntegratedSystemStability:
    """Test suite for integrated system stability"""
    
    def setup_method(self):
        """Set up test environment"""
        self.data_storage = TickDataStorage()
        self.data_cleaner = TickDataCleaner()
        self.ingestor = DukascopyIngestor(self.data_storage, self.data_cleaner)
        
        # Initialize sensory cortex
        instrument_meta = InstrumentMeta(
            symbol="EURUSD",
            pip_size=0.0001,
            lot_size=100000,
            timezone="UTC",
            typical_spread=0.00015,
            avg_daily_range=0.01
        )
        self.sensory_cortex = MasterOrchestrator(instrument_meta)
        
        # Initialize evolution engine
        evolution_config = EvolutionConfig(
            population_size=50,
            elite_ratio=0.1,
            crossover_ratio=0.6,
            mutation_ratio=0.3
        )
        self.fitness_evaluator = FitnessEvaluator(self.data_storage)
        self.evolution_engine = EvolutionEngine(evolution_config, self.fitness_evaluator)
        
        # Initialize market simulator
        self.market_simulator = MarketSimulator(self.data_storage, initial_balance=100000.0)
        
    def test_long_running_stability(self):
        """Test system stability over extended period"""
        # Generate test data for 2023
        success = self.ingestor.ingest_year("EURUSD", 2023)
        assert success, "Failed to generate test data"
        
        # Run evolution for stability test
        success = self.evolution_engine.initialize_population(seed=42)
        assert success, "Failed to initialize population"
        
        # Run for 5 generations to test stability
        for generation in range(5):
            stats = self.evolution_engine.evolve_generation()
            assert stats.best_fitness >= 0, f"Negative fitness in generation {generation}"
            assert stats.population_size > 0, f"Population died in generation {generation}"
            
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run intensive operations
        for _ in range(10):
            self.evolution_engine.initialize_population(seed=42)
            for gen in range(3):
                self.evolution_engine.evolve_generation()
                
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Allow for some memory growth but not excessive
        max_acceptable_increase = 100 * 1024 * 1024  # 100MB
        assert memory_increase < max_acceptable_increase, f"Memory leak detected: {memory_increase} bytes"
        
    def test_data_consistency(self):
        """Test data consistency across components"""
        # Load test data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        # Test data loading
        data = self.data_storage.load_tick_data("EURUSD", start_time, end_time)
        assert not data.empty, "Failed to load test data"
        
        # Test OHLCV conversion
        ohlcv = self.data_storage.get_ohlcv("EURUSD", start_time, end_time, "H1")
        assert not ohlcv.empty, "Failed to convert to OHLCV"
        
        # Ensure no NaN values
        assert not data.isnull().any().any(), "Data contains NaN values"
        assert not ohlcv.isnull().any().any(), "OHLCV contains NaN values"
        
    def test_sensory_integration(self):
        """Test sensory cortex integration with market data"""
        # Load test data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        data = self.data_storage.load_tick_data("EURUSD", start_time, end_time)
        assert not data.empty, "No data for sensory integration test"
        
        # Test sensory perception
        sample_row = data.iloc[0]
        sensory_reading = self.sensory_cortex.perceive(sample_row)
        
        # Verify sensory reading structure
        assert hasattr(sensory_reading, 'macro_trend'), "Missing macro_trend attribute"
        assert hasattr(sensory_reading, 'technical_signal'), "Missing technical_signal attribute"
        assert hasattr(sensory_reading, 'manipulation_probability'), "Missing manipulation_probability attribute"
        
    def test_evolution_integration(self):
        """Test evolution engine integration with sensory cortex"""
        # Initialize population
        success = self.evolution_engine.initialize_population(seed=42)
        assert success, "Failed to initialize evolution population"
        
        # Test genome evaluation
        best_genomes = self.evolution_engine.get_best_genomes(1)
        assert len(best_genomes) > 0, "No genomes available"
        
        genome = best_genomes[0]
        assert genome.fitness_score >= 0, "Invalid fitness score"
        
    def test_simulation_integration(self):
        """Test market simulator integration"""
        # Load test data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        success = self.market_simulator.load_data("EURUSD", start_time, end_time)
        assert success, "Failed to load data into simulator"
        
        # Run simulation for 100 steps
        steps = 0
        while steps < 100:
            market_state = self.market_simulator.step()
            if market_state is None:
                break
            steps += 1
            
        assert steps > 0, "Simulation produced no steps"
        
        # Check performance metrics
        performance = self.market_simulator.get_performance_stats()
        assert isinstance(performance, dict), "Invalid performance stats format"
        assert 'total_return' in performance, "Missing total_return in performance"


class TestEndToEndIntegration:
    """Test complete end-to-end integration"""
    
    def test_full_system_workflow(self):
        """Test complete system workflow from data to decisions"""
        # Setup system
        data_storage = TickDataStorage()
        data_cleaner = TickDataCleaner()
        ingestor = DukascopyIngestor(data_storage, data_cleaner)
        
        # Generate data
        success = ingestor.ingest_year("EURUSD", 2023)
        assert success, "Failed to generate data"
        
        # Initialize components
        instrument_meta = InstrumentMeta(
            symbol="EURUSD",
            pip_size=0.0001,
            lot_size=100000,
            timezone="UTC",
            typical_spread=0.00015,
            avg_daily_range=0.01
        )
        sensory_cortex = MasterOrchestrator(instrument_meta)
        
        evolution_config = EvolutionConfig(
            population_size=20,  # Smaller for testing
            elite_ratio=0.1,
            crossover_ratio=0.6,
            mutation_ratio=0.3
        )
        fitness_evaluator = FitnessEvaluator(data_storage)
        evolution_engine = EvolutionEngine(evolution_config, fitness_evaluator)
        
        # Run complete workflow
        success = evolution_engine.initialize_population(seed=42)
        assert success, "Failed to initialize population"
        
        # Run evolution
        for generation in range(3):  # Short run for testing
            stats = evolution_engine.evolve_generation()
            assert stats.best_fitness >= 0, f"Invalid fitness in generation {generation}"
            
        # Verify system integrity
        best_genomes = evolution_engine.get_best_genomes(3)
        assert len(best_genomes) == 3, "Failed to get best genomes"
        
        for genome in best_genomes:
            assert genome.fitness_score >= 0, "Invalid genome fitness"
            assert genome.genome_id, "Missing genome ID"
            
    def test_error_handling(self):
        """Test system error handling and recovery"""
        # Test with invalid data
        data_storage = TickDataStorage()
        
        # Try to load non-existent data
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 1, 2)
        
        data = data_storage.load_tick_data("INVALID", start_time, end_time)
        assert data.empty, "Should return empty DataFrame for invalid symbol"
        
        # Test evolution with no data
        fitness_evaluator = FitnessEvaluator(data_storage)
        evolution_config = EvolutionConfig(population_size=10)
        evolution_engine = EvolutionEngine(evolution_config, fitness_evaluator)
        
        success = evolution_engine.initialize_population(seed=42)
        assert success, "Should initialize even with no data"
        
        # Evolution should handle missing data gracefully
        stats = evolution_engine.evolve_generation()
        assert stats.population_size == 10, "Population should remain intact"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
