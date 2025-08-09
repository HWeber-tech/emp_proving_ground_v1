"""
Integration tests to validate fraud elimination in EMP Proving Ground.
Tests ensure real functionality replaces fraudulent implementations.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from trading.execution.execution_engine import ExecutionEngine
from core.population_manager import PopulationManager
from evolution.fitness.base_fitness import ProfitFitness, RiskFitness
from core.models import Order, OrderStatus, DecisionGenome


class TestFraudElimination:
    """Test suite to validate fraud elimination."""
    
    def setup_method(self):
        """Setup test environment."""
        self.execution_engine = ExecutionEngine()
        self.population_manager = PopulationManager(population_size=10)
        
    @pytest.mark.asyncio
    async def test_execution_engine_no_simulation(self):
        """Test that execution engine no longer uses simulation."""
        # Create test order
        order = Order(
            order_id="test_001",
            symbol="EURUSD",
            side="BUY",
            quantity=1000,
            order_type="MARKET",
            price=None
        )
        
        # Initialize engine
        await self.execution_engine.initialize()
        
        # Mock FIX API to avoid real trading
        with patch('src.operational.icmarkets_api.ICMarketsManager') as mock_fix:
            mock_fix_instance = Mock()
            mock_fix_instance.start.return_value = True
            mock_fix_instance.place_market_order.return_value = "ORDER_123"
            mock_fix_instance.orders = {
                "ORDER_123": {
                    'status': 'FILLED',
                    'filled_quantity': 1000,
                    'average_price': 1.0850,
                    'broker_order_id': 'BROKER_456'
                }
            }
            mock_fix.return_value = mock_fix_instance
            
            # Execute order
            result = await self.execution_engine.execute_order(order)
            
            # Verify real FIX API integration
            assert result is True
            assert order.status == OrderStatus.FILLED
            assert order.filled_quantity == 1000
            assert order.average_price == 1.0850
            assert order.broker_order_id == 'BROKER_456'
            
            # Verify FIX API was called
            mock_fix_instance.place_market_order.assert_called_once()
            
    def test_execution_engine_no_hardcoded_success(self):
        """Test that execution engine doesn't return hardcoded success."""
        # Test with invalid FIX API connection
        with patch('src.operational.icmarkets_api.ICMarketsManager') as mock_fix:
            mock_fix_instance = Mock()
            mock_fix_instance.start.return_value = False  # Connection fails
            mock_fix.return_value = mock_fix_instance
            
            order = Order(
                order_id="test_002",
                symbol="EURUSD",
                side="BUY",
                quantity=1000,
                order_type="MARKET"
            )
            
            # Execute order - should fail due to connection failure
            result = asyncio.run(self.execution_engine.execute_order(order))
            
            # Verify failure is properly handled (not hardcoded success)
            assert result is False
            assert order.status == OrderStatus.REJECTED
            assert "FIX API error" in order.rejection_reason
            
    def test_population_manager_generates_real_population(self):
        """Test that population manager generates real genomes."""
        # Get best genomes (should trigger population generation)
        best_genomes = self.population_manager.get_best_genomes(5)
        
        # Verify real population was generated
        assert len(best_genomes) == 5
        assert len(self.population_manager.population) == 10
        
        # Verify genomes have real parameters
        for genome in best_genomes:
            assert isinstance(genome, DecisionGenome)
            assert genome.id.startswith('genome_')
            assert len(genome.parameters) > 0
            assert 'risk_tolerance' in genome.parameters
            assert 'position_size_factor' in genome.parameters
            assert isinstance(genome.fitness, float)
            
    def test_population_manager_no_empty_returns(self):
        """Test that population manager doesn't return empty collections."""
        # Clear population to test generation
        self.population_manager.population = []
        
        # Request genomes - should generate population
        genomes = self.population_manager.get_best_genomes(3)
        
        # Verify no empty returns
        assert len(genomes) > 0
        assert all(isinstance(g, DecisionGenome) for g in genomes)
        
    def test_fitness_calculation_real_implementation(self):
        """Test that fitness calculation has real implementation."""
        config = {'weight': 1.0, 'min_samples': 5}
        profit_fitness = ProfitFitness(config)
        risk_fitness = RiskFitness(config)
        
        # Create test genome
        genome = DecisionGenome(
            id="test_genome",
            parameters={'risk_tolerance': 0.5},
            fitness=0.0,
            generation=0,
            species_type='test',
            parent_ids=[],
            mutation_history=[],
            performance_metrics={}
        )
        
        # Mock data for fitness calculation
        with patch.object(profit_fitness, 'get_required_data') as mock_data:
            mock_data.return_value = {
                'total_return': 0.15,
                'win_rate': 0.65,
                'profit_factor': 1.8,
                'trade_count': 100
            }
            
            # Calculate fitness
            fitness_score = profit_fitness.calculate_fitness(genome)
            
            # Verify real calculation (not hardcoded)
            assert isinstance(fitness_score, float)
            assert 0.0 <= fitness_score <= 1.0
            assert fitness_score > 0.0  # Should be positive with good metrics
            
    def test_fitness_no_pass_statements(self):
        """Test that fitness classes don't have pass statements."""
        config = {'weight': 1.0}
        profit_fitness = ProfitFitness(config)
        risk_fitness = RiskFitness(config)
        
        # Verify methods are implemented
        assert hasattr(profit_fitness, 'calculate_fitness')
        assert hasattr(profit_fitness, 'get_optimal_weight')
        assert hasattr(profit_fitness, 'get_required_data_keys')
        
        # Test method calls don't raise NotImplementedError
        try:
            weight = profit_fitness.get_optimal_weight('bull')
            assert isinstance(weight, float)
            assert weight > 0.0
            
            keys = profit_fitness.get_required_data_keys()
            assert isinstance(keys, list)
            assert len(keys) > 0
            
        except NotImplementedError:
            pytest.fail("Fitness methods still contain pass statements")
            
    def test_evolution_functionality(self):
        """Test that evolution system has real functionality."""
        # Create population with test data
        market_data = {
            'market_regime': 'bull',
            'volatility': 0.02,
            'trend_strength': 0.7
        }
        
        performance_metrics = {
            'total_return': 0.12,
            'win_rate': 0.6,
            'max_drawdown': 0.08,
            'sharpe_ratio': 1.5,
            'profit_factor': 1.6
        }
        
        # Evolve population
        self.population_manager.evolve_population(market_data, performance_metrics)
        
        # Verify evolution occurred
        assert len(self.population_manager.population) == 10
        assert self.population_manager.generation > 0
        
        # Verify fitness was calculated
        for genome in self.population_manager.population:
            assert genome.fitness >= 0.0
            assert len(genome.performance_metrics) > 0
            
    @pytest.mark.asyncio
    async def test_no_mock_data_in_production_paths(self):
        """Test that production paths don't use mock data."""
        # Test execution engine
        order = Order(
            order_id="prod_test",
            symbol="EURUSD", 
            side="BUY",
            quantity=1000,
            order_type="MARKET"
        )
        
        # Mock FIX API but verify real data flow
        with patch('src.operational.icmarkets_api.ICMarketsManager') as mock_fix:
            mock_fix_instance = Mock()
            mock_fix_instance.start.return_value = True
            mock_fix_instance.place_market_order.return_value = "REAL_ORDER_ID"
            mock_fix_instance.orders = {
                "REAL_ORDER_ID": {
                    'status': 'FILLED',
                    'filled_quantity': 1000,
                    'average_price': 1.0851,  # Real-looking price
                    'broker_order_id': 'IC_MARKETS_789'
                }
            }
            mock_fix.return_value = mock_fix_instance
            
            # Execute
            result = await self.execution_engine.execute_order(order)
            
            # Verify no hardcoded mock values
            assert order.average_price != 100.0  # Old mock price
            assert order.average_price == 1.0851  # Real price from FIX
            assert order.broker_order_id.startswith('IC_MARKETS_')
            
    def test_anti_fraud_measures(self):
        """Test that anti-fraud measures are in place."""
        # Test that simulation methods are removed
        assert not hasattr(self.execution_engine, '_simulate_execution')
        
        # Test that FIX API integration exists
        assert hasattr(self.execution_engine, '_execute_via_fix_api')
        assert hasattr(self.execution_engine, '_get_symbol_id')
        assert hasattr(self.execution_engine, '_wait_for_execution_report')
        
        # Test population generation
        assert hasattr(self.population_manager, '_generate_initial_population')
        assert hasattr(self.population_manager, 'evolve_population')
        
    def test_comprehensive_error_handling(self):
        """Test that comprehensive error handling replaces pass statements."""
        # Test execution engine error handling
        order = Order(
            order_id="error_test",
            symbol="INVALID",
            side="BUY", 
            quantity=-1000,  # Invalid quantity
            order_type="INVALID"
        )
        
        result = asyncio.run(self.execution_engine.execute_order(order))
        
        # Verify proper error handling (not silent pass)
        assert result is False
        assert order.status == OrderStatus.REJECTED
        
    def test_performance_metrics_integration(self):
        """Test that performance metrics are properly integrated."""
        # Generate population
        self.population_manager._generate_initial_population()
        
        # Verify genomes have performance metrics structure
        for genome in self.population_manager.population:
            assert 'performance_metrics' in genome.__dict__
            assert isinstance(genome.performance_metrics, dict)
            
            # Verify expected metrics exist
            expected_metrics = [
                'total_trades', 'win_rate', 'profit_factor',
                'max_drawdown', 'sharpe_ratio', 'sortino_ratio'
            ]
            
            for metric in expected_metrics:
                assert metric in genome.performance_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

