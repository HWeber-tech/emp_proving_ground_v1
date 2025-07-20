#!/usr/bin/env python3
"""
EMP Evolution Integration Test v1.1

Tests the complete evolution integration across all layers
including genetic engine, human approval, and strategy registry.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.configuration import Configuration
from src.core.event_bus import event_bus
from src.core.interfaces import MarketData, SensorySignal
from src.sensory.integration.sensory_cortex import SensoryCortex
from src.sensory.organs.price_organ import PriceOrgan
from src.thinking.patterns.trend_detector import TrendDetector
from src.thinking.analysis.risk_analyzer import RiskAnalyzer
from src.thinking.analysis.performance_analyzer import PerformanceAnalyzer
from src.evolution.engine.genetic_engine import GeneticEngine, EvolutionConfig
from src.evolution.engine.population_manager import PopulationManager
from src.governance.human_gateway import HumanApprovalGateway, ApprovalLevel
from src.governance.strategy_registry import StrategyRegistry
from src.governance.fitness_store import FitnessStore
from src.genome.models.genome import DecisionGenome

logger = logging.getLogger(__name__)


class EvolutionIntegrationTest:
    """Test the complete evolution integration."""
    
    def __init__(self):
        self.config = None
        self.sensory_cortex = None
        self.trend_detector = None
        self.risk_analyzer = None
        self.performance_analyzer = None
        self.genetic_engine = None
        self.population_manager = None
        self.human_gateway = None
        self.strategy_registry = None
        self.fitness_store = None
        self.test_results = {}
        
    async def setup(self):
        """Set up the test environment."""
        logger.info("Setting up evolution integration test")
        
        # Create test configuration
        self.config = Configuration(
            system_name="EMP Evolution Test",
            system_version="1.1.0",
            environment="test",
            debug=True
        )
        
        # Initialize sensory layer
        self.sensory_cortex = SensoryCortex()
        price_organ = PriceOrgan()
        self.sensory_cortex.register_organ('price', price_organ)
        await self.sensory_cortex.start()
        
        # Initialize thinking layer
        self.trend_detector = TrendDetector()
        self.risk_analyzer = RiskAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Initialize evolution components
        evolution_config = EvolutionConfig(
            population_size=20,
            elite_size=2,
            tournament_size=3,
            mutation_rate=0.1,
            crossover_rate=0.8,
            max_generations=5,
            fitness_threshold=0.8
        )
        
        self.genetic_engine = GeneticEngine(evolution_config)
        self.population_manager = PopulationManager(max_population_size=50)
        
        # Set up fitness evaluator
        self.genetic_engine.fitness_evaluator = self._evaluate_genome_fitness
        
        # Initialize governance components
        self.human_gateway = HumanApprovalGateway()
        self.strategy_registry = StrategyRegistry()
        self.fitness_store = FitnessStore()
        
        # Start event bus
        await event_bus.start()
        
        logger.info("Evolution test environment setup complete")
        
    async def teardown(self):
        """Clean up the test environment."""
        logger.info("Tearing down evolution test environment")
        
        # Stop sensory cortex
        if self.sensory_cortex:
            await self.sensory_cortex.stop()
            
        # Stop event bus
        await event_bus.stop()
        
        logger.info("Evolution test environment cleanup complete")
        
    def create_test_market_data(self) -> MarketData:
        """Create test market data."""
        return MarketData(
            symbol="EURUSD",
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=1000,
            bid=1.1004,
            ask=1.1006,
            source="test",
            latency_ms=1.0
        )
        
    def create_test_sensory_signals(self) -> List[SensorySignal]:
        """Create test sensory signals."""
        return [
            SensorySignal(
                timestamp=datetime.now(),
                signal_type="price_composite",
                value=0.6,
                confidence=0.8,
                metadata={'organ_id': 'price_organ'}
            ),
            SensorySignal(
                timestamp=datetime.now(),
                signal_type="volume_composite",
                value=0.4,
                confidence=0.7,
                metadata={'organ_id': 'volume_organ'}
            )
        ]
        
    async def _evaluate_genome_fitness(self, genome: DecisionGenome) -> float:
        """Evaluate fitness for a genome using sensory and thinking layers."""
        try:
            # Create test market data
            market_data = self.create_test_market_data()
            
            # Process through sensory layer
            sensory_reading = await self.sensory_cortex.process_market_data(market_data)
            
            # Create sensory signals
            sensory_signals = self.create_test_sensory_signals()
            
            # Apply genome weights to sensory signals
            weighted_signals = self._apply_genome_weights(genome, sensory_signals)
            
            # Analyze through thinking layer
            trend_analysis = self.trend_detector.analyze(weighted_signals)
            risk_analysis = self.risk_analyzer.analyze(weighted_signals)
            
            # Create mock trading data
            trading_data = {
                'equity_curve': [100000, 100100, 100200, 100150, 100300, 100250, 100400],
                'trade_history': [
                    {
                        'entry_time': datetime.now(),
                        'exit_time': datetime.now(),
                        'entry_price': 1.1000,
                        'exit_price': 1.1010,
                        'quantity': 10000,
                        'pnl': 100,
                        'type': 'LONG_CLOSE'
                    }
                ]
            }
            
            performance_analysis = self.performance_analyzer.analyze(trading_data)
            
            # Calculate fitness based on analysis results
            fitness_score = self._calculate_fitness_score(
                genome, trend_analysis, risk_analysis, performance_analysis
            )
            
            return fitness_score
            
        except Exception as e:
            logger.error(f"Error evaluating genome fitness: {e}")
            return 0.0
            
    def _apply_genome_weights(self, genome: DecisionGenome, 
                            signals: List[SensorySignal]) -> List[SensorySignal]:
        """Apply genome weights to sensory signals."""
        weighted_signals = []
        
        for signal in signals:
            # Apply sensory weights based on signal type
            weight = 1.0
            if 'price' in signal.signal_type:
                weight = genome.sensory.price_weight
            elif 'volume' in signal.signal_type:
                weight = genome.sensory.volume_weight
            elif 'orderbook' in signal.signal_type:
                weight = genome.sensory.orderbook_weight
                
            # Create weighted signal
            weighted_signal = SensorySignal(
                timestamp=signal.timestamp,
                signal_type=signal.signal_type,
                value=signal.value * weight,
                confidence=signal.confidence * weight,
                metadata=signal.metadata
            )
            weighted_signals.append(weighted_signal)
            
        return weighted_signals
        
    def _calculate_fitness_score(self, genome: DecisionGenome,
                               trend_analysis, risk_analysis, 
                               performance_analysis) -> float:
        """Calculate fitness score from analysis results."""
        try:
            # Extract analysis results
            trend_score = trend_analysis.result.get('trend_strength', 0.0)
            risk_score = risk_analysis.result.get('risk_level', 0.5)
            performance_score = performance_analysis.result.get('total_return', 0.0)
            
            # Apply genome thinking weights
            weighted_trend = trend_score * genome.thinking.trend_analysis_weight
            weighted_risk = (1.0 - risk_score) * genome.thinking.risk_analysis_weight
            weighted_performance = max(0, performance_score) * genome.thinking.performance_analysis_weight
            
            # Calculate composite fitness
            fitness_score = (weighted_trend + weighted_risk + weighted_performance) / 3.0
            
            # Apply risk tolerance adjustment
            risk_adjustment = 1.0 - (risk_score * genome.risk.risk_tolerance)
            fitness_score *= risk_adjustment
            
            # Normalize to [0, 1]
            fitness_score = max(0.0, min(1.0, fitness_score))
            
            return fitness_score
            
        except Exception as e:
            logger.error(f"Error calculating fitness score: {e}")
            return 0.0
            
    async def test_genetic_evolution(self):
        """Test the genetic evolution process."""
        logger.info("Testing genetic evolution")
        
        try:
            # Initialize population
            await self.genetic_engine.initialize_population()
            
            # Run evolution
            best_genome = await self.genetic_engine.evolve(max_generations=3)
            
            # Verify evolution results
            assert best_genome is not None
            assert best_genome.fitness_score > 0
            assert best_genome.generation > 0
            
            # Check evolution summary
            summary = self.genetic_engine.get_evolution_summary()
            assert summary['total_generations'] > 0
            assert summary['best_fitness'] > 0
            
            self.test_results['genetic_evolution'] = {
                'status': 'PASS',
                'best_fitness': best_genome.fitness_score,
                'total_generations': summary['total_generations'],
                'population_size': summary['population_size']
            }
            
            logger.info("Genetic evolution test PASSED")
            
        except Exception as e:
            logger.error(f"Genetic evolution test FAILED: {e}")
            self.test_results['genetic_evolution'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_human_approval(self):
        """Test the human approval process."""
        logger.info("Testing human approval process")
        
        try:
            # Create test genome
            genome = DecisionGenome(genome_id="test_approval_genome")
            
            # Create test data
            risk_assessment = {
                'overall_risk': 0.3,
                'var_95': -0.02,
                'max_drawdown': 0.08
            }
            
            performance_metrics = {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'win_rate': 0.6
            }
            
            compliance_check = {
                'compliance_score': 0.9,
                'regulatory_checks': ['passed']
            }
            
            # Request approval
            request_id = await self.human_gateway.request_approval(
                genome, "test_user", risk_assessment, performance_metrics, compliance_check
            )
            
            # Verify request was created
            request = self.human_gateway.get_request(request_id)
            assert request is not None
            assert request.genome_id == genome.genome_id
            
            # Test approval
            approval_success = await self.human_gateway.approve_request(
                request_id, "test_approver", "Test approval"
            )
            assert approval_success is True
            
            # Verify approval
            updated_request = self.human_gateway.get_request(request_id)
            assert updated_request.status.value == 'approved'
            
            self.test_results['human_approval'] = {
                'status': 'PASS',
                'request_id': request_id,
                'approval_level': request.approval_level.value,
                'final_status': updated_request.status.value
            }
            
            logger.info("Human approval test PASSED")
            
        except Exception as e:
            logger.error(f"Human approval test FAILED: {e}")
            self.test_results['human_approval'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_strategy_registry(self):
        """Test the strategy registry functionality."""
        logger.info("Testing strategy registry")
        
        try:
            # Create test genome
            genome = DecisionGenome(genome_id="test_registry_genome")
            
            # Create strategy metadata
            metadata = {
                'name': 'Test Evolved Strategy',
                'description': 'Test strategy for registry',
                'version': '1.0.0',
                'author': 'Test User',
                'tags': ['test', 'evolved'],
                'risk_level': 'moderate',
                'expected_return': 0.12,
                'max_drawdown': 0.08,
                'sharpe_ratio': 1.1,
                'sortino_ratio': 1.3,
                'win_rate': 0.58,
                'profit_factor': 1.4,
                'total_trades': 100,
                'avg_trade_duration': 2.5,
                'instruments': ['EURUSD', 'GBPUSD'],
                'timeframes': ['1H', '4H'],
                'constraints': {'max_position_size': 0.1},
                'dependencies': []
            }
            
            # Register strategy
            strategy_id = await self.strategy_registry.register_strategy(genome, metadata)
            
            # Verify registration
            strategy = self.strategy_registry.get_strategy(strategy_id)
            assert strategy is not None
            assert strategy.name == 'Test Evolved Strategy'
            
            # Approve strategy
            approval_success = await self.strategy_registry.approve_strategy(
                strategy_id, "test_approver"
            )
            assert approval_success is True
            
            # Activate strategy
            activation_success = await self.strategy_registry.activate_strategy(
                strategy_id, "test_activator"
            )
            assert activation_success is True
            
            # Verify activation
            active_strategies = self.strategy_registry.get_active_strategies()
            assert len(active_strategies) > 0
            assert any(s.strategy_id == strategy_id for s in active_strategies)
            
            self.test_results['strategy_registry'] = {
                'status': 'PASS',
                'strategy_id': strategy_id,
                'active_strategies': len(active_strategies),
                'total_strategies': len(self.strategy_registry.strategies)
            }
            
            logger.info("Strategy registry test PASSED")
            
        except Exception as e:
            logger.error(f"Strategy registry test FAILED: {e}")
            self.test_results['strategy_registry'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_end_to_end_evolution(self):
        """Test complete end-to-end evolution workflow."""
        logger.info("Testing end-to-end evolution workflow")
        
        try:
            # Step 1: Run genetic evolution
            await self.genetic_engine.initialize_population()
            best_genome = await self.genetic_engine.evolve(max_generations=2)
            
            # Step 2: Create mock analysis data
            risk_assessment = {
                'overall_risk': 0.25,
                'var_95': -0.015,
                'max_drawdown': 0.06
            }
            
            performance_metrics = {
                'total_return': 0.18,
                'sharpe_ratio': 1.4,
                'win_rate': 0.62
            }
            
            compliance_check = {
                'compliance_score': 0.95,
                'regulatory_checks': ['passed']
            }
            
            # Step 3: Request human approval
            request_id = await self.human_gateway.request_approval(
                best_genome, "evolution_engine", risk_assessment, 
                performance_metrics, compliance_check
            )
            
            # Step 4: Approve strategy
            await self.human_gateway.approve_request(
                request_id, "test_approver", "Auto-approved for testing"
            )
            
            # Step 5: Register strategy
            metadata = {
                'name': f'Evolved Strategy {best_genome.genome_id}',
                'description': 'Strategy evolved through genetic algorithm',
                'version': '1.0.0',
                'author': 'EMP Evolution Engine',
                'tags': ['evolved', 'genetic', 'automated'],
                'risk_level': 'moderate',
                'expected_return': performance_metrics['total_return'],
                'max_drawdown': risk_assessment['max_drawdown'],
                'sharpe_ratio': performance_metrics['sharpe_ratio'],
                'win_rate': performance_metrics['win_rate'],
                'instruments': ['EURUSD'],
                'timeframes': ['1H']
            }
            
            strategy_id = await self.strategy_registry.register_strategy(best_genome, metadata)
            
            # Step 6: Activate strategy
            await self.strategy_registry.activate_strategy(strategy_id, "test_activator")
            
            # Verify end-to-end success
            strategy = self.strategy_registry.get_strategy(strategy_id)
            assert strategy is not None
            assert strategy.status.value == 'active'
            
            active_strategies = self.strategy_registry.get_active_strategies()
            assert len(active_strategies) > 0
            
            self.test_results['end_to_end_evolution'] = {
                'status': 'PASS',
                'best_genome_fitness': best_genome.fitness_score,
                'strategy_id': strategy_id,
                'active_strategies': len(active_strategies),
                'workflow_completed': True
            }
            
            logger.info("End-to-end evolution test PASSED")
            
        except Exception as e:
            logger.error(f"End-to-end evolution test FAILED: {e}")
            self.test_results['end_to_end_evolution'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def run_all_tests(self):
        """Run all evolution integration tests."""
        logger.info("Starting EMP Evolution Integration Tests")
        
        try:
            await self.setup()
            
            # Run individual component tests
            await self.test_genetic_evolution()
            await self.test_human_approval()
            await self.test_strategy_registry()
            await self.test_end_to_end_evolution()
            
            # Generate test report
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"Evolution integration test failed: {e}")
        finally:
            await self.teardown()
            
    def generate_test_report(self):
        """Generate a comprehensive test report."""
        logger.info("Generating evolution integration test report")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*70)
        print("EMP EVOLUTION INTEGRATION TEST REPORT v1.1")
        print("="*70)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*70)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['status'] == 'PASS' else "❌ FAIL"
            print(f"{test_name:25} {status}")
            if result['status'] == 'FAIL':
                print(f"  Error: {result['error']}")
            else:
                for key, value in result.items():
                    if key != 'status':
                        print(f"  {key}: {value}")
            print()
            
        print("="*70)
        
        if failed_tests == 0:
            print("🎉 ALL EVOLUTION TESTS PASSED - INTEGRATION SUCCESSFUL!")
            print("🚀 EMP Ultimate Architecture v1.1 Evolution Integration Complete!")
        else:
            print(f"⚠️  {failed_tests} TESTS FAILED - EVOLUTION INTEGRATION ISSUES DETECTED")
        print("="*70)


async def main():
    """Main test runner."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run evolution integration tests
    test_runner = EvolutionIntegrationTest()
    await test_runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 