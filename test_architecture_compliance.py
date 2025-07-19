"""
EMP Architecture Compliance Test v1.1

Comprehensive test to validate EMP Ultimate Architecture v1.1 compliance
and integration across all layers.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.events import MarketData, TradeIntent, FitnessReport, MarketUnderstanding
from core.event_bus import EventBus, start_event_bus, stop_event_bus
from sensory.integration.sensory_cortex import SensoryCortex, SensoryOrgan
from thinking.patterns.regime_classifier import RegimeClassifier
from thinking.analysis.performance_analyzer import PerformanceAnalyzer
from thinking.analysis.risk_analyzer import RiskAnalyzer
from simulation.evaluation.fitness_evaluator import FitnessEvaluator, EvaluationContext
from evolution.engine.genetic_engine import GeneticEngine
from evolution.selection.selection_strategies import SelectionFactory
from evolution.variation.variation_strategies import VariationFactory
from governance.fitness_store import FitnessStore
from governance.strategy_registry import StrategyRegistry, StrategyRecord, StrategyStatus
from genome.models.genome import StrategyGenome

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArchitectureComplianceTest:
    """Test suite for EMP v1.1 architecture compliance."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all architecture compliance tests."""
        logger.info("Starting EMP v1.1 Architecture Compliance Tests")
        
        try:
            # Test 1: Core Infrastructure
            await self.test_core_infrastructure()
            
            # Test 2: Sensory Layer
            await self.test_sensory_layer()
            
            # Test 3: Thinking Layer
            await self.test_thinking_layer()
            
            # Test 4: Simulation Envelope
            await self.test_simulation_envelope()
            
            # Test 5: Adaptive Core
            await self.test_adaptive_core()
            
            # Test 6: Governance Layer
            await self.test_governance_layer()
            
            # Test 7: Integration Tests
            await self.test_integration()
            
            # Test 8: Event Flow
            await self.test_event_flow()
            
            # Generate compliance report
            compliance_report = self.generate_compliance_report()
            
            logger.info("All architecture compliance tests completed")
            return compliance_report
            
        except Exception as e:
            logger.error(f"Error in architecture compliance tests: {e}")
            return {"error": str(e), "compliance": 0}
            
    async def test_core_infrastructure(self):
        """Test core infrastructure components."""
        logger.info("Testing Core Infrastructure...")
        
        # Test event bus
        event_bus = EventBus()
        await event_bus.start()
        
        # Test event creation
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000
        )
        
        # Test event publishing
        await event_bus.publish(market_data)
        
        await event_bus.stop()
        
        self.test_results["core_infrastructure"] = "PASS"
        logger.info("✓ Core Infrastructure: PASS")
        
    async def test_sensory_layer(self):
        """Test sensory layer components."""
        logger.info("Testing Sensory Layer...")
        
        # Create sensory cortex
        cortex = SensoryCortex()
        
        # Register sensory organs
        price_organ = SensoryOrgan("price_organ", "price_organ", weight=1.0)
        volume_organ = SensoryOrgan("volume_organ", "volume_organ", weight=0.8)
        
        cortex.register_organ(price_organ)
        cortex.register_organ(volume_organ)
        
        # Test market data processing
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000
        )
        
        understanding = await cortex.process_market_data(market_data)
        
        # Validate understanding
        assert isinstance(understanding, MarketUnderstanding)
        assert understanding.symbol == "AAPL"
        assert len(understanding.signals) > 0
        
        self.test_results["sensory_layer"] = "PASS"
        logger.info("✓ Sensory Layer: PASS")
        
    async def test_thinking_layer(self):
        """Test thinking layer components."""
        logger.info("Testing Thinking Layer...")
        
        # Test regime classifier
        classifier = RegimeClassifier()
        
        # Create test market data
        market_data_list = []
        for i in range(100):
            data = MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open=150.0 + i * 0.1,
                high=155.0 + i * 0.1,
                low=149.0 + i * 0.1,
                close=153.0 + i * 0.1,
                volume=1000000
            )
            market_data_list.append(data)
            
        # Test regime classification
        regime_result = classifier.classify_regime(market_data_list)
        
        assert regime_result.analysis_type == "market_regime_classification"
        assert "primary_regime" in regime_result.result
        
        # Test performance analyzer
        analyzer = PerformanceAnalyzer()
        
        # Create test trade history
        trade_history = []
        for i in range(10):
            trade = TradeIntent(
                timestamp=datetime.now(),
                symbol="AAPL",
                action="BUY" if i % 2 == 0 else "SELL",
                quantity=100,
                price=150.0 + i,
                strategy_id="test_strategy",
                genome_id="test_genome"
            )
            trade_history.append(trade)
            
        # Test performance analysis
        performance_result = analyzer.analyze_performance(trade_history)
        
        assert performance_result.analysis_type == "performance_analysis"
        assert "performance_metrics" in performance_result.result
        
        # Test risk analyzer
        risk_analyzer = RiskAnalyzer()
        risk_result = risk_analyzer.analyze_risk(trade_history)
        
        assert risk_result.analysis_type == "risk_analysis"
        assert "risk_metrics" in risk_result.result
        
        self.test_results["thinking_layer"] = "PASS"
        logger.info("✓ Thinking Layer: PASS")
        
    async def test_simulation_envelope(self):
        """Test simulation envelope components."""
        logger.info("Testing Simulation Envelope...")
        
        # Create fitness evaluator
        evaluator = FitnessEvaluator()
        
        # Create test trade history
        trade_history = []
        for i in range(10):
            trade = TradeIntent(
                timestamp=datetime.now(),
                symbol="AAPL",
                action="BUY" if i % 2 == 0 else "SELL",
                quantity=100,
                price=150.0 + i,
                strategy_id="test_strategy",
                genome_id="test_genome"
            )
            trade_history.append(trade)
            
        # Create evaluation context
        context = EvaluationContext(
            strategy_id="test_strategy",
            genome_id="test_genome",
            generation=1,
            initial_capital=100000.0,
            evaluation_period=252
        )
        
        # Test fitness evaluation
        fitness_report = await evaluator.evaluate_fitness(trade_history, context)
        
        assert isinstance(fitness_report, FitnessReport)
        assert fitness_report.strategy_id == "test_strategy"
        assert fitness_report.genome_id == "test_genome"
        
        self.test_results["simulation_envelope"] = "PASS"
        logger.info("✓ Simulation Envelope: PASS")
        
    async def test_adaptive_core(self):
        """Test adaptive core components."""
        logger.info("Testing Adaptive Core...")
        
        # Test selection strategies
        tournament_selection = SelectionFactory.create_strategy("tournament")
        roulette_selection = SelectionFactory.create_strategy("roulette_wheel")
        
        assert tournament_selection.name == "tournament_selection"
        assert roulette_selection.name == "roulette_wheel_selection"
        
        # Test variation strategies
        uniform_crossover = VariationFactory.create_crossover_strategy("uniform")
        gaussian_mutation = VariationFactory.create_mutation_strategy("gaussian")
        
        assert uniform_crossover.name == "uniform_crossover"
        assert gaussian_mutation.name == "gaussian_mutation"
        
        # Test genetic engine
        engine = GeneticEngine(
            population_size=50,
            elite_size=5,
            selection_strategy="tournament",
            crossover_strategy="uniform",
            mutation_strategy="gaussian"
        )
        
        # Test genome factory
        def genome_factory():
            return StrategyGenome()
            
        engine.initialize_population(genome_factory)
        
        assert len(engine.get_population()) == 50
        
        # Test evolution
        fitness_reports = []
        for i in range(10):
            report = FitnessReport(
                timestamp=datetime.now(),
                genome_id=f"genome_{i}",
                strategy_id=f"strategy_{i}",
                performance_metrics=None,
                risk_metrics=None,
                fitness_score=0.5 + i * 0.05,
                generation=1
            )
            fitness_reports.append(report)
            
        new_population = await engine.evolve_generation(fitness_reports)
        
        assert len(new_population) == 50
        
        self.test_results["adaptive_core"] = "PASS"
        logger.info("✓ Adaptive Core: PASS")
        
    async def test_governance_layer(self):
        """Test governance layer components."""
        logger.info("Testing Governance Layer...")
        
        # Test fitness store
        fitness_store = FitnessStore()
        fitness_store.load_definitions()
        
        definitions = fitness_store.list_definitions()
        assert len(definitions) > 0
        
        # Test strategy registry
        registry = StrategyRegistry()
        
        # Test strategy registration
        success = registry.register_strategy(
            strategy_id="test_strategy",
            genome_id="test_genome",
            name="Test Strategy",
            description="Test strategy for compliance testing",
            fitness_score=0.75,
            performance_metrics={"total_return": 0.15},
            risk_metrics={"max_drawdown": 0.10}
        )
        
        assert success
        
        # Test strategy retrieval
        strategy = registry.get_strategy("test_strategy")
        assert strategy is not None
        assert strategy.name == "Test Strategy"
        
        self.test_results["governance_layer"] = "PASS"
        logger.info("✓ Governance Layer: PASS")
        
    async def test_integration(self):
        """Test integration between layers."""
        logger.info("Testing Layer Integration...")
        
        # Test sensory -> thinking integration
        cortex = SensoryCortex()
        classifier = RegimeClassifier()
        
        # Process market data through sensory layer
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000
        )
        
        understanding = await cortex.process_market_data(market_data)
        
        # Use thinking layer to analyze
        market_data_list = [market_data] * 100
        regime_result = classifier.classify_regime(market_data_list)
        
        assert understanding is not None
        assert regime_result is not None
        
        # Test thinking -> simulation integration
        analyzer = PerformanceAnalyzer()
        evaluator = FitnessEvaluator()
        
        trade_history = []
        for i in range(10):
            trade = TradeIntent(
                timestamp=datetime.now(),
                symbol="AAPL",
                action="BUY" if i % 2 == 0 else "SELL",
                quantity=100,
                price=150.0 + i,
                strategy_id="test_strategy",
                genome_id="test_genome"
            )
            trade_history.append(trade)
            
        performance_result = analyzer.analyze_performance(trade_history)
        
        context = EvaluationContext(
            strategy_id="test_strategy",
            genome_id="test_genome",
            generation=1,
            initial_capital=100000.0,
            evaluation_period=252
        )
        
        fitness_report = await evaluator.evaluate_fitness(trade_history, context)
        
        assert performance_result is not None
        assert fitness_report is not None
        
        self.test_results["integration"] = "PASS"
        logger.info("✓ Layer Integration: PASS")
        
    async def test_event_flow(self):
        """Test event flow through the system."""
        logger.info("Testing Event Flow...")
        
        # Start event bus
        await start_event_bus()
        
        # Test event publishing and subscription
        events_received = []
        
        async def event_handler(event):
            events_received.append(event)
            
        # Subscribe to events
        from core.event_bus import subscribe_to_event
        subscribe_to_event(EventType.MARKET_UNDERSTANDING, event_handler)
        
        # Create and publish events
        cortex = SensoryCortex()
        market_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000
        )
        
        understanding = await cortex.process_market_data(market_data)
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        # Stop event bus
        await stop_event_bus()
        
        assert len(events_received) > 0
        
        self.test_results["event_flow"] = "PASS"
        logger.info("✓ Event Flow: PASS")
        
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Calculate compliance score
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASS")
        compliance_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "compliance_score": compliance_score,
                "test_duration_seconds": duration
            },
            "test_results": self.test_results,
            "architecture_compliance": {
                "sensory_layer": "COMPLIANT" if self.test_results.get("sensory_layer") == "PASS" else "NON_COMPLIANT",
                "thinking_layer": "COMPLIANT" if self.test_results.get("thinking_layer") == "PASS" else "NON_COMPLIANT",
                "simulation_envelope": "COMPLIANT" if self.test_results.get("simulation_envelope") == "PASS" else "NON_COMPLIANT",
                "adaptive_core": "COMPLIANT" if self.test_results.get("adaptive_core") == "PASS" else "NON_COMPLIANT",
                "governance_layer": "COMPLIANT" if self.test_results.get("governance_layer") == "PASS" else "NON_COMPLIANT",
                "integration": "COMPLIANT" if self.test_results.get("integration") == "PASS" else "NON_COMPLIANT",
                "event_flow": "COMPLIANT" if self.test_results.get("event_flow") == "PASS" else "NON_COMPLIANT"
            },
            "timestamp": end_time.isoformat(),
            "version": "1.1.0"
        }
        
        return report


async def main():
    """Main test execution."""
    test_suite = ArchitectureComplianceTest()
    report = await test_suite.run_all_tests()
    
    # Print results
    print("\n" + "="*60)
    print("EMP ULTIMATE ARCHITECTURE v1.1 COMPLIANCE REPORT")
    print("="*60)
    
    summary = report.get("test_summary", {})
    print(f"Total Tests: {summary.get('total_tests', 0)}")
    print(f"Passed Tests: {summary.get('passed_tests', 0)}")
    print(f"Failed Tests: {summary.get('failed_tests', 0)}")
    print(f"Compliance Score: {summary.get('compliance_score', 0):.1f}%")
    print(f"Test Duration: {summary.get('test_duration_seconds', 0):.2f} seconds")
    
    print("\nArchitecture Compliance:")
    compliance = report.get("architecture_compliance", {})
    for layer, status in compliance.items():
        print(f"  {layer.replace('_', ' ').title()}: {status}")
        
    print("\n" + "="*60)
    
    # Save report
    import json
    with open("ARCHITECTURE_COMPLIANCE_REPORT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
        
    print("Compliance report saved to ARCHITECTURE_COMPLIANCE_REPORT.json")


if __name__ == "__main__":
    asyncio.run(main()) 