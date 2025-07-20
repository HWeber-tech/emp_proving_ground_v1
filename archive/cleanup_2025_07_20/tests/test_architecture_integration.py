#!/usr/bin/env python3
"""
EMP Ultimate Architecture v1.1 - Integration Test

Tests the integration of all layers in the new architecture
to ensure proper communication and functionality.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.configuration import Configuration
from src.core.event_bus import event_bus
from src.core.interfaces import MarketData
from src.sensory.integration.sensory_cortex import SensoryCortex
from src.sensory.organs.price_organ import PriceOrgan
from src.sensory.organs.volume_organ import VolumeOrgan
from src.sensory.organs.orderbook_organ import OrderbookOrgan
from src.thinking.patterns.trend_detector import TrendDetector
from src.thinking.analysis.risk_analyzer import RiskAnalyzer
from src.thinking.analysis.performance_analyzer import PerformanceAnalyzer
from src.governance.fitness_store import FitnessStore
from src.operational.state_store import StateStore
from src.genome.models.genome import DecisionGenome

logger = logging.getLogger(__name__)


class ArchitectureIntegrationTest:
    """Test the integration of all architecture layers."""
    
    def __init__(self):
        self.config = None
        self.sensory_cortex = None
        self.trend_detector = None
        self.risk_analyzer = None
        self.performance_analyzer = None
        self.fitness_store = None
        self.state_store = None
        self.test_results = {}
        
    async def setup(self):
        """Set up the test environment."""
        logger.info("Setting up architecture integration test")
        
        # Create test configuration
        self.config = Configuration(
            system_name="EMP Test",
            system_version="1.1.0",
            environment="test",
            debug=True
        )
        
        # Initialize operational backbone
        self.state_store = StateStore({
            'connection': {
                'host': 'localhost',
                'port': 6379,
                'db': 1  # Use different DB for testing
            }
        })
        
        # Initialize governance
        self.fitness_store = FitnessStore()
        
        # Initialize sensory layer
        self.sensory_cortex = SensoryCortex()
        
        # Register sensory organs
        price_organ = PriceOrgan()
        volume_organ = VolumeOrgan()
        orderbook_organ = OrderbookOrgan()
        
        self.sensory_cortex.register_organ('price', price_organ)
        self.sensory_cortex.register_organ('volume', volume_organ)
        self.sensory_cortex.register_organ('orderbook', orderbook_organ)
        
        # Initialize thinking layer
        self.trend_detector = TrendDetector()
        self.risk_analyzer = RiskAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Start event bus
        await event_bus.start()
        
        # Start sensory cortex
        await self.sensory_cortex.start()
        
        logger.info("Test environment setup complete")
        
    async def teardown(self):
        """Clean up the test environment."""
        logger.info("Tearing down test environment")
        
        # Stop sensory cortex
        if self.sensory_cortex:
            await self.sensory_cortex.stop()
            
        # Stop event bus
        await event_bus.stop()
        
        # Close state store
        if self.state_store:
            self.state_store.close()
            
        logger.info("Test environment cleanup complete")
        
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
        
    def create_test_trading_data(self) -> Dict[str, Any]:
        """Create test trading data."""
        return {
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
                },
                {
                    'entry_time': datetime.now(),
                    'exit_time': datetime.now(),
                    'entry_price': 1.1010,
                    'exit_price': 1.1005,
                    'quantity': 10000,
                    'pnl': -50,
                    'type': 'SHORT_CLOSE'
                }
            ]
        }
        
    async def test_sensory_layer(self):
        """Test the sensory layer functionality."""
        logger.info("Testing sensory layer")
        
        try:
            # Create test market data
            market_data = self.create_test_market_data()
            
            # Process through sensory cortex
            sensory_reading = await self.sensory_cortex.process_market_data(market_data)
            
            # Verify sensory reading
            assert sensory_reading is not None
            assert sensory_reading.signal_type == "volume_composite"  # Last organ processed
            assert len(sensory_reading.metadata['signals']) > 0
            
            # Test organ calibration
            organ_status = self.sensory_cortex.get_organ_status()
            assert len(organ_status) == 3  # price, volume, orderbook
            
            self.test_results['sensory_layer'] = {
                'status': 'PASS',
                'organs_registered': len(organ_status),
                'signals_generated': len(sensory_reading.metadata['signals'])
            }
            
            logger.info("Sensory layer test PASSED")
            
        except Exception as e:
            logger.error(f"Sensory layer test FAILED: {e}")
            self.test_results['sensory_layer'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_thinking_layer(self):
        """Test the thinking layer functionality."""
        logger.info("Testing thinking layer")
        
        try:
            # Create test sensory signals
            from src.core.interfaces import SensorySignal
            
            sensory_signals = [
                SensorySignal(
                    timestamp=datetime.now(),
                    signal_type="price_composite",
                    value=0.5,
                    confidence=0.8,
                    metadata={'organ_id': 'price_organ'}
                ),
                SensorySignal(
                    timestamp=datetime.now(),
                    signal_type="volume_composite",
                    value=0.3,
                    confidence=0.7,
                    metadata={'organ_id': 'volume_organ'}
                )
            ]
            
            # Test trend detection
            trend_analysis = self.trend_detector.analyze(sensory_signals)
            assert trend_analysis is not None
            assert trend_analysis.analysis_type == "trend_detection"
            
            # Test risk analysis
            risk_analysis = self.risk_analyzer.analyze(sensory_signals)
            assert risk_analysis is not None
            assert risk_analysis.analysis_type == "risk_analysis"
            
            # Test performance analysis
            trading_data = self.create_test_trading_data()
            performance_analysis = self.performance_analyzer.analyze(trading_data)
            assert performance_analysis is not None
            assert performance_analysis.analysis_type == "performance_analysis"
            
            self.test_results['thinking_layer'] = {
                'status': 'PASS',
                'trend_analysis': trend_analysis.result['trend_direction'],
                'risk_analysis': risk_analysis.result['risk_level'],
                'performance_analysis': performance_analysis.result['total_return']
            }
            
            logger.info("Thinking layer test PASSED")
            
        except Exception as e:
            logger.error(f"Thinking layer test FAILED: {e}")
            self.test_results['thinking_layer'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_governance_layer(self):
        """Test the governance layer functionality."""
        logger.info("Testing governance layer")
        
        try:
            # Test fitness store
            definitions = self.fitness_store.list_definitions()
            assert len(definitions) > 0
            
            active_definition = self.fitness_store.get_active_definition()
            assert active_definition is not None
            
            # Test fitness calculation
            performance_metrics = {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
                'win_rate': 0.6,
                'profit_factor': 1.5
            }
            
            fitness_score = self.fitness_store.calculate_fitness(performance_metrics)
            assert 0 <= fitness_score <= 1
            
            self.test_results['governance_layer'] = {
                'status': 'PASS',
                'definitions_loaded': len(definitions),
                'fitness_score': fitness_score
            }
            
            logger.info("Governance layer test PASSED")
            
        except Exception as e:
            logger.error(f"Governance layer test FAILED: {e}")
            self.test_results['governance_layer'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_operational_layer(self):
        """Test the operational layer functionality."""
        logger.info("Testing operational layer")
        
        try:
            # Test state store health
            health_status = self.state_store.health_check()
            assert health_status is True
            
            # Test state storage and retrieval
            test_data = {'test_key': 'test_value', 'timestamp': datetime.now().isoformat()}
            
            # Store test data
            success = self.state_store.store_genome('test_genome', test_data)
            assert success is True
            
            # Retrieve test data
            retrieved_data = self.state_store.get_genome('test_genome')
            assert retrieved_data is not None
            assert retrieved_data['test_key'] == 'test_value'
            
            # Test stats
            stats = self.state_store.get_stats()
            assert 'total_keys' in stats
            
            self.test_results['operational_layer'] = {
                'status': 'PASS',
                'health_status': health_status,
                'storage_test': success,
                'retrieval_test': retrieved_data is not None
            }
            
            logger.info("Operational layer test PASSED")
            
        except Exception as e:
            logger.error(f"Operational layer test FAILED: {e}")
            self.test_results['operational_layer'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_genome_model(self):
        """Test the genome model functionality."""
        logger.info("Testing genome model")
        
        try:
            # Create test genome
            genome = DecisionGenome(genome_id="test_genome_001")
            
            # Test genome validation
            is_valid = genome.validate()
            assert is_valid is True
            
            # Test genome serialization
            genome_dict = genome.to_dict()
            assert genome_dict['genome_id'] == "test_genome_001"
            assert 'strategy' in genome_dict
            assert 'risk' in genome_dict
            assert 'timing' in genome_dict
            assert 'sensory' in genome_dict
            assert 'thinking' in genome_dict
            
            # Test genome deserialization
            reconstructed_genome = DecisionGenome.from_dict(genome_dict)
            assert reconstructed_genome.genome_id == genome.genome_id
            assert reconstructed_genome.strategy.strategy_type == genome.strategy.strategy_type
            
            # Test genome mutation
            mutated_genome = genome.mutate(mutation_rate=0.5)
            assert mutated_genome.genome_id != genome.genome_id
            assert mutated_genome.parent_ids == [genome.genome_id]
            assert mutated_genome.mutation_count == genome.mutation_count + 1
            
            self.test_results['genome_model'] = {
                'status': 'PASS',
                'validation': is_valid,
                'serialization': True,
                'deserialization': True,
                'mutation': True
            }
            
            logger.info("Genome model test PASSED")
            
        except Exception as e:
            logger.error(f"Genome model test FAILED: {e}")
            self.test_results['genome_model'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def test_event_communication(self):
        """Test event-driven communication between layers."""
        logger.info("Testing event communication")
        
        try:
            events_received = []
            
            # Subscribe to events
            async def event_handler(event):
                events_received.append(event)
                
            event_bus.subscribe('sensory.signal.received', event_handler)
            event_bus.subscribe('thinking.analysis.completed', event_handler)
            
            # Create and process market data
            market_data = self.create_test_market_data()
            await self.sensory_cortex.process_market_data(market_data)
            
            # Wait for events to be processed
            await asyncio.sleep(0.1)
            
            # Verify events were received
            assert len(events_received) > 0
            
            self.test_results['event_communication'] = {
                'status': 'PASS',
                'events_received': len(events_received)
            }
            
            logger.info("Event communication test PASSED")
            
        except Exception as e:
            logger.error(f"Event communication test FAILED: {e}")
            self.test_results['event_communication'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            
    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info("Starting EMP Ultimate Architecture v1.1 Integration Tests")
        
        try:
            await self.setup()
            
            # Run individual layer tests
            await self.test_sensory_layer()
            await self.test_thinking_layer()
            await self.test_governance_layer()
            await self.test_operational_layer()
            await self.test_genome_model()
            await self.test_event_communication()
            
            # Generate test report
            self.generate_test_report()
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
        finally:
            await self.teardown()
            
    def generate_test_report(self):
        """Generate a comprehensive test report."""
        logger.info("Generating test report")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*60)
        print("EMP ULTIMATE ARCHITECTURE v1.1 - INTEGRATION TEST REPORT")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("="*60)
        
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
            
        print("="*60)
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED - ARCHITECTURE INTEGRATION SUCCESSFUL!")
        else:
            print(f"⚠️  {failed_tests} TESTS FAILED - ARCHITECTURE INTEGRATION ISSUES DETECTED")
        print("="*60)


async def main():
    """Main test runner."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run integration tests
    test_runner = ArchitectureIntegrationTest()
    await test_runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 