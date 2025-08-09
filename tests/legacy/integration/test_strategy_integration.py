#!/usr/bin/env python3
"""
Test Strategy Integration - Phase 2.1 Validation

This script tests the complete integration between evolved strategies from the genetic engine
and the live trading executor, validating the strategy manager and signal generation.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.trading.strategy_manager import StrategyManager, StrategySignal
from src.trading.live_trading_executor import LiveTradingExecutor, TradingSignal
from src.evolution.real_genetic_engine import RealGeneticEngine, TradingStrategy
from src.core.interfaces import DecisionGenome
from src.trading.mock_ctrader_interface import TradingConfig, TradingMode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StrategyIntegrationTester:
    """Test the complete strategy integration system."""
    
    def __init__(self):
        self.strategy_manager = StrategyManager()
        self.genetic_engine = RealGeneticEngine(data_source="real")
        self.test_results = {}
    
    async def run_all_tests(self):
        """Run all strategy integration tests."""
        logger.info("🚀 Starting Strategy Integration Tests (Phase 2.1)")
        
        tests = [
            ("Test Strategy Manager Creation", self.test_strategy_manager_creation),
            ("Test Strategy Loading", self.test_strategy_loading),
            ("Test Strategy Evaluation", self.test_strategy_evaluation),
            ("Test Strategy Selection", self.test_strategy_selection),
            ("Test Performance Tracking", self.test_performance_tracking),
            ("Test Live Trading Integration", self.test_live_trading_integration),
            ("Test Signal Generation", self.test_signal_generation),
            ("Test Strategy Conversion", self.test_strategy_conversion)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n📋 Running: {test_name}")
                
                # Check if test function is async
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()
                    
                self.test_results[test_name] = result
                
                if result['passed']:
                    logger.info(f"✅ PASSED: {test_name}")
                    passed += 1
                else:
                    logger.error(f"❌ FAILED: {test_name} - {result['error']}")
                    
            except Exception as e:
                logger.error(f"❌ ERROR in {test_name}: {e}")
                self.test_results[test_name] = {'passed': False, 'error': str(e)}
        
        # Print summary
        logger.info(f"\n📊 Test Summary:")
        logger.info(f"Passed: {passed}/{total}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            logger.info("🎉 All tests passed! Strategy integration is working correctly.")
        else:
            logger.error("⚠️ Some tests failed. Check the logs above for details.")
        
        return passed == total
    
    def test_strategy_manager_creation(self):
        """Test strategy manager initialization."""
        try:
            # Test basic creation
            manager = StrategyManager()
            assert manager is not None
            assert hasattr(manager, 'strategies')
            assert hasattr(manager, 'performance')
            assert hasattr(manager, 'add_strategy')
            assert hasattr(manager, 'evaluate_strategies')
            assert hasattr(manager, 'select_best_strategy')
            
            # Test configuration
            assert manager.max_strategies == 10
            assert manager.min_confidence_threshold == 0.6
            assert manager.selection_method == 'performance_ranked'
            
            return {'passed': True, 'message': 'Strategy manager created successfully'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_strategy_loading(self):
        """Test loading strategies into the manager."""
        try:
            # Create a mock strategy
            mock_strategy = TradingStrategy(
                id="test_strategy_001",
                name="Test Strategy",
                parameters={
                    'buy_threshold': 0.6,
                    'sell_threshold': 0.4,
                    'stop_loss_pct': 0.01,
                    'take_profit_pct': 0.02,
                    'position_size': 0.01
                },
                indicators=['SMA', 'RSI'],
                entry_rules=['sma_crossover', 'rsi_oversold'],
                exit_rules=['stop_loss', 'take_profit'],
                risk_management={'max_risk': 0.02},
                fitness_score=0.75,
                generation=1
            )
            
            # Convert to genome and add to manager
            genome = self._convert_strategy_to_genome(mock_strategy)
            success = self.strategy_manager.add_strategy(genome)
            
            assert success == True
            assert mock_strategy.id in self.strategy_manager.strategies
            assert mock_strategy.id in self.strategy_manager.performance
            
            return {'passed': True, 'message': f'Strategy {mock_strategy.id} loaded successfully'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_strategy_evaluation(self):
        """Test strategy evaluation with market data."""
        try:
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
            
            # Evaluate strategies
            signals = self.strategy_manager.evaluate_strategies('EURUSD', market_data)
            
            # Check that we get a list of signals (may be empty if no strategies)
            assert isinstance(signals, list)
            
            # If we have signals, check their structure
            for signal in signals:
                assert isinstance(signal, StrategySignal)
                assert hasattr(signal, 'strategy_id')
                assert hasattr(signal, 'symbol')
                assert hasattr(signal, 'action')
                assert hasattr(signal, 'confidence')
                assert signal.symbol == 'EURUSD'
                assert signal.action in ['buy', 'sell', 'hold']
                assert 0 <= signal.confidence <= 1
            
            return {'passed': True, 'message': f'Evaluated {len(signals)} strategy signals'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_strategy_selection(self):
        """Test strategy selection methods."""
        try:
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
            
            # Test different selection methods
            selection_methods = ['performance_ranked', 'ensemble', 'regime_based']
            
            for method in selection_methods:
                self.strategy_manager.selection_method = method
                best_signal = self.strategy_manager.select_best_strategy('EURUSD', market_data)
                
                # Should return either a signal or None
                assert best_signal is None or isinstance(best_signal, StrategySignal)
                
                if best_signal:
                    assert best_signal.symbol == 'EURUSD'
                    assert best_signal.action in ['buy', 'sell', 'hold']
                    assert best_signal.confidence >= self.strategy_manager.min_confidence_threshold
            
            return {'passed': True, 'message': 'Strategy selection methods working correctly'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        try:
            # Add a test strategy if none exists
            if not self.strategy_manager.strategies:
                mock_strategy = TradingStrategy(
                    id="perf_test_001",
                    name="Performance Test Strategy",
                    parameters={'buy_threshold': 0.5},
                    indicators=['SMA'],
                    entry_rules=['sma_crossover'],
                    exit_rules=['stop_loss'],
                    risk_management={'max_risk': 0.02},
                    fitness_score=0.6,
                    generation=1
                )
                genome = self._convert_strategy_to_genome(mock_strategy)
                self.strategy_manager.add_strategy(genome)
            
            # Update performance with mock trades
            strategy_id = list(self.strategy_manager.strategies.keys())[0]
            
            # Simulate winning trade
            self.strategy_manager.update_strategy_performance(strategy_id, 100.0)
            
            # Simulate losing trade
            self.strategy_manager.update_strategy_performance(strategy_id, -50.0)
            
            # Check performance metrics
            perf = self.strategy_manager.get_strategy_performance(strategy_id)
            assert perf is not None
            assert perf.total_trades == 2
            assert perf.winning_trades == 1
            assert perf.losing_trades == 1
            assert perf.net_profit == 50.0
            assert perf.win_rate == 0.5
            
            # Test top strategies
            top_strategies = self.strategy_manager.get_top_strategies(count=3)
            assert isinstance(top_strategies, list)
            
            # Test summary
            summary = self.strategy_manager.get_strategy_summary()
            assert isinstance(summary, dict)
            assert 'total_strategies' in summary
            assert 'active_strategies' in summary
            
            return {'passed': True, 'message': 'Performance tracking working correctly'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def test_live_trading_integration(self):
        """Test integration with live trading executor."""
        try:
            # Create trading config
            config = TradingConfig(
                client_id="test_client",
                client_secret="test_secret",
                access_token="test_token",
                refresh_token="test_refresh",
                account_id=12345,
                mode=TradingMode.DEMO
            )
            
            # Create live trading executor
            executor = LiveTradingExecutor(
                config=config,
                symbols=['EURUSD'],
                max_positions=3,
                max_risk_per_trade=0.02
            )
            
            # Check that strategy manager is initialized
            assert hasattr(executor, 'strategy_manager')
            assert isinstance(executor.strategy_manager, StrategyManager)
            
            # Check that genetic engine is initialized
            assert hasattr(executor, 'genetic_engine')
            assert isinstance(executor.genetic_engine, RealGeneticEngine)
            
            return {'passed': True, 'message': 'Live trading integration working correctly'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_signal_generation(self):
        """Test signal generation from evolved strategies."""
        try:
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
            
            # Add a strategy with known behavior
            mock_strategy = TradingStrategy(
                id="signal_test_001",
                name="Signal Test Strategy",
                parameters={
                    'buy_threshold': 0.3,  # Low threshold for testing
                    'sell_threshold': 0.7,
                    'stop_loss_pct': 0.01,
                    'take_profit_pct': 0.02,
                    'position_size': 0.01
                },
                indicators=['SMA', 'RSI'],
                entry_rules=['sma_crossover', 'rsi_oversold'],
                exit_rules=['stop_loss', 'take_profit'],
                risk_management={'max_risk': 0.02},
                fitness_score=0.8,
                generation=1
            )
            
            genome = self._convert_strategy_to_genome(mock_strategy)
            self.strategy_manager.add_strategy(genome)
            
            # Generate signals
            signals = self.strategy_manager.evaluate_strategies('EURUSD', market_data)
            
            # Check signal structure
            for signal in signals:
                assert isinstance(signal, StrategySignal)
                assert signal.symbol == 'EURUSD'
                assert signal.action in ['buy', 'sell', 'hold']
                assert 0 <= signal.confidence <= 1
                assert signal.entry_price is not None
                assert signal.stop_loss is not None
                assert signal.take_profit is not None
                assert signal.volume > 0
            
            return {'passed': True, 'message': f'Generated {len(signals)} valid signals'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_strategy_conversion(self):
        """Test conversion between TradingStrategy and DecisionGenome."""
        try:
            # Create a complex trading strategy
            original_strategy = TradingStrategy(
                id="convert_test_001",
                name="Conversion Test Strategy",
                parameters={
                    'buy_threshold': 0.6,
                    'sell_threshold': 0.4,
                    'momentum_weight': 0.3,
                    'trend_weight': 0.4,
                    'institutional_weight': 0.3,
                    'stop_loss_pct': 0.015,
                    'take_profit_pct': 0.025,
                    'position_size': 0.02
                },
                indicators=['SMA', 'EMA', 'RSI', 'MACD', 'BOLLINGER'],
                entry_rules=['sma_crossover', 'rsi_oversold', 'macd_bullish'],
                exit_rules=['stop_loss', 'take_profit', 'rsi_overbought'],
                risk_management={
                    'max_risk': 0.02,
                    'max_positions': 3,
                    'correlation_limit': 0.7
                },
                fitness_score=0.85,
                generation=5,
                parent_ids=['parent_001', 'parent_002']
            )
            
            # Convert to genome
            genome = self._convert_strategy_to_genome(original_strategy)
            
            # Verify conversion
            assert isinstance(genome, DecisionGenome)
            assert genome.genome_id == original_strategy.id
            assert genome.fitness_score == original_strategy.fitness_score
            assert genome.generation == original_strategy.generation
            assert genome.parent_ids == original_strategy.parent_ids
            
            # Check decision tree structure
            assert 'parameters' in genome.decision_tree
            assert 'indicators' in genome.decision_tree
            assert 'entry_rules' in genome.decision_tree
            assert 'exit_rules' in genome.decision_tree
            assert 'risk_management' in genome.decision_tree
            
            # Verify parameters are preserved
            for key, value in original_strategy.parameters.items():
                assert genome.decision_tree['parameters'][key] == value
            
            return {'passed': True, 'message': 'Strategy conversion working correctly'}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _convert_strategy_to_genome(self, strategy) -> DecisionGenome:
        """Convert TradingStrategy to DecisionGenome format."""
        # Create decision tree from strategy parameters
        decision_tree = {
            'parameters': strategy.parameters,
            'indicators': strategy.indicators,
            'entry_rules': strategy.entry_rules,
            'exit_rules': strategy.exit_rules,
            'risk_management': strategy.risk_management
        }
        
        # Create DecisionGenome
        genome = DecisionGenome(
            genome_id=strategy.id,
            decision_tree=decision_tree,
            fitness_score=strategy.fitness_score,
            generation=strategy.generation,
            parent_ids=strategy.parent_ids
        )
        
        return genome


async def main():
    """Run the strategy integration tests."""
    print("🧬 EMP Strategy Integration Test Suite")
    print("=" * 50)
    
    tester = StrategyIntegrationTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Phase 2.1: Strategy Integration - COMPLETED SUCCESSFULLY!")
        print("\n✅ Key Achievements:")
        print("   • Strategy Manager: Real-time strategy evaluation and selection")
        print("   • Live Trading Integration: Evolved strategies connected to trading executor")
        print("   • Performance Tracking: Individual strategy performance monitoring")
        print("   • Signal Generation: Real-time trading signals from evolved strategies")
        print("   • Strategy Conversion: Seamless conversion between strategy formats")
        print("\n🚀 Ready to proceed to Phase 2.2: Advanced Risk Management")
    else:
        print("\n❌ Phase 2.1: Strategy Integration - FAILED")
        print("Please check the test results above and fix any issues.")
    
    return success


if __name__ == "__main__":
    asyncio.run(main()) 
