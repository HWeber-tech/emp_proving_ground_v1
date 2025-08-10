#!/usr/bin/env python3
"""
Component Integration Test Suite - Phase 2B
==========================================

Comprehensive integration tests for component communication and data flow.
Tests real component interactions using actual market data.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.trading.risk.advanced_risk_manager import AdvancedRiskManager
from src.trading.strategies.strategy_manager import StrategyManager
from src.trading.risk.market_regime_detector import MarketRegimeDetector, MarketRegime
from src.data_integration.real_data_integration import RealDataManager
from src.sensory.organs.yahoo_finance_organ import YahooFinanceOrgan
try:
    from src.core.interfaces import DecisionGenome  # legacy
except Exception:  # pragma: no cover
    DecisionGenome = None  # type: ignore

logger = logging.getLogger(__name__)


class ComponentIntegrationTester:
    """Test suite for component integration"""
    
    def __init__(self):
        self.test_results = []
        self.real_data_manager = None
        self.yahoo_organ = None
        
    async def setup_components(self):
        """Initialize all components for testing"""
        try:
            # Initialize real data components
            self.real_data_manager = RealDataManager({'fallback_to_mock': False})
            self.yahoo_organ = YahooFinanceOrgan()
            
            # Initialize trading components
            self.strategy_manager = StrategyManager()
            self.regime_detector = MarketRegimeDetector()
            self.risk_manager = AdvancedRiskManager(
                strategy_manager=self.strategy_manager,
                regime_detector=self.regime_detector
            )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Component setup failed: {e}")
            return False
    
    async def test_yahoo_finance_integration(self):
        """Test Yahoo Finance data integration"""
        try:
            # Test data retrieval
            data = self.yahoo_organ.fetch_data('EURUSD=X')
            
            if data is None or len(data) == 0:
                return False, "No data retrieved from Yahoo Finance"
            
            # Validate data structure
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                return False, f"Missing columns: {missing_columns}"
            
            # Test data quality
            if data.isnull().any().any():
                null_counts = data.isnull().sum()
                return False, f"Null values found: {null_counts.to_dict()}"
            
            return True, f"Successfully retrieved {len(data)} rows of EURUSD data"
            
        except Exception as e:
            return False, f"Yahoo Finance integration error: {str(e)}"
    
    async def test_market_regime_detection(self):
        """Test market regime detection with real data"""
        try:
            # Get real market data
            data = self.yahoo_organ.fetch_data('EURUSD=X')
            if data is None or len(data) < 20:
                return False, "Insufficient data for regime detection"
            
            # Test regime detection
            regime_result = await self.regime_detector.detect_regime(data)
            
            if regime_result is None:
                return False, "Regime detection returned None"
            
            # Validate regime result
            if not hasattr(regime_result, 'regime'):
                return False, "Invalid regime result structure"
            
            if regime_result.confidence < 0.1:
                return False, f"Low confidence: {regime_result.confidence}"
            
            return True, f"Detected regime: {regime_result.regime.value} with {regime_result.confidence:.2f} confidence"
            
        except Exception as e:
            return False, f"Regime detection error: {str(e)}"
    
    async def test_strategy_manager_integration(self):
        """Test strategy manager with real data"""
        try:
            # Create test strategy
            test_genome = DecisionGenome(
                genome_id="test_integration_strategy",
                decision_tree={
                    "type": "test_strategy",
                    "parameters": {
                        "buy_threshold": 0.6,
                        "sell_threshold": 0.6,
                        "momentum_weight": 0.5,
                        "trend_weight": 0.5,
                        "institutional_weight": 0.5
                    }
                }
            )
            test_genome.generation = 1
            test_genome.fitness_score = 0.75
            test_genome.robustness_score = 0.8
            
            # Add strategy
            success = self.strategy_manager.add_strategy(test_genome)
            if not success:
                return False, "Failed to add test strategy"
            
            # Get real market data
            data = self.yahoo_organ.fetch_data('EURUSD=X')
            if data is None or len(data) == 0:
                return False, "No market data available"
            
            # Create market data dict
            market_data = {
                'symbol': 'EURUSD',
                'timestamp': data.iloc[-1]['timestamp'],
                'open': data.iloc[-1]['open'],
                'high': data.iloc[-1]['high'],
                'low': data.iloc[-1]['low'],
                'close': data.iloc[-1]['close'],
                'volume': data.iloc[-1]['volume']
            }
            
            # Test strategy evaluation
            signals = self.strategy_manager.evaluate_strategies('EURUSD', market_data)
            
            # Validate signals
            if not isinstance(signals, list):
                return False, "Invalid signals format"
            
            return True, f"Strategy manager generated {len(signals)} signals"
            
        except Exception as e:
            return False, f"Strategy manager integration error: {str(e)}"
    
    async def test_risk_manager_integration(self):
        """Test advanced risk manager integration"""
        try:
            # Get real market data
            data = self.yahoo_organ.fetch_data('EURUSD=X')
            if data is None or len(data) < 20:
                return False, "Insufficient data for risk manager"
            
            # Create test signal
            from src.trading.strategies.strategy_manager import StrategySignal
            
            test_signal = StrategySignal(
                strategy_id="test_risk_strategy",
                symbol="EURUSD",
                action="buy",
                confidence=0.75,
                entry_price=1.1000,
                stop_loss=1.0980,
                take_profit=1.1040,
                volume=0.01
            )
            
            # Create market data dict
            market_data = {
                'EURUSD': data.iloc[-1].to_dict()
            }
            
            # Test signal validation
            is_valid, reason, metadata = await self.risk_manager.validate_signal(
                test_signal, market_data
            )
            
            if not is_valid:
                return False, f"Signal validation failed: {reason}"
            
            # Test position sizing
            position_size = await self.risk_manager.calculate_position_size(
                test_signal, 10000.0, market_data
            )
            
            if position_size <= 0:
                return False, f"Invalid position size: {position_size}"
            
            return True, f"Risk manager validated signal with position size: {position_size:.4f}"
            
        except Exception as e:
            return False, f"Risk manager integration error: {str(e)}"
    
    async def test_end_to_end_data_flow(self):
        """Test complete data flow from data source to risk management"""
        try:
            # Step 1: Get real market data
            data = self.yahoo_organ.fetch_data('EURUSD=X')
            if data is None or len(data) < 20:
                return False, "Failed to retrieve market data"
            
            # Step 2: Detect market regime
            regime_result = await self.regime_detector.detect_regime(data)
            if regime_result is None:
                return False, "Failed to detect regime"
            
            # Step 3: Create test strategy and signal
            test_genome = DecisionGenome(
                genome_id="e2e_test_strategy",
                decision_tree={
                    "type": "e2e_test",
                    "parameters": {
                        "buy_threshold": 0.7,
                        "sell_threshold": 0.7,
                        "momentum_weight": 0.6,
                        "trend_weight": 0.4,
                        "institutional_weight": 0.5
                    }
                }
            )
            test_genome.generation = 1
            test_genome.fitness_score = 0.8
            
            self.strategy_manager.add_strategy(test_genome)
            
            market_data = {
                'symbol': 'EURUSD',
                'timestamp': data.iloc[-1]['timestamp'],
                'open': data.iloc[-1]['open'],
                'high': data.iloc[-1]['high'],
                'low': data.iloc[-1]['low'],
                'close': data.iloc[-1]['close'],
                'volume': data.iloc[-1]['volume']
            }
            
            signals = self.strategy_manager.evaluate_strategies('EURUSD', market_data)
            if not signals:
                return False, "No signals generated"
            
            # Step 4: Validate signal with risk manager
            signal = signals[0]
            market_dict = {'EURUSD': data.iloc[-1].to_dict()}
            
            is_valid, reason, metadata = await self.risk_manager.validate_signal(
                signal, market_dict, regime_result
            )
            
            if not is_valid:
                return False, f"End-to-end validation failed: {reason}"
            
            return True, "End-to-end data flow completed successfully"
            
        except Exception as e:
            return False, f"End-to-end flow error: {str(e)}"
    
    async def test_error_handling(self):
        """Test error handling and graceful degradation"""
        try:
            # Test with invalid symbol
            data = self.yahoo_organ.fetch_data('INVALID_SYMBOL')
            if data is not None:
                return False, "Should return None for invalid symbol"
            
            # Test with empty data
            empty_data = pd.DataFrame()
            regime_result = await self.regime_detector.detect_regime(empty_data)
            if regime_result is not None:
                return False, "Should return None for empty data"
            
            # Test with invalid market data format
            invalid_data = pd.DataFrame({'invalid': [1, 2, 3]})
            regime_result = await self.regime_detector.detect_regime(invalid_data)
            if regime_result is not None:
                return False, "Should return None for invalid data format"
            
            return True, "Error handling working correctly"
            
        except Exception as e:
            return False, f"Error handling test failed: {str(e)}"


async def run_integration_tests():
    """Run all integration tests"""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting component integration tests...")
    
    tester = ComponentIntegrationTester()
    
    # Setup components
    setup_success = await tester.setup_components()
    if not setup_success:
        logger.error("Failed to setup components")
        return
    
    # Run tests
    tests = [
        ("Yahoo Finance Integration", tester.test_yahoo_finance_integration),
        ("Market Regime Detection", tester.test_market_regime_detection),
        ("Strategy Manager Integration", tester.test_strategy_manager_integration),
        ("Risk Manager Integration", tester.test_risk_manager_integration),
        ("End-to-End Data Flow", tester.test_end_to_end_data_flow),
        ("Error Handling", tester.test_error_handling),
    ]
    
    results = []
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"Running {test_name}...")
            success, message = await test_func()
            results.append((test_name, success, message))
            
            if success:
                logger.info(f"✅ {test_name}: PASS - {message}")
                passed_tests += 1
            else:
                logger.error(f"❌ {test_name}: FAIL - {message}")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {str(e)}")
            results.append((test_name, False, str(e)))
    
    # Print summary
    total_tests = len(tests)
    print("\n" + "="*60)
    print("COMPONENT INTEGRATION TEST RESULTS")
    print("="*60)
    
    for test_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
    
    print(f"\nSummary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All integration tests passed! Phase 2B is complete.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Review and fix issues.")
    
    return {
        'passed_tests': passed_tests,
        'total_tests': total_tests,
        'results': results
    }


if __name__ == "__main__":
    asyncio.run(run_integration_tests())
