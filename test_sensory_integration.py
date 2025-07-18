#!/usr/bin/env python3
"""
Test Sensory Integration - Market Regime Detection and Pattern Recognition

This test verifies that market regime detection and pattern recognition have been
properly integrated into the sensory system, eliminating the redundant analysis folder.
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from dataclasses import dataclass

# Import integrated sensory components
from src.sensory.core.base import MarketData, MarketRegime
from src.sensory.dimensions.enhanced_when_dimension import TemporalAnalyzer
from src.sensory.dimensions.enhanced_anomaly_dimension import PatternRecognitionDetector, PatternType, AnomalyType

# Import trading components to verify integration
from src.trading.live_trading_executor import LiveTradingExecutor, TradingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result structure"""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None


class SensoryIntegrationTester:
    """Test suite for sensory integration"""
    
    def __init__(self):
        self.results = []
        
    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        logger.info("Starting Sensory Integration Tests")
        
        # Test 1: Verify analysis folder is removed
        self._test_analysis_folder_removed()
        
        # Test 2: Test market regime detection integration
        self._test_market_regime_detection()
        
        # Test 3: Test pattern recognition integration
        self._test_pattern_recognition()
        
        # Test 4: Test live trading executor integration
        self._test_live_trading_integration()
        
        # Test 5: Test sensory system coherence
        self._test_sensory_coherence()
        
        # Report results
        return self._report_results()
    
    def _test_analysis_folder_removed(self):
        """Test that the analysis folder has been removed"""
        test_name = "Analysis Folder Removal"
        
        try:
            # Check if analysis folder exists
            analysis_path = os.path.join("src", "analysis")
            if os.path.exists(analysis_path):
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    message="Analysis folder still exists",
                    details={"path": analysis_path}
                ))
                return
            
            # Check for any remaining imports
            import_patterns = [
                "from src.analysis",
                "import src.analysis",
                "from .analysis"
            ]
            
            remaining_imports = []
            for root, dirs, files in os.walk("src"):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                                for pattern in import_patterns:
                                    if pattern in content:
                                        remaining_imports.append(f"{file_path}: {pattern}")
                        except Exception as e:
                            logger.warning(f"Could not read {file_path}: {e}")
            
            if remaining_imports:
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    message="Found remaining analysis imports",
                    details={"remaining_imports": remaining_imports}
                ))
            else:
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=True,
                    message="Analysis folder successfully removed and imports cleaned up"
                ))
                
        except Exception as e:
            self.results.append(TestResult(
                test_name=test_name,
                passed=False,
                message=f"Error testing analysis folder removal: {e}"
            ))
    
    def _test_market_regime_detection(self):
        """Test market regime detection integration"""
        test_name = "Market Regime Detection Integration"
        
        try:
            # Create temporal analyzer
            temporal_analyzer = TemporalAnalyzer()
            
            # Generate test market data
            test_data = self._generate_test_market_data(100)
            
            # Test regime detection
            regimes_detected = []
            for market_data in test_data:
                temporal_analyzer.update_market_data(market_data)
                temporal_analyzer.update_temporal_data(market_data)
                
                regime = temporal_analyzer.detect_market_regime()
                regimes_detected.append(regime)
            
            # Verify regime detection works
            if not regimes_detected:
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    message="No regimes detected"
                ))
                return
            
            # Check that we get valid regimes
            valid_regimes = [r for r in regimes_detected if r != MarketRegime.UNKNOWN]
            regime_diversity = len(set(valid_regimes))
            
            if regime_diversity > 0:
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=True,
                    message=f"Market regime detection working - detected {regime_diversity} different regimes",
                    details={
                        "total_regimes": len(regimes_detected),
                        "valid_regimes": len(valid_regimes),
                        "regime_diversity": regime_diversity,
                        "sample_regimes": list(set(valid_regimes))[:3]
                    }
                ))
            else:
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    message="Only UNKNOWN regimes detected"
                ))
                
        except Exception as e:
            self.results.append(TestResult(
                test_name=test_name,
                passed=False,
                message=f"Error testing market regime detection: {e}"
            ))
    
    def _test_pattern_recognition(self):
        """Test pattern recognition integration"""
        test_name = "Pattern Recognition Integration"
        
        try:
            # Create pattern recognition detector
            pattern_detector = PatternRecognitionDetector()
            
            # Generate test market data with patterns
            test_data = self._generate_test_market_data_with_patterns(200)
            
            # Test pattern detection
            patterns_detected = []
            for market_data in test_data:
                pattern_detector.update_data(market_data)
                
                patterns = pattern_detector.detect_patterns(market_data)
                if patterns:
                    patterns_detected.extend(patterns)
            
            # Verify pattern detection works
            if patterns_detected:
                pattern_types = [p.anomaly_type for p in patterns_detected]
                unique_patterns = len(set(pattern_types))
                
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=True,
                    message=f"Pattern recognition working - detected {len(patterns_detected)} patterns",
                    details={
                        "total_patterns": len(patterns_detected),
                        "unique_pattern_types": unique_patterns,
                        "pattern_types": list(set(pattern_types))
                    }
                ))
            else:
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    message="No patterns detected"
                ))
                
        except Exception as e:
            self.results.append(TestResult(
                test_name=test_name,
                passed=False,
                message=f"Error testing pattern recognition: {e}"
            ))
    
    def _test_live_trading_integration(self):
        """Test live trading executor integration"""
        test_name = "Live Trading Integration"
        
        try:
            # Create trading config with correct parameters
            config = TradingConfig(
                client_id="test_client_id",
                client_secret="test_client_secret",
                access_token="test_access_token",
                refresh_token="test_refresh_token",
                account_id=12345
            )
            
            # Create live trading executor
            symbols = ["EURUSD", "GBPUSD"]
            executor = LiveTradingExecutor(config, symbols)
            
            # Verify that the executor uses the integrated components
            if hasattr(executor, 'regime_detector') and isinstance(executor.regime_detector, TemporalAnalyzer):
                regime_integration = True
            else:
                regime_integration = False
            
            if hasattr(executor, 'pattern_recognition') and isinstance(executor.pattern_recognition, PatternRecognitionDetector):
                pattern_integration = True
            else:
                pattern_integration = False
            
            if regime_integration and pattern_integration:
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=True,
                    message="Live trading executor successfully integrated with sensory system"
                ))
            else:
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    message="Live trading executor not properly integrated",
                    details={
                        "regime_integration": regime_integration,
                        "pattern_integration": pattern_integration
                    }
                ))
                
        except Exception as e:
            self.results.append(TestResult(
                test_name=test_name,
                passed=False,
                message=f"Error testing live trading integration: {e}"
            ))
    
    def _test_sensory_coherence(self):
        """Test sensory system coherence"""
        test_name = "Sensory System Coherence"
        
        try:
            # Test that all sensory dimensions work together
            from src.sensory.orchestration.master_orchestrator import MasterOrchestrator
            from src.sensory.core.base import InstrumentMeta
            
            # Create instrument metadata
            instrument_meta = InstrumentMeta(
                symbol="EURUSD",
                pip_size=0.0001,
                lot_size=100000
            )
            
            # Create master orchestrator with instrument_meta
            engine = MasterOrchestrator(instrument_meta)
            
            # Generate test data
            test_data = self._generate_test_market_data(50)
            
            # Test dimensional analysis
            readings = []
            for market_data in test_data:
                try:
                    reading = asyncio.run(engine.update(market_data))
                    readings.append(reading)
                except Exception as e:
                    logger.warning(f"Error in dimensional analysis: {e}")
            
            if readings:
                # Check that all dimensions are working
                dimensions = [r.dimension for r in readings if hasattr(r, 'dimension')]
                unique_dimensions = set(dimensions)
                
                expected_dimensions = {'WHY', 'HOW', 'WHAT', 'WHEN', 'ANOMALY'}
                missing_dimensions = expected_dimensions - unique_dimensions
                
                if not missing_dimensions:
                    self.results.append(TestResult(
                        test_name=test_name,
                        passed=True,
                        message="All sensory dimensions working coherently",
                        details={
                            "total_readings": len(readings),
                            "dimensions_detected": list(unique_dimensions),
                            "sample_regimes": list(set([r.regime for r in readings if hasattr(r, 'regime') and r.regime]))
                        }
                    ))
                else:
                    self.results.append(TestResult(
                        test_name=test_name,
                        passed=False,
                        message=f"Missing sensory dimensions: {missing_dimensions}"
                    ))
            else:
                self.results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    message="No sensory readings generated"
                ))
                
        except Exception as e:
            self.results.append(TestResult(
                test_name=test_name,
                passed=False,
                message=f"Error testing sensory coherence: {e}"
            ))
    
    def _generate_test_market_data(self, count: int) -> list:
        """Generate test market data"""
        data = []
        base_price = 1.1000
        
        for i in range(count):
            # Simulate price movement
            price_change = np.random.normal(0, 0.001)
            base_price += price_change
            
            # Create OHLCV data
            high = base_price + abs(np.random.normal(0, 0.0005))
            low = base_price - abs(np.random.normal(0, 0.0005))
            open_price = base_price + np.random.normal(0, 0.0002)
            close_price = base_price + np.random.normal(0, 0.0002)
            
            # Use timezone-naive timestamp
            timestamp = datetime.now() + timedelta(minutes=i)
            
            market_data = MarketData(
                symbol="EURUSD",
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=np.random.uniform(1000, 5000),
                bid=close_price - 0.0001,
                ask=close_price + 0.0001
            )
            data.append(market_data)
        
        return data
    
    def _generate_test_market_data_with_patterns(self, count: int) -> list:
        """Generate test market data with patterns"""
        data = []
        base_price = 1.1000
        
        # Create some pattern-like movements
        for i in range(count):
            # Create triangle pattern
            if i < 50:
                # Ascending triangle
                base_price += 0.0001
                high = base_price + 0.001
                low = base_price - 0.0005 + (i * 0.00001)
            elif i < 100:
                # Descending triangle
                base_price -= 0.0001
                high = base_price + 0.001 - (i * 0.00001)
                low = base_price - 0.0005
            else:
                # Random movement
                price_change = np.random.normal(0, 0.001)
                base_price += price_change
                high = base_price + abs(np.random.normal(0, 0.0005))
                low = base_price - abs(np.random.normal(0, 0.0005))
            
            open_price = base_price + np.random.normal(0, 0.0002)
            close_price = base_price + np.random.normal(0, 0.0002)
            
            # Use timezone-naive timestamp
            timestamp = datetime.now() + timedelta(minutes=i)
            
            market_data = MarketData(
                symbol="EURUSD",
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=np.random.uniform(1000, 5000),
                bid=close_price - 0.0001,
                ask=close_price + 0.0001
            )
            data.append(market_data)
        
        return data
    
    def _report_results(self) -> bool:
        """Report test results"""
        logger.info("\n" + "="*60)
        logger.info("SENSORY INTEGRATION TEST RESULTS")
        logger.info("="*60)
        
        passed_tests = 0
        total_tests = len(self.results)
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            logger.info(f"{status}: {result.test_name}")
            logger.info(f"  {result.message}")
            
            if result.details:
                for key, value in result.details.items():
                    logger.info(f"    {key}: {value}")
            
            if result.passed:
                passed_tests += 1
            
            logger.info("")
        
        logger.info(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Sensory integration successful!")
        else:
            logger.error(f"❌ {total_tests - passed_tests} tests failed")
        
        logger.info("="*60)
        
        return passed_tests == total_tests


async def main():
    """Main test function"""
    tester = SensoryIntegrationTester()
    success = tester.run_all_tests()
    
    if success:
        logger.info("✅ Sensory integration verification complete - system is properly integrated!")
        return 0
    else:
        logger.error("❌ Sensory integration verification failed - check the issues above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 