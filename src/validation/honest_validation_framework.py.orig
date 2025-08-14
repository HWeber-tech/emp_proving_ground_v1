#!/usr/bin/env python3
"""
Honest Validation Framework - Phase 2A
======================================

Real validation framework that uses actual component testing and real data.
Replaces the fraudulent validation scripts with honest, transparent validation.
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from src.sensory.organs.yahoo_finance_organ import YahooFinanceOrgan
try:
    from src.trading.risk.market_regime_detector import MarketRegimeDetector  # deprecated
except Exception:  # pragma: no cover
    MarketRegimeDetector = None  # type: ignore
from src.data_integration.real_data_integration import RealDataManager
try:
    from src.core.interfaces import DecisionGenome  # legacy
except Exception:  # pragma: no cover
    DecisionGenome = object  # type: ignore

logger = logging.getLogger(__name__)


class HonestValidationResult:
    """Honest validation result with actual metrics"""
    
    def __init__(self, test_name: str, passed: bool, value: float, 
                 threshold: float, unit: str, details: str = ""):
        self.test_name = test_name
        self.passed = passed
        self.value = value
        self.threshold = threshold
        self.unit = unit
        self.details = details
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'value': self.value,
            'threshold': self.threshold,
            'unit': self.unit,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


class HonestValidationFramework:
    """
    Honest validation framework that uses real components and actual data.
    No synthetic data, no hardcoded results, no fraud.
    """
    
    def __init__(self):
        self.results: List[HonestValidationResult] = []
        self.yahoo_organ = YahooFinanceOrgan()
        self.regime_detector = MarketRegimeDetector()
        self.strategy_manager = None # Placeholder for StrategyManager
        self.real_data_manager = RealDataManager({'fallback_to_mock': False})
        
    async def validate_data_integrity(self) -> HonestValidationResult:
        """Validate that real market data can be retrieved and processed"""
        try:
            # Test actual data retrieval
            start_time = time.time()
            data = self.yahoo_organ.fetch_data('EURUSD=X', period="1d", interval="1m")
            retrieval_time = time.time() - start_time
            
            if data is None or len(data) == 0:
                return HonestValidationResult(
                    test_name="data_integrity",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="Failed to retrieve real market data"
                )
            
            # Validate data structure
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                return HonestValidationResult(
                    test_name="data_integrity",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details=f"Missing required columns: {missing_columns}"
                )
            
            # Check data quality
            null_count = data[required_columns].isnull().sum().sum()
            if null_count > 0:
                return HonestValidationResult(
                    test_name="data_integrity",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details=f"Found {null_count} null values in data"
                )
            
            return HonestValidationResult(
                test_name="data_integrity",
                passed=True,
                value=retrieval_time,
                threshold=5.0,
                unit="seconds",
                details=f"Successfully retrieved {len(data)} rows of EURUSD data in {retrieval_time:.2f}s"
            )
            
        except Exception as e:
            return HonestValidationResult(
                test_name="data_integrity",
                passed=False,
                value=0.0,
                threshold=1.0,
                unit="boolean",
                details=f"Data integrity validation failed: {str(e)}"
            )
    
    async def validate_regime_detection(self) -> HonestValidationResult:
        """Validate market regime detection with real data"""
        try:
            # Get real market data
            data = self.yahoo_organ.fetch_data('EURUSD=X', period="1d", interval="1h")
            if data is None or len(data) < 20:
                return HonestValidationResult(
                    test_name="regime_detection",
                    passed=False,
                    value=0.0,
                    threshold=0.8,
                    unit="accuracy",
                    details="Insufficient data for regime detection"
                )
            
            # Test regime detection
            start_time = time.time()
            regime_result = await self.regime_detector.detect_regime(data)
            detection_time = time.time() - start_time
            
            if regime_result is None:
                return HonestValidationResult(
                    test_name="regime_detection",
                    passed=False,
                    value=0.0,
                    threshold=0.8,
                    unit="accuracy",
                    details="Regime detection returned None"
                )
            
            # Validate confidence
            confidence = regime_result.confidence
            passed = confidence >= 0.5  # Reasonable threshold for real data
            
            return HonestValidationResult(
                test_name="regime_detection",
                passed=passed,
                value=confidence,
                threshold=0.5,
                unit="confidence",
                details=f"Detected {regime_result.regime.value} with {confidence:.2f} confidence in {detection_time:.2f}s"
            )
            
        except Exception as e:
            return HonestValidationResult(
                test_name="regime_detection",
                passed=False,
                value=0.0,
                threshold=0.8,
                unit="accuracy",
                details=f"Regime detection failed: {str(e)}"
            )
    
    async def validate_strategy_integration(self) -> HonestValidationResult:
        """Validate strategy manager integration with real data"""
        try:
            # Create test strategy
            test_genome = DecisionGenome()
            test_genome.genome_id = "honest_test_strategy"
            test_genome.generation = 1
            test_genome.fitness_score = 0.75
            test_genome.robustness_score = 0.8
            
            # Add strategy
            success = self.strategy_manager.add_strategy(test_genome)
            if not success:
                return HonestValidationResult(
                    test_name="strategy_integration",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="Failed to add test strategy"
                )
            
            # Get real market data
            data = self.yahoo_organ.fetch_data('EURUSD=X', period="1d", interval="1h")
            if data is None or len(data) == 0:
                return HonestValidationResult(
                    test_name="strategy_integration",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="No market data available"
                )
            
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
            start_time = time.time()
            signals = self.strategy_manager.evaluate_strategies('EURUSD', market_data)
            evaluation_time = time.time() - start_time
            
            # Validate signals
            if not isinstance(signals, list):
                return HonestValidationResult(
                    test_name="strategy_integration",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="Invalid signals format"
                )
            
            # Check if we got any signals
            signal_count = len(signals)
            passed = signal_count > 0
            
            return HonestValidationResult(
                test_name="strategy_integration",
                passed=passed,
                value=evaluation_time,
                threshold=2.0,
                unit="seconds",
                details=f"Strategy manager generated {signal_count} signals in {evaluation_time:.2f}s"
            )
            
        except Exception as e:
            return HonestValidationResult(
                test_name="strategy_integration",
                passed=False,
                value=0.0,
                threshold=1.0,
                unit="boolean",
                details=f"Strategy integration failed: {str(e)}"
            )
    
    async def validate_real_data_sources(self) -> HonestValidationResult:
        """Validate that real data sources are working"""
        try:
            # Test Yahoo Finance
            yahoo_data = self.yahoo_organ.fetch_data('EURUSD=X', period="1d", interval="1h")
            yahoo_works = yahoo_data is not None and len(yahoo_data) > 0
            
            # Test RealDataManager
            market_data = await self.real_data_manager.get_market_data('EURUSD=X')
            real_manager_works = market_data is not None
            
            # Overall success
            success = yahoo_works and real_manager_works
            
            return HonestValidationResult(
                test_name="real_data_sources",
                passed=success,
                value=1.0 if success else 0.0,
                threshold=1.0,
                unit="boolean",
                details=f"Yahoo Finance: {'✅' if yahoo_works else '❌'}, "
                       f"RealDataManager: {'✅' if real_manager_works else '❌'}"
            )
            
        except Exception as e:
            return HonestValidationResult(
                test_name="real_data_sources",
                passed=False,
                value=0.0,
                threshold=1.0,
                unit="boolean",
                details=f"Real data sources validation failed: {str(e)}"
            )
    
    async def validate_no_synthetic_data(self) -> HonestValidationResult:
        """Validate that we're not using synthetic data"""
        try:
            # Check if we're using real data
            data = self.yahoo_organ.fetch_data('EURUSD=X', period="1d", interval="1h")
            
            if data is None:
                return HonestValidationResult(
                    test_name="no_synthetic_data",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="No real data available - system may be using synthetic data"
                )
            
            # Check for obvious synthetic patterns
            # Real data should have some randomness and not be perfectly smooth
            price_changes = data['close'].pct_change().dropna()
            volatility = price_changes.std()
            
            # Real EURUSD data typically has volatility > 0.0001
            is_real = volatility > 0.0001
            
            return HonestValidationResult(
                test_name="no_synthetic_data",
                passed=is_real,
                value=volatility,
                threshold=0.0001,
                unit="volatility",
                details=f"Data volatility: {volatility:.6f} - {'Real data' if is_real else 'Synthetic data detected'}"
            )
            
        except Exception as e:
            return HonestValidationResult(
                test_name="no_synthetic_data",
                passed=False,
                value=0.0,
                threshold=1.0,
                unit="boolean",
                details=f"Cannot determine data source: {str(e)}"
            )
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all honest validations"""
        logger.info("Starting honest validation framework...")
        
        # Run all validation tests
        validations = [
            self.validate_data_integrity(),
            self.validate_regime_detection(),
            self.validate_strategy_integration(),
            self.validate_real_data_sources(),
            self.validate_no_synthetic_data()
        ]
        
        # Execute all validations
        results = await asyncio.gather(*validations)
        
        # Calculate summary
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        # Create final report
        report = {
            'timestamp': datetime.now().isoformat(),
            'framework': 'Honest Validation Framework',
            'version': '1.0.0',
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'success_rate': passed / total if total > 0 else 0.0,
            'results': [r.to_dict() for r in results],
            'summary': {
                'status': 'PASSED' if passed == total else 'FAILED',
                'message': f"{passed}/{total} validations passed"
            }
        }
        
        # Save results
        with open('honest_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print comprehensive validation report"""
        print("\n" + "="*80)
        print("HONEST VALIDATION FRAMEWORK REPORT")
        print("="*80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Framework: {report['framework']} v{report['version']}")
        print(f"Status: {report['summary']['status']}")
        print(f"Success Rate: {report['success_rate']:.2%}")
        print()
        
        print("VALIDATION RESULTS:")
        print("-" * 40)
        for result in report['results']:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['test_name']}: {result['details']}")
            print(f"  Value: {result['value']} {result['unit']}")
            print(f"  Threshold: {result['threshold']} {result['unit']}")
            print()
        
        print("="*80)
        print(report['summary']['message'])
        print("="*80)


async def main():
    """Run honest validation framework"""
    logging.basicConfig(level=logging.INFO)
    
    framework = HonestValidationFramework()
    report = await framework.run_all_validations()
    framework.print_report(report)
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if report['success_rate'] >= 0.8 else 1)


if __name__ == "__main__":
    asyncio.run(main())
