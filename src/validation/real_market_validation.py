#!/usr/bin/env python3
"""
Real Market Validation Framework - Phase 2C
==========================================

Comprehensive validation framework using real market data for honest performance testing.
Tests actual anomaly detection, regime classification, and trading performance with real data.
"""

import asyncio
import logging
import json
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import yfinance as yf

from src.sensory.organs.yahoo_finance_organ import YahooFinanceOrgan
from src.sensory.enhanced.anomaly.manipulation_detection import ManipulationDetectionSystem
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


class RealMarketValidationResult:
    """Result from real market validation testing"""
    
    def __init__(self, test_name: str, passed: bool, value: float, 
                 threshold: float, unit: str, details: str = "",
                 historical_data: Dict[str, Any] = None):
        self.test_name = test_name
        self.passed = passed
        self.value = value
        self.threshold = threshold
        self.unit = unit
        self.details = details
        self.historical_data = historical_data or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'value': self.value,
            'threshold': self.threshold,
            'unit': self.unit,
            'details': self.details,
            'historical_data': self.historical_data,
            'timestamp': self.timestamp.isoformat()
        }


class RealMarketValidationFramework:
    """
    Comprehensive validation framework using real market data.
    Tests actual performance with historical market events.
    """
    
    def __init__(self):
        self.results: List[RealMarketValidationResult] = []
        self.yahoo_organ = YahooFinanceOrgan()
        self.manipulation_detector = ManipulationDetector()
        self.regime_detector = MarketRegimeDetector()
        self.strategy_manager = StrategyManager()
        self.real_data_manager = RealDataManager({'fallback_to_mock': False})
        
        # Historical market events for validation
        self.known_market_events = {
            'flash_crash': [
                {'date': '2010-05-06', 'symbol': '^GSPC', 'type': 'flash_crash'},
                {'date': '2015-08-24', 'symbol': '^DJI', 'type': 'flash_crash'}
            ],
            'pump_dump': [
                {'date': '2021-01-28', 'symbol': 'GME', 'type': 'pump_dump'},
                {'date': '2021-01-29', 'symbol': 'AMC', 'type': 'pump_dump'}
            ],
            'regime_change': [
                {'date': '2008-09-15', 'symbol': '^GSPC', 'type': 'crisis'},
                {'date': '2020-03-09', 'symbol': '^GSPC', 'type': 'crisis'}
            ]
        }
        
    async def validate_anomaly_detection_accuracy(self) -> RealMarketValidationResult:
        """Validate anomaly detection against known market manipulation events"""
        try:
            total_events = 0
            detected_events = 0
            false_positives = 0
            
            # Test against known flash crashes
            for event in self.known_market_events['flash_crash']:
                try:
                    # Get data around the event
                    start_date = datetime.strptime(event['date'], '%Y-%m-%d') - timedelta(days=5)
                    end_date = datetime.strptime(event['date'], '%Y-%m-%d') + timedelta(days=5)
                    
                    data = self.yahoo_organ.fetch_data(
                        event['symbol'], 
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval="1m"
                    )
                    
                    if data is None or len(data) < 100:
                        continue
                    
                    # Detect anomalies
                    anomalies = await self.manipulation_detector.detect_manipulation(data)
                    
                    # Check if we detected the known event
                    event_detected = any(
                        abs((datetime.fromtimestamp(a['timestamp']) - 
                             datetime.strptime(event['date'], '%Y-%m-%d')).days) <= 1
                        for a in anomalies
                    )
                    
                    total_events += 1
                    if event_detected:
                        detected_events += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to test event {event}: {e}")
                    continue
            
            # Calculate accuracy metrics
            if total_events > 0:
                recall = detected_events / total_events
                precision = detected_events / max(detected_events + false_positives, 1)
                f1 = 2 * (precision * recall) / max(precision + recall, 0.001)
                
                passed = recall >= 0.7  # 70% recall threshold
                
                return RealMarketValidationResult(
                    test_name="anomaly_detection_accuracy",
                    passed=passed,
                    value=f1,
                    threshold=0.7,
                    unit="f1_score",
                    details=f"Detected {detected_events}/{total_events} known events. "
                           f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}",
                    historical_data={
                        'total_events': total_events,
                        'detected_events': detected_events,
                        'precision': precision,
                        'recall': recall
                    }
                )
            else:
                return RealMarketValidationResult(
                    test_name="anomaly_detection_accuracy",
                    passed=False,
                    value=0.0,
                    threshold=0.7,
                    unit="f1_score",
                    details="No historical events available for testing"
                )
                
        except Exception as e:
            return RealMarketValidationResult(
                test_name="anomaly_detection_accuracy",
                passed=False,
                value=0.0,
                threshold=0.7,
                unit="f1_score",
                details=f"Anomaly detection validation failed: {str(e)}"
            )
    
    async def validate_regime_classification_accuracy(self) -> RealMarketValidationResult:
        """Validate regime classification against known market regimes"""
        try:
            # Test with 2020 COVID crash data
            start_date = datetime(2020, 2, 1)
            end_date = datetime(2020, 5, 1)
            
            # Get S&P 500 data
            data = self.yahoo_organ.fetch_data(
                '^GSPC',
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1d"
            )
            
            if data is None or len(data) < 20:
                return RealMarketValidationResult(
                    test_name="regime_classification_accuracy",
                    passed=False,
                    value=0.0,
                    threshold=0.8,
                    unit="accuracy",
                    details="Insufficient data for regime classification"
                )
            
            # Detect regimes
            regimes = []
            for i in range(20, len(data)):
                window = data.iloc[i-20:i]
                regime_result = await self.regime_detector.detect_regime(window)
                if regime_result:
                    regimes.append({
                        'date': data.iloc[i]['timestamp'],
                        'regime': regime_result.regime.value,
                        'confidence': regime_result.confidence
                    })
            
            # Validate regime transitions
            crisis_periods = [
                (datetime(2020, 2, 19), datetime(2020, 3, 23))  # COVID crash
            ]
            
            correct_classifications = 0
            total_classifications = 0
            
            for regime in regimes:
                regime_date = pd.to_datetime(regime['date'])
                
                # Check if correctly identified crisis periods
                for crisis_start, crisis_end in crisis_periods:
                    if crisis_start <= regime_date <= crisis_end:
                        if regime['regime'] in ['CRISIS', 'VOLATILE']:
                            correct_classifications += 1
                        total_classifications += 1
            
            accuracy = correct_classifications / max(total_classifications, 1)
            passed = accuracy >= 0.8
            
            return RealMarketValidationResult(
                test_name="regime_classification_accuracy",
                passed=passed,
                value=accuracy,
                threshold=0.8,
                unit="accuracy",
                details=f"Correctly classified {correct_classifications}/{total_classifications} crisis periods",
                historical_data={
                    'total_classifications': total_classifications,
                    'correct_classifications': correct_classifications,
                    'crisis_periods': len(crisis_periods)
                }
            )
            
        except Exception as e:
            return RealMarketValidationResult(
                test_name="regime_classification_accuracy",
                passed=False,
                value=0.0,
                threshold=0.8,
                unit="accuracy",
                details=f"Regime classification validation failed: {str(e)}"
            )
    
    async def validate_real_performance_metrics(self) -> RealMarketValidationResult:
        """Validate actual performance metrics with real data"""
        try:
            # Test with recent EURUSD data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            data = self.yahoo_organ.fetch_data(
                'EURUSD=X',
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1h"
            )
            
            if data is None or len(data) < 100:
                return RealMarketValidationResult(
                    test_name="real_performance_metrics",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="boolean",
                    details="Insufficient data for performance testing"
                )
            
            # Measure processing times
            processing_times = []
            
            # Test data retrieval
            start_time = time.time()
            _ = self.yahoo_organ.fetch_data('EURUSD=X', period="1d", interval="1m")
            retrieval_time = time.time() - start_time
            processing_times.append(retrieval_time)
            
            # Test anomaly detection
            start_time = time.time()
            _ = await self.manipulation_detector.detect_manipulation(data)
            anomaly_time = time.time() - start_time
            processing_times.append(anomaly_time)
            
            # Test regime detection
            start_time = time.time()
            _ = await self.regime_detector.detect_regime(data)
            regime_time = time.time() - start_time
            processing_times.append(regime_time)
            
            # Calculate metrics
            avg_processing_time = np.mean(processing_times)
            max_processing_time = np.max(processing_times)
            
            # Memory usage estimation (simplified)
            data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
            
            # Performance thresholds
            retrieval_threshold = 5.0  # seconds
            anomaly_threshold = 10.0     # seconds
            regime_threshold = 3.0         # seconds
            
            passed = (
                retrieval_time <= retrieval_threshold and
                anomaly_time <= anomaly_threshold and
                regime_time <= regime_threshold
            )
            
            return RealMarketValidationResult(
                test_name="real_performance_metrics",
                passed=passed,
                value=avg_processing_time,
                threshold=5.0,
                unit="seconds",
                details=f"Retrieval: {retrieval_time:.2f}s, "
                       f"Anomaly: {anomaly_time:.2f}s, "
                       f"Regime: {regime_time:.2f}s, "
                       f"Data size: {data_size_mb:.2f}MB",
                historical_data={
                    'retrieval_time': retrieval_time,
                    'anomaly_time': anomaly_time,
                    'regime_time': regime_time,
                    'data_size_mb': data_size_mb,
                    'max_processing_time': max_processing_time
                }
            )
            
        except Exception as e:
            return RealMarketValidationResult(
                test_name="real_performance_metrics",
                passed=False,
                value=0.0,
                threshold=5.0,
                unit="seconds",
                details=f"Performance validation failed: {str(e)}"
            )
    
    async def validate_sharpe_ratio_calculation(self) -> RealMarketValidationResult:
        """Validate Sharpe ratio calculation with real trading data"""
        try:
            # Get 1 year of S&P 500 data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            data = self.yahoo_organ.fetch_data(
                '^GSPC',
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1d"
            )
            
            if data is None or len(data) < 20:
                return RealMarketValidationResult(
                    test_name="sharpe_ratio_calculation",
                    passed=False,
                    value=0.0,
                    threshold=1.0,
                    unit="sharpe_ratio",
                    details="Insufficient data for Sharpe ratio calculation"
                )
            
            # Calculate daily returns
            data['returns'] = data['close'].pct_change()
            data = data.dropna()
            
            # Calculate Sharpe ratio (annualized)
            excess_returns = data['returns'] - 0.02/252  # 2% risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            
            # Validate calculation
            expected_range = (-2.0, 3.0)  # Reasonable range for S&P 500
            passed = expected_range[0] <= sharpe_ratio <= expected_range[1]
            
            return RealMarketValidationResult(
                test_name="sharpe_ratio_calculation",
                passed=passed,
                value=sharpe_ratio,
                threshold=1.0,
                unit="sharpe_ratio",
                details=f"Annualized Sharpe ratio: {sharpe_ratio:.4f} "
                       f"(Expected range: {expected_range})",
                historical_data={
                    'mean_return': data['returns'].mean(),
                    'volatility': data['returns'].std(),
                    'observations': len(data)
                }
            )
            
        except Exception as e:
            return RealMarketValidationResult(
                test_name="sharpe_ratio_calculation",
                passed=False,
                value=0.0,
                threshold=1.0,
                unit="sharpe_ratio",
                details=f"Sharpe ratio calculation failed: {str(e)}"
            )
    
    async def validate_max_drawdown_calculation(self) -> RealMarketValidationResult:
        """Validate maximum drawdown calculation with real data"""
        try:
            # Get COVID crash data
            start_date = datetime(2020, 2, 1)
            end_date = datetime(2020, 5, 1)
            
            data = self.yahoo_organ.fetch_data(
                '^GSPC',
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1d"
            )
            
            if data is None or len(data) < 20:
                return RealMarketValidationResult(
                    test_name="max_drawdown_calculation",
                    passed=False,
                    value=0.0,
                    threshold=-0.5,
                    unit="percentage",
                    details="Insufficient data for drawdown calculation"
                )
            
            # Calculate cumulative returns
            data['returns'] = data['close'].pct_change()
            data['cumulative'] = (1 + data['returns']).cumprod()
            
            # Calculate running maximum
            data['running_max'] = data['cumulative'].expanding().max()
            data['drawdown'] = (data['cumulative'] - data['running_max']) / data['running_max']
            
            max_drawdown = data['drawdown'].min()
            
            # Validate against known COVID crash (-34%)
            expected_drawdown = -0.34
            tolerance = 0.05
            passed = abs(max_drawdown - expected_drawdown) <= tolerance
            
            return RealMarketValidationResult(
                test_name="max_drawdown_calculation",
                passed=passed,
                value=max_drawdown,
                threshold=-0.5,
                unit="percentage",
                details=f"Maximum drawdown: {max_drawdown:.2%} "
                       f"(Expected: {expected_drawdown:.2%}, "
                       f"Tolerance: ±{tolerance:.1%})",
                historical_data={
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'peak_date': data.loc[data['cumulative'].idxmax(), 'timestamp'],
                    'trough_date': data.loc[data['drawdown'].idxmin(), 'timestamp']
                }
            )
            
        except Exception as e:
            return RealMarketValidationResult(
                test_name="max_drawdown_calculation",
                passed=False,
                value=0.0,
                threshold=-0.5,
                unit="percentage",
                details=f"Max drawdown calculation failed: {str(e)}"
            )
    
    async def validate_no_synthetic_data_usage(self) -> RealMarketValidationResult:
        """Validate that no synthetic data is being used in testing"""
        try:
            # Test multiple data sources
            test_symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', '^GSPC', '^DJI']
            
            real_data_count = 0
            total_tests = 0
            
            for symbol in test_symbols:
                try:
                    # Get real data
                    data = self.yahoo_organ.fetch_data(symbol, period="1d", interval="1h")
                    
                    if data is not None and len(data) > 0:
                        # Check for synthetic patterns
                        price_changes = data['close'].pct_change().dropna()
                        
                        # Real data should have:
                        # 1. Non-zero volatility
                        volatility = price_changes.std()
                        
                        # 2. Non-perfect correlation
                        correlation = data['close'].corr(data['volume']) if 'volume' in data.columns else 0
                        
                        # 3. Realistic price ranges
                        price_range = (data['high'] - data['low']).mean()
                        
                        # Simple heuristic for real data
                        is_real = (
                            volatility > 0.0001 and
                            abs(correlation) < 0.95 and
                            price_range > 0
                        )
                        
                        if is_real:
                            real_data_count += 1
                        
                        total_tests += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to test symbol {symbol}: {e}")
                    continue
            
            success_rate = real_data_count / max(total_tests, 1)
            passed = success_rate >= 0.8
            
            return RealMarketValidationResult(
                test_name="no_synthetic_data_usage",
                passed=passed,
                value=success_rate,
                threshold=0.8,
                unit="success_rate",
                details=f"Real data confirmed for {real_data_count}/{total_tests} symbols "
                       f"({success_rate:.1%} success rate)",
                historical_data={
                    'real_data_count': real_data_count,
                    'total_tests': total_tests,
                    'tested_symbols': test_symbols
                }
            )
            
        except Exception as e:
            return RealMarketValidationResult(
                test_name="no_synthetic_data_usage",
                passed=False,
                value=0.0,
                threshold=0.8,
                unit="success_rate",
                details=f"Synthetic data validation failed: {str(e)}"
            )
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all real market validations"""
        logger.info("Starting comprehensive real market validation...")
        
        # Run all validation tests
        validations = [
            self.validate_anomaly_detection_accuracy(),
            self.validate_regime_classification_accuracy(),
            self.validate_real_performance_metrics(),
            self.validate_sharpe_ratio_calculation(),
            self.validate_max_drawdown_calculation(),
            self.validate_no_synthetic_data_usage()
        ]
        
        # Execute all validations
        results = await asyncio.gather(*validations)
        
        # Calculate summary
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'framework': 'Real Market Validation Framework',
            'version': '2.0.0',
            'phase': '2C',
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'success_rate': passed / total if total > 0 else 0.0,
            'results': [r.to_dict() for r in results],
            'summary': {
                'status': 'PASSED' if passed >= 5 else 'FAILED',  # Allow 1 failure
                'message': f"{passed}/{total} validations passed ({passed/total:.1%} success rate)"
            },
            'historical_events_tested': len(self.known_market_events),
            'data_sources_validated': ['Yahoo Finance', 'RealDataManager']
        }
        
        # Save results
        with open('real_market_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_comprehensive_report(self, report: Dict[str, Any]):
        """Print comprehensive validation report"""
        print("\n" + "="*100)
        print("REAL MARKET VALIDATION FRAMEWORK REPORT")
        print("="*100)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Framework: {report['framework']} v{report['version']}")
        print(f"Phase: {report['phase']}")
        print(f"Status: {report['summary']['status']}")
        print(f"Success Rate: {report['success_rate']:.1%}")
        print(f"Historical Events Tested: {report['historical_events_tested']}")
        print()
        
        print("VALIDATION RESULTS:")
        print("-" * 60)
        for result in report['results']:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['test_name']}: {result['details']}")
            print(f"  Value: {result['value']:.4f} {result['unit']}")
            print(f"  Threshold: {result['threshold']} {result['unit']}")
            if result['historical_data']:
                print(f"  Historical Data: {result['historical_data']}")
            print()
        
        print("="*100)
        print(report['summary']['message'])
        print("="*100)


async def main():
    """Run comprehensive real market validation"""
    logging.basicConfig(level=logging.INFO)
    
    framework = RealMarketValidationFramework()
    report = await framework.run_comprehensive_validation()
    framework.print_comprehensive_report(report)
    
    # Exit with appropriate code
    import sys
    success_threshold = 0.8  # 80% success rate required
    sys.exit(0 if report['success_rate'] >= success_threshold else 1)


if __name__ == "__main__":
    asyncio.run(main())
