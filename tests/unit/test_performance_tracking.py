#!/usr/bin/env python3
"""
Performance Tracking Integration Test Suite

Tests the integration of the performance tracker with the live trading executor
and verifies all performance metrics, reporting, and alerting functionality.
"""

import sys
import os
import asyncio
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.trading.performance_tracker import PerformanceTracker, PerformanceMetrics
from src.trading.live_trading_executor import LiveTradingExecutor
from src.trading.mock_ctrader_interface import TradingConfig, MarketData, Order, Position, OrderSide
from src.evolution.real_genetic_engine import RealGeneticEngine
from src.sensory.dimensions.enhanced_when_dimension import MarketRegimeDetector
from src.sensory.dimensions.enhanced_anomaly_dimension import AdvancedPatternRecognition

class PerformanceTrackingTester:
    """Test suite for performance tracking integration."""
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
        
    def run_test(self, test_name: str, test_func):
        """Run a test and record results."""
        self.total_tests += 1
        print(f"\n🧪 Running test: {test_name}")
        
        try:
            result = test_func()
            if result:
                print(f"✅ PASSED: {test_name}")
                self.passed_tests += 1
                self.test_results.append({"test": test_name, "status": "PASSED"})
            else:
                print(f"❌ FAILED: {test_name}")
                self.test_results.append({"test": test_name, "status": "FAILED"})
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            self.test_results.append({"test": test_name, "status": "ERROR", "error": str(e)})
    
    def test_performance_tracker_initialization(self):
        """Test performance tracker initialization."""
        tracker = PerformanceTracker(initial_balance=100000.0)
        
        # Check initial state
        assert tracker.initial_balance == 100000.0
        assert tracker.current_balance == 100000.0
        assert len(tracker.positions_history) == 0
        assert len(tracker.trades_history) == 0
        assert len(tracker.daily_equity) == 0
        
        return True
    
    def test_position_tracking(self):
        """Test position tracking functionality."""
        tracker = PerformanceTracker(initial_balance=100000.0)
        
        # Add test positions
        position1 = {
            'symbol': 'EURUSD',
            'volume': 0.1,
            'entry_price': 1.1000,
            'current_price': 1.1050,
            'pnl': 50.0,
            'side': 'buy'
        }
        
        position2 = {
            'symbol': 'GBPUSD',
            'volume': 0.05,
            'entry_price': 1.2500,
            'current_price': 1.2450,
            'pnl': -25.0,
            'side': 'sell'
        }
        
        tracker.update_position(position1)
        tracker.update_position(position2)
        
        # Verify positions recorded
        assert len(tracker.positions_history) == 2
        assert tracker.current_balance == 100025.0  # 100000 + 50 - 25
        
        return True
    
    def test_trade_recording(self):
        """Test trade recording functionality."""
        tracker = PerformanceTracker(initial_balance=100000.0)
        
        # Record test trades
        trade1 = {
            'symbol': 'EURUSD',
            'action': 'buy',
            'entry_price': 1.1000,
            'exit_price': 1.1050,
            'size': 0.1,
            'strategy': 'evolutionary',
            'entry_time': datetime.now(),
            'exit_time': datetime.now() + timedelta(hours=2)
        }
        
        trade2 = {
            'symbol': 'GBPUSD',
            'action': 'sell',
            'entry_price': 1.2500,
            'exit_price': 1.2450,
            'size': 0.05,
            'strategy': 'momentum',
            'entry_time': datetime.now(),
            'exit_time': datetime.now() + timedelta(hours=1)
        }
        
        tracker.record_trade(trade1)
        tracker.record_trade(trade2)
        
        # Verify trades recorded
        assert len(tracker.trades_history) == 2
        assert tracker.trades_history[0]['trade_id'] == 1
        assert tracker.trades_history[1]['trade_id'] == 2
        
        # Verify strategy performance tracking
        assert 'evolutionary' in tracker.strategy_performance
        assert 'momentum' in tracker.strategy_performance
        assert tracker.strategy_performance['evolutionary']['trades'] == 1
        assert tracker.strategy_performance['momentum']['trades'] == 1
        
        return True
    
    def test_daily_equity_tracking(self):
        """Test daily equity tracking."""
        tracker = PerformanceTracker(initial_balance=100000.0)
        
        # Add daily equity updates
        dates = [
            datetime.now() - timedelta(days=4),
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=2),
            datetime.now() - timedelta(days=1),
            datetime.now()
        ]
        
        equities = [100000, 100500, 99500, 101000, 100750]
        
        for date, equity in zip(dates, equities):
            tracker.update_daily_equity(equity, date)
        
        # Verify equity tracking
        assert len(tracker.daily_equity) == 5
        assert tracker.daily_equity[0]['equity'] == 100000
        assert tracker.daily_equity[-1]['equity'] == 100750
        
        return True
    
    def test_regime_performance_tracking(self):
        """Test regime performance tracking."""
        tracker = PerformanceTracker(initial_balance=100000.0)
        
        # Update regime performance
        tracker.update_regime_performance('trending', 0.05)
        tracker.update_regime_performance('trending', 0.03)
        tracker.update_regime_performance('ranging', -0.02)
        tracker.update_regime_performance('volatile', 0.08)
        
        # Verify regime tracking
        assert 'trending' in tracker.regime_performance
        assert 'ranging' in tracker.regime_performance
        assert 'volatile' in tracker.regime_performance
        
        # Check calculations
        trending_data = tracker.regime_performance['trending']
        assert trending_data['trades'] == 2
        assert abs(trending_data['avg_return'] - 0.04) < 0.001  # (0.05 + 0.03) / 2
        
        return True
    
    def test_metrics_calculation(self):
        """Test comprehensive metrics calculation."""
        tracker = PerformanceTracker(initial_balance=100000.0)
        
        # Add test data
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        equities = [100000 + i * 10 for i in range(30)]  # Increasing equity
        
        for date, equity in zip(dates, equities):
            tracker.update_daily_equity(equity, date)
        
        # Add some trades
        for i in range(10):
            trade = {
                'symbol': 'EURUSD',
                'action': 'buy' if i % 2 == 0 else 'sell',
                'entry_price': 1.1000,
                'exit_price': 1.1050 if i % 2 == 0 else 1.0950,
                'size': 0.1,
                'strategy': 'test',
                'entry_time': datetime.now(),
                'exit_time': datetime.now() + timedelta(hours=1)
            }
            tracker.record_trade(trade)
        
        # Calculate metrics
        metrics = tracker.calculate_metrics()
        
        # Verify metrics structure
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return > 0  # Should be positive due to increasing equity
        assert metrics.total_trades == 10
        assert metrics.winning_trades == 5  # Half should be winning
        assert metrics.losing_trades == 5
        assert metrics.win_rate == 0.5
        
        return True
    
    def test_performance_reports(self):
        """Test performance report generation."""
        tracker = PerformanceTracker(initial_balance=100000.0)
        
        # Add test data
        for i in range(10):
            tracker.update_daily_equity(100000 + i * 100, datetime.now() - timedelta(days=10-i))
            
            trade = {
                'symbol': 'EURUSD',
                'action': 'buy',
                'entry_price': 1.1000,
                'exit_price': 1.1050,
                'size': 0.1,
                'strategy': 'test',
                'entry_time': datetime.now(),
                'exit_time': datetime.now() + timedelta(hours=1)
            }
            tracker.record_trade(trade)
        
        # Test different report types
        summary_report = tracker.generate_report("summary")
        detailed_report = tracker.generate_report("detailed")
        comprehensive_report = tracker.generate_report("comprehensive")
        
        # Verify report structures
        assert 'total_return' in summary_report
        assert 'returns' in detailed_report
        assert 'summary' in comprehensive_report
        assert 'detailed_metrics' in comprehensive_report
        
        return True
    
    def test_data_export(self):
        """Test data export functionality."""
        tracker = PerformanceTracker(initial_balance=100000.0)
        
        # Add test data
        tracker.update_daily_equity(100000, datetime.now())
        tracker.record_trade({
            'symbol': 'EURUSD',
            'action': 'buy',
            'entry_price': 1.1000,
            'exit_price': 1.1050,
            'size': 0.1,
            'strategy': 'test',
            'entry_time': datetime.now(),
            'exit_time': datetime.now() + timedelta(hours=1)
        })
        
        # Test JSON export
        json_data = tracker.export_data("json")
        data = json.loads(json_data)
        
        assert 'metrics' in data
        assert 'trades_history' in data
        assert 'daily_equity' in data
        
        return True
    
    def test_performance_alerts(self):
        """Test performance alert generation."""
        tracker = PerformanceTracker(initial_balance=100000.0)
        
        # Add data that should trigger alerts
        # High drawdown scenario
        for i in range(20):
            equity = 100000 - (i * 1000)  # Decreasing equity
            tracker.update_daily_equity(equity, datetime.now() - timedelta(days=20-i))
        
        # Low win rate scenario
        for i in range(10):
            trade = {
                'symbol': 'EURUSD',
                'action': 'buy',
                'entry_price': 1.1000,
                'exit_price': 1.0950,  # Loss
                'size': 0.1,
                'strategy': 'test',
                'entry_time': datetime.now(),
                'exit_time': datetime.now() + timedelta(hours=1)
            }
            tracker.record_trade(trade)
        
        # Get alerts
        alerts = tracker.get_performance_alerts()
        
        # Should have alerts for high drawdown and low win rate
        assert len(alerts) > 0
        
        # Check alert types
        alert_types = [alert['type'] for alert in alerts]
        assert 'risk' in alert_types or 'trading' in alert_types
        
        return True
    
    def test_live_trading_integration(self):
        """Test integration with live trading executor."""
        # Create mock config
        config = TradingConfig(
            client_id="test_client",
            client_secret="test_secret",
            access_token="test_token",
            refresh_token="test_refresh",
            account_id=12345
        )
        
        # Create executor
        executor = LiveTradingExecutor(config, ["EURUSD", "GBPUSD"])
        
        # Verify performance tracker is initialized
        assert hasattr(executor, 'performance_tracker')
        assert isinstance(executor.performance_tracker, PerformanceTracker)
        
        # Test performance report methods
        summary = executor.get_comprehensive_performance_report("summary")
        assert isinstance(summary, dict)
        
        alerts = executor.get_performance_alerts()
        assert isinstance(alerts, list)
        
        return True
    
    def test_strategy_performance_tracking(self):
        """Test strategy-specific performance tracking."""
        try:
            tracker = PerformanceTracker(initial_balance=100000.0)
            
            # Add equity data first
            for i in range(10):
                tracker.update_daily_equity(100000 + i * 100, datetime.now() - timedelta(days=10-i))
            
            # Add trades for different strategies
            strategies = ['evolutionary', 'momentum', 'mean_reversion']
            
            for i, strategy in enumerate(strategies):
                for j in range(5):  # 5 trades per strategy
                    trade = {
                        'symbol': 'EURUSD',
                        'action': 'buy',
                        'entry_price': 1.1000,
                        'exit_price': 1.1050 if j % 2 == 0 else 1.0950,
                        'size': 0.1,
                        'strategy': strategy,
                        'entry_time': datetime.now(),
                        'exit_time': datetime.now() + timedelta(hours=1)
                    }
                    tracker.record_trade(trade)
            
            # Calculate metrics
            metrics = tracker.calculate_metrics()
            
            # Verify strategy performance
            strategy_perf = metrics.strategy_performance
            assert len(strategy_perf) == 3
            
            for strategy in strategies:
                assert strategy in strategy_perf
                assert strategy_perf[strategy]['trade_count'] == 5
            
            return True
        except Exception as e:
            print(f"Error in strategy performance tracking: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_correlation_matrix(self):
        """Test correlation matrix calculation."""
        tracker = PerformanceTracker(initial_balance=100000.0)
        
        # Add trades for multiple strategies
        strategies = ['strategy_a', 'strategy_b', 'strategy_c']
        
        for strategy in strategies:
            for i in range(10):
                trade = {
                    'symbol': 'EURUSD',
                    'action': 'buy',
                    'entry_price': 1.1000,
                    'exit_price': 1.1050,
                    'size': 0.1,
                    'strategy': strategy,
                    'entry_time': datetime.now(),
                    'exit_time': datetime.now() + timedelta(hours=1)
                }
                tracker.record_trade(trade)
        
        # Calculate metrics
        metrics = tracker.calculate_metrics()
        
        # Verify correlation matrix
        if not metrics.correlation_matrix.empty:
            assert metrics.correlation_matrix.shape[0] == len(strategies)
            assert metrics.correlation_matrix.shape[1] == len(strategies)
        
        return True
    
    def run_all_tests(self):
        """Run all performance tracking tests."""
        print("🚀 Starting Performance Tracking Integration Tests")
        print("=" * 60)
        
        # Core functionality tests
        self.run_test("Performance Tracker Initialization", self.test_performance_tracker_initialization)
        self.run_test("Position Tracking", self.test_position_tracking)
        self.run_test("Trade Recording", self.test_trade_recording)
        self.run_test("Daily Equity Tracking", self.test_daily_equity_tracking)
        self.run_test("Regime Performance Tracking", self.test_regime_performance_tracking)
        
        # Metrics and analysis tests
        self.run_test("Metrics Calculation", self.test_metrics_calculation)
        self.run_test("Performance Reports", self.test_performance_reports)
        self.run_test("Data Export", self.test_data_export)
        self.run_test("Performance Alerts", self.test_performance_alerts)
        
        # Integration tests
        self.run_test("Live Trading Integration", self.test_live_trading_integration)
        self.run_test("Strategy Performance Tracking", self.test_strategy_performance_tracking)
        self.run_test("Correlation Matrix", self.test_correlation_matrix)
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE TRACKING TEST RESULTS")
        print("=" * 60)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED! Performance tracking system is fully functional.")
        else:
            print("\n⚠️  Some tests failed. Check the results above.")
        
        # Print detailed results
        print("\n📋 Detailed Results:")
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(f"  {status_icon} {result['test']}: {result['status']}")
            if "error" in result:
                print(f"     Error: {result['error']}")
        
        return self.passed_tests == self.total_tests

def main():
    """Main test runner."""
    tester = PerformanceTrackingTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 Performance tracking integration is ready for production!")
        print("   - Real-time metrics calculation ✓")
        print("   - Comprehensive reporting ✓")
        print("   - Performance alerts ✓")
        print("   - Data export capabilities ✓")
        print("   - Strategy performance tracking ✓")
        print("   - Live trading integration ✓")
    else:
        print("\n🔧 Performance tracking needs attention before production use.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 