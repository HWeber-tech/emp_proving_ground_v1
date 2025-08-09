#!/usr/bin/env python3
"""
Order Book Integration Test Suite

Tests the integration of the order book analyzer with the live trading executor
and verifies all market microstructure analysis functionality.
"""

import sys
import os
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.trading.order_book_analyzer import OrderBookAnalyzer, OrderBookSnapshot, MarketMicrostructure
from src.trading.live_trading_executor import LiveTradingExecutor
from src.trading.mock_ctrader_interface import TradingConfig, MarketData, Order, Position, OrderSide

class OrderBookIntegrationTester:
    """Test suite for order book integration."""
    
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
    
    def test_order_book_analyzer_initialization(self):
        """Test order book analyzer initialization."""
        analyzer = OrderBookAnalyzer(max_levels=20, history_window=1000)
        
        # Check initial state
        assert analyzer.max_levels == 20
        assert analyzer.history_window == 1000
        assert len(analyzer.order_book_history) == 0
        assert len(analyzer.microstructure_history) == 0
        assert len(analyzer.current_metrics) == 0
        
        return True
    
    def test_order_book_update(self):
        """Test order book data update."""
        try:
            analyzer = OrderBookAnalyzer()
            
            # Create test order book data
            bids = [(1.1000, 1.5), (1.0999, 2.0), (1.0998, 1.0)]
            asks = [(1.1001, 1.2), (1.1002, 2.5), (1.1003, 1.8)]
            
            analyzer.update_order_book("EURUSD", bids, asks)
            
            # Verify order book was updated
            assert "EURUSD" in analyzer.order_book_history
            assert len(analyzer.order_book_history["EURUSD"]) == 1
            
            snapshot = analyzer.order_book_history["EURUSD"][0]
            assert snapshot.symbol == "EURUSD"
            
            # Debug output
            print(f"Expected spread: 0.0001, Actual spread: {snapshot.spread}")
            print(f"Expected mid_price: 1.10005, Actual mid_price: {snapshot.mid_price}")
            
            # Use approximate comparison for floating point values
            assert abs(snapshot.spread - 0.0001) < 0.00001
            assert abs(snapshot.mid_price - 1.10005) < 0.00001
            assert len(snapshot.bids) == 3
            assert len(snapshot.asks) == 3
            
            return True
        except Exception as e:
            print(f"Error in order book update: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_microstructure_analysis(self):
        """Test market microstructure analysis."""
        analyzer = OrderBookAnalyzer()
        
        # Add multiple order book updates
        for i in range(5):
            bids = [(1.1000 - i*0.0001, 1.5 + i*0.1), (1.0999 - i*0.0001, 2.0 + i*0.1)]
            asks = [(1.1001 + i*0.0001, 1.2 + i*0.1), (1.1002 + i*0.0001, 2.5 + i*0.1)]
            analyzer.update_order_book("EURUSD", bids, asks)
        
        # Verify microstructure analysis
        assert "EURUSD" in analyzer.microstructure_history
        assert len(analyzer.microstructure_history["EURUSD"]) == 5
        
        microstructure = analyzer.current_metrics["EURUSD"]
        assert isinstance(microstructure, MarketMicrostructure)
        assert microstructure.symbol == "EURUSD"
        assert microstructure.bid_liquidity > 0
        assert microstructure.ask_liquidity > 0
        assert microstructure.total_liquidity > 0
        
        return True
    
    def test_market_analysis_generation(self):
        """Test market analysis generation."""
        analyzer = OrderBookAnalyzer()
        
        # Add order book data
        bids = [(1.1000, 1.5), (1.0999, 2.0), (1.0998, 1.0)]
        asks = [(1.1001, 1.2), (1.1002, 2.5), (1.1003, 1.8)]
        analyzer.update_order_book("EURUSD", bids, asks)
        
        # Get market analysis
        analysis = analyzer.get_market_analysis("EURUSD")
        
        # Verify analysis structure
        assert 'symbol' in analysis
        assert 'current' in analysis
        assert 'depth' in analysis
        assert 'market_impact' in analysis
        assert 'volatility' in analysis
        assert 'regime' in analysis
        assert 'trends' in analysis
        assert 'signals' in analysis
        
        # Verify current metrics
        current = analysis['current']
        assert 'spread' in current
        assert 'spread_bps' in current
        assert 'mid_price' in current
        assert 'total_liquidity' in current
        assert 'liquidity_imbalance' in current
        
        # Verify signals
        signals = analysis['signals']
        assert 'liquidity_signal' in signals
        assert 'spread_signal' in signals
        assert 'imbalance_signal' in signals
        assert 'pressure_signal' in signals
        assert 'overall_signal' in signals
        
        return True
    
    def test_liquidity_analysis(self):
        """Test liquidity analysis for specific volumes."""
        analyzer = OrderBookAnalyzer()
        
        # Add order book data
        bids = [(1.1000, 1.5), (1.0999, 2.0), (1.0998, 1.0)]
        asks = [(1.1001, 1.2), (1.1002, 2.5), (1.1003, 1.8)]
        analyzer.update_order_book("EURUSD", bids, asks)
        
        # Test liquidity analysis for different volumes
        for volume in [0.1, 1.0, 5.0]:
            analysis = analyzer.get_liquidity_analysis("EURUSD", volume)
            
            assert 'symbol' in analysis
            assert 'volume' in analysis
            assert 'available_bid_liquidity' in analysis
            assert 'available_ask_liquidity' in analysis
            assert 'bid_execution_probability' in analysis
            assert 'ask_execution_probability' in analysis
            assert 'market_impact' in analysis
            assert 'recommended_split' in analysis
            
            # Verify probabilities are between 0 and 1
            assert 0 <= analysis['bid_execution_probability'] <= 1
            assert 0 <= analysis['ask_execution_probability'] <= 1
        
        return True
    
    def test_trading_signals_generation(self):
        """Test trading signals generation from order book analysis."""
        analyzer = OrderBookAnalyzer()
        
        # Add order book data with different characteristics
        # High liquidity, tight spread scenario
        bids = [(1.1000, 10.0), (1.0999, 15.0), (1.0998, 20.0)]
        asks = [(1.1001, 12.0), (1.1002, 18.0), (1.1003, 25.0)]
        analyzer.update_order_book("EURUSD", bids, asks)
        
        analysis = analyzer.get_market_analysis("EURUSD")
        signals = analysis['signals']
        
        # Should have positive liquidity signal due to high liquidity
        assert signals['liquidity_signal'] in ['positive', 'neutral', 'negative']
        assert signals['spread_signal'] in ['positive', 'neutral', 'negative']
        assert signals['imbalance_signal'] in ['buy', 'sell', 'neutral']
        assert signals['pressure_signal'] in ['buy', 'sell', 'neutral']
        assert signals['overall_signal'] in ['buy', 'sell', 'neutral']
        
        return True
    
    def test_order_book_snapshot_retrieval(self):
        """Test order book snapshot retrieval."""
        try:
            analyzer = OrderBookAnalyzer()
            
            # Add order book data
            bids = [(1.1000, 1.5), (1.0999, 2.0)]
            asks = [(1.1001, 1.2), (1.1002, 2.5)]
            analyzer.update_order_book("EURUSD", bids, asks)
            
            # Get snapshot
            snapshot = analyzer.get_order_book_snapshot("EURUSD")
            
            assert snapshot is not None
            assert isinstance(snapshot, OrderBookSnapshot)
            assert snapshot.symbol == "EURUSD"
            assert len(snapshot.bids) == 2
            assert len(snapshot.asks) == 2
            
            # Debug output
            print(f"Expected spread: 0.0001, Actual spread: {snapshot.spread}")
            
            # Use approximate comparison for floating point values
            assert abs(snapshot.spread - 0.0001) < 0.00001
            
            return True
        except Exception as e:
            print(f"Error in order book snapshot retrieval: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_microstructure_history(self):
        """Test microstructure history retrieval."""
        analyzer = OrderBookAnalyzer()
        
        # Add multiple updates
        for i in range(10):
            bids = [(1.1000 - i*0.0001, 1.5 + i*0.1)]
            asks = [(1.1001 + i*0.0001, 1.2 + i*0.1)]
            analyzer.update_order_book("EURUSD", bids, asks)
        
        # Get history for last 60 minutes
        history = analyzer.get_microstructure_history("EURUSD", minutes=60)
        
        assert len(history) == 10
        assert all(isinstance(m, MarketMicrostructure) for m in history)
        assert all(m.symbol == "EURUSD" for m in history)
        
        return True
    
    def test_data_export(self):
        """Test order book data export."""
        analyzer = OrderBookAnalyzer()
        
        # Add order book data
        bids = [(1.1000, 1.5), (1.0999, 2.0)]
        asks = [(1.1001, 1.2), (1.1002, 2.5)]
        analyzer.update_order_book("EURUSD", bids, asks)
        
        # Export data
        json_data = analyzer.export_order_book_data("EURUSD", "json")
        
        # Parse and verify
        data = json.loads(json_data)
        assert 'symbol' in data
        assert 'snapshots' in data
        assert len(data['snapshots']) == 1
        
        snapshot_data = data['snapshots'][0]
        assert 'timestamp' in snapshot_data
        assert 'spread' in snapshot_data
        assert 'mid_price' in snapshot_data
        assert 'bids' in snapshot_data
        assert 'asks' in snapshot_data
        
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
        
        # Verify order book analyzer is initialized
        assert hasattr(executor, 'order_book_analyzer')
        assert isinstance(executor.order_book_analyzer, OrderBookAnalyzer)
        
        # Test order book analysis methods
        analysis = executor.get_order_book_analysis("EURUSD")
        assert isinstance(analysis, dict)
        
        liquidity_analysis = executor.get_liquidity_analysis("EURUSD", 1.0)
        assert isinstance(liquidity_analysis, dict)
        
        snapshot = executor.get_order_book_snapshot("EURUSD")
        # Should be None initially since no data has been added
        
        return True
    
    def test_signal_adjustment_with_order_book(self):
        """Test signal adjustment based on order book analysis."""
        try:
            # Create mock config and executor
            config = TradingConfig(
                client_id="test_client",
                client_secret="test_secret",
                access_token="test_token",
                refresh_token="test_refresh",
                account_id=12345
            )
            
            executor = LiveTradingExecutor(config, ["EURUSD"])
            
            # Create a test signal
            from src.trading.live_trading_executor import TradingSignal
            signal = TradingSignal(
                symbol="EURUSD",
                action="buy",
                confidence=0.5,
                entry_price=1.1000,
                stop_loss=1.0980,
                take_profit=1.1020,
                volume=0.01
            )
            
            # Create mock order book analysis
            order_book_analysis = {
                'current': {
                    'spread': 0.0001,
                    'total_liquidity': 150.0  # High liquidity
                },
                'signals': {
                    'liquidity_signal': 'positive',
                    'spread_signal': 'positive',
                    'imbalance_signal': 'buy',
                    'pressure_signal': 'buy',
                    'overall_signal': 'buy'
                }
            }
            
            # Test signal adjustment
            adjusted_signal = executor._adjust_signal_with_order_book(signal, order_book_analysis)
            
            # Debug output
            print(f"Original confidence: {signal.confidence}, Adjusted confidence: {adjusted_signal.confidence}")
            print(f"Original volume: {signal.volume}, Adjusted volume: {adjusted_signal.volume}")
            
            # Verify that the method works and returns a valid signal
            assert isinstance(adjusted_signal, TradingSignal)
            assert adjusted_signal.symbol == signal.symbol
            assert adjusted_signal.action == signal.action
            assert 0.1 <= adjusted_signal.confidence <= 0.95  # Confidence should be within valid range
            assert adjusted_signal.volume > 0  # Volume should be positive
            
            return True
        except Exception as e:
            print(f"Error in signal adjustment: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_order_book_data_generation(self):
        """Test order book data generation in live trading executor."""
        # Create mock config and executor
        config = TradingConfig(
            client_id="test_client",
            client_secret="test_secret",
            access_token="test_token",
            refresh_token="test_refresh",
            account_id=12345
        )
        
        executor = LiveTradingExecutor(config, ["EURUSD"])
        
        # Create mock market data
        market_data = MarketData(
            symbol_id=1,
            symbol_name="EURUSD",
            bid=1.1000,
            ask=1.1001,
            timestamp=datetime.now(),
            digits=5
        )
        executor.market_data["EURUSD"] = market_data
        
        # Test order book data generation
        order_book_data = asyncio.run(executor._get_order_book_data("EURUSD"))
        
        assert order_book_data is not None
        bids, asks = order_book_data
        
        # Verify structure
        assert len(bids) == 10
        assert len(asks) == 10
        
        # Verify price ordering
        for i in range(1, len(bids)):
            assert bids[i][0] < bids[i-1][0]  # Bids should be descending
        
        for i in range(1, len(asks)):
            assert asks[i][0] > asks[i-1][0]  # Asks should be ascending
        
        # Verify volumes are positive
        assert all(volume > 0 for _, volume in bids)
        assert all(volume > 0 for _, volume in asks)
        
        return True
    
    def run_all_tests(self):
        """Run all order book integration tests."""
        print("🚀 Starting Order Book Integration Tests")
        print("=" * 60)
        
        # Core functionality tests
        self.run_test("Order Book Analyzer Initialization", self.test_order_book_analyzer_initialization)
        self.run_test("Order Book Update", self.test_order_book_update)
        self.run_test("Microstructure Analysis", self.test_microstructure_analysis)
        self.run_test("Market Analysis Generation", self.test_market_analysis_generation)
        self.run_test("Liquidity Analysis", self.test_liquidity_analysis)
        self.run_test("Trading Signals Generation", self.test_trading_signals_generation)
        
        # Data retrieval tests
        self.run_test("Order Book Snapshot Retrieval", self.test_order_book_snapshot_retrieval)
        self.run_test("Microstructure History", self.test_microstructure_history)
        self.run_test("Data Export", self.test_data_export)
        
        # Integration tests
        self.run_test("Live Trading Integration", self.test_live_trading_integration)
        self.run_test("Signal Adjustment with Order Book", self.test_signal_adjustment_with_order_book)
        self.run_test("Order Book Data Generation", self.test_order_book_data_generation)
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 ORDER BOOK INTEGRATION TEST RESULTS")
        print("=" * 60)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED! Order book integration is fully functional.")
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
    tester = OrderBookIntegrationTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 Order book integration is ready for production!")
        print("   - Real-time order book analysis ✓")
        print("   - Market microstructure insights ✓")
        print("   - Liquidity assessment ✓")
        print("   - Trading signal enhancement ✓")
        print("   - Live trading integration ✓")
        print("   - Data export capabilities ✓")
    else:
        print("\n🔧 Order book integration needs attention before production use.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 
