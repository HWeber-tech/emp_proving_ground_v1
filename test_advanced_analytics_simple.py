#!/usr/bin/env python3
"""
Simplified Advanced Analytics Test Suite
Tests the advanced analytics functionality without complex trading dependencies.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.advanced_analytics import (
    AdvancedAnalytics, SentimentAnalysis, AdvancedIndicators, 
    MarketCorrelation, SentimentType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAdvancedAnalyticsTest:
    """Simplified test suite for advanced analytics."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.test_results = []
        self.passed_tests = 0
        self.total_tests = 0
        
        # Initialize advanced analytics
        self.advanced_analytics = AdvancedAnalytics()
        
        # Test configuration
        self.test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        self.test_period = 60  # days
        
        logger.info("Simple Advanced Analytics Test Suite initialized")
    
    def run_all_tests(self):
        """Run all simplified tests."""
        logger.info("=" * 60)
        logger.info("SIMPLE ADVANCED ANALYTICS TEST SUITE")
        logger.info("=" * 60)
        
        # Test 1: Core Functionality
        self._test_core_functionality()
        
        # Test 2: Sentiment Analysis
        self._test_sentiment_analysis()
        
        # Test 3: Technical Indicators
        self._test_technical_indicators()
        
        # Test 4: Market Correlation
        self._test_market_correlation()
        
        # Test 5: Comprehensive Analysis
        self._test_comprehensive_analysis()
        
        # Test 6: Performance and Caching
        self._test_performance_and_caching()
        
        # Test 7: Error Handling
        self._test_error_handling()
        
        # Test 8: Trading Signal Generation
        self._test_trading_signal_generation()
        
        # Print results
        self._print_results()
    
    def _test_core_functionality(self):
        """Test core advanced analytics functionality."""
        logger.info("\n--- Test 1: Core Functionality ---")
        
        try:
            # Test initialization
            analytics = AdvancedAnalytics()
            assert analytics is not None, "AdvancedAnalytics initialization failed"
            self._log_test_result("Initialization", True)
            
            # Test configuration
            assert hasattr(analytics, 'news_cache'), "Missing news_cache attribute"
            assert hasattr(analytics, 'sentiment_cache'), "Missing sentiment_cache attribute"
            assert hasattr(analytics, 'indicators_cache'), "Missing indicators_cache attribute"
            assert hasattr(analytics, 'correlation_cache'), "Missing correlation_cache attribute"
            self._log_test_result("Configuration", True)
            
            # Test market symbols
            assert len(analytics.market_symbols) > 0, "No market symbols configured"
            assert 'SPY' in analytics.market_symbols, "SPY not in market symbols"
            self._log_test_result("Market Symbols", True)
            
            logger.info("✓ Core Functionality: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Core Functionality: FAILED - {e}")
            self._log_test_result("Core Functionality", False)
    
    def _test_sentiment_analysis(self):
        """Test sentiment analysis."""
        logger.info("\n--- Test 2: Sentiment Analysis ---")
        
        try:
            for symbol in self.test_symbols:
                # Test sentiment analysis
                sentiment = self.advanced_analytics.analyze_sentiment(symbol)
                
                # Validate sentiment structure
                assert isinstance(sentiment, SentimentAnalysis), "Invalid sentiment type"
                assert sentiment.symbol == symbol, "Symbol mismatch"
                assert isinstance(sentiment.overall_sentiment, float), "Invalid sentiment score"
                assert sentiment.sentiment_type in SentimentType, "Invalid sentiment type"
                assert sentiment.news_count >= 0, "Invalid news count"
                assert isinstance(sentiment.top_keywords, list), "Invalid keywords"
                
                # Test sentiment caching
                cached_sentiment = self.advanced_analytics.analyze_sentiment(symbol)
                assert cached_sentiment.timestamp == sentiment.timestamp, "Caching not working"
                
                # Test force refresh
                refreshed_sentiment = self.advanced_analytics.analyze_sentiment(symbol, force_refresh=True)
                assert refreshed_sentiment.timestamp > sentiment.timestamp, "Force refresh not working"
                
                logger.info(f"✓ Sentiment analysis for {symbol}: {sentiment.sentiment_type.value} ({sentiment.overall_sentiment:.3f})")
            
            self._log_test_result("Sentiment Analysis", True)
            logger.info("✓ Sentiment Analysis: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Sentiment Analysis: FAILED - {e}")
            self._log_test_result("Sentiment Analysis", False)
    
    def _test_technical_indicators(self):
        """Test technical indicators."""
        logger.info("\n--- Test 3: Technical Indicators ---")
        
        try:
            # Generate test data
            test_data = self._generate_test_data()
            
            for symbol in self.test_symbols:
                # Test advanced indicators calculation
                indicators = self.advanced_analytics.calculate_advanced_indicators(symbol, test_data)
                
                # Validate indicators structure
                assert isinstance(indicators, AdvancedIndicators), "Invalid indicators type"
                assert indicators.symbol == symbol, "Symbol mismatch"
                
                # Test momentum indicators
                assert 0 <= indicators.rsi <= 100, "RSI out of range"
                assert 0 <= indicators.stochastic_k <= 100, "Stochastic K out of range"
                assert 0 <= indicators.stochastic_d <= 100, "Stochastic D out of range"
                assert -100 <= indicators.williams_r <= 0, "Williams %R out of range"
                
                # Test volatility indicators
                assert indicators.bollinger_upper >= indicators.bollinger_middle, "Bollinger bands invalid"
                assert indicators.bollinger_middle >= indicators.bollinger_lower, "Bollinger bands invalid"
                assert indicators.atr >= 0, "ATR negative"
                
                # Test trend indicators
                assert isinstance(indicators.macd, float), "Invalid MACD"
                assert isinstance(indicators.macd_signal, float), "Invalid MACD signal"
                assert isinstance(indicators.macd_histogram, float), "Invalid MACD histogram"
                assert 0 <= indicators.adx <= 100, "ADX out of range"
                
                # Test volume indicators
                assert isinstance(indicators.obv, (int, float)), "OBV not numeric"
                assert indicators.vwap >= 0, "VWAP negative"
                assert 0 <= indicators.money_flow_index <= 100, "MFI out of range"
                
                # Test custom indicators
                assert indicators.support_level >= 0, "Support level negative"
                assert indicators.resistance_level >= 0, "Resistance level negative"
                assert indicators.pivot_point >= 0, "Pivot point negative"
                assert isinstance(indicators.fibonacci_retracement, dict), "Invalid Fibonacci levels"
                
                logger.info(f"✓ Technical indicators for {symbol}: RSI={indicators.rsi:.1f}, MACD={indicators.macd:.3f}")
            
            self._log_test_result("Technical Indicators", True)
            logger.info("✓ Technical Indicators: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Technical Indicators: FAILED - {e}")
            self._log_test_result("Technical Indicators", False)
    
    def _test_market_correlation(self):
        """Test market correlation."""
        logger.info("\n--- Test 4: Market Correlation ---")
        
        try:
            # Generate test data
            test_data = self._generate_test_data()
            
            for symbol in self.test_symbols:
                # Test market correlation analysis
                correlation = self.advanced_analytics.analyze_market_correlation(symbol, test_data)
                
                # Validate correlation structure
                assert isinstance(correlation, MarketCorrelation), "Invalid correlation type"
                assert correlation.symbol == symbol, "Symbol mismatch"
                
                # Test correlation metrics
                assert isinstance(correlation.beta, float), "Invalid beta"
                assert isinstance(correlation.alpha, float), "Invalid alpha"
                assert isinstance(correlation.sharpe_ratio, float), "Invalid Sharpe ratio"
                assert correlation.volatility >= 0, "Negative volatility"
                
                # Test correlations dictionary
                assert isinstance(correlation.correlations, dict), "Invalid correlations"
                for market_symbol, corr_value in correlation.correlations.items():
                    assert -1 <= corr_value <= 1, f"Correlation out of range: {corr_value}"
                
                # Test correlation matrix
                assert isinstance(correlation.correlation_matrix, pd.DataFrame), "Invalid correlation matrix"
                
                # Test sector and market correlations
                assert -1 <= correlation.sector_correlation <= 1, "Sector correlation out of range"
                assert -1 <= correlation.market_correlation <= 1, "Market correlation out of range"
                
                logger.info(f"✓ Market correlation for {symbol}: beta={correlation.beta:.2f}, sharpe={correlation.sharpe_ratio:.2f}")
            
            self._log_test_result("Market Correlation", True)
            logger.info("✓ Market Correlation: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Market Correlation: FAILED - {e}")
            self._log_test_result("Market Correlation", False)
    
    def _test_comprehensive_analysis(self):
        """Test comprehensive analysis."""
        logger.info("\n--- Test 5: Comprehensive Analysis ---")
        
        try:
            # Generate test data
            test_data = self._generate_test_data()
            
            for symbol in self.test_symbols:
                # Test comprehensive analysis
                analysis = self.advanced_analytics.get_comprehensive_analysis(symbol, test_data)
                
                # Validate analysis structure
                assert isinstance(analysis, dict), "Invalid analysis type"
                assert analysis['symbol'] == symbol, "Symbol mismatch"
                assert 'timestamp' in analysis, "Missing timestamp"
                assert 'sentiment' in analysis, "Missing sentiment"
                assert 'technical_indicators' in analysis, "Missing technical indicators"
                assert 'market_correlation' in analysis, "Missing market correlation"
                assert 'trading_signals' in analysis, "Missing trading signals"
                
                # Test sentiment section
                sentiment = analysis['sentiment']
                assert 'overall_sentiment' in sentiment, "Missing overall sentiment"
                assert 'sentiment_type' in sentiment, "Missing sentiment type"
                assert 'news_count' in sentiment, "Missing news count"
                assert 'sentiment_trend' in sentiment, "Missing sentiment trend"
                
                # Test technical indicators section
                indicators = analysis['technical_indicators']
                assert 'momentum' in indicators, "Missing momentum indicators"
                assert 'volatility' in indicators, "Missing volatility indicators"
                assert 'trend' in indicators, "Missing trend indicators"
                assert 'volume' in indicators, "Missing volume indicators"
                assert 'support_resistance' in indicators, "Missing support/resistance"
                
                # Test market correlation section
                correlation = analysis['market_correlation']
                assert 'beta' in correlation, "Missing beta"
                assert 'alpha' in correlation, "Missing alpha"
                assert 'sharpe_ratio' in correlation, "Missing Sharpe ratio"
                assert 'volatility' in correlation, "Missing volatility"
                
                # Test trading signals section
                signals = analysis['trading_signals']
                assert 'sentiment_signal' in signals, "Missing sentiment signal"
                assert 'technical_signal' in signals, "Missing technical signal"
                assert 'correlation_signal' in signals, "Missing correlation signal"
                assert 'overall_signal' in signals, "Missing overall signal"
                assert 'confidence' in signals, "Missing confidence"
                
                logger.info(f"✓ Comprehensive analysis for {symbol}: {signals['overall_signal']} (confidence: {signals['confidence']:.2f})")
            
            self._log_test_result("Comprehensive Analysis", True)
            logger.info("✓ Comprehensive Analysis: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Comprehensive Analysis: FAILED - {e}")
            self._log_test_result("Comprehensive Analysis", False)
    
    def _test_performance_and_caching(self):
        """Test performance and caching."""
        logger.info("\n--- Test 6: Performance and Caching ---")
        
        try:
            import time
            
            # Test caching performance
            symbol = self.test_symbols[0]
            
            # First call (no cache)
            start_time = time.time()
            sentiment1 = self.advanced_analytics.analyze_sentiment(symbol)
            first_call_time = time.time() - start_time
            
            # Second call (cached)
            start_time = time.time()
            sentiment2 = self.advanced_analytics.analyze_sentiment(symbol)
            second_call_time = time.time() - start_time
            
            # Cached call should be faster
            assert second_call_time < first_call_time, "Caching not improving performance"
            assert sentiment1.timestamp == sentiment2.timestamp, "Cache not working"
            
            # Test cache invalidation
            time.sleep(0.1)  # Small delay
            sentiment3 = self.advanced_analytics.analyze_sentiment(symbol, force_refresh=True)
            assert sentiment3.timestamp > sentiment1.timestamp, "Force refresh not working"
            
            logger.info(f"✓ Performance: First call: {first_call_time:.3f}s, Cached call: {second_call_time:.3f}s")
            self._log_test_result("Performance and Caching", True)
            logger.info("✓ Performance and Caching: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Performance and Caching: FAILED - {e}")
            self._log_test_result("Performance and Caching", False)
    
    def _test_error_handling(self):
        """Test error handling and resilience."""
        logger.info("\n--- Test 7: Error Handling ---")
        
        try:
            # Test with empty data
            empty_data = pd.DataFrame()
            indicators = self.advanced_analytics.calculate_advanced_indicators("TEST", empty_data)
            assert indicators is not None, "Should handle empty data gracefully"
            
            # Test with invalid symbol
            sentiment = self.advanced_analytics.analyze_sentiment("INVALID_SYMBOL_12345")
            assert sentiment is not None, "Should handle invalid symbols gracefully"
            
            # Test with None data
            correlation = self.advanced_analytics.analyze_market_correlation("TEST", None)
            assert correlation is not None, "Should handle None data gracefully"
            
            # Test with malformed data
            malformed_data = pd.DataFrame({'Invalid': [1, 2, 3]})
            indicators = self.advanced_analytics.calculate_advanced_indicators("TEST", malformed_data)
            assert indicators is not None, "Should handle malformed data gracefully"
            
            logger.info("✓ Error handling: All error conditions handled gracefully")
            self._log_test_result("Error Handling", True)
            logger.info("✓ Error Handling: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Error Handling: FAILED - {e}")
            self._log_test_result("Error Handling", False)
    
    def _test_trading_signal_generation(self):
        """Test trading signal generation."""
        logger.info("\n--- Test 8: Trading Signal Generation ---")
        
        try:
            # Generate test data
            test_data = self._generate_test_data()
            
            for symbol in self.test_symbols:
                # Get comprehensive analysis
                analysis = self.advanced_analytics.get_comprehensive_analysis(symbol, test_data)
                
                # Extract trading signals
                signals = analysis['trading_signals']
                
                # Validate signal enhancement
                assert signals['overall_signal'] in ['buy', 'sell', 'neutral'], "Invalid overall signal"
                assert 0 <= signals['confidence'] <= 1, "Confidence out of range"
                
                # Test signal consistency
                sentiment_signal = signals['sentiment_signal']
                technical_signal = signals['technical_signal']
                correlation_signal = signals['correlation_signal']
                
                # Count signal types
                buy_count = sum(1 for s in [sentiment_signal, technical_signal, correlation_signal] if s == 'buy')
                sell_count = sum(1 for s in [sentiment_signal, technical_signal, correlation_signal] if s == 'sell')
                
                # Validate signal logic
                if buy_count > sell_count:
                    assert signals['overall_signal'] == 'buy', "Signal aggregation logic error"
                elif sell_count > buy_count:
                    assert signals['overall_signal'] == 'sell', "Signal aggregation logic error"
                
                logger.info(f"✓ Signal generation for {symbol}: {signals['overall_signal']} "
                           f"(confidence: {signals['confidence']:.2f})")
            
            self._log_test_result("Trading Signal Generation", True)
            logger.info("✓ Trading Signal Generation: PASSED")
            
        except Exception as e:
            logger.error(f"✗ Trading Signal Generation: FAILED - {e}")
            self._log_test_result("Trading Signal Generation", False)
    
    def _generate_test_data(self) -> pd.DataFrame:
        """Generate synthetic test data."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=self.test_period), 
                            end=datetime.now(), freq='D')
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        base_price = 1.2000
        returns = np.random.normal(0, 0.01, len(dates))  # 1% daily volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'Close': prices,
            'Volume': np.random.uniform(1000, 10000, len(dates))
        })
        
        # Ensure High >= Low
        data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
        
        return data
    
    def _log_test_result(self, test_name: str, passed: bool):
        """Log test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        self.test_results.append((test_name, passed))
    
    def _print_results(self):
        """Print test results summary."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        for test_name, passed in self.test_results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info("-" * 60)
        logger.info(f"Total Tests: {self.total_tests}")
        logger.info(f"Passed: {self.passed_tests}")
        logger.info(f"Failed: {self.total_tests - self.passed_tests}")
        logger.info(f"Success Rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        
        if self.passed_tests == self.total_tests:
            logger.info("\n🎉 ALL TESTS PASSED! Advanced Analytics is working correctly.")
        else:
            logger.info(f"\n⚠ {self.total_tests - self.passed_tests} tests failed. Please review the issues above.")
        
        logger.info("=" * 60)

def main():
    """Run the simple advanced analytics test suite."""
    try:
        # Create and run test suite
        test_suite = SimpleAdvancedAnalyticsTest()
        test_suite.run_all_tests()
        
        # Return exit code based on test results
        if test_suite.passed_tests == test_suite.total_tests:
            return 0
        else:
            return 1
            
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 