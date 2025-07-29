#!/usr/bin/env python3
"""
Test Market Regime Detection

This test verifies that the market regime detector can identify
different market conditions using real market data.
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_market_regime_detection():
    """Test the market regime detector functionality."""
    print("🧪 Testing Market Regime Detection...")
    
    try:
        # Import the market regime detector
        from src.sensory.dimensions.enhanced_when_dimension import MarketRegimeDetector, MarketRegime
        from src.data.real_data_ingestor import RealDataIngestor
        
        # Get real market data
        ingestor = RealDataIngestor()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # More data for better analysis
        
        data = ingestor.load_symbol_data("EURUSD", start_date, end_date)
        
        if data is None or data.empty:
            print("   ❌ No data available for regime detection")
            return False
        
        print(f"   ✅ Loaded {len(data)} records for analysis")
        
        # Initialize detector
        detector = MarketRegimeDetector(lookback_period=30)
        
        # Detect current regime
        print("   Detecting current market regime...")
        regime_result = detector.detect_regime(data, "EURUSD")
        
        if regime_result.regime == MarketRegime.UNKNOWN:
            print("   ⚠️  Regime detection returned UNKNOWN")
        else:
            print(f"   ✅ Detected regime: {regime_result.regime.value.upper()}")
            print(f"   Confidence: {regime_result.confidence:.2%}")
            print(f"   Description: {regime_result.description}")
        
        # Show regime metrics
        print(f"   📊 Regime Metrics:")
        for key, value in regime_result.metrics.items():
            print(f"     {key}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing regime detection: {e}")
        return False


def test_regime_history():
    """Test regime history detection."""
    print("\n🧪 Testing Regime History...")
    
    try:
        from src.sensory.dimensions.enhanced_when_dimension import MarketRegimeDetector
        from src.data.real_data_ingestor import RealDataIngestor
        
        # Get real market data
        ingestor = RealDataIngestor()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # More data for history
        
        data = ingestor.load_symbol_data("EURUSD", start_date, end_date)
        
        if data is None or data.empty:
            print("   ❌ No data available for regime history")
            return False
        
        # Initialize detector
        detector = MarketRegimeDetector(lookback_period=20)
        
        # Detect regime history
        print("   Detecting regime history...")
        regime_history = detector.detect_regime_history(data, window_size=30, step_size=5)
        
        if not regime_history:
            print("   ❌ No regime history detected")
            return False
        
        print(f"   ✅ Detected {len(regime_history)} regime periods")
        
        # Analyze regime distribution
        regime_counts = {}
        for result in regime_history:
            regime = result.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        print(f"   📈 Regime Distribution:")
        for regime, count in regime_counts.items():
            percentage = (count / len(regime_history)) * 100
            print(f"     {regime}: {count} periods ({percentage:.1f}%)")
        
        # Find regime transitions
        transitions = detector.get_regime_transitions(regime_history)
        if transitions:
            print(f"   🔄 Found {len(transitions)} regime transitions")
            for transition in transitions[-3:]:  # Show last 3
                print(f"     {transition['transition_time'].date()}: {transition['description']}")
        else:
            print("   ⚠️  No regime transitions detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing regime history: {e}")
        return False


def test_multiple_instruments():
    """Test regime detection on multiple instruments."""
    print("\n🧪 Testing Multiple Instruments...")
    
    try:
        from src.sensory.dimensions.enhanced_when_dimension import MarketRegimeDetector
        from src.data.real_data_ingestor import RealDataIngestor
        
        ingestor = RealDataIngestor()
        detector = MarketRegimeDetector(lookback_period=25)
        
        instruments = ["EURUSD", "GBPUSD", "USDJPY"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        results = {}
        
        for instrument in instruments:
            print(f"   Analyzing {instrument}...")
            try:
                data = ingestor.load_symbol_data(instrument, start_date, end_date)
                
                if data is not None and not data.empty:
                    regime_result = detector.detect_regime(data, instrument)
                    results[instrument] = {
                        'regime': regime_result.regime.value,
                        'confidence': regime_result.confidence,
                        'data_points': len(data)
                    }
                    print(f"     ✅ {regime_result.regime.value.upper()} (confidence: {regime_result.confidence:.2%})")
                else:
                    results[instrument] = {'regime': 'unknown', 'confidence': 0, 'data_points': 0}
                    print(f"     ❌ No data available")
                    
            except Exception as e:
                results[instrument] = {'regime': 'error', 'confidence': 0, 'data_points': 0}
                print(f"     ❌ Error: {e}")
        
        # Summary
        successful_instruments = sum(1 for r in results.values() if r['data_points'] > 0)
        print(f"\n   📊 Results: {successful_instruments}/{len(instruments)} instruments analyzed")
        
        return successful_instruments > 0
        
    except Exception as e:
        print(f"   ❌ Error testing multiple instruments: {e}")
        return False


def test_regime_metrics():
    """Test regime-specific metrics calculation."""
    print("\n🧪 Testing Regime Metrics...")
    
    try:
        from src.sensory.dimensions.enhanced_when_dimension import MarketRegimeDetector
        from src.data.real_data_ingestor import RealDataIngestor
        
        ingestor = RealDataIngestor()
        detector = MarketRegimeDetector(lookback_period=30)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=45)
        
        data = ingestor.load_symbol_data("EURUSD", start_date, end_date)
        
        if data is None or data.empty:
            print("   ❌ No data available for metrics testing")
            return False
        
        # Detect regime and get metrics
        regime_result = detector.detect_regime(data, "EURUSD")
        
        if regime_result.metrics:
            print(f"   ✅ Regime metrics calculated:")
            for key, value in regime_result.metrics.items():
                print(f"     {key}: {value:.4f}")
            
            # Check if metrics are reasonable
            avg_vol = regime_result.metrics.get('avg_volatility', 0)
            if 0 <= avg_vol <= 1:  # Volatility should be between 0 and 1
                print(f"   ✅ Volatility metric is reasonable: {avg_vol:.4f}")
            else:
                print(f"   ⚠️  Volatility metric seems unusual: {avg_vol:.4f}")
            
            return True
        else:
            print("   ❌ No metrics calculated")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing regime metrics: {e}")
        return False


def main():
    """Run all market regime detection tests."""
    print("🚀 MARKET REGIME DETECTION TEST SUITE")
    print("=" * 50)
    
    # Test 1: Basic regime detection
    test1_passed = test_market_regime_detection()
    
    # Test 2: Regime history
    test2_passed = test_regime_history()
    
    # Test 3: Multiple instruments
    test3_passed = test_multiple_instruments()
    
    # Test 4: Regime metrics
    test4_passed = test_regime_metrics()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    print(f"Basic Regime Detection: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Regime History: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Multiple Instruments: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print(f"Regime Metrics: {'✅ PASSED' if test4_passed else '❌ FAILED'}")
    
    total_passed = sum([test1_passed, test2_passed, test3_passed, test4_passed])
    print(f"\nOverall: {total_passed}/4 tests passed")
    
    if total_passed >= 3:
        print("🎉 Market regime detection is working!")
        print("   The system can now identify different market conditions using real data.")
    else:
        print("⚠️  Market regime detection needs improvement.")
        print("   The system is still using basic analysis methods.")
    
    return total_passed >= 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
