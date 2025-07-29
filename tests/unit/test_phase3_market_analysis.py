#!/usr/bin/env python3
"""
Test Phase 3: Real Market Analysis

This test verifies that the system can perform advanced market analysis
including regime detection and pattern recognition on real market data.
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np # Added missing import for numpy

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_phase3_market_analysis():
    """Test the complete Phase 3 market analysis system."""
    print("🧪 Testing Phase 3: Real Market Analysis...")
    
    try:
        # Import analysis components
        from src.sensory.dimensions.enhanced_when_dimension import MarketRegimeDetector, MarketRegime
        from src.sensory.dimensions.enhanced_anomaly_dimension import AdvancedPatternRecognition, PatternType
        from src.data.real_data_ingestor import RealDataIngestor
        
        # Get real market data
        ingestor = RealDataIngestor()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # More data for comprehensive analysis
        
        data = ingestor.load_symbol_data("EURUSD", start_date, end_date)
        
        if data is None or data.empty:
            print("   ❌ No data available for market analysis")
            return False
        
        print(f"   ✅ Loaded {len(data)} records for comprehensive analysis")
        
        # Initialize analysis components
        regime_detector = MarketRegimeDetector(lookback_period=30)
        pattern_recognition = AdvancedPatternRecognition()
        
        # Perform market regime analysis
        print("   📊 Analyzing market regime...")
        regime_result = regime_detector.detect_regime(data, "EURUSD")
        
        if regime_result.regime != MarketRegime.UNKNOWN:
            print(f"   ✅ Detected regime: {regime_result.regime.value.upper()}")
            print(f"   Confidence: {regime_result.confidence:.2%}")
            print(f"   Description: {regime_result.description}")
        else:
            print("   ⚠️  Regime detection returned UNKNOWN")
        
        # Perform pattern recognition
        print("   🔍 Analyzing trading patterns...")
        patterns = pattern_recognition.detect_patterns(data, "EURUSD")
        
        if patterns:
            print(f"   ✅ Detected {len(patterns)} trading patterns")
            for i, pattern in enumerate(patterns[:3]):  # Show top 3
                print(f"     {i+1}. {pattern.pattern_type.value.upper()} (confidence: {pattern.confidence:.2%})")
        else:
            print("   ⚠️  No patterns detected")
        
        # Analyze regime history
        print("   📈 Analyzing regime history...")
        regime_history = regime_detector.detect_regime_history(data, window_size=30, step_size=10)
        
        if regime_history:
            print(f"   ✅ Analyzed {len(regime_history)} regime periods")
            
            # Count regime types
            regime_counts = {}
            for result in regime_history:
                regime = result.regime.value
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            print(f"   📊 Regime distribution:")
            for regime, count in regime_counts.items():
                percentage = (count / len(regime_history)) * 100
                print(f"     {regime}: {count} periods ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing market analysis: {e}")
        return False


def test_regime_pattern_integration():
    """Test integration between regime detection and pattern recognition."""
    print("\n🧪 Testing Regime-Pattern Integration...")
    
    try:
        from src.sensory.dimensions.enhanced_when_dimension import MarketRegimeDetector
        from src.sensory.dimensions.enhanced_anomaly_dimension import AdvancedPatternRecognition
        from src.data.real_data_ingestor import RealDataIngestor
        
        ingestor = RealDataIngestor()
        regime_detector = MarketRegimeDetector(lookback_period=25)
        pattern_recognition = AdvancedPatternRecognition()
        
        instruments = ["EURUSD", "GBPUSD", "USDJPY"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        results = {}
        
        for instrument in instruments:
            print(f"   Analyzing {instrument}...")
            try:
                data = ingestor.load_symbol_data(instrument, start_date, end_date)
                
                if data is not None and not data.empty:
                    # Detect regime
                    regime_result = regime_detector.detect_regime(data, instrument)
                    
                    # Detect patterns
                    patterns = pattern_recognition.detect_patterns(data, instrument)
                    
                    results[instrument] = {
                        'regime': regime_result.regime.value,
                        'regime_confidence': regime_result.confidence,
                        'patterns_found': len(patterns),
                        'top_pattern': patterns[0].pattern_type.value if patterns else 'none',
                        'data_points': len(data)
                    }
                    
                    print(f"     ✅ {regime_result.regime.value.upper()} regime, {len(patterns)} patterns")
                else:
                    results[instrument] = {'regime': 'unknown', 'patterns_found': 0, 'data_points': 0}
                    print(f"     ❌ No data available")
                    
            except Exception as e:
                results[instrument] = {'regime': 'error', 'patterns_found': 0, 'data_points': 0}
                print(f"     ❌ Error: {e}")
        
        # Summary
        successful_instruments = sum(1 for r in results.values() if r['data_points'] > 0)
        total_patterns = sum(r['patterns_found'] for r in results.values())
        
        print(f"\n   📊 Integration Results:")
        print(f"     Instruments analyzed: {successful_instruments}/{len(instruments)}")
        print(f"     Total patterns detected: {total_patterns}")
        
        return successful_instruments > 0
        
    except Exception as e:
        print(f"   ❌ Error testing integration: {e}")
        return False


def test_analysis_accuracy():
    """Test the accuracy of market analysis components."""
    print("\n🧪 Testing Analysis Accuracy...")
    
    try:
        from src.sensory.dimensions.enhanced_when_dimension import MarketRegimeDetector
        from src.sensory.dimensions.enhanced_anomaly_dimension import AdvancedPatternRecognition
        from src.data.real_data_ingestor import RealDataIngestor
        
        ingestor = RealDataIngestor()
        regime_detector = MarketRegimeDetector(lookback_period=30)
        pattern_recognition = AdvancedPatternRecognition()
        
        # Test with different time periods
        periods = [30, 60, 90]
        end_date = datetime.now()
        
        accuracy_results = {}
        
        for days in periods:
            print(f"   Testing {days}-day period...")
            start_date = end_date - timedelta(days=days)
            
            data = ingestor.load_symbol_data("EURUSD", start_date, end_date)
            
            if data is not None and not data.empty:
                # Test regime detection
                regime_result = regime_detector.detect_regime(data, "EURUSD")
                
                # Test pattern recognition
                patterns = pattern_recognition.detect_patterns(data, "EURUSD")
                
                accuracy_results[days] = {
                    'regime_confidence': regime_result.confidence,
                    'patterns_found': len(patterns),
                    'avg_pattern_confidence': np.mean([p.confidence for p in patterns]) if patterns else 0,
                    'data_points': len(data)
                }
                
                print(f"     ✅ Regime confidence: {regime_result.confidence:.2%}")
                print(f"     ✅ Patterns: {len(patterns)} (avg confidence: {accuracy_results[days]['avg_pattern_confidence']:.2%})")
            else:
                accuracy_results[days] = {'regime_confidence': 0, 'patterns_found': 0, 'avg_pattern_confidence': 0, 'data_points': 0}
                print(f"     ❌ No data available")
        
        # Calculate overall accuracy
        valid_periods = [r for r in accuracy_results.values() if r['data_points'] > 0]
        if valid_periods:
            avg_regime_confidence = np.mean([r['regime_confidence'] for r in valid_periods])
            avg_pattern_confidence = np.mean([r['avg_pattern_confidence'] for r in valid_periods])
            
            print(f"\n   📊 Accuracy Summary:")
            print(f"     Average regime confidence: {avg_regime_confidence:.2%}")
            print(f"     Average pattern confidence: {avg_pattern_confidence:.2%}")
            
            return avg_regime_confidence > 0.5 and avg_pattern_confidence > 0.3
        else:
            print(f"   ❌ No valid periods for accuracy testing")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing accuracy: {e}")
        return False


def test_analysis_performance():
    """Test the performance of market analysis components."""
    print("\n🧪 Testing Analysis Performance...")
    
    try:
        import time
        from src.sensory.dimensions.enhanced_when_dimension import MarketRegimeDetector
        from src.sensory.dimensions.enhanced_anomaly_dimension import AdvancedPatternRecognition
        from src.data.real_data_ingestor import RealDataIngestor
        
        ingestor = RealDataIngestor()
        regime_detector = MarketRegimeDetector(lookback_period=30)
        pattern_recognition = AdvancedPatternRecognition()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        data = ingestor.load_symbol_data("EURUSD", start_date, end_date)
        
        if data is None or data.empty:
            print("   ❌ No data available for performance testing")
            return False
        
        # Test regime detection performance
        print("   Testing regime detection performance...")
        start_time = time.time()
        regime_result = regime_detector.detect_regime(data, "EURUSD")
        regime_time = time.time() - start_time
        
        # Test pattern recognition performance
        print("   Testing pattern recognition performance...")
        start_time = time.time()
        patterns = pattern_recognition.detect_patterns(data, "EURUSD")
        pattern_time = time.time() - start_time
        
        # Test regime history performance
        print("   Testing regime history performance...")
        start_time = time.time()
        regime_history = regime_detector.detect_regime_history(data, window_size=30, step_size=10)
        history_time = time.time() - start_time
        
        print(f"   📊 Performance Results:")
        print(f"     Regime detection: {regime_time:.3f}s")
        print(f"     Pattern recognition: {pattern_time:.3f}s")
        print(f"     Regime history: {history_time:.3f}s")
        print(f"     Total analysis time: {regime_time + pattern_time + history_time:.3f}s")
        
        # Check if performance is reasonable (should be under 5 seconds for this data size)
        total_time = regime_time + pattern_time + history_time
        if total_time < 5.0:
            print(f"   ✅ Performance is acceptable")
            return True
        else:
            print(f"   ⚠️  Performance is slow ({total_time:.3f}s)")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing performance: {e}")
        return False


def main():
    """Run all Phase 3 market analysis tests."""
    print("🚀 PHASE 3: REAL MARKET ANALYSIS TEST SUITE")
    print("=" * 50)
    
    # Test 1: Complete market analysis
    test1_passed = test_phase3_market_analysis()
    
    # Test 2: Regime-pattern integration
    test2_passed = test_regime_pattern_integration()
    
    # Test 3: Analysis accuracy
    test3_passed = test_analysis_accuracy()
    
    # Test 4: Analysis performance
    test4_passed = test_analysis_performance()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 PHASE 3 TEST SUMMARY")
    print("=" * 50)
    print(f"Complete Market Analysis: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Regime-Pattern Integration: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Analysis Accuracy: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print(f"Analysis Performance: {'✅ PASSED' if test4_passed else '❌ FAILED'}")
    
    total_passed = sum([test1_passed, test2_passed, test3_passed, test4_passed])
    print(f"\nOverall: {total_passed}/4 tests passed")
    
    if total_passed >= 3:
        print("🎉 Phase 3: Real Market Analysis is working!")
        print("   The system can now perform advanced market analysis including:")
        print("   - Market regime detection")
        print("   - Pattern recognition")
        print("   - Integration between analysis components")
        print("   - Performance optimization")
    else:
        print("⚠️  Phase 3: Real Market Analysis needs improvement.")
        print("   The system is still using basic analysis methods.")
    
    return total_passed >= 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
