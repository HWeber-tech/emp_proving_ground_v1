#!/usr/bin/env python3
"""
EMP System - Reality Check Tests

These tests provide honest assessment of actual functionality vs. mock implementations.
Many tests are EXPECTED TO FAIL until real integrations are implemented.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal

def test_real_data_availability():
    """Test: Can we access real market data?"""
    print("\n🔍 REALITY CHECK: Data Availability")
    
    # Check for real data files
    data_files = [
        "data/raw/EURUSD_2023.csv",
        "data/raw/EURUSD_2024.csv",
        "data/processed/EURUSD_ohlcv.csv"
    ]
    
    real_data_found = False
    for path in data_files:
        if os.path.exists(path):
            print(f"✅ Found real data: {path}")
            real_data_found = True
        else:
            print(f"❌ Missing real data: {path}")
    
    if not real_data_found:
        print("⚠️  SYSTEM STATUS: Using synthetic data only")
        print("⚠️  IMPACT: Evolution runs on fake patterns")
        return False
    
    return True

def test_economic_data_integration():
    """Test: Can we fetch real economic data?"""
    print("\n🔍 REALITY CHECK: Economic Data Integration")
    
    try:
        # Test FRED API integration
        from src.sensory.dimensions.why_engine import WHYEngine
        from src.sensory.core.base import InstrumentMeta
        
        # Create mock instrument meta for testing
        instrument_meta = InstrumentMeta(symbol="EURUSD")
        
        # This should fail if no real API integration
        engine = WHYEngine(instrument_meta)
        
        # Check if economic data is real or synthetic
        if hasattr(engine, '_fetch_fred_data'):
            print("✅ FRED API integration exists")
            return True
        else:
            print("❌ FRED API integration not implemented")
            print("⚠️  SYSTEM STATUS: Using synthetic economic data")
            return False
            
    except Exception as e:
        print(f"❌ Economic data integration failed: {e}")
        print("⚠️  SYSTEM STATUS: No real economic data available")
        return False

def test_broker_integration():
    """Test: Can we connect to a real broker?"""
    print("\n🔍 REALITY CHECK: Broker Integration")
    
    try:
        # Test if any broker integration exists
        from src.simulation import MarketSimulator
        from src.data import TickDataStorage
        
        # Create mock data storage for testing
        data_storage = TickDataStorage()
        simulator = MarketSimulator(data_storage)
        
        if hasattr(simulator, 'connect_broker'):
            print("✅ Broker connection method exists")
            return True
        else:
            print("❌ Broker integration not implemented")
            print("⚠️  SYSTEM STATUS: Simulation only, no live trading")
            return False
            
    except Exception as e:
        print(f"❌ Broker integration test failed: {e}")
        print("⚠️  SYSTEM STATUS: No live trading capability")
        return False

def test_real_strategy_discovery():
    """Test: Can we discover strategies from real market patterns?"""
    print("\n🔍 REALITY CHECK: Strategy Discovery")
    
    try:
        from src.evolution import DecisionGenome, FitnessEvaluator
        
        # Create a genome with required parameters
        genome = DecisionGenome(genome_id="test", decision_tree={})
        
        # Test if it can evaluate on real data
        if hasattr(genome, 'evaluate_real_data'):
            print("✅ Real data evaluation method exists")
            return True
        else:
            print("❌ Real data evaluation not implemented")
            print("⚠️  SYSTEM STATUS: Strategies evolved on synthetic data only")
            return False
            
    except Exception as e:
        print(f"❌ Strategy discovery test failed: {e}")
        print("⚠️  SYSTEM STATUS: No real strategy discovery")
        return False

def test_sentiment_analysis():
    """Test: Can we analyze real market sentiment?"""
    print("\n🔍 REALITY CHECK: Sentiment Analysis")
    
    try:
        from src.sensory.dimensions.why_engine import WHYEngine
        
        engine = WHYEngine("EURUSD")
        
        # Check if sentiment analysis uses real data
        if hasattr(engine, '_fetch_news_sentiment'):
            print("✅ News sentiment API integration exists")
            return True
        else:
            print("❌ Real sentiment analysis not implemented")
            print("⚠️  SYSTEM STATUS: Using synthetic sentiment data")
            return False
            
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        print("⚠️  SYSTEM STATUS: No real sentiment analysis")
        return False

def test_order_book_data():
    """Test: Can we access real order book data?"""
    print("\n🔍 REALITY CHECK: Order Book Data")
    
    try:
        from src.sensory.dimensions.how_engine import HOWEngine
        
        engine = HOWEngine("EURUSD")
        
        # Check if order book analysis uses real data
        if hasattr(engine, '_fetch_order_book'):
            print("✅ Order book data integration exists")
            return True
        else:
            print("❌ Real order book data not implemented")
            print("⚠️  SYSTEM STATUS: Using synthetic order book data")
            return False
            
    except Exception as e:
        print(f"❌ Order book test failed: {e}")
        print("⚠️  SYSTEM STATUS: No real order book analysis")
        return False

def test_live_trading_capability():
    """Test: Can we execute real trades?"""
    print("\n🔍 REALITY CHECK: Live Trading Capability")
    
    try:
        from src.simulation import MarketSimulator
        
        simulator = MarketSimulator()
        
        # Check if live trading methods exist
        live_methods = [
            'place_real_order',
            'get_real_positions',
            'get_real_account_balance'
        ]
        
        implemented_methods = 0
        for method in live_methods:
            if hasattr(simulator, method):
                implemented_methods += 1
                print(f"✅ {method} exists")
            else:
                print(f"❌ {method} not implemented")
        
        if implemented_methods == len(live_methods):
            print("✅ Live trading fully implemented")
            return True
        else:
            print(f"⚠️  SYSTEM STATUS: {implemented_methods}/{len(live_methods)} live trading methods implemented")
            return False
            
    except Exception as e:
        print(f"❌ Live trading test failed: {e}")
        print("⚠️  SYSTEM STATUS: No live trading capability")
        return False

def test_performance_validation():
    """Test: Can we validate performance on real data?"""
    print("\n🔍 REALITY CHECK: Performance Validation")
    
    try:
        # Test if we can run out-of-sample validation
        from src.evolution import EvolutionEngine, FitnessEvaluator
        
        # This should fail if no real validation methods
        if hasattr(FitnessEvaluator, 'validate_out_of_sample'):
            print("✅ Out-of-sample validation exists")
            return True
        else:
            print("❌ Out-of-sample validation not implemented")
            print("⚠️  SYSTEM STATUS: No real performance validation")
            return False
            
    except Exception as e:
        print(f"❌ Performance validation test failed: {e}")
        print("⚠️  SYSTEM STATUS: No real performance validation")
        return False

def generate_honest_report():
    """Generate an honest assessment report"""
    print("\n" + "="*60)
    print("EMP SYSTEM - HONEST REALITY CHECK")
    print("="*60)
    
    tests = [
        ("Real Data Availability", test_real_data_availability),
        ("Economic Data Integration", test_economic_data_integration),
        ("Broker Integration", test_broker_integration),
        ("Real Strategy Discovery", test_real_strategy_discovery),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Order Book Data", test_order_book_data),
        ("Live Trading Capability", test_live_trading_capability),
        ("Performance Validation", test_performance_validation)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("HONEST ASSESSMENT SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Real functionality implemented: {passed}/{total}")
    
    if passed == 0:
        print("\n🚨 BRUTAL HONESTY:")
        print("   The system is a sophisticated mock framework.")
        print("   It can run complex simulations but has ZERO real trading capability.")
        print("   All 'success' tests are celebrating the ability to run on fake data.")
        
    elif passed < total:
        print(f"\n⚠️  PARTIAL REALITY:")
        print(f"   {passed}/{total} real integrations implemented.")
        print("   System has some real functionality but is not production-ready.")
        
    else:
        print("\n✅ FULL REALITY:")
        print("   All real integrations implemented.")
        print("   System is production-ready for live trading.")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if passed == 0:
        print("1. STOP celebrating mock successes")
        print("2. Implement real data integration FIRST")
        print("3. Add broker API integration")
        print("4. Validate on real market data")
        print("5. Test with paper trading before live")
    else:
        print("1. Complete remaining real integrations")
        print("2. Add comprehensive real data testing")
        print("3. Implement live trading validation")
        print("4. Add performance monitoring")
    
    return results

if __name__ == "__main__":
    results = generate_honest_report()
    
    # Exit with appropriate code
    if sum(results.values()) == 0:
        print("\n❌ SYSTEM STATUS: Mock framework only")
        sys.exit(1)
    elif sum(results.values()) < len(results):
        print("\n⚠️  SYSTEM STATUS: Partial real functionality")
        sys.exit(2)
    else:
        print("\n✅ SYSTEM STATUS: Production ready")
        sys.exit(0) 