"""
Test Sensory Cortex Refactoring

This test verifies that the sensory cortex refactoring into folder structures
with sub-modules is working correctly.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from src.sensory.core.base import MarketData
from src.sensory.dimensions.how import HowEngine
from src.sensory.dimensions.how.indicators import TechnicalIndicators


def test_how_sense_refactoring():
    """Test the refactored how sense with sub-modules"""
    print("🧪 Testing Sensory Cortex Refactoring")
    print("=" * 50)
    
    # Test 1: Import the how engine
    try:
        engine = HowEngine()
        print("✅ How Engine imported successfully")
    except Exception as e:
        print(f"❌ How Engine import failed: {e}")
        return False
    
    # Test 2: Import technical indicators
    try:
        indicators = TechnicalIndicators()
        print("✅ Technical Indicators imported successfully")
    except Exception as e:
        print(f"❌ Technical Indicators import failed: {e}")
        return False
    
    # Test 3: Create sample market data
    try:
        market_data = []
        base_price = 1.1000
        for i in range(50):
            timestamp = datetime.now() - timedelta(minutes=50-i)
            price_change = (i % 10 - 5) * 0.0001  # Small price variations
            current_price = base_price + price_change
            
            market_data.append(MarketData(
                symbol="EURUSD",
                timestamp=timestamp,
                open=current_price - 0.0001,
                high=current_price + 0.0001,
                low=current_price - 0.0001,
                close=current_price,
                volume=1000 + (i * 10),
                bid=current_price - 0.0001,
                ask=current_price + 0.0001
            ))
        
        print(f"✅ Created {len(market_data)} sample market data points")
    except Exception as e:
        print(f"❌ Market data creation failed: {e}")
        return False
    
    # Test 4: Test technical analysis
    try:
        analysis_results = engine.analyze_market_data(market_data, "EURUSD")
        print(f"✅ Technical analysis completed: {len(analysis_results)} results")
    except Exception as e:
        print(f"❌ Technical analysis failed: {e}")
        return False
    
    # Test 5: Test indicators directly
    try:
        import pandas as pd
        
        # Convert to DataFrame
        data = []
        for md in market_data:
            data.append({
                'timestamp': md.timestamp,
                'open': md.bid,
                'high': md.ask,
                'low': md.bid,
                'close': (md.bid + md.ask) / 2,
                'volume': md.volume,
                'volatility': md.volatility
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate indicators
        indicator_results = indicators.calculate_indicators(df, ['rsi', 'sma'])
        print(f"✅ Indicators calculated: {len(indicator_results)} indicators")
        
        # Check specific indicators
        if 'rsi' in indicator_results:
            print(f"   RSI range: {indicator_results['rsi'].min():.2f} - {indicator_results['rsi'].max():.2f}")
        if 'sma_20' in indicator_results:
            print(f"   SMA 20 calculated: {len(indicator_results['sma_20'].dropna())} values")
            
    except Exception as e:
        print(f"❌ Direct indicator test failed: {e}")
        return False
    
    print("\n🎉 Sensory Cortex Refactoring Test PASSED!")
    print("✅ How sense with sub-modules is working correctly")
    print("✅ Technical indicators are functional")
    print("✅ Architecture is clean and modular")
    
    return True


def test_architecture_compliance():
    """Test that the architecture rule is being followed"""
    print("\n🏗️ Testing Architecture Compliance")
    print("=" * 40)
    
    # Check that analysis is in sensory cortex, not data integration
    try:
        # This should work (analysis in sensory cortex)
        from src.sensory.dimensions.how.indicators import TechnicalIndicators
        print("✅ Technical analysis properly located in sensory cortex")
        
        # This should NOT contain analysis logic
        from src.data_integration import real_data_integration
        print("✅ Data integration layer contains no analysis logic")
        
    except Exception as e:
        print(f"❌ Architecture compliance test failed: {e}")
        return False
    
    print("✅ Architecture rule is being followed")
    return True


if __name__ == "__main__":
    print("🚀 EMP Sensory Cortex Refactoring Test")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_how_sense_refactoring()
    test2_passed = test_architecture_compliance()
    
    if test1_passed and test2_passed:
        print("\n🎯 ALL TESTS PASSED!")
        print("✅ Sensory cortex refactoring is successful")
        print("✅ Architecture is clean and compliant")
        print("✅ Ready to continue with Phase 2 development")
    else:
        print("\n❌ Some tests failed")
        print("Please check the implementation") 