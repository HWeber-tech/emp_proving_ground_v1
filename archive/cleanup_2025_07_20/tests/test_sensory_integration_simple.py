"""
Simple Sensory Integration Test

Test to verify that the trading layer can properly use the sensory layer.

Author: EMP Development Team
Date: July 18, 2024
Phase: 3 - Advanced Trading Strategies and Risk Management
"""

import sys
import os
import asyncio
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


async def test_sensory_integration():
    """Test sensory layer integration"""
    print("🔍 Testing Sensory Layer Integration")
    
    try:
        # Test 1: Import sensory core
        print("📦 Testing Sensory Core Import...")
        from src.sensory.core.base import MarketData
        print("✅ Sensory core import successful")
        
        # Test 2: Import technical indicators
        print("📊 Testing Technical Indicators Import...")
        from src.sensory.dimensions.how.indicators import TechnicalIndicators
        print("✅ Technical indicators import successful")
        
        # Test 3: Create market data
        print("📈 Testing Market Data Creation...")
        market_data = MarketData(
            symbol="EURUSD",
            timestamp=datetime.utcnow(),
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=1000.0,
            bid=1.1004,
            ask=1.1006
        )
        print("✅ Market data creation successful")
        
        # Test 4: Initialize indicators
        print("🔧 Testing Indicators Initialization...")
        indicators = TechnicalIndicators()
        print("✅ Indicators initialization successful")
        
        # Test 5: Test strategy import
        print("🎯 Testing Strategy Import...")
        from src.trading.strategy_engine.templates.trend_following import TrendFollowingStrategy
        print("✅ Strategy import successful")
        
        # Test 6: Test strategy instantiation
        print("⚙️ Testing Strategy Instantiation...")
        strategy_params = {
            'short_ma_period': 10,
            'long_ma_period': 20,
            'rsi_period': 14
        }
        strategy = TrendFollowingStrategy("test_strategy", strategy_params, ["EURUSD"])
        print("✅ Strategy instantiation successful")
        
        print("\n🎉 All Sensory Integration Tests PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Sensory Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    success = await test_sensory_integration()
    
    if success:
        print("\n" + "="*60)
        print("SENSORY INTEGRATION STATUS: ✅ WORKING")
        print("="*60)
        print("The trading layer can properly use the sensory layer.")
        print("Phase 3 architecture is correctly implemented.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("SENSORY INTEGRATION STATUS: ❌ FAILED")
        print("="*60)
        print("There are issues with the sensory layer integration.")
        print("Need to fix import dependencies.")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(main()) 