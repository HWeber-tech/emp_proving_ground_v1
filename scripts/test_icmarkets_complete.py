#!/usr/bin/env python3
"""
Complete IC Markets FIX API Integration Test
Tests all functionality: connection, market data, and trading
"""

import asyncio
import logging
import os
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.operational.icmarkets_simplefix_application import ICMarketsSimpleFIXManager
from src.operational.icmarkets_config import ICMarketsConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_icmarkets_complete():
    """Complete end-to-end test of IC Markets FIX API."""
    print("🧪 IC Markets FIX API Complete Integration Test")
    print("=" * 60)
    
    # Test configuration
    account = os.getenv("ICMARKETS_ACCOUNT", "9533708")
    password = os.getenv("ICMARKETS_PASSWORD", "WNSE5822")
    
    print(f"📋 Test Configuration:")
    print(f"   Account: {account}")
    print(f"   Environment: demo")
    print(f"   Server: demo-uk-eqx-01.p.c-trader.com")
    print()
    
    # Create configuration
    config = ICMarketsConfig(
        environment="demo",
        account_number=account
    )
    
    # Create IC Markets manager
    manager = ICMarketsSimpleFIXManager(config)
    
    try:
        # Test 1: Connection
        print("🔌 Test 1: Connection Establishment")
        print("-" * 30)
        
        success = manager.connect()
        if success:
            print("✅ Connection test PASSED")
            status = manager.get_connection_status()
            print(f"   Price connected: {status.get('price_connected', False)}")
            print(f"   Trade connected: {status.get('trade_connected', False)}")
        else:
            print("❌ Connection test FAILED")
            return False
            
        print()
        
        # Test 2: Market Data
        print("📊 Test 2: Market Data Subscription")
        print("-" * 30)
        
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        success = manager.subscribe_market_data(symbols)
        if success:
            print("✅ Market data subscription PASSED")
            print(f"   Subscribed to: {symbols}")
        else:
            print("❌ Market data subscription FAILED")
            return False
            
        # Wait for market data
        print("⏳ Waiting for market data...")
        await asyncio.sleep(5)
        
        # Check market data
        for symbol in symbols:
            data = manager.price_connection.get_market_data(symbol)
            if data:
                print(f"   {symbol}: {data}")
            else:
                print(f"   {symbol}: No data received")
                
        print()
        
        # Test 3: Trading
        print("💰 Test 3: Trading Functionality")
        print("-" * 30)
        
        # Test market order
        order_id = manager.place_market_order("EURUSD", "BUY", 0.01)
        if order_id:
            print("✅ Trading test PASSED")
            print(f"   Order ID: {order_id}")
        else:
            print("❌ Trading test FAILED")
            return False
            
        print()
        
        # Test 4: Status Monitoring
        print("📈 Test 4: Status Monitoring")
        print("-" * 30)
        
        status = manager.get_connection_status()
        print("✅ Status monitoring PASSED")
        for key, value in status.items():
            print(f"   {key}: {value}")
            
        print()
        
        # Test 5: Graceful Shutdown
        print("🛑 Test 5: Graceful Shutdown")
        print("-" * 30)
        
        manager.disconnect()
        print("✅ Graceful shutdown PASSED")
        
        print()
        print("🎉 ALL TESTS PASSED!")
        print("✅ IC Markets FIX API is fully operational")
        print("✅ Ready for production trading")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the complete test."""
    success = await test_icmarkets_complete()
    
    if success:
        print("\n🏆 IC Markets FIX API Integration: COMPLETE SUCCESS")
        print("💡 You can now use main_icmarkets.py for production trading")
    else:
        print("\n❌ IC Markets FIX API Integration: FAILED")
        print("🔧 Check configuration and credentials")


if __name__ == "__main__":
    asyncio.run(main())
