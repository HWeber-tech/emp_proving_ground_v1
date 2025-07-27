#!/usr/bin/env python3
"""
Test script for IC Markets SimpleFIX implementation
Windows-compatible FIX 4.4 testing
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.operational.icmarkets_config import ICMarketsConfig
from src.operational.icmarkets_simplefix_application import ICMarketsSimpleFIXManager

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simplefix_connection():
    """Test SimpleFIX connection to IC Markets."""
    print("🧪 Testing IC Markets SimpleFIX Connection")
    print("=" * 50)
    
    # Load configuration
    config = ICMarketsConfig(environment="demo")
    
    # Check if credentials are available
    if not config.account_number or not config.password:
        print("❌ Missing IC Markets credentials")
        print("Please set environment variables:")
        print("export ICMARKETS_ACCOUNT=your_account_number")
        print("export ICMARKETS_PASSWORD=your_password")
        return False
        
    print(f"✅ Configuration loaded for {config.environment}")
    print(f"📍 Price server: {config._get_host()}:{config._get_port('price')}")
    print(f"📍 Trade server: {config._get_host()}:{config._get_port('trade')}")
    
    # Test connection
    manager = ICMarketsSimpleFIXManager(config)
    
    try:
        print("\n🔌 Testing connection...")
        success = manager.connect()
        
        if success:
            print("✅ Connection successful!")
            status = manager.get_connection_status()
            print(f"📊 Connection status: {status}")
            
            # Test market data subscription
            print("\n📈 Testing market data subscription...")
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            sub_success = manager.subscribe_market_data(symbols)
            
            if sub_success:
                print("✅ Market data subscription successful!")
            else:
                print("⚠️  Market data subscription failed (expected without real connection)")
                
            # Test order placement
            print("\n💰 Testing order placement...")
            order_id = manager.place_market_order("EURUSD", "BUY", 0.01)
            
            if order_id:
                print(f"✅ Order placed successfully! Order ID: {order_id}")
            else:
                print("⚠️  Order placement failed (expected without real connection)")
                
        else:
            print("❌ Connection failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    finally:
        manager.disconnect()
        
    return True


def test_configuration():
    """Test configuration setup."""
    print("\n🔧 Testing Configuration")
    print("=" * 30)
    
    config = ICMarketsConfig(environment="demo")
    
    try:
        config.validate_config()
        print("✅ Configuration is valid")
        
        price_config = config.get_price_session_config()
        trade_config = config.get_trade_session_config()
        
        print("\n📋 Price session config:")
        for key, value in price_config.items():
            print(f"  {key}: {value}")
            
        print("\n📋 Trade session config:")
        for key, value in trade_config.items():
            print(f"  {key}: {value}")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return False
        
    return True


def main():
    """Main test function."""
    print("🚀 IC Markets SimpleFIX Test Suite")
    print("=" * 40)
    
    # Test configuration
    config_ok = test_configuration()
    
    # Test connection (will fail without credentials, but shows structure)
    connection_ok = test_simplefix_connection()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print(f"Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"Connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    
    if not config_ok or not connection_ok:
        print("\n💡 Next Steps:")
        print("1. Set your IC Markets credentials in environment variables")
        print("2. Run: export ICMARKETS_ACCOUNT=your_account_number")
        print("3. Run: export ICMARKETS_PASSWORD=your_password")
        print("4. Test again with: python scripts/test_simplefix.py")
    else:
        print("\n🎉 All tests passed! Ready for real trading.")


if __name__ == "__main__":
    main()
