#!/usr/bin/env python3
"""
Test Real cTrader Integration

This script tests the real cTrader OpenAPI integration to ensure
it can connect, authenticate, and perform basic operations.

Usage:
    python test_real_ctrader_integration.py --config configs/ctrader_config.yaml

Requirements:
    - Valid cTrader OAuth credentials
    - IC Markets demo or live account
    - Internet connection
"""

import asyncio
import argparse
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from trading.real_ctrader_interface import (
        RealCTraderInterface, create_demo_config, create_live_config
    )
    REAL_CTRADER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Real cTrader interface not available: {e}")
    print("   This is expected if websockets or aiohttp are not installed.")
    REAL_CTRADER_AVAILABLE = False

from trading.mock_ctrader_interface import CTraderInterface as MockCTraderInterface


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/ctrader_test.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load cTrader configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variables
        import os
        for section in ['demo', 'live']:
            if 'oauth' in config and section in config['oauth']:
                oauth_config = config['oauth'][section]
                oauth_config['client_id'] = os.getenv(
                    oauth_config['client_id'].strip('${}'), 
                    oauth_config['client_id']
                )
                oauth_config['client_secret'] = os.getenv(
                    oauth_config['client_secret'].strip('${}'), 
                    oauth_config['client_secret']
                )
        
        return config
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return {}


async def test_real_ctrader_connection(config: dict) -> bool:
    """Test real cTrader connection and basic operations."""
    print("\n🔗 Testing Real cTrader Connection...")
    
    try:
        # Create demo config
        demo_config = config['oauth']['demo']
        trading_config = create_demo_config(
            client_id=demo_config['client_id'],
            client_secret=demo_config['client_secret']
        )
        
        # Check if we have valid credentials
        if trading_config.client_id.startswith('${') or trading_config.client_secret.startswith('${'):
            print("⚠️  No real OAuth credentials found in environment variables")
            print("   Set CTRADER_DEMO_CLIENT_ID and CTRADER_DEMO_CLIENT_SECRET")
            return False
        
        # Create interface
        ctrader = RealCTraderInterface(trading_config)
        
        # Test connection
        print("   Connecting to cTrader API...")
        connected = await ctrader.connect()
        
        if not connected:
            print("❌ Failed to connect to cTrader API")
            return False
        
        print("✅ Successfully connected to cTrader API")
        
        # Test account info
        account_info = await ctrader.get_account_info()
        if account_info:
            print(f"✅ Account: {account_info.get('accountName', 'Unknown')}")
            print(f"   Balance: {account_info.get('balance', 'Unknown')}")
            print(f"   Currency: {account_info.get('currency', 'Unknown')}")
        
        # Test symbol loading
        symbols = list(ctrader.symbol_map.keys())
        print(f"✅ Loaded {len(symbols)} symbols")
        
        # Test market data subscription
        test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        available_symbols = [s for s in test_symbols if s in symbols]
        
        if available_symbols:
            print(f"   Subscribing to market data for {available_symbols}...")
            subscribed = await ctrader.subscribe_market_data(available_symbols)
            
            if subscribed:
                print("✅ Market data subscription successful")
                
                # Wait for some market data
                print("   Waiting for market data...")
                await asyncio.sleep(5)
                
                # Check received data
                for symbol in available_symbols:
                    market_data = ctrader.get_market_data(symbol)
                    if market_data:
                        print(f"✅ {symbol}: Bid={market_data.bid:.5f}, Ask={market_data.ask:.5f}")
                    else:
                        print(f"⚠️  No market data received for {symbol}")
            else:
                print("❌ Failed to subscribe to market data")
        else:
            print("⚠️  No test symbols available")
        
        # Test order placement (with minimal volume)
        if available_symbols:
            symbol = available_symbols[0]
            print(f"   Testing order placement for {symbol}...")
            
            # Get current market data
            market_data = ctrader.get_market_data(symbol)
            if market_data:
                # Place a limit order slightly away from market
                from trading.real_ctrader_interface import OrderType, OrderSide
                
                # Use a very small volume and price away from market
                test_volume = 0.01
                test_price = market_data.bid - 0.001  # 1 pip below bid
                
                order_id = await ctrader.place_order(
                    symbol_name=symbol,
                    order_type=OrderType.LIMIT,
                    side=OrderSide.BUY,
                    volume=test_volume,
                    price=test_price
                )
                
                if order_id:
                    print(f"✅ Test order placed: {order_id}")
                    
                    # Cancel the test order
                    cancelled = await ctrader.cancel_order(order_id)
                    if cancelled:
                        print("✅ Test order cancelled successfully")
                    else:
                        print("⚠️  Failed to cancel test order")
                else:
                    print("⚠️  Failed to place test order (this may be normal)")
            else:
                print("⚠️  No market data available for order test")
        
        # Disconnect
        await ctrader.disconnect()
        print("✅ Disconnected from cTrader API")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing real cTrader: {e}")
        return False


async def test_mock_ctrader_fallback():
    """Test mock cTrader interface as fallback."""
    print("\n🎭 Testing Mock cTrader Fallback...")
    
    try:
        # Create mock interface
        from trading.mock_ctrader_interface import TradingConfig, TradingMode
        
        config = TradingConfig(
            client_id="mock_client_id",
            client_secret="mock_client_secret",
            access_token="mock_access_token",
            refresh_token="mock_refresh_token",
            account_id=12345
        )
        
        ctrader = MockCTraderInterface(config)
        
        # Test connection
        connected = await ctrader.connect()
        if connected:
            print("✅ Mock cTrader connected successfully")
            
            # Test basic operations
            positions = ctrader.get_positions()
            orders = ctrader.get_orders()
            market_data = ctrader.get_market_data("EURUSD")
            
            print(f"✅ Mock positions: {len(positions)}")
            print(f"✅ Mock orders: {len(orders)}")
            print(f"✅ Mock market data: {market_data is not None}")
            
            await ctrader.disconnect()
            print("✅ Mock cTrader disconnected")
            
            return True
        else:
            print("❌ Mock cTrader connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing mock cTrader: {e}")
        return False


async def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test cTrader integration")
    parser.add_argument("--config", default="configs/ctrader_config.yaml", 
                       help="Path to cTrader config file")
    parser.add_argument("--test-mock", action="store_true", 
                       help="Test mock interface only")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    print("🚀 cTrader Integration Test")
    print("=" * 50)
    
    # Check if real interface is available
    if not REAL_CTRADER_AVAILABLE:
        print("⚠️  Real cTrader interface not available")
        print("   Installing required dependencies:")
        print("   pip install websockets aiohttp")
        print("\n   Falling back to mock interface...")
        
        success = await test_mock_ctrader_fallback()
        if success:
            print("\n✅ Mock cTrader test passed")
        else:
            print("\n❌ Mock cTrader test failed")
        return
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("❌ Failed to load configuration")
        return
    
    # Test real cTrader if not mock-only
    if not args.test_mock:
        real_success = await test_real_ctrader_connection(config)
        
        if real_success:
            print("\n✅ Real cTrader integration test passed")
        else:
            print("\n❌ Real cTrader integration test failed")
            print("   Falling back to mock interface...")
    
    # Test mock interface
    mock_success = await test_mock_ctrader_fallback()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    if REAL_CTRADER_AVAILABLE:
        print(f"Real cTrader Interface: {'✅ Available' if REAL_CTRADER_AVAILABLE else '❌ Not Available'}")
    
    if not args.test_mock and REAL_CTRADER_AVAILABLE:
        print(f"Real cTrader Test: {'✅ PASSED' if real_success else '❌ FAILED'}")
    
    print(f"Mock cTrader Test: {'✅ PASSED' if mock_success else '❌ FAILED'}")
    
    if REAL_CTRADER_AVAILABLE and not args.test_mock:
        if real_success:
            print("\n🎉 SUCCESS: Real cTrader integration is working!")
            print("   The system can now perform real trading operations.")
        else:
            print("\n⚠️  WARNING: Real cTrader integration failed")
            print("   The system will use mock interface for testing.")
    else:
        print("\n📝 NOTE: Using mock interface for testing")
        print("   To enable real trading, configure OAuth credentials.")


if __name__ == "__main__":
    asyncio.run(main()) 