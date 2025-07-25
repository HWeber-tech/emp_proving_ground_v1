#!/usr/bin/env python3
"""
Test script for IC Markets FIX implementation
Tests connection, market data, and trading functionality
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.fix.icmarkets_config import ICMarketsConfig
from src.operational.icmarkets_fix_application import ICMarketsFIXManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ICMarketsTestSuite:
    """Test suite for IC Markets FIX implementation."""
    
    def __init__(self):
        self.config = ICMarketsConfig(
            environment="demo",
            account_number=os.getenv("ICMARKETS_ACCOUNT", "12345")
        )
        self.config.password = os.getenv("ICMARKETS_PASSWORD", "password")
        self.fix_manager = ICMarketsFIXManager(self.config)
        
    async def run_tests(self):
        """Run all tests."""
        logger.info("🧪 Starting IC Markets FIX Test Suite")
        
        try:
            # Test 1: Configuration validation
            await self.test_configuration()
            
            # Test 2: Connection establishment
            await self.test_connection()
            
            # Test 3: Market data subscription
            await self.test_market_data()
            
            # Test 4: Order placement (if connected)
            await self.test_order_placement()
            
            logger.info("✅ All tests completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
            
    async def test_configuration(self):
        """Test configuration validation."""
        logger.info("🔍 Testing configuration...")
        
        try:
            self.config.validate_config()
            logger.info("✅ Configuration validated")
            
            # Print configuration details
            price_config = self.config.get_price_session_config()
            trade_config = self.config.get_trade_session_config()
            
            logger.info(f"📊 Price Session: {price_config['SocketConnectHost']}:{price_config['SocketConnectPort']}")
            logger.info(f"📊 Trade Session: {trade_config['SocketConnectHost']}:{trade_config['SocketConnectPort']}")
            logger.info(f"🔑 SenderCompID: {price_config['SenderCompID']}")
            
        except Exception as e:
            logger.error(f"❌ Configuration test failed: {e}")
            raise
            
    async def test_connection(self):
        """Test FIX connection establishment."""
        logger.info("🔗 Testing connection...")
        
        try:
            # Start sessions
            self.fix_manager.start_sessions()
            
            # Wait for connection
            connected = self.fix_manager.wait_for_connection(timeout=30)
            
            if connected:
                logger.info("✅ Connection established")
                status = self.fix_manager.get_connection_status()
                logger.info(f"📊 Connection status: {status}")
            else:
                logger.error("❌ Connection timeout")
                raise RuntimeError("Failed to establish connection")
                
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            raise
            
    async def test_market_data(self):
        """Test market data subscription."""
        logger.info("📈 Testing market data...")
        
        try:
            if not self.fix_manager.price_app or not self.fix_manager.price_app.is_connected():
                logger.warning("⚠️  Price session not connected, skipping market data test")
                return
                
            # Subscribe to market data
            symbols = ["EURUSD", "GBPUSD", "USDJPY"]
            self.fix_manager.price_app.subscribe_market_data(symbols)
            
            # Wait for data
            await asyncio.sleep(5)
            
            # Check if we received data
            for symbol in symbols:
                data = self.fix_manager.price_app.get_market_data(symbol)
                if data:
                    logger.info(f"✅ Received data for {symbol}: Bid={data.bid}, Ask={data.ask}")
                else:
                    logger.warning(f"⚠️  No data received for {symbol}")
                    
        except Exception as e:
            logger.error(f"❌ Market data test failed: {e}")
            raise
            
    async def test_order_placement(self):
        """Test order placement."""
        logger.info("💰 Testing order placement...")
        
        try:
            if not self.fix_manager.trade_app or not self.fix_manager.trade_app.is_connected():
                logger.warning("⚠️  Trade session not connected, skipping order test")
                return
                
            # Place a test order
            symbol = "EURUSD"
            cl_ord_id = self.fix_manager.trade_app.place_market_order(
                symbol=symbol,
                side="1",  # Buy
                quantity=1000
            )
            
            logger.info(f"✅ Order placed: {cl_ord_id}")
            
            # Wait for execution report
            await asyncio.sleep(3)
            
            # Check order status
            order_status = self.fix_manager.trade_app.get_order_status(cl_ord_id)
            if order_status:
                logger.info(f"✅ Order status: {order_status.status}")
            else:
                logger.warning("⚠️  No order status received")
                
        except Exception as e:
            logger.error(f"❌ Order placement test failed: {e}")
            raise
            
    def cleanup(self):
        """Clean up resources."""
        logger.info("🧹 Cleaning up...")
        self.fix_manager.stop_sessions()


async def main():
    """Main test runner."""
    logger.info("🚀 IC Markets FIX Test Suite")
    logger.info("=" * 50)
    
    # Check environment variables
    account = os.getenv("ICMARKETS_ACCOUNT")
    password = os.getenv("ICMARKETS_PASSWORD")
    
    if not account or not password:
        logger.error("❌ Missing environment variables:")
        logger.error("   Set ICMARKETS_ACCOUNT and ICMARKETS_PASSWORD")
        logger.error("   Example: export ICMARKETS_ACCOUNT=12345")
        logger.error("   Example: export ICMARKETS_PASSWORD=your_password")
        return
        
    # Run tests
    test_suite = ICMarketsTestSuite()
    
    try:
        await test_suite.run_tests()
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
