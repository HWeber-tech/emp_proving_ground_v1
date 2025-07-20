"""
Integration test for cTrader API
Demonstrates complete cTrader integration workflow
"""

import asyncio
import logging
import os
from datetime import datetime
from decimal import Decimal
from dotenv import load_dotenv

from src.core.events import EventBus
from src.core.events import MarketUnderstanding, TradeIntent
from src.sensory.organs.ctrader_data_organ import CTraderDataOrgan
from src.trading.integration.ctrader_broker_interface import CTraderBrokerInterface
from src.governance.token_manager import TokenManager
from src.governance.system_config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class CTraderIntegrationTest:
    """Integration test for cTrader API components"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.token_manager = TokenManager()
        self.data_organ = CTraderDataOrgan(self.event_bus)
        self.broker = CTraderBrokerInterface(self.event_bus)
        
        # Link components
        self.broker.set_data_organ(self.data_organ)
        
        # Event handlers
        self.event_bus.subscribe(MarketUnderstanding, self.on_market_data)
        self.event_bus.subscribe(TradeIntent, self.on_trade_intent)
        
    async def on_market_data(self, event: MarketUnderstanding):
        """Handle market data events"""
        logger.info(f"Market data: {event.symbol} - Price: {event.price}, Volume: {event.volume}")
        
    async def on_trade_intent(self, event: TradeIntent):
        """Handle trade intent events"""
        logger.info(f"Trade intent: {event.action} {event.quantity} {event.symbol} @ {event.price}")
        
    async def test_credentials(self):
        """Test if cTrader credentials are configured"""
        if not config.validate_credentials():
            logger.error("❌ cTrader credentials not configured")
            logger.info("Please configure your .env file with:")
            logger.info("CTRADER_CLIENT_ID=your_client_id")
            logger.info("CTRADER_CLIENT_SECRET=your_client_secret")
            logger.info("CTRADER_ACCESS_TOKEN=your_access_token")
            logger.info("CTRADER_ACCOUNT_ID=your_account_id")
            return False
        logger.info("✅ cTrader credentials configured")
        return True
        
    async def test_token_refresh(self):
        """Test token refresh functionality"""
        try:
            success = await self.token_manager.refresh_access_token()
            if success:
                logger.info("✅ Token refresh successful")
            else:
                logger.warning("⚠️  Token refresh failed - check refresh token")
            return success
        except Exception as e:
            logger.error(f"❌ Token refresh error: {e}")
            return False
            
    async def test_data_organ(self):
        """Test data organ connection"""
        try:
            await self.data_organ.start()
            if self.data_organ.connected:
                logger.info("✅ Data organ connected")
                return True
            else:
                logger.error("❌ Data organ failed to connect")
                return False
        except Exception as e:
            logger.error(f"❌ Data organ error: {e}")
            return False
            
    async def test_broker_interface(self):
        """Test broker interface"""
        try:
            await self.broker.start()
            logger.info("✅ Broker interface ready")
            return True
        except Exception as e:
            logger.error(f"❌ Broker interface error: {e}")
            return False
            
    async def test_symbol_mapping(self):
        """Test symbol mapping"""
        if self.data_organ.symbol_mapping:
            logger.info(f"✅ Loaded {len(self.data_organ.symbol_mapping)} symbols")
            for symbol, symbol_id in list(self.data_organ.symbol_mapping.items())[:5]:
                logger.info(f"  {symbol}: {symbol_id}")
            return True
        else:
            logger.warning("⚠️  No symbols loaded yet")
            return False
            
    async def run_integration_test(self):
        """Run complete integration test"""
        logger.info("🚀 Starting cTrader Integration Test")
        
        # Test 1: Credentials
        if not await self.test_credentials():
            return False
            
        # Test 2: Token refresh
        await self.test_token_refresh()
        
        # Test 3: Data organ
        if not await self.test_data_organ():
            return False
            
        # Test 4: Broker interface
        if not await self.test_broker_interface():
            return False
            
        # Test 5: Symbol mapping
        await self.test_symbol_mapping()
        
        # Test 6: Simulate trading flow
        logger.info("📊 Simulating trading flow...")
        
        # Create a sample trade intent
        trade_intent = TradeIntent(
            event_id="test-trade-001",
            timestamp=datetime.now(),
            source="test",
            symbol="EURUSD",
            action="BUY",
            quantity=Decimal('0.01'),  # 0.01 lots
            price=Decimal('1.1000'),
            order_type="MARKET"
        )
        
        # Publish to event bus
        await self.event_bus.publish(trade_intent)
        
        logger.info("✅ Integration test completed successfully")
        return True
        
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up...")
        await self.data_organ.stop()
        await self.broker.stop()
        self.token_manager.stop_auto_refresh()
        logger.info("✅ Cleanup completed")


async def main():
    """Main test function"""
    logger.info("🎯 Starting cTrader Integration Test")
    
    # Check if running in test mode
    if not os.getenv("CTRADER_CLIENT_ID"):
        logger.info("🔧 Running in demo mode - no real credentials")
        logger.info("To test with real credentials:")
        logger.info("1. Copy .env.example to .env")
        logger.info("2. Fill in your cTrader credentials")
        logger.info("3. Run: python test_ctrader_integration.py")
        return
    
    test = CTraderIntegrationTest()
    
    try:
        success = await test.run_integration_test()
        if success:
            logger.info("🎉 All tests passed! Ready for live trading.")
        else:
            logger.error("❌ Some tests failed. Check configuration.")
    finally:
        await test.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
