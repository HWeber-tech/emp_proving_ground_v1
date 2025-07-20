#!/usr/bin/env python3
"""
Sprint 2 Complete Test - Live Trading Cycle
Demonstrates the complete live trading cycle with cTrader API
"""

import asyncio
import logging
import os
from datetime import datetime
from decimal import Decimal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import EMP components
from src.core.events import EventBus
from src.sensory.organs.ctrader_data_organ import CTraderDataOrgan
from src.trading.integration.ctrader_broker_interface import CTraderBrokerInterface
from src.core.events import TradeIntent, ExecutionReport, MarketUnderstanding


class Sprint2IntegrationTest:
    """Complete integration test for Sprint 2: Live Trading Cycle"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.data_organ = CTraderDataOrgan(self.event_bus)
        self.broker_interface = CTraderBrokerInterface(self.event_bus)
        self.broker_interface.set_data_organ(self.data_organ)
        
        # Track events
        self.market_data_received = []
        self.executions_received = []
        
    async def on_market_data(self, event: MarketUnderstanding):
        """Handle market data events"""
        self.market_data_received.append(event)
        logger.info(f"📊 Market: {event.symbol} - Bid: {event.bid}, Ask: {event.ask}")
        
    async def on_execution_report(self, event: ExecutionReport):
        """Handle execution reports"""
        self.executions_received.append(event)
        logger.info(f"💰 Trade: {event.symbol} {event.action} {event.quantity} @ {event.price}")
        
    async def run_live_test(self):
        """Run complete live trading test"""
        try:
            logger.info("🚀 Starting Sprint 2 Complete Test")
            
            # Subscribe to events
            self.event_bus.subscribe(MarketUnderstanding, self.on_market_data)
            self.event_bus.subscribe(ExecutionReport, self.on_execution_report)
            
            # Start data organ
            await self.data_organ.start()
            
            # Wait for connection and symbols
            await asyncio.sleep(5)
            
            # Test trade execution
            if self.data_organ.connected:
                logger.info("✅ Connected to cTrader - Testing trade execution")
                
                # Create test trade intent
                test_trade = TradeIntent(
                    event_id="test-trade-001",
                    timestamp=datetime.now(),
                    source="test",
                    symbol="EURUSD",
                    action="BUY",
                    quantity=Decimal('0.01'),  # 0.01 lots
                    price=None,  # Market order
                    order_type="MARKET"
                )
                
                # Send trade
                await self.broker_interface.place_order(test_trade)
                logger.info("📤 Trade intent sent to cTrader")
                
                # Wait for execution
                await asyncio.sleep(10)
                
                # Check results
                logger.info(f"📊 Test Results:")
                logger.info(f"   Market data received: {len(self.market_data_received)} events")
                logger.info(f"   Executions received: {len(self.executions_received)} events")
                
                if self.executions_received:
                    logger.info("✅ Trade execution confirmed!")
                    for exec in self.executions_received:
                        logger.info(f"   - {exec.symbol} {exec.action} {exec.quantity} @ {exec.price}")
                else:
                    logger.info("⏳ No executions yet (expected in demo mode)")
                    
            else:
                logger.warning("❌ Not connected to cTrader - running in mock mode")
                
            # Keep running for observation
            logger.info("🔄 Keeping connection alive for 30 seconds...")
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
        finally:
            await self.data_organ.stop()
            logger.info("✅ Sprint 2 Complete Test finished")


async def main():
    """Main test runner"""
    test = Sprint2IntegrationTest()
    await test.run_live_test()


if __name__ == "__main__":
    asyncio.run(main())
