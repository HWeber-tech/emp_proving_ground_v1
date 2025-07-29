#!/usr/bin/env python3
"""
EMP v4.0 Professional Predator - IC Markets FIX Integration
Production-ready trading system with real IC Markets FIX API
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.governance.system_config import SystemConfig
from src.operational.icmarkets_api import FinalFIXTester
from src.operational.icmarkets_config import ICMarketsConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EMPProfessionalPredatorICMarkets:
    """Production-ready IC Markets trading system."""
    
    def __init__(self):
        self.config = None
        self.event_bus = None
        self.icmarkets_manager = None
        self.running = False
        
    async def initialize(self, config_path: str = None):
        """Initialize the IC Markets trading system."""
        try:
            logger.info("🚀 Initializing EMP v4.0 IC Markets Professional Predator")
            
            # Load configuration
            self.config = SystemConfig()
            logger.info(f"✅ Configuration loaded: EMP v4.0 IC Markets Professional Predator")
            logger.info(f"🔧 Protocol: {self.config.connection_protocol}")
            
            # Initialize event bus
            self.event_bus = EventBus()
            logger.info("✅ Event bus initialized")
            
            # Setup IC Markets components
            await self._setup_icmarkets_components()
            
            logger.info("🎉 IC Markets Professional Predator initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Error initializing IC Markets Professional Predator: {e}")
            raise
            
    async def _setup_icmarkets_components(self):
        """Setup IC Markets FIX components."""
        logger.info("🔧 Setting up IC Markets FIX components")
        
        # Create IC Markets configuration
        icmarkets_config = ICMarketsConfig(
            environment="demo",
            account_number=os.getenv("ICMARKETS_ACCOUNT", "9533708")
        )
        
        # Validate configuration
        icmarkets_config.validate_config()
        
        # Create IC Markets manager
        self.icmarkets_manager = ICMarketsSimpleFIXManager(icmarkets_config)
        
        # Connect to IC Markets
        if self.icmarkets_manager.connect():
            logger.info("✅ IC Markets FIX connections established")
            
            # Subscribe to market data
            symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
            if self.icmarkets_manager.subscribe_market_data(symbols):
                logger.info(f"✅ Subscribed to market data: {symbols}")
            else:
                logger.warning("⚠️ Failed to subscribe to market data")
        else:
            logger.error("❌ Failed to connect to IC Markets")
            raise ConnectionError("Cannot establish IC Markets FIX connections")
            
    async def run(self):
        """Run the IC Markets trading system."""
        try:
            self.running = True
            logger.info("🎯 IC Markets Professional Predator system started")
            
            # Display system status
            summary = await self.get_system_summary()
            logger.info("📊 System Summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")
                
            logger.info("🎉 IC Markets Professional Predator running successfully!")
            
            # Keep running for monitoring
            while self.running:
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("⏹️ Received shutdown signal")
        except Exception as e:
            logger.error(f"❌ IC Markets Professional Predator error: {e}")
        finally:
            await self.shutdown()
            
    async def test_trading(self):
        """Test basic trading functionality."""
        logger.info("🧪 Testing IC Markets trading functionality")
        
        if not self.icmarkets_manager:
            logger.error("❌ IC Markets manager not initialized")
            return False
            
        # Test market order
        order_id = self.icmarkets_manager.place_market_order("EURUSD", "BUY", 0.1)
        if order_id:
            logger.info(f"✅ Market order placed successfully: {order_id}")
            return True
        else:
            logger.error("❌ Failed to place market order")
            return False
            
    async def shutdown(self):
        """Shutdown the IC Markets trading system."""
        try:
            logger.info("🛑 Shutting down IC Markets Professional Predator")
            self.running = False
            
            if self.icmarkets_manager:
                self.icmarkets_manager.disconnect()
                logger.info("✅ IC Markets connections stopped")
                
            if self.event_bus:
                logger.info("✅ Event bus shutdown")
                
            logger.info("✅ IC Markets Professional Predator shutdown complete")
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
            
    async def get_system_summary(self) -> dict:
        """Get comprehensive system summary."""
        status = {}
        if self.icmarkets_manager:
            status.update(self.icmarkets_manager.get_connection_status())
            
        return {
            'version': '4.0',
            'protocol': 'IC Markets FIX 4.4',
            'status': 'RUNNING',
            'timestamp': datetime.now().isoformat(),
            'icmarkets_status': status,
            'account': os.getenv("ICMARKETS_ACCOUNT", "9533708"),
            'environment': 'demo'
        }


async def main():
    """Main entry point for IC Markets Professional Predator."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run IC Markets Professional Predator
    system = EMPProfessionalPredatorICMarkets()
    
    try:
        await system.initialize()
        
        # Test trading functionality
        await system.test_trading()
        
        await system.run()
    except Exception as e:
        logger.error(f"❌ IC Markets Professional Predator failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
