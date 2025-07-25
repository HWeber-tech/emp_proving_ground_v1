#!/usr/bin/env python3
"""
EMP v4.0 Professional Predator - Master Switch Integration

Complete system with configurable protocol selection (FIX vs OpenAPI)
This is the final integration of Sprint 1: The Professional Upgrade
"""

import asyncio
import logging
import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.governance.system_config import SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager
from src.operational.event_bus import EventBus

# Protocol-specific components
from src.sensory.organs.fix_sensory_organ import FIXSensoryOrgan
from src.sensory.organs.ctrader_data_organ import CTraderDataOrgan
from src.trading.integration.ctrader_broker_interface import CTraderBrokerInterface

logger = logging.getLogger(__name__)


class EMPProfessionalPredator:
    """Professional-grade trading system with configurable protocol selection."""
    
    def __init__(self):
        self.config = None
        self.event_bus = None
        self.fix_connection_manager = None
        self.sensory_organ = None
        self.broker_interface = None
        self.running = False
        
    async def initialize(self, config_path: str = None):
        """Initialize the professional predator system."""
        try:
            logger.info("🚀 Initializing EMP v4.0 Professional Predator")
            
            # Load configuration
            self.config = SystemConfig()
            logger.info(f"✅ Configuration loaded: EMP v4.0 Professional Predator")
            logger.info(f"🔧 Protocol: {self.config.connection_protocol}")
            
            # Initialize event bus
            self.event_bus = EventBus()
            logger.info("✅ Event bus initialized")
            
            # Setup protocol-specific components
            await self._setup_live_components()
            
            logger.info("🎉 Professional Predator initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Professional Predator: {e}")
            raise
            
    async def _setup_live_components(self):
        """Dynamically sets up sensory and trading layers based on protocol."""
        logger.info(f"🔧 Setting up LIVE components using '{self.config.connection_protocol}' protocol")
        
        if self.config.connection_protocol == "fix":
            # --- SETUP FOR PROFESSIONAL FIX PROTOCOL ---
            logger.info("🎯 Configuring FIX protocol components")
            
            # 1. Start the FIX Connection Manager
            self.fix_connection_manager = FIXConnectionManager(self.config)
            self.fix_connection_manager.start_sessions()
            
            # 2. Create message queues for thread-safe communication
            price_queue = asyncio.Queue()
            trade_queue = asyncio.Queue()
            
            # 3. Configure FIX applications with queues
            price_app = self.fix_connection_manager.get_application("price")
            trade_app = self.fix_connection_manager.get_application("trade")
            
            if price_app:
                price_app.set_message_queue(price_queue)
            if trade_app:
                trade_app.set_message_queue(trade_queue)
            
            # 4. Instantiate FIX-based components
            self.sensory_organ = FIXSensoryOrgan(self.event_bus, price_queue, self.config)
            self.broker_interface = FIXBrokerInterface(
                self.event_bus, 
                trade_queue, 
                self.fix_connection_manager.get_initiator("trade")
            )
            
            logger.info("✅ FIX components configured successfully")
            
        elif self.config.CONNECTION_PROTOCOL == "openapi":
            # --- SETUP FOR LEGACY OPENAPI PROTOCOL ---
            logger.info("🔄 Configuring OpenAPI components (fallback mode)")
            
            # Instantiate OpenAPI components
            self.sensory_organ = CTraderDataOrgan(self.event_bus, self.config)
            self.broker_interface = CTraderBrokerInterface(self.event_bus, self.config)
            
            logger.info("✅ OpenAPI components configured successfully")
            
        else:
            raise ValueError(f"❌ Unsupported connection protocol: {self.config.CONNECTION_PROTOCOL}")
        
        # 5. Inject chosen components into protocol-agnostic managers
        # Note: These would be integrated with the actual system managers
        logger.info(f"✅ Successfully configured {self.sensory_organ.__class__.__name__} and {self.broker_interface.__class__.__name__}")
        
    async def run(self):
        """Run the professional predator system."""
        try:
            self.running = True
            logger.info("🎯 Professional Predator system started")
            
            # Display system status
            summary = await self.get_system_summary()
            logger.info("📊 System Summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")
                
            logger.info("🎉 Professional Predator running successfully!")
            
            # Keep running for monitoring
            while self.running:
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("⏹️ Received shutdown signal")
        except Exception as e:
            logger.error(f"❌ Professional Predator error: {e}")
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Shutdown the professional predator system."""
        try:
            logger.info("🛑 Shutting down Professional Predator")
            self.running = False
            
            if self.fix_connection_manager:
                self.fix_connection_manager.stop_sessions()
                logger.info("✅ FIX connections stopped")
                
            if self.event_bus:
                logger.info("✅ Event bus shutdown")
                
            logger.info("✅ Professional Predator shutdown complete")
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")
            
    async def get_system_summary(self) -> dict:
        """Get comprehensive system summary."""
        return {
            'version': '4.0',
            'protocol': self.config.connection_protocol,
            'status': 'RUNNING',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'sensory_organ': self.sensory_organ.__class__.__name__ if self.sensory_organ else None,
                'broker_interface': self.broker_interface.__class__.__name__ if self.broker_interface else None,
                'fix_manager': 'FIXConnectionManager' if self.fix_connection_manager else None
            }
        }


async def main():
    """Main entry point for Professional Predator."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run Professional Predator
    system = EMPProfessionalPredator()
    
    try:
        await system.initialize()
        await system.run()
    except Exception as e:
        logger.error(f"❌ Professional Predator failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
