#!/usr/bin/env python3
"""
EMP Ultimate Architecture v1.1 - Main Entry Point

This is the main entry point for the EMP system using the new
layered architecture with proper separation of concerns.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.configuration import load_config
from src.core.event_bus import event_bus
from src.core.models import InstrumentMeta
from src.data_integration.real_data_integration import RealDataManager
from src.governance.fitness_store import FitnessStore
from src.operational.state_store import StateStore

logger = logging.getLogger(__name__)


class EMPSystem:
    """Main EMP system orchestrator."""
    
    def __init__(self):
        self.config = None
        self.data_manager = None
        self.fitness_store = None
        self.state_store = None
        self.running = False
        
    async def initialize(self, config_path: str = None):
        """Initialize the EMP system."""
        try:
            logger.info("Initializing EMP Ultimate Architecture v1.1")
            
            # Load configuration
            self.config = load_config(config_path)
            logger.info(f"Configuration loaded: {self.config.system_name} v{self.config.system_version}")
            
            # Create instrument metadata
            instrument_config = InstrumentMeta(
                symbol="EURUSD",
                pip_size=0.0001,
                lot_size=100000,
                commission=0.0,
                spread=0.0001
            )
            logger.info(f"Instrument metadata created: {instrument_config.symbol}")
            
            # Initialize operational backbone
            self.state_store = StateStore(self.config.operational.get('redis', {}))
            logger.info("State store initialized")
            
            # Initialize governance
            self.fitness_store = FitnessStore()
            logger.info("Fitness store initialized")
            
            # Initialize data manager with config
            self.data_manager = RealDataManager(self.config.data_sources or {})
            logger.info("Data manager initialized")
            
            # Start event bus
            await event_bus.start()
            logger.info("Event bus started")
            
            logger.info("EMP system initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing EMP system: {e}")
            raise
            
    async def run(self):
        """Run the EMP system."""
        try:
            self.running = True
            logger.info("EMP system started")
            
            # Test data manager
            market_data = await self.data_manager.get_market_data("EURUSD=X")
            if market_data:
                logger.info(f"Successfully retrieved market data: {market_data}")
            else:
                logger.warning("Failed to retrieve market data")
            
            # Main system loop
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in main system loop: {e}")
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Shutdown the EMP system."""
        try:
            logger.info("Shutting down EMP system")
            
            self.running = False
            
            # Stop event bus
            await event_bus.stop()
            
            logger.info("EMP system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
    def get_system_status(self) -> dict:
        """Get system status."""
        return {
            'system_name': self.config.system_name if self.config else 'Unknown',
            'system_version': self.config.system_version if self.config else 'Unknown',
            'environment': self.config.environment if self.config else 'Unknown',
            'running': self.running,
            'data_sources': self.data_manager.get_available_sources() if self.data_manager else [],
            'fitness_definitions': self.fitness_store.list_definitions() if self.fitness_store else [],
        }


async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run EMP system
    emp_system = EMPSystem()
    
    try:
        await emp_system.initialize()
        await emp_system.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
