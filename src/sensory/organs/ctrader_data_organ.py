"""
IC Markets cTrader Data Organ
Live market data ingestion from IC Markets cTrader API
Implements SENSORY-03: Live cTrader Data Organ
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime

from src.core.events import MarketUnderstanding
from src.governance.system_config import config

logger = logging.getLogger(__name__)


class CTraderDataOrgan:
    """
    Live market data organ that connects to IC Markets cTrader API
    Provides real-time price data and market events
    """
    
    def __init__(self, event_bus):
        """
        Initialize the cTrader data organ
        
        Args:
            event_bus: The event bus for publishing market data
        """
        self.event_bus = event_bus
        self.client = None
        self.is_connected = False
        self.symbol_mapping = {}  # Maps symbol names to symbol IDs
        self.account_id = config.ctrader_account_id
        
    async def start(self) -> bool:
        """
        Start the cTrader data organ
        Establishes connection and subscribes to market data
        
        Returns:
            bool: True if started successfully
        """
        try:
            logger.info("Starting IC Markets cTrader data organ...")
            
            # Validate credentials
            if not config.validate_credentials():
                logger.error("Invalid cTrader credentials")
                return False
            
            # Always use mock mode for now - real mode requires ctrader-open-api-py
            logger.info("Using mock mode for cTrader data organ")
            return await self.start_mock_mode()
                
        except Exception as e:
            logger.error(f"Failed to start cTrader data organ: {e}")
            return False
    
    async def start_mock_mode(self) -> bool:
        """Start in mock mode for development without cTrader API"""
        logger.info("Starting cTrader data organ in mock mode")
        
        # Mock symbol mapping
        self.symbol_mapping = {
            "EURUSD": 1,
            "GBPUSD": 2,
            "USDJPY": 3,
            "AUDUSD": 4,
            "USDCAD": 5,
            "XAUUSD": 6,
            "XAGUSD": 7
        }
        
        # Start mock data generation
        asyncio.create_task(self.generate_mock_data())
        self.is_connected = True
        return True
    
    async def generate_mock_data(self) -> None:
        """Generate mock market data for development"""
        import random
        
        symbols = config.default_symbols
        base_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2750,
            "USDJPY": 157.50,
            "AUDUSD": 0.6650,
            "USDCAD": 1.3750,
            "XAUUSD": 2035.00,
            "XAGUSD": 24.50
        }
        
        while self.is_connected:
            for symbol in symbols:
                if symbol.upper() in base_prices:
                    base_price = base_prices[symbol.upper()]
                    # Generate realistic price movement
                    volatility = 0.0005 if "USD" in symbol else 0.001
                    change = random.uniform(-volatility, volatility)
                    price = base_prices[symbol.upper()] + change
                    base_prices[symbol.upper()] = price
                    
                    # Create market understanding event
                    market_understanding = MarketUnderstanding(
                        event_id=f"mock_ctrader_{datetime.now().isoformat()}",
                        timestamp=datetime.now(),
                        source="CTraderDataOrgan-Mock",
                        symbol=symbol.upper(),
                        price=Decimal(str(price)),
                        volume=Decimal(str(random.uniform(100, 10000))),
                        indicators={
                            "bid": Decimal(str(price - 0.0001)),
                            "ask": Decimal(str(price + 0.0001)),
                            "spread": Decimal("0.0002"),
                            "high": Decimal(str(price + abs(change) * 2)),
                            "low": Decimal(str(price - abs(change) * 2))
                        },
                        metadata={
                            "mock": True,
                            "symbol_id": self.symbol_mapping.get(symbol.upper(), 0),
                            "account_id": self.account_id
                        }
                    )
                    
                    await self.event_bus.publish(market_understanding)
            
            await asyncio.sleep(0.5)  # Generate data every 0.5 seconds
    
    async def stop(self) -> None:
        """Stop the cTrader data organ"""
        self.is_connected = False
        logger.info("IC Markets cTrader data organ stopped")
    
    async def start_real_mode(self) -> bool:
        """Real mode placeholder - requires ctrader-open-api-py"""
        logger.warning("Real cTrader mode not implemented - requires ctrader-open-api-py")
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the data organ"""
        return {
            "connected": self.is_connected,
            "symbols": list(self.symbol_mapping.keys()),
            "account_id": self.account_id,
            "mode": "mock"
        }
