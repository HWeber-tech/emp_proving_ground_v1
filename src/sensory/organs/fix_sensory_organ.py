"""
FIX Sensory Organ for IC Markets
Processes market data from FIX protocol
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import simplefix

logger = logging.getLogger(__name__)


class FIXSensoryOrgan:
    """Processes market data from FIX protocol."""
    
    def __init__(self, event_bus, price_queue, config):
        """
        Initialize FIX sensory organ.
        
        Args:
            event_bus: Event bus for system communication
            price_queue: Queue for price messages
            config: System configuration
        """
        self.event_bus = event_bus
        self.price_queue = price_queue
        self.config = config
        self.running = False
        self.market_data = {}
        self.symbols = []
        
    async def start(self):
        """Start the sensory organ."""
        self.running = True
        logger.info("FIX sensory organ started")
        
        # Start message processing
        asyncio.create_task(self._process_price_messages())
        
    async def stop(self):
        """Stop the sensory organ."""
        self.running = False
        logger.info("FIX sensory organ stopped")
        
    async def _process_price_messages(self):
        """Process price messages from the queue."""
        while self.running:
            try:
                # Get message from queue
                message = await self.price_queue.get()
                
                # Process based on message type
                msg_type = message.get(35)
                
                if msg_type == b"W":  # MarketDataSnapshotFullRefresh
                    await self._handle_market_data_snapshot(message)
                elif msg_type == b"X":  # MarketDataIncrementalRefresh
                    await self._handle_market_data_incremental(message)
                    
            except Exception as e:
                logger.error(f"Error processing price message: {e}")
                
    async def _handle_market_data_snapshot(self, message):
        """Handle market data snapshot messages."""
        try:
            symbol = message.get(55).decode() if message.get(55) else None
            if not symbol:
                return
                
            # Extract market data
            market_data = self._extract_market_data(message)
            
            if market_data:
                # Update local cache
                self.market_data[symbol] = market_data
                
                # Emit event for system
                await self.event_bus.emit("market_data_update", {
                    "symbol": symbol,
                    "data": market_data,
                    "timestamp": datetime.utcnow()
                })
                
                logger.debug(f"Market data updated for {symbol}: {market_data}")
                
        except Exception as e:
            logger.error(f"Error handling market data snapshot: {e}")
            
    async def _handle_market_data_incremental(self, message):
        """Handle incremental market data updates."""
        try:
            symbol = message.get(55).decode() if message.get(55) else None
            if not symbol or symbol not in self.market_data:
                return
                
            # Update existing market data
            updates = self._extract_market_data(message)
            if updates:
                self.market_data[symbol].update(updates)
                
                # Emit event for system
                await self.event_bus.emit("market_data_incremental", {
                    "symbol": symbol,
                    "updates": updates,
                    "timestamp": datetime.utcnow()
                })
                
                logger.debug(f"Market data incremental update for {symbol}: {updates}")
                
        except Exception as e:
            logger.error(f"Error handling market data incremental: {e}")
            
    def _extract_market_data(self, message) -> Dict[str, Any]:
        """Extract market data from FIX message."""
        data = {}
        
        try:
            # Extract bid/ask prices
            if message.get(270):
                # This is a simplified extraction
                # Real implementation would parse MDEntry groups
                bid_price = None
                ask_price = None
                bid_size = None
                ask_size = None
                
                # For now, extract basic fields
                data = {
                    "bid": bid_price,
                    "ask": ask_price,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "timestamp": datetime.utcnow()
                }
                
        except Exception as e:
            logger.error(f"Error extracting market data: {e}")
            
        return data
        
    def subscribe_to_symbols(self, symbols: List[str]):
        """Subscribe to market data for symbols."""
        self.symbols = symbols
        logger.info(f"Subscribed to symbols: {symbols}")
        
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data for a symbol."""
        return self.market_data.get(symbol, {})
        
    def get_all_market_data(self) -> Dict[str, Dict[str, Any]]:
        """Get all market data."""
        return self.market_data.copy()
        
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of subscribed symbols."""
        return self.symbols.copy()
