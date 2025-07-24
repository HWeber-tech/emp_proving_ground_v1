"""
FIX Sensory Organ
Specialist organ for high-resolution market vision via FIX protocol
"""

import logging
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import simplefix

from src.core.events import MarketUnderstanding, OrderBook, OrderBookLevel
from src.governance.system_config import SystemConfig

log = logging.getLogger(__name__)


class FIXSensoryOrgan:
    """
    Specialist sensory organ for FIX market data ingestion.
    
    This organ transforms raw FIX market data into rich MarketUnderstanding events
    with full order book depth, providing the predator with institutional-grade
    market vision.
    """
    
    def __init__(self, event_bus, config: SystemConfig, fix_app):
        """
        Initialize the FIX sensory organ.
        
        Args:
            event_bus: NATS event bus for publishing MarketUnderstanding events
            config: System configuration with symbol mappings
            fix_app: FIX application instance for sending messages
        """
        self.event_bus = event_bus
        self.config = config
        self.fix_app = fix_app
        self.subscriptions: Dict[str, str] = {}  # symbol -> subscription_id
        self.order_books: Dict[str, OrderBook] = {}  # symbol -> current order book
        
    async def subscribe_to_market_data(self, symbol: str) -> bool:
        """
        Subscribe to market data for a specific symbol.
        
        Args:
            symbol: Human-readable symbol name (e.g., "EURUSD")
            
        Returns:
            True if subscription successful, False otherwise
        """
        try:
            # Map symbol to FIX symbolId
            symbol_id = self.config.fix_symbol_map.get(symbol)
            if not symbol_id:
                log.error(f"Unknown symbol: {symbol}")
                return False
            
            # Generate unique subscription ID
            subscription_id = str(uuid.uuid4())
            
            # Create market data request
            msg = simplefix.FixMessage()
            msg.append_pair(35, "V")  # MarketDataRequest
            msg.append_pair(262, subscription_id)  # MDReqID
            msg.append_pair(263, "1")  # SubscriptionRequestType = SNAPSHOT_PLUS_UPDATES
            msg.append_pair(264, 0)  # MarketDepth = 0 (full depth)
            msg.append_pair(265, 0)  # MDUpdateType = FULL_REFRESH
            
            # Add symbol to repeating group
            msg.append_pair(146, 1)  # NoRelatedSym = 1
            msg.append_pair(55, str(symbol_id))  # Symbol
            
            # Add MDEntryTypes (Bid and Ask)
            msg.append_pair(267, 2)  # NoMDEntryTypes = 2
            msg.append_pair(269, 0)  # MDEntryType = BID
            msg.append_pair(269, 1)  # MDEntryType = OFFER
            
            # Send the request
            if self.fix_app.send_message(msg):
                self.subscriptions[symbol] = subscription_id
                log.info(f"Subscribed to market data for {symbol} (ID: {symbol_id})")
                return True
            else:
                log.error(f"Failed to send market data request for {symbol}")
                return False
                
        except Exception as e:
            log.error(f"Error subscribing to market data for {symbol}: {e}")
            return False
    
    async def subscribe_to_market_data_limited(self, symbol: str, depth: int = 10) -> bool:
        """
        Subscribe to market data with limited depth (fallback).
        
        Args:
            symbol: Human-readable symbol name
            depth: Maximum depth to request
            
        Returns:
            True if subscription successful, False otherwise
        """
        try:
            symbol_id = self.config.fix_symbol_map.get(symbol)
            if not symbol_id:
                log.error(f"Unknown symbol: {symbol}")
                return False
            
            subscription_id = str(uuid.uuid4())
            
            msg = simplefix.FixMessage()
            msg.append_pair(35, "V")
            msg.append_pair(262, subscription_id)
            msg.append_pair(263, "1")
            msg.append_pair(264, depth)  # Limited depth
            msg.append_pair(265, 0)
            msg.append_pair(146, 1)
            msg.append_pair(55, str(symbol_id))
            msg.append_pair(267, 2)
            msg.append_pair(269, 0)
            msg.append_pair(269, 1)
            
            if self.fix_app.send_message(msg):
                self.subscriptions[symbol] = subscription_id
                log.info(f"Subscribed to limited market data for {symbol} (depth: {depth})")
                return True
            else:
                log.error(f"Failed to send limited market data request for {symbol}")
                return False
                
        except Exception as e:
            log.error(f"Error subscribing to limited market data for {symbol}: {e}")
            return False
    
    async def process_market_data_snapshot(self, message: simplefix.FixMessage):
        """
        Process MarketDataSnapshotFullRefresh (W) messages.
        
        Args:
            message: FIX message containing snapshot data
        """
        try:
            # Extract symbol
            symbol_id = int(message.get(55, b"0").decode())
            symbol = self._get_symbol_from_id(symbol_id)
            if not symbol:
                log.error(f"Unknown symbol ID: {symbol_id}")
                return
            
            # Create new order book
            order_book = OrderBook()
            
            # Parse MDEntries
            no_md_entries = int(message.get(268, b"0").decode())
            
            for i in range(no_md_entries):
                # Extract entry data
                entry_type = message.get(269 + i * 10, b"").decode()
                price = float(message.get(270 + i * 10, b"0").decode())
                size = float(message.get(271 + i * 10, b"0").decode())
                
                if entry_type == "0":  # BID
                    order_book.add_bid(price, size)
                elif entry_type == "1":  # OFFER
                    order_book.add_ask(price, size)
            
            # Update current order book
            self.order_books[symbol] = order_book
            
            # Create and publish MarketUnderstanding event
            market_event = MarketUnderstanding(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                best_bid=order_book.best_bid or 0.0,
                best_ask=order_book.best_ask or 0.0,
                order_book=order_book,
                source="FIX",
                sequence_number=int(message.get(34, b"0").decode())
            )
            
            # Publish to event bus
            if hasattr(self.event_bus, 'publish'):
                asyncio.create_task(self.event_bus.publish(market_event))
            else:
                log.warning("Event bus does not have publish method")
                
            log.debug(f"Published MarketUnderstanding for {symbol}: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
            
        except Exception as e:
            log.error(f"Error processing market data snapshot: {e}")
    
    async def process_market_data_incremental(self, message: simplefix.FixMessage):
        """
        Process MarketDataIncrementalRefresh (X) messages.
        
        Args:
            message: FIX message containing incremental updates
        """
        try:
            # Extract symbol
            symbol_id = int(message.get(55, b"0").decode())
            symbol = self._get_symbol_from_id(symbol_id)
            if not symbol:
                log.error(f"Unknown symbol ID: {symbol_id}")
                return
            
            # Get current order book
            order_book = self.order_books.get(symbol, OrderBook())
            
            # Parse MDEntries
            no_md_entries = int(message.get(268, b"0").decode())
            
            for i in range(no_md_entries):
                entry_type = message.get(269 + i * 10, b"").decode()
                action = message.get(279 + i * 10, b"").decode()  # MDUpdateAction
                price = float(message.get(270 + i * 10, b"0").decode())
                size = float(message.get(271 + i * 10, b"0").decode())
                
                if action == "0":  # NEW
                    if entry_type == "0":  # BID
                        order_book.add_bid(price, size)
                    elif entry_type == "1":  # OFFER
                        order_book.add_ask(price, size)
                elif action == "1":  # CHANGE
                    # Update existing level
                    levels = order_book.bids if entry_type == "0" else order_book.asks
                    for level in levels:
                        if abs(level.price - price) < 0.00001:  # Floating point comparison
                            level.size = size
                            break
                elif action == "2":  # DELETE
                    # Remove level
                    levels = order_book.bids if entry_type == "0" else order_book.asks
                    levels[:] = [level for level in levels if abs(level.price - price) > 0.00001]
            
            # Update current order book
            self.order_books[symbol] = order_book
            
            # Create and publish MarketUnderstanding event
            market_event = MarketUnderstanding(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                best_bid=order_book.best_bid or 0.0,
                best_ask=order_book.best_ask or 0.0,
                order_book=order_book,
                source="FIX",
                sequence_number=int(message.get(34, b"0").decode())
            )
            
            # Publish to event bus
            if hasattr(self.event_bus, 'publish'):
                asyncio.create_task(self.event_bus.publish(market_event))
            else:
                log.warning("Event bus does not have publish method")
                
            log.debug(f"Updated MarketUnderstanding for {symbol}: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
            
        except Exception as e:
            log.error(f"Error processing market data incremental: {e}")
    
    async def process_market_data_reject(self, message: simplefix.FixMessage):
        """
        Process MarketDataRequestReject (Y) messages.
        
        Args:
            message: FIX rejection message
        """
        try:
            md_req_id = message.get(262, b"").decode()
            text = message.get(58, b"").decode()
            
            log.warning(f"Market data request rejected: {text} (ID: {md_req_id})")
            
            # Check if this is a depth rejection
            if "depth" in text.lower() or "market depth" in text.lower():
                # Find the symbol for this subscription
                for symbol, sub_id in self.subscriptions.items():
                    if sub_id == md_req_id:
                        log.info(f"Retrying {symbol} with limited depth due to rejection")
                        await self.subscribe_to_market_data_limited(symbol, depth=10)
                        break
            
        except Exception as e:
            log.error(f"Error processing market data reject: {e}")
    
    def _get_symbol_from_id(self, symbol_id: int) -> Optional[str]:
        """Map FIX symbolId back to human-readable symbol."""
        for symbol, sid in self.config.fix_symbol_map.items():
            if sid == symbol_id:
                return symbol
        return None
    
    def get_current_order_book(self, symbol: str) -> Optional[OrderBook]:
        """
        Get the current order book for a symbol.
        
        Args:
            symbol: Human-readable symbol name
            
        Returns:
            Current OrderBook or None if not available
        """
        return self.order_books.get(symbol)
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols."""
        return list(self.subscriptions.keys())
    
    async def unsubscribe_all(self):
        """Unsubscribe from all market data."""
        for symbol in list(self.subscriptions.keys()):
            await self.unsubscribe_from_market_data(symbol)
    
    async def unsubscribe_from_market_data(self, symbol: str) -> bool:
        """
        Unsubscribe from market data for a symbol.
        
        Args:
            symbol: Human-readable symbol name
            
        Returns:
            True if unsubscribe successful, False otherwise
        """
        try:
            if symbol not in self.subscriptions:
                log.warning(f"Not subscribed to {symbol}")
                return False
            
            subscription_id = self.subscriptions[symbol]
            symbol_id = self.config.fix_symbol_map.get(symbol)
            
            if not symbol_id:
                log.error(f"Unknown symbol: {symbol}")
                return False
            
            # Create unsubscribe request
            msg = simplefix.FixMessage()
            msg.append_pair(35, "V")
            msg.append_pair(262, subscription_id)
            msg.append_pair(263, "2")  # SubscriptionRequestType = UNSUBSCRIBE
            msg.append_pair(146, 1)
            msg.append_pair(55, str(symbol_id))
            
            if self.fix_app.send_message(msg):
                del self.subscriptions[symbol]
                if symbol in self.order_books:
                    del self.order_books[symbol]
                log.info(f"Unsubscribed from market data for {symbol}")
                return True
            else:
                log.error(f"Failed to send unsubscribe request for {symbol}")
                return False
                
        except Exception as e:
            log.error(f"Error unsubscribing from market data for {symbol}: {e}")
            return False
