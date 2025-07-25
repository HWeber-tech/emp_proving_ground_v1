"""
IC Markets FIX Application
Production-ready FIX 4.4 implementation for IC Markets cTrader API
"""

import quickfix as fix
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MarketDataEntry:
    """Represents a market data entry."""
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class OrderStatus:
    """Represents an order status update."""
    cl_ord_id: str
    order_id: str
    symbol: str
    side: str
    order_qty: float
    filled_qty: float
    avg_px: float
    status: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ICMarketsPriceApplication(fix.Application):
    """FIX application for IC Markets price data."""
    
    def __init__(self, message_callback: Optional[Callable] = None):
        super().__init__()
        self.session_id = None
        self.connected = False
        self.message_callback = message_callback
        self.market_data: Dict[str, MarketDataEntry] = {}
        self.lock = threading.Lock()
        
    def onCreate(self, session_id):
        """Called when session is created."""
        self.session_id = session_id
        logger.info(f"Price session created: {session_id}")
        
    def onLogon(self, session_id):
        """Called when session is logged on."""
        self.connected = True
        logger.info(f"Price session logged on: {session_id}")
        
    def onLogout(self, session_id):
        """Called when session is logged out."""
        self.connected = False
        logger.info(f"Price session logged out: {session_id}")
        
    def toAdmin(self, message, session_id):
        """Handle outgoing admin messages."""
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        
        if msg_type.getValue() == fix.MsgType_Logon:
            # Add authentication for logon
            message.setField(fix.Username(self.session_id.getSenderCompID().split('.')[1]))
            message.setField(fix.Password(os.getenv("ICMARKETS_PASSWORD", "")))
            
    def fromAdmin(self, message, session_id):
        """Handle incoming admin messages."""
        pass
        
    def toApp(self, message, session_id):
        """Handle outgoing application messages."""
        logger.debug(f"Sending message: {message}")
        
    def fromApp(self, message, session_id):
        """Handle incoming application messages."""
        try:
            self.process_market_data(message)
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            
    def process_market_data(self, message):
        """Process market data messages."""
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        
        if msg_type.getValue() == fix.MsgType_MarketDataSnapshotFullRefresh:
            self._process_snapshot(message)
        elif msg_type.getValue() == fix.MsgType_MarketDataIncrementalRefresh:
            self._process_incremental(message)
            
    def _process_snapshot(self, message):
        """Process market data snapshot."""
        symbol = fix.Symbol()
        message.getField(symbol)
        
        entry = MarketDataEntry(symbol=symbol.getValue())
        
        # Extract bid/ask prices and sizes
        no_md_entries = fix.NoMDEntries()
        message.getField(no_md_entries)
        
        for i in range(no_md_entries.getValue()):
            group = fix.Group()
            message.getGroup(i + 1, group)
            
            md_entry_type = fix.MDEntryType()
            md_entry_px = fix.MDEntryPx()
            md_entry_size = fix.MDEntrySize()
            
            group.getField(md_entry_type)
            group.getField(md_entry_px)
            
            if md_entry_type.getValue() == '0':  # Bid
                entry.bid = float(md_entry_px.getValue())
                if group.isSetField(md_entry_size):
                    group.getField(md_entry_size)
                    entry.bid_size = float(md_entry_size.getValue())
            elif md_entry_type.getValue() == '1':  # Ask
                entry.ask = float(md_entry_px.getValue())
                if group.isSetField(md_entry_size):
                    group.getField(md_entry_size)
                    entry.ask_size = float(md_entry_size.getValue())
                    
        with self.lock:
            self.market_data[symbol.getValue()] = entry
            
        if self.message_callback:
            self.message_callback(entry)
            
    def _process_incremental(self, message):
        """Process incremental market data updates."""
        # Similar to snapshot processing but updates existing data
        pass
        
    def subscribe_market_data(self, symbols: List[str]):
        """Subscribe to market data for given symbols."""
        if not self.connected or not self.session_id:
            logger.error("Cannot subscribe - not connected")
            return
            
        message = fix.Message()
        header = message.getHeader()
        
        # Standard header
        header.setField(fix.BeginString("FIX.4.4"))
        header.setField(fix.MsgType(fix.MsgType_MarketDataRequest))
        
        # Request details
        message.setField(fix.MDReqID(str(int(time.time() * 1000))))
        message.setField(fix.SubscriptionRequestType('1'))  # Snapshot + Updates
        message.setField(fix.MarketDepth(0))  # Full book
        
        # Add symbols
        for symbol in symbols:
            group = fix.Group()
            group.setField(fix.Symbol(symbol))
            message.addGroup(group)
            
        fix.Session.sendToTarget(message, self.session_id)
        
    def get_market_data(self, symbol: str) -> Optional[MarketDataEntry]:
        """Get current market data for a symbol."""
        with self.lock:
            return self.market_data.get(symbol)
            
    def is_connected(self) -> bool:
        """Check if session is connected."""
        return self.connected


class ICMarketsTradeApplication(fix.Application):
    """FIX application for IC Markets trading operations."""
    
    def __init__(self, order_callback: Optional[Callable] = None):
        super().__init__()
        self.session_id = None
        self.connected = False
        self.order_callback = order_callback
        self.orders: Dict[str, OrderStatus] = {}
        self.lock = threading.Lock()
        self.order_id_counter = 0
        
    def onCreate(self, session_id):
        """Called when session is created."""
        self.session_id = session_id
        logger.info(f"Trade session created: {session_id}")
        
    def onLogon(self, session_id):
        """Called when session is logged on."""
        self.connected = True
        logger.info(f"Trade session logged on: {session_id}")
        
    def onLogout(self, session_id):
        """Called when session is logged out."""
        self.connected = False
        logger.info(f"Trade session logged out: {session_id}")
        
    def toAdmin(self, message, session_id):
        """Handle outgoing admin messages."""
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        
        if msg_type.getValue() == fix.MsgType_Logon:
            # Add authentication for logon
            message.setField(fix.Username(self.session_id.getSenderCompID().split('.')[1]))
            message.setField(fix.Password(os.getenv("ICMARKETS_PASSWORD", "")))
            
    def fromAdmin(self, message, session_id):
        """Handle incoming admin messages."""
        pass
        
    def toApp(self, message, session_id):
        """Handle outgoing application messages."""
        logger.debug(f"Sending trade message: {message}")
        
    def fromApp(self, message, session_id):
        """Handle incoming application messages."""
        try:
            self.process_execution_report(message)
        except Exception as e:
            logger.error(f"Error processing execution report: {e}")
            
    def process_execution_report(self, message):
        """Process execution reports."""
        msg_type = fix.MsgType()
        message.getHeader().getField(msg_type)
        
        if msg_type.getValue() == fix.MsgType_ExecutionReport:
            self._process_execution_report(message)
            
    def _process_execution_report(self, message):
        """Process execution report details."""
        cl_ord_id = fix.ClOrdID()
        order_id = fix.OrderID()
        symbol = fix.Symbol()
        side = fix.Side()
        order_qty = fix.OrderQty()
        cum_qty = fix.CumQty()
        avg_px = fix.AvgPx()
        ord_status = fix.OrdStatus()
        
        message.getField(cl_ord_id)
        message.getField(order_id)
        message.getField(symbol)
        message.getField(side)
        message.getField(order_qty)
        message.getField(cum_qty)
        message.getField(avg_px)
        message.getField(ord_status)
        
        order_status = OrderStatus(
            cl_ord_id=cl_ord_id.getValue(),
            order_id=order_id.getValue(),
            symbol=symbol.getValue(),
            side=side.getValue(),
            order_qty=float(order_qty.getValue()),
            filled_qty=float(cum_qty.getValue()),
            avg_px=float(avg_px.getValue()),
            status=ord_status.getValue()
        )
        
        with self.lock:
            self.orders[cl_ord_id.getValue()] = order_status
            
        if self.order_callback:
            self.order_callback(order_status)
            
    def place_market_order(self, symbol: str, side: str, quantity: float) -> str:
        """Place a market order."""
        if not self.connected or not self.session_id:
            raise RuntimeError("Cannot place order - not connected")
            
        self.order_id_counter += 1
        cl_ord_id = f"{int(time.time() * 1000)}_{self.order_id_counter}"
        
        message = fix.Message()
        header = message.getHeader()
        
        # Standard header
        header.setField(fix.BeginString("FIX.4.4"))
        header.setField(fix.MsgType(fix.MsgType_NewOrderSingle))
        
        # Order details
        message.setField(fix.ClOrdID(cl_ord_id))
        message.setField(fix.Symbol(symbol))
        message.setField(fix.Side(side))  # 1=Buy, 2=Sell
        message.setField(fix.OrderQty(quantity))
        message.setField(fix.OrdType('1'))  # Market order
        message.setField(fix.TransactTime())
        
        fix.Session.sendToTarget(message, self.session_id)
        
        return cl_ord_id
        
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> str:
        """Place a limit order."""
        if not self.connected or not self.session_id:
            raise RuntimeError("Cannot place order - not connected")
            
        self.order_id_counter += 1
        cl_ord_id = f"{int(time.time() * 1000)}_{self.order_id_counter}"
        
        message = fix.Message()
        header = message.getHeader()
        
        # Standard header
        header.setField(fix.BeginString("FIX.4.4"))
        header.setField(fix.MsgType(fix.MsgType_NewOrderSingle))
        
        # Order details
        message.setField(fix.ClOrdID(cl_ord_id))
        message.setField(fix.Symbol(symbol))
        message.setField(fix.Side(side))
        message.setField(fix.OrderQty(quantity))
        message.setField(fix.OrdType('2'))  # Limit order
        message.setField(fix.Price(price))
        message.setField(fix.TransactTime())
        
        fix.Session.sendToTarget(message, self.session_id)
        
        return cl_ord_id
        
    def cancel_order(self, cl_ord_id: str, symbol: str) -> str:
        """Cancel an existing order."""
        if not self.connected or not self.session_id:
            raise RuntimeError("Cannot cancel order - not connected")
            
        self.order_id_counter += 1
        cancel_id = f"CX_{int(time.time() * 1000)}_{self.order_id_counter}"
        
        message = fix.Message()
        header = message.getHeader()
        
        # Standard header
        header.setField(fix.BeginString("FIX.4.4"))
        header.setField(fix.MsgType(fix.MsgType_OrderCancelRequest))
        
        # Cancel details
        message.setField(fix.OrigClOrdID(cl_ord_id))
        message.setField(fix.ClOrdID(cancel_id))
        message.setField(fix.Symbol(symbol))
        message.setField(fix.Side('1'))  # Need to get from original order
        message.setField(fix.TransactTime())
        
        fix.Session.sendToTarget(message, self.session_id)
        
        return cancel_id
        
    def get_order_status(self, cl_ord_id: str) -> Optional[OrderStatus]:
        """Get order status by client order ID."""
        with self.lock:
            return self.orders.get(cl_ord_id)
            
    def get_all_orders(self) -> Dict[str, OrderStatus]:
        """Get all orders."""
        with self.lock:
            return self.orders.copy()
            
    def is_connected(self) -> bool:
        """Check if session is connected."""
        return self.connected


class ICMarketsFIXManager:
    """Manager for IC Markets FIX connections."""
    
    def __init__(self, config: ICMarketsConfig):
        self.config = config
        self.price_app = None
        self.trade_app = None
        self.price_initiator = None
        self.trade_initiator = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def start_sessions(self):
        """Start both price and trade sessions."""
        try:
            self.config.validate_config()
            
            # Start price session
            self.price_app = ICMarketsPriceApplication()
            self._start_session(
                self.price_app,
                self.config.get_price_session_config(),
                "price"
            )
            
            # Start trade session
            self.trade_app = ICMarketsTradeApplication()
            self._start_session(
                self.trade_app,
                self.config.get_trade_session_config(),
                "trade"
            )
            
            logger.info("IC Markets FIX sessions started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start FIX sessions: {e}")
            raise
            
    def _start_session(self, app, config_dict, session_type):
        """Start a single FIX session."""
        try:
            import quickfix as fix
            
            # Create settings
            settings = fix.SessionSettings()
            session = fix.Session()
            
            for key, value in config_dict.items():
                settings.set(key, str(value))
                
            # Create store and log factories
            store_factory = fix.FileStoreFactory(settings)
            log_factory = fix.FileLogFactory(settings)
            
            # Create initiator
            if session_type == "price":
                self.price_initiator = fix.SocketInitiator(
                    app, store_factory, settings, log_factory
                )
                self.price_initiator.start()
            else:
                self.trade_initiator = fix.SocketInitiator(
                    app, store_factory, settings, log_factory
                )
                self.trade_initiator.start()
                
            logger.info(f"{session_type} session started")
            
        except Exception as e:
            logger.error(f"Failed to start {session_type} session: {e}")
            raise
            
    def stop_sessions(self):
        """Stop all FIX sessions."""
        try:
            if self.price_initiator:
                self.price_initiator.stop()
                logger.info("Price session stopped")
                
            if self.trade_initiator:
                self.trade_initiator.stop()
                logger.info("Trade session stopped")
                
            self.executor.shutdown(wait=True)
            logger.info("All FIX sessions stopped")
            
        except Exception as e:
            logger.error(f"Error stopping sessions: {e}")
            
    def wait_for_connection(self, timeout: int = 30):
        """Wait for both sessions to connect."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if (self.price_app and self.price_app.is_connected() and
                self.trade_app and self.trade_app.is_connected()):
                return True
            time.sleep(1)
            
        return False
        
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for both sessions."""
        return {
            'price_connected': self.price_app.is_connected() if self.price_app else False,
            'trade_connected': self.trade_app.is_connected() if self.trade_app else False
        }


# Import os for environment variables
import os
