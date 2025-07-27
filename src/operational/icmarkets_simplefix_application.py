"""
IC Markets SimpleFIX Application
Windows-compatible FIX 4.4 implementation using simplefix
"""

import socket
import ssl
import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import simplefix

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


class ICMarketsSimpleFIXConnection:
    """SimpleFIX-based connection to IC Markets."""
    
    def __init__(self, config: 'ICMarketsConfig'):
        self.config = config
        self.price_socket = None
        self.trade_socket = None
        self.price_connected = False
        self.trade_connected = False
        self.sequence_number = 1
        self.market_data: Dict[str, MarketDataEntry] = {}
        self.orders: Dict[str, OrderStatus] = {}
        self.lock = threading.Lock()
        
    def connect_price_session(self) -> bool:
        """Connect to IC Markets price session."""
        try:
            self.config.validate_config()
            
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            # Create socket connection
            self.price_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.price_socket.settimeout(10)
            
            # Wrap with SSL
            self.price_socket = context.wrap_socket(
                self.price_socket,
                server_hostname=self.config._get_host()
            )
            
            # Connect to IC Markets price server
            host = self.config._get_host()
            port = self.config._get_port('price')
            self.price_socket.connect((host, port))
            
            # Send logon message
            if self._send_logon(self.price_socket, 'QUOTE'):
                self.price_connected = True
                logger.info("Price session connected successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect price session: {e}")
            return False
            
    def connect_trade_session(self) -> bool:
        """Connect to IC Markets trade session."""
        try:
            self.config.validate_config()
            
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            # Create socket connection
            self.trade_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.trade_socket.settimeout(10)
            
            # Wrap with SSL
            self.trade_socket = context.wrap_socket(
                self.trade_socket,
                server_hostname=self.config._get_host()
            )
            
            # Connect to IC Markets trade server
            host = self.config._get_host()
            port = self.config._get_port('trade')
            self.trade_socket.connect((host, port))
            
            # Send logon message
            if self._send_logon(self.trade_socket, 'TRADE'):
                self.trade_connected = True
                logger.info("Trade session connected successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect trade session: {e}")
            return False
            
    def _send_logon(self, sock: socket.socket, target_sub_id: str) -> bool:
        """Send logon message to IC Markets."""
        try:
            # Create logon message
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")  # BeginString
            msg.append_pair(35, "A")  # MsgType = Logon
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")  # SenderCompID
            msg.append_pair(56, "cServer")  # TargetCompID
            msg.append_pair(57, target_sub_id)  # TargetSubID
            msg.append_pair(34, self.sequence_number)  # MsgSeqNum
            msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])  # SendingTime
            msg.append_pair(98, 0)  # EncryptMethod = None
            msg.append_pair(108, 30)  # HeartBtInt = 30 seconds
            msg.append_pair(553, self.config.account_number)  # Username
            msg.append_pair(554, self.config.password)  # Password
            
            # Send message
            message_bytes = msg.encode()
            sock.send(message_bytes)
            
            # Receive response
            response = sock.recv(1024)
            if response:
                # Use proper SimpleFIX buffer-based parsing
                parser = simplefix.FixParser()
                parser.append_buffer(response)
                response_msg = parser.get_message()
                if response_msg and response_msg.get(35) == b'A':  # Logon response
                    self.sequence_number += 1
                    return True
                    
        except Exception as e:
            logger.error(f"Logon failed: {e}")
            return False
            
    def subscribe_market_data(self, symbols: List[str]) -> bool:
        """Subscribe to market data for given symbols."""
        if not self.price_connected or not self.price_socket:
            logger.error("Cannot subscribe - price session not connected")
            return False
            
        try:
            # Create market data request
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "V")  # MarketDataRequest
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "QUOTE")
            msg.append_pair(34, self.sequence_number)
            msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
            msg.append_pair(262, str(int(time.time())))  # MDReqID
            msg.append_pair(263, "1")  # SubscriptionRequestType = Snapshot + Updates
            msg.append_pair(264, "0")  # MarketDepth = Full book
            
            # Add symbols
            for symbol in symbols:
                msg.append_pair(146, "1")  # NoRelatedSym
                msg.append_pair(55, symbol)  # Symbol
                
            # Send message
            message_bytes = msg.encode()
            self.price_socket.send(message_bytes)
            self.sequence_number += 1
            
            # Start receiving market data
            threading.Thread(target=self._receive_market_data, daemon=True).start()
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {e}")
            return False
            
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        """Place a market order."""
        if not self.trade_connected or not self.trade_socket:
            logger.error("Cannot place order - trade session not connected")
            return None
            
        try:
            cl_ord_id = str(int(time.time() * 1000))
            
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "D")  # NewOrderSingle
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "TRADE")
            msg.append_pair(34, self.sequence_number)
            msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
            msg.append_pair(11, cl_ord_id)  # ClOrdID
            msg.append_pair(55, symbol)  # Symbol
            msg.append_pair(54, "1" if side.upper() == "BUY" else "2")  # Side
            msg.append_pair(38, str(quantity))  # OrderQty
            msg.append_pair(40, "1")  # OrdType = Market
            msg.append_pair(60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])  # TransactTime
            
            # Send message
            message_bytes = msg.encode()
            self.trade_socket.send(message_bytes)
            self.sequence_number += 1
            
            return cl_ord_id
            
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None
            
    def _receive_market_data(self):
        """Receive and process market data."""
        parser = simplefix.FixParser()
        while self.price_connected:
            try:
                data = self.price_socket.recv(4096)
                if data:
                    # Use proper buffer-based parsing
                    parser.append_buffer(data)
                    msg = parser.get_message()
                    
                    while msg:
                        if msg.get(35) == b"W":  # MarketDataSnapshotFullRefresh
                            self._process_market_data_snapshot(msg)
                        elif msg.get(35) == b"X":  # MarketDataIncrementalRefresh
                            self._process_market_data_incremental(msg)
                        
                        # Get next message if available
                        msg = parser.get_message()
                        
            except Exception as e:
                logger.error(f"Error receiving market data: {e}")
                break
                
    def _process_market_data_snapshot(self, msg):
        """Process market data snapshot."""
        symbol = msg.get(55).decode() if msg.get(55) else ""
        if not symbol:
            return
            
        entry = MarketDataEntry(symbol=symbol)
        
        # Extract bid/ask prices
        # This is a simplified version - real implementation would parse groups
        if msg.get(270):
            # Handle price data
            pass
            
        with self.lock:
            self.market_data[symbol] = entry
            
    def _process_market_data_incremental(self, msg):
        """Process incremental market data updates."""
        # Implementation for incremental updates
        pass
        
    def get_market_data(self, symbol: str) -> Optional[MarketDataEntry]:
        """Get current market data for a symbol."""
        with self.lock:
            return self.market_data.get(symbol)
            
    def disconnect(self):
        """Disconnect from IC Markets."""
        if self.price_socket:
            self.price_socket.close()
            self.price_connected = False
            
        if self.trade_socket:
            self.trade_socket.close()
            self.trade_connected = False
            
    def is_connected(self) -> Dict[str, bool]:
        """Get connection status."""
        return {
            'price_connected': self.price_connected,
            'trade_connected': self.trade_connected
        }


class ICMarketsSimpleFIXManager:
    """Manager for IC Markets SimpleFIX connections."""
    
    def __init__(self, config: 'ICMarketsConfig'):
        self.config = config
        self.price_connection = None
        self.trade_connection = None
        
    def connect(self) -> bool:
        """Connect to both price and trade sessions."""
        try:
            self.config.validate_config()
            
            # Create connections
            self.price_connection = ICMarketsSimpleFIXConnection(self.config)
            self.trade_connection = ICMarketsSimpleFIXConnection(self.config)
            
            # Connect sessions
            price_success = self.price_connection.connect_price_session()
            trade_success = self.trade_connection.connect_trade_session()
            
            if price_success and trade_success:
                logger.info("IC Markets SimpleFIX connections established")
                return True
            else:
                logger.error("Failed to establish connections")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
            
    def subscribe_market_data(self, symbols: List[str]) -> bool:
        """Subscribe to market data."""
        if self.price_connection:
            return self.price_connection.subscribe_market_data(symbols)
        return False
        
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        """Place a market order."""
        if self.trade_connection:
            return self.trade_connection.place_market_order(symbol, side, quantity)
        return None
        
    def disconnect(self):
        """Disconnect all sessions."""
        if self.price_connection:
            self.price_connection.disconnect()
        if self.trade_connection:
            self.trade_connection.disconnect()
            
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status."""
        status = {}
        if self.price_connection:
            status.update(self.price_connection.is_connected())
        if self.trade_connection:
            status.update(self.trade_connection.is_connected())
        return status
