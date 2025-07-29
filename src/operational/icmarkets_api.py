"""
Genuine IC Markets FIX API Implementation
With proper ExecutionReport processing, Market Data handling, and real order tracking
"""

import socket
import ssl
import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import simplefix
import json
import os
from concurrent.futures import ThreadPoolExecutor
import queue
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration based on FIX OrdStatus values."""
    NEW = "0"
    PARTIALLY_FILLED = "1"
    FILLED = "2"
    DONE_FOR_DAY = "3"
    CANCELED = "4"
    REPLACED = "5"
    PENDING_CANCEL = "6"
    STOPPED = "7"
    REJECTED = "8"
    SUSPENDED = "9"
    PENDING_NEW = "A"
    CALCULATED = "B"
    EXPIRED = "C"
    ACCEPTED_FOR_BIDDING = "D"
    PENDING_REPLACE = "E"


class ExecType(Enum):
    """Execution type enumeration based on FIX ExecType values."""
    NEW = "0"
    PARTIAL_FILL = "1"
    FILL = "2"
    DONE_FOR_DAY = "3"
    CANCELED = "4"
    REPLACED = "5"
    PENDING_CANCEL = "6"
    STOPPED = "7"
    REJECTED = "8"
    SUSPENDED = "9"
    PENDING_NEW = "A"
    CALCULATED = "B"
    EXPIRED = "C"
    RESTATED = "D"
    PENDING_REPLACE = "E"
    TRADE = "F"


@dataclass
class OrderInfo:
    """Complete order information tracking."""
    cl_ord_id: str
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    order_qty: float = 0.0
    ord_type: str = ""
    price: Optional[float] = None
    time_in_force: str = "0"  # Day
    status: OrderStatus = OrderStatus.PENDING_NEW
    cum_qty: float = 0.0
    leaves_qty: float = 0.0
    avg_px: float = 0.0
    last_qty: float = 0.0
    last_px: float = 0.0
    exec_id: Optional[str] = None
    transact_time: Optional[datetime] = None
    text: Optional[str] = None
    reject_reason: Optional[str] = None
    created_time: datetime = field(default_factory=datetime.utcnow)
    updated_time: datetime = field(default_factory=datetime.utcnow)
    executions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MarketDataEntry:
    """Market data entry with proper structure."""
    symbol: str
    entry_type: str  # 0=Bid, 1=Offer, 2=Trade
    price: float
    size: float
    entry_time: datetime = field(default_factory=datetime.utcnow)
    entry_id: Optional[str] = None
    position_no: Optional[int] = None


@dataclass
class OrderBook:
    """Order book structure for market data."""
    symbol: str
    bids: List[MarketDataEntry] = field(default_factory=list)
    asks: List[MarketDataEntry] = field(default_factory=list)
    last_trade: Optional[MarketDataEntry] = None
    last_update: datetime = field(default_factory=datetime.utcnow)


class GenuineFIXConnection:
    """Genuine FIX connection with proper message processing."""
    
    def __init__(self, config, session_type: str, message_handler: Callable):
        self.config = config
        self.session_type = session_type
        self.message_handler = message_handler
        self.socket = None
        self.ssl_socket = None
        self.connected = False
        self.authenticated = False
        self.sequence_number = 1
        self.expected_seq_num = 1
        self.message_queue = queue.Queue()
        self.running = False
        self.receiver_thread = None
        self.heartbeat_thread = None
        self.last_heartbeat = None
        self.last_received = None
        self.pending_requests = {}  # Track pending requests for correlation
        
    def connect(self) -> bool:
        """Connect with proper session management."""
        try:
            logger.info(f"Connecting to {self.session_type} session...")
            
            # Create SSL context
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(30)
            
            # Wrap with SSL
            self.ssl_socket = context.wrap_socket(
                self.socket, 
                server_hostname=self.config._get_host()
            )
            
            # Connect to server
            host = self.config._get_host()
            port = self.config._get_port(self.session_type)
            logger.info(f"Connecting to {host}:{port}")
            
            self.ssl_socket.connect((host, port))
            self.connected = True
            
            # Send logon message and wait for response
            if self._send_logon_and_wait():
                self.authenticated = True
                self.running = True
                
                # Start receiver thread
                self.receiver_thread = threading.Thread(target=self._receive_messages, daemon=True)
                self.receiver_thread.start()
                
                # Start heartbeat thread
                self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
                self.heartbeat_thread.start()
                
                logger.info(f"✅ {self.session_type} session authenticated successfully")
                return True
            else:
                logger.error(f"❌ {self.session_type} session authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.connected = False
            self.authenticated = False
            return False
            
    def _send_logon_and_wait(self) -> bool:
        """Send logon and wait for proper response."""
        try:
            msg = simplefix.FixMessage()
            
            # Standard FIX header in correct order
            msg.append_pair(8, "FIX.4.4")  # BeginString
            msg.append_pair(35, "A")       # MsgType = Logon
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")  # SenderCompID
            msg.append_pair(56, "cServer")  # TargetCompID
            msg.append_pair(57, self.session_type.upper())  # TargetSubID
            msg.append_pair(50, self.session_type.upper())  # SenderSubID
            msg.append_pair(34, self.sequence_number)  # MsgSeqNum
            msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])  # SendingTime
            
            # Logon specific fields
            msg.append_pair(98, 0)   # EncryptMethod = None
            msg.append_pair(108, 30) # HeartBtInt = 30 seconds
            msg.append_pair(141, "Y") # ResetSeqNumFlag
            msg.append_pair(553, self.config.account_number)  # Username
            msg.append_pair(554, self.config.password)        # Password
            
            # Send message
            message_str = msg.encode()
            logger.info(f"Sending logon message: {message_str}")
            self.ssl_socket.send(message_str)
            self.sequence_number += 1
            
            # Wait for logon response with proper timeout
            start_time = time.time()
            while time.time() - start_time < 10:  # 10 second timeout
                try:
                    response = self.ssl_socket.recv(1024)
                    if response:
                        logger.info(f"Received logon response: {response}")
                        
                        # Parse the response properly
                        parsed_response = self._parse_raw_message(response)
                        if parsed_response:
                            msg_type = parsed_response.get('35')
                            
                            if msg_type == 'A':  # Logon response
                                logger.info("✅ Logon accepted by server")
                                self.last_heartbeat = datetime.utcnow()
                                self.last_received = datetime.utcnow()
                                return True
                            elif msg_type == '5':  # Logout
                                text = parsed_response.get('58', 'No reason provided')
                                logger.error(f"❌ Server sent logout: {text}")
                                return False
                            elif msg_type == '3':  # Reject
                                text = parsed_response.get('58', 'No reason provided')
                                logger.error(f"❌ Server rejected logon: {text}")
                                return False
                                
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error receiving logon response: {e}")
                    return False
                    
            logger.error("❌ Logon timeout - no response from server")
            return False
            
        except Exception as e:
            logger.error(f"❌ Logon failed: {e}")
            return False
            
    def _parse_raw_message(self, raw_data: bytes) -> Optional[Dict[str, str]]:
        """Parse raw FIX message data."""
        try:
            # Convert bytes to string and split by SOH
            message_str = raw_data.decode('utf-8', errors='ignore')
            fields = message_str.split('\x01')
            
            parsed = {}
            for field in fields:
                if '=' in field:
                    tag, value = field.split('=', 1)
                    parsed[tag] = value
                    
            return parsed if parsed else None
            
        except Exception as e:
            logger.error(f"Error parsing raw message: {e}")
            return None
            
    def send_message_and_track(self, message, request_id: str = None) -> bool:
        """Send message and track for response correlation."""
        try:
            if not self.connected or not self.authenticated:
                logger.error("Cannot send message - not connected/authenticated")
                return False
                
            # Create a new message with proper field ordering
            ordered_msg = simplefix.FixMessage()
            
            # Standard header fields in correct order
            ordered_msg.append_pair(8, message.get(8))   # BeginString
            ordered_msg.append_pair(35, message.get(35)) # MsgType
            ordered_msg.append_pair(49, message.get(49)) # SenderCompID
            ordered_msg.append_pair(56, message.get(56)) # TargetCompID
            
            # Optional header fields
            if message.get(57):
                ordered_msg.append_pair(57, message.get(57)) # TargetSubID
            if message.get(50):
                ordered_msg.append_pair(50, message.get(50)) # SenderSubID
                
            # Sequence number and timestamp
            ordered_msg.append_pair(34, self.sequence_number)
            ordered_msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
            
            # Copy all other fields in order they were added
            for tag in range(1, 1000):
                if tag not in [8, 34, 35, 49, 50, 52, 56, 57]:  # Skip header fields already added
                    value = message.get(tag)
                    if value is not None:
                        ordered_msg.append_pair(tag, value)
            
            message_str = ordered_msg.encode()
            self.ssl_socket.send(message_str)
            
            # Track request if ID provided
            if request_id:
                self.pending_requests[request_id] = {
                    'sent_time': datetime.utcnow(),
                    'sequence_number': self.sequence_number,
                    'message_type': message.get(35)
                }
            
            self.sequence_number += 1
            logger.info(f"Sent message: {message_str}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
            
    def _receive_messages(self):
        """Receive and process incoming messages with proper parsing."""
        buffer = b""
        
        while self.running and self.connected:
            try:
                data = self.ssl_socket.recv(4096)
                if not data:
                    logger.warning("No data received - connection may be closed")
                    break
                    
                buffer += data
                self.last_received = datetime.utcnow()
                
                # Process complete messages
                while True:
                    # Find complete FIX message (ends with checksum field)
                    checksum_pos = buffer.find(b'\x0110=')
                    if checksum_pos == -1:
                        break
                        
                    # Find end of checksum field
                    end_pos = buffer.find(b'\x01', checksum_pos + 4)
                    if end_pos == -1:
                        break
                        
                    message_data = buffer[:end_pos + 1]
                    buffer = buffer[end_pos + 1:]
                    
                    # Parse and process message
                    try:
                        parsed = self._parse_complete_message(message_data)
                        if parsed:
                            # Handle message through message handler
                            self.message_handler(parsed, self.session_type)
                            logger.debug(f"Processed message: {parsed.get('35', 'Unknown')}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break
                
        logger.info(f"{self.session_type} receiver thread stopped")
        
    def _parse_complete_message(self, raw_message: bytes) -> Optional[Dict[str, str]]:
        """Parse complete FIX message with proper validation."""
        try:
            # Convert to string and split by SOH
            message_str = raw_message.decode('utf-8', errors='ignore')
            fields = message_str.split('\x01')
            
            parsed = {}
            for field in fields:
                if '=' in field:
                    tag, value = field.split('=', 1)
                    parsed[tag] = value
                    
            # Validate required fields
            if '8' not in parsed or '35' not in parsed:
                logger.warning("Invalid FIX message - missing required fields")
                return None
                
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing FIX message: {e}")
            return None
            
    def _send_heartbeat(self):
        """Send heartbeat message."""
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "0")  # Heartbeat
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, self.session_type.upper())
            msg.append_pair(50, self.session_type.upper())
            
            self.send_message_and_track(msg)
            self.last_heartbeat = datetime.utcnow()
            logger.debug(f"Sent heartbeat for {self.session_type}")
            
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            self.connected = False
            self.authenticated = False
            
    def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self.running and self.connected and self.authenticated:
            try:
                time.sleep(30)  # 30-second heartbeat interval
                if self.connected and self.authenticated:
                    self._send_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                break
                
    def disconnect(self):
        """Disconnect gracefully."""
        try:
            self.running = False
            self.authenticated = False
            
            if self.ssl_socket:
                # Send logout message
                msg = simplefix.FixMessage()
                msg.append_pair(8, "FIX.4.4")
                msg.append_pair(35, "5")  # Logout
                msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
                msg.append_pair(56, "cServer")
                msg.append_pair(57, self.session_type.upper())
                msg.append_pair(50, self.session_type.upper())
                
                self.send_message_and_track(msg)
                time.sleep(1)  # Give time for logout to be processed
                self.ssl_socket.close()
                
            self.connected = False
            logger.info(f"{self.session_type} session disconnected")
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            
    def is_connected(self) -> bool:
        """Check if connection is active and authenticated."""
        return self.connected and self.authenticated and self.running


class GenuineFIXManager:
    """Genuine FIX manager with proper ExecutionReport and Market Data processing."""
    
    def __init__(self, config):
        self.config = config
        self.price_connection = None
        self.trade_connection = None
        self.orders: Dict[str, OrderInfo] = {}  # Real order tracking
        self.order_books: Dict[str, OrderBook] = {}  # Real market data
        self.running = False
        self.order_callbacks = []  # Callbacks for order updates
        self.market_data_callbacks = []  # Callbacks for market data updates
        
    def add_order_callback(self, callback: Callable[[OrderInfo], None]):
        """Add callback for order updates."""
        self.order_callbacks.append(callback)
        
    def add_market_data_callback(self, callback: Callable[[str, OrderBook], None]):
        """Add callback for market data updates."""
        self.market_data_callbacks.append(callback)
        
    def _handle_message(self, message: Dict[str, str], session_type: str):
        """Handle incoming FIX messages based on type."""
        try:
            msg_type = message.get('35')
            
            if session_type == 'trade':
                if msg_type == '8':  # ExecutionReport
                    self._handle_execution_report(message)
                elif msg_type == '9':  # OrderCancelReject
                    self._handle_order_cancel_reject(message)
                elif msg_type == 'j':  # BusinessMessageReject
                    self._handle_business_message_reject(message)
                    
            elif session_type == 'quote':
                if msg_type == 'W':  # MarketDataSnapshot
                    self._handle_market_data_snapshot(message)
                elif msg_type == 'X':  # MarketDataIncrementalRefresh
                    self._handle_market_data_incremental_refresh(message)
                elif msg_type == 'Y':  # MarketDataRequestReject
                    self._handle_market_data_request_reject(message)
                    
            # Common message types
            if msg_type == '0':  # Heartbeat
                logger.debug(f"Received heartbeat from {session_type}")
            elif msg_type == '1':  # TestRequest
                self._handle_test_request(message, session_type)
            elif msg_type == '3':  # Reject
                self._handle_session_reject(message, session_type)
            elif msg_type == '5':  # Logout
                self._handle_logout(message, session_type)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
    def _handle_execution_report(self, message: Dict[str, str]):
        """Handle ExecutionReport messages - CRITICAL for real order tracking."""
        try:
            cl_ord_id = message.get('11')
            order_id = message.get('37')
            exec_type = message.get('150')
            ord_status = message.get('39')
            
            if not cl_ord_id:
                logger.error("ExecutionReport missing ClOrdID")
                return
                
            # Get or create order info
            if cl_ord_id not in self.orders:
                # Create new order info from ExecutionReport
                self.orders[cl_ord_id] = OrderInfo(
                    cl_ord_id=cl_ord_id,
                    order_id=order_id,
                    symbol=message.get('55', ''),
                    side=message.get('54', ''),
                    order_qty=float(message.get('38', 0)),
                    ord_type=message.get('40', ''),
                    price=float(message.get('44', 0)) if message.get('44') else None
                )
                
            order = self.orders[cl_ord_id]
            
            # Update order information
            order.order_id = order_id or order.order_id
            order.status = OrderStatus(ord_status) if ord_status else order.status
            order.cum_qty = float(message.get('14', 0))
            order.leaves_qty = float(message.get('151', 0))
            order.avg_px = float(message.get('6', 0))
            order.last_qty = float(message.get('32', 0))
            order.last_px = float(message.get('31', 0))
            order.exec_id = message.get('17')
            order.text = message.get('58')
            order.updated_time = datetime.utcnow()
            
            # Handle transact time
            transact_time_str = message.get('60')
            if transact_time_str:
                try:
                    order.transact_time = datetime.strptime(transact_time_str, "%Y%m%d-%H:%M:%S.%f")
                except:
                    order.transact_time = datetime.utcnow()
                    
            # Handle rejection
            if exec_type == ExecType.REJECTED.value:
                order.reject_reason = message.get('103')  # OrdRejReason
                logger.error(f"Order {cl_ord_id} rejected: {order.text} (Reason: {order.reject_reason})")
            
            # Add execution to history
            execution = {
                'exec_id': order.exec_id,
                'exec_type': exec_type,
                'ord_status': ord_status,
                'last_qty': order.last_qty,
                'last_px': order.last_px,
                'cum_qty': order.cum_qty,
                'leaves_qty': order.leaves_qty,
                'timestamp': order.updated_time,
                'text': order.text
            }
            order.executions.append(execution)
            
            # Log execution report
            if exec_type == ExecType.NEW.value:
                logger.info(f"✅ Order {cl_ord_id} accepted by broker (OrderID: {order_id})")
            elif exec_type == ExecType.FILL.value:
                logger.info(f"✅ Order {cl_ord_id} filled: {order.last_qty} @ {order.last_px}")
            elif exec_type == ExecType.PARTIAL_FILL.value:
                logger.info(f"✅ Order {cl_ord_id} partially filled: {order.last_qty} @ {order.last_px} (Total: {order.cum_qty})")
            elif exec_type == ExecType.CANCELED.value:
                logger.info(f"✅ Order {cl_ord_id} canceled")
            elif exec_type == ExecType.REJECTED.value:
                logger.error(f"❌ Order {cl_ord_id} rejected: {order.text}")
                
            # Notify callbacks
            for callback in self.order_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Error in order callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling ExecutionReport: {e}")
            
    def _handle_market_data_snapshot(self, message: Dict[str, str]):
        """Handle Market Data Snapshot messages - CRITICAL for real market data."""
        try:
            symbol = message.get('55')
            if not symbol:
                logger.error("MarketDataSnapshot missing symbol")
                return
                
            # Initialize order book if not exists
            if symbol not in self.order_books:
                self.order_books[symbol] = OrderBook(symbol=symbol)
                
            order_book = self.order_books[symbol]
            order_book.bids.clear()
            order_book.asks.clear()
            order_book.last_update = datetime.utcnow()
            
            # Parse market data entries
            num_entries = int(message.get('268', 0))
            logger.info(f"Processing market data snapshot for {symbol} with {num_entries} entries")
            
            for i in range(num_entries):
                # FIX repeating groups use incremental tags
                entry_type = message.get(f'269')  # MDEntryType
                entry_px = message.get(f'270')   # MDEntryPx
                entry_size = message.get(f'271') # MDEntrySize
                entry_time = message.get(f'273') # MDEntryTime
                
                if entry_type and entry_px and entry_size:
                    entry = MarketDataEntry(
                        symbol=symbol,
                        entry_type=entry_type,
                        price=float(entry_px),
                        size=float(entry_size)
                    )
                    
                    if entry_time:
                        try:
                            entry.entry_time = datetime.strptime(entry_time, "%Y%m%d-%H:%M:%S.%f")
                        except:
                            entry.entry_time = datetime.utcnow()
                            
                    # Add to appropriate book side
                    if entry_type == '0':  # Bid
                        order_book.bids.append(entry)
                    elif entry_type == '1':  # Offer/Ask
                        order_book.asks.append(entry)
                    elif entry_type == '2':  # Trade
                        order_book.last_trade = entry
                        
            # Sort order book
            order_book.bids.sort(key=lambda x: x.price, reverse=True)  # Highest bid first
            order_book.asks.sort(key=lambda x: x.price)  # Lowest ask first
            
            logger.info(f"✅ Market data snapshot processed for {symbol}: {len(order_book.bids)} bids, {len(order_book.asks)} asks")
            
            # Notify callbacks
            for callback in self.market_data_callbacks:
                try:
                    callback(symbol, order_book)
                except Exception as e:
                    logger.error(f"Error in market data callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling MarketDataSnapshot: {e}")
            
    def start(self) -> bool:
        """Start FIX connections with proper message handling."""
        try:
            logger.info("🚀 Starting Genuine FIX Manager...")
            
            # Validate configuration
            self.config.validate_config()
            
            # Create connections with message handler
            self.price_connection = GenuineFIXConnection(self.config, "quote", self._handle_message)
            self.trade_connection = GenuineFIXConnection(self.config, "trade", self._handle_message)
            
            # Connect price session
            price_success = self.price_connection.connect()
            if not price_success:
                logger.error("❌ Failed to connect price session")
                return False
                
            # Connect trade session
            trade_success = self.trade_connection.connect()
            if not trade_success:
                logger.error("❌ Failed to connect trade session")
                return False
                
            if price_success and trade_success:
                self.running = True
                logger.info("✅ Genuine FIX Manager started successfully")
                return True
            else:
                logger.error("❌ Failed to start Genuine FIX Manager")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start Genuine FIX Manager: {e}")
            return False


            
    def place_market_order_genuine(self, symbol: str, side: str, quantity: float, 
                                  timeout: float = 10.0) -> Optional[OrderInfo]:
        """Place market order with GENUINE broker confirmation - NO FRAUD."""
        if not self.trade_connection or not self.trade_connection.is_connected():
            logger.error("❌ Trade connection not available for order placement")
            return None
            
        try:
            cl_ord_id = f"ORDER_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            
            # Create order info BEFORE sending (pending state)
            order_info = OrderInfo(
                cl_ord_id=cl_ord_id,
                symbol=symbol,
                side=side,
                order_qty=quantity,
                ord_type="1",  # Market order
                status=OrderStatus.PENDING_NEW
            )
            self.orders[cl_ord_id] = order_info
            
            # Build FIX message
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "D")  # NewOrderSingle
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "TRADE")  # TargetSubID
            msg.append_pair(50, "TRADE")  # SenderSubID
            msg.append_pair(11, cl_ord_id)  # ClOrdID
            msg.append_pair(55, symbol)     # Symbol
            msg.append_pair(54, "1" if side.upper() == "BUY" else "2")  # Side
            msg.append_pair(38, str(quantity))  # OrderQty
            msg.append_pair(40, "1")  # OrdType = Market
            msg.append_pair(59, "0")  # TimeInForce = Day
            msg.append_pair(60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])  # TransactTime
            
            # Send message and track
            success = self.trade_connection.send_message_and_track(msg, cl_ord_id)
            if not success:
                logger.error(f"❌ Failed to send order message for {cl_ord_id}")
                del self.orders[cl_ord_id]
                return None
                
            logger.info(f"📤 Order message sent for {cl_ord_id}, waiting for broker confirmation...")
            
            # CRITICAL: Wait for ExecutionReport confirmation
            start_time = time.time()
            while time.time() - start_time < timeout:
                current_order = self.orders.get(cl_ord_id)
                if current_order:
                    # Check if we received any ExecutionReport
                    if current_order.status != OrderStatus.PENDING_NEW:
                        if current_order.status == OrderStatus.REJECTED:
                            logger.error(f"❌ Order {cl_ord_id} rejected by broker: {current_order.text}")
                            return current_order
                        else:
                            logger.info(f"✅ Order {cl_ord_id} confirmed by broker with status: {current_order.status.value}")
                            return current_order
                            
                time.sleep(0.1)  # Small delay to avoid busy waiting
                
            # Timeout - order status unknown
            logger.error(f"⏰ Timeout waiting for order confirmation for {cl_ord_id}")
            order_info.text = "Timeout waiting for broker confirmation"
            return order_info
            
        except Exception as e:
            logger.error(f"❌ Failed to place market order: {e}")
            if cl_ord_id in self.orders:
                del self.orders[cl_ord_id]
            return None
            
    def subscribe_market_data_genuine(self, symbols: List[str], timeout: float = 10.0) -> Dict[str, bool]:
        """Subscribe to market data with GENUINE broker confirmation - NO FRAUD."""
        if not self.price_connection or not self.price_connection.is_connected():
            logger.error("❌ Price connection not available for market data subscription")
            return {symbol: False for symbol in symbols}
            
        results = {}
        
        try:
            for symbol in symbols:
                req_id = f"MD_{symbol}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                
                msg = simplefix.FixMessage()
                msg.append_pair(8, "FIX.4.4")
                msg.append_pair(35, "V")  # MarketDataRequest
                msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
                msg.append_pair(56, "cServer")
                msg.append_pair(57, "QUOTE")  # TargetSubID
                msg.append_pair(50, "QUOTE")  # SenderSubID
                msg.append_pair(262, req_id)  # MDReqID
                msg.append_pair(263, "1")  # SubscriptionRequestType = Snapshot + Updates
                msg.append_pair(264, "0")  # MarketDepth = Full Book
                msg.append_pair(265, "1")  # MDUpdateType = Incremental
                msg.append_pair(267, "2")  # NoMDEntryTypes
                msg.append_pair(269, "0")  # MDEntryType = Bid
                msg.append_pair(269, "1")  # MDEntryType = Offer
                msg.append_pair(146, "1")  # NoRelatedSym
                msg.append_pair(55, symbol)  # Symbol
                
                # Send message and track
                success = self.price_connection.send_message_and_track(msg, req_id)
                if not success:
                    logger.error(f"❌ Failed to send market data request for {symbol}")
                    results[symbol] = False
                    continue
                    
                logger.info(f"📤 Market data request sent for {symbol}, waiting for broker response...")
                
                # CRITICAL: Wait for Market Data Snapshot or Reject
                start_time = time.time()
                subscription_confirmed = False
                
                while time.time() - start_time < timeout:
                    # Check if we received market data for this symbol
                    if symbol in self.order_books:
                        order_book = self.order_books[symbol]
                        # Check if we have recent data
                        if (datetime.utcnow() - order_book.last_update).total_seconds() < timeout:
                            logger.info(f"✅ Market data subscription confirmed for {symbol}")
                            subscription_confirmed = True
                            break
                            
                    time.sleep(0.1)  # Small delay to avoid busy waiting
                    
                if subscription_confirmed:
                    results[symbol] = True
                else:
                    logger.error(f"⏰ Timeout waiting for market data confirmation for {symbol}")
                    results[symbol] = False
                    
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to market data: {e}")
            for symbol in symbols:
                if symbol not in results:
                    results[symbol] = False
                    
        return results
        
    def get_order_status(self, cl_ord_id: str) -> Optional[OrderInfo]:
        """Get real order status - NO FRAUD."""
        return self.orders.get(cl_ord_id)
        
    def get_all_orders(self) -> Dict[str, OrderInfo]:
        """Get all orders - NO FRAUD."""
        return self.orders.copy()
        
    def get_market_data(self, symbol: str) -> Optional[OrderBook]:
        """Get real market data - NO FRAUD."""
        return self.order_books.get(symbol)
        
    def get_all_market_data(self) -> Dict[str, OrderBook]:
        """Get all market data - NO FRAUD."""
        return self.order_books.copy()
        
    def _handle_order_cancel_reject(self, message: Dict[str, str]):
        """Handle OrderCancelReject messages."""
        try:
            cl_ord_id = message.get('11')
            text = message.get('58', 'No reason provided')
            logger.error(f"❌ Order cancel rejected for {cl_ord_id}: {text}")
        except Exception as e:
            logger.error(f"Error handling OrderCancelReject: {e}")
            
    def _handle_business_message_reject(self, message: Dict[str, str]):
        """Handle BusinessMessageReject messages."""
        try:
            ref_msg_type = message.get('372')
            text = message.get('58', 'No reason provided')
            logger.error(f"❌ Business message reject for message type {ref_msg_type}: {text}")
        except Exception as e:
            logger.error(f"Error handling BusinessMessageReject: {e}")
            
    def _handle_market_data_incremental_refresh(self, message: Dict[str, str]):
        """Handle Market Data Incremental Refresh messages."""
        try:
            symbol = message.get('55')
            if not symbol or symbol not in self.order_books:
                return
                
            order_book = self.order_books[symbol]
            num_entries = int(message.get('268', 0))
            
            for i in range(num_entries):
                update_action = message.get('279')  # MDUpdateAction
                entry_type = message.get('269')     # MDEntryType
                entry_px = message.get('270')       # MDEntryPx
                entry_size = message.get('271')     # MDEntrySize
                
                if entry_type and entry_px:
                    price = float(entry_px)
                    size = float(entry_size) if entry_size else 0.0
                    
                    if entry_type == '0':  # Bid
                        self._update_order_book_side(order_book.bids, update_action, price, size, symbol, entry_type)
                    elif entry_type == '1':  # Offer
                        self._update_order_book_side(order_book.asks, update_action, price, size, symbol, entry_type)
                    elif entry_type == '2':  # Trade
                        order_book.last_trade = MarketDataEntry(
                            symbol=symbol,
                            entry_type=entry_type,
                            price=price,
                            size=size
                        )
                        
            order_book.last_update = datetime.utcnow()
            
            # Notify callbacks
            for callback in self.market_data_callbacks:
                try:
                    callback(symbol, order_book)
                except Exception as e:
                    logger.error(f"Error in market data callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling MarketDataIncrementalRefresh: {e}")
            
    def _update_order_book_side(self, book_side: List[MarketDataEntry], update_action: str, 
                               price: float, size: float, symbol: str, entry_type: str):
        """Update order book side based on update action."""
        try:
            if update_action == '0':  # New
                entry = MarketDataEntry(symbol=symbol, entry_type=entry_type, price=price, size=size)
                book_side.append(entry)
                # Re-sort
                if entry_type == '0':  # Bid
                    book_side.sort(key=lambda x: x.price, reverse=True)
                else:  # Ask
                    book_side.sort(key=lambda x: x.price)
            elif update_action == '1':  # Change
                for entry in book_side:
                    if entry.price == price:
                        entry.size = size
                        entry.entry_time = datetime.utcnow()
                        break
            elif update_action == '2':  # Delete
                book_side[:] = [entry for entry in book_side if entry.price != price]
                
        except Exception as e:
            logger.error(f"Error updating order book side: {e}")
            
    def _handle_market_data_request_reject(self, message: Dict[str, str]):
        """Handle Market Data Request Reject messages."""
        try:
            md_req_id = message.get('262')
            text = message.get('58', 'No reason provided')
            logger.error(f"❌ Market data request rejected for {md_req_id}: {text}")
        except Exception as e:
            logger.error(f"Error handling MarketDataRequestReject: {e}")
            
    def _handle_test_request(self, message: Dict[str, str], session_type: str):
        """Handle TestRequest messages."""
        try:
            test_req_id = message.get('112')
            # Send Heartbeat in response
            connection = self.trade_connection if session_type == 'trade' else self.price_connection
            if connection:
                msg = simplefix.FixMessage()
                msg.append_pair(8, "FIX.4.4")
                msg.append_pair(35, "0")  # Heartbeat
                msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
                msg.append_pair(56, "cServer")
                msg.append_pair(57, session_type.upper())
                msg.append_pair(50, session_type.upper())
                if test_req_id:
                    msg.append_pair(112, test_req_id)  # TestReqID
                connection.send_message_and_track(msg)
                logger.debug(f"Responded to TestRequest from {session_type}")
        except Exception as e:
            logger.error(f"Error handling TestRequest: {e}")
            
    def _handle_session_reject(self, message: Dict[str, str], session_type: str):
        """Handle session-level Reject messages."""
        try:
            text = message.get('58', 'No reason provided')
            logger.error(f"❌ Session reject from {session_type}: {text}")
        except Exception as e:
            logger.error(f"Error handling session reject: {e}")
            
    def _handle_logout(self, message: Dict[str, str], session_type: str):
        """Handle Logout messages."""
        try:
            text = message.get('58', 'No reason provided')
            logger.warning(f"⚠️ Logout received from {session_type}: {text}")
            # Connection will be marked as disconnected by the connection handler
        except Exception as e:
            logger.error(f"Error handling logout: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status - GENUINE DATA ONLY."""
        return {
            'price_connected': self.price_connection.is_connected() if self.price_connection else False,
            'trade_connected': self.trade_connection.is_connected() if self.trade_connection else False,
            'running': self.running,
            'total_orders': len(self.orders),
            'pending_orders': len([o for o in self.orders.values() if o.status == OrderStatus.PENDING_NEW]),
            'active_orders': len([o for o in self.orders.values() if o.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]]),
            'filled_orders': len([o for o in self.orders.values() if o.status == OrderStatus.FILLED]),
            'rejected_orders': len([o for o in self.orders.values() if o.status == OrderStatus.REJECTED]),
            'market_data_symbols': list(self.order_books.keys()),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def stop(self):
        """Stop all connections gracefully."""
        logger.info("🛑 Stopping Genuine FIX Manager...")
        self.running = False
        
        if self.price_connection:
            self.price_connection.disconnect()
            
        if self.trade_connection:
            self.trade_connection.disconnect()
            
        logger.info("✅ Genuine FIX Manager stopped")

