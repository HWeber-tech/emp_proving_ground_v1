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
        # Sequence management
        self.expected_seq_num = 1  # Next expected inbound MsgSeqNum (34)
        
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
            
            # Preserve original body field order to keep repeating groups valid
            # Parse the original message encoding to get ordered pairs
            try:
                raw = message.encode()
                raw_fields = raw.decode('utf-8', errors='ignore').split('\x01')
                body_pairs = []
                for f in raw_fields:
                    if '=' in f:
                        t, v = f.split('=', 1)
                        if t not in ('8','35','49','56','57','50','34','52','10'):
                            body_pairs.append((t, v))
                for t, v in body_pairs:
                    try:
                        ordered_msg.append_pair(int(t), v)
                    except Exception:
                        # Fallback for any unexpected non-int tags
                        pass
            except Exception:
                # Fallback to naive copy if parsing fails
                for tag in range(1, 1000):
                    if tag not in [8, 34, 35, 49, 50, 52, 56, 57]:
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

    def _send_resend_request(self, begin_seq_no: int, end_seq_no: int) -> None:
        """Send ResendRequest (35=2) for a range of missing messages."""
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "2")  # ResendRequest
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            # Sub-IDs per session
            sender_sub = self.session_type.upper()
            target_sub = self.session_type.upper()
            msg.append_pair(57, target_sub)
            msg.append_pair(50, sender_sub)
            # Range
            msg.append_pair(7, str(begin_seq_no))   # BeginSeqNo
            msg.append_pair(16, str(end_seq_no))    # EndSeqNo (0 = infinity)
            self.send_message_and_track(msg)
            logger.warning(f"ResendRequest sent for range [{begin_seq_no}, {end_seq_no}]")
        except Exception as e:
            logger.error(f"Failed to send ResendRequest: {e}")

    def _send_sequence_reset_gapfill(self, new_seq_no: int) -> None:
        """Send SequenceReset (35=4) with GapFillFlag=Y to advance counterparty's expected seq."""
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "4")  # SequenceReset
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            # Sub-IDs per session
            sender_sub = self.session_type.upper()
            target_sub = self.session_type.upper()
            msg.append_pair(57, target_sub)
            msg.append_pair(50, sender_sub)
            msg.append_pair(123, "Y")  # GapFillFlag
            msg.append_pair(36, str(new_seq_no))  # NewSeqNo
            self.send_message_and_track(msg)
            logger.warning(f"SequenceReset(GapFill) sent with NewSeqNo={new_seq_no}")
        except Exception as e:
            logger.error(f"Failed to send SequenceReset GapFill: {e}")
            
    def _receive_messages(self):
        """Receive and process incoming messages with proper parsing."""
        buffer = b""
        
        while self.running and self.connected:
            try:
                data = self.ssl_socket.recv(4096)
                if not data:
                    logger.warning("No data received - connection may be closed")
                    # Mark as disconnected to trigger supervisor
                    self.connected = False
                    self.authenticated = False
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
                            # Session-level sequence handling
                            try:
                                msg_type = parsed.get('35')
                                seq_str = parsed.get('34')
                                poss_dup = parsed.get('43')
                                seq_num = int(seq_str) if seq_str and seq_str.isdigit() else None

                                # Handle incoming ResendRequest (counterparty missed our messages)
                                if msg_type == '2':  # ResendRequest
                                    # Minimal compliance: advance with GapFill to our current outbound seq
                                    try:
                                        self._send_sequence_reset_gapfill(self.sequence_number)
                                    except Exception:
                                        pass
                                    # Do not forward to app layer
                                    continue

                                # Handle incoming SequenceReset (GapFill)
                                if msg_type == '4':
                                    try:
                                        new_seq = int(parsed.get('36', str(self.expected_seq_num)))
                                    except Exception:
                                        new_seq = self.expected_seq_num
                                    gapfill = parsed.get('123') == 'Y'
                                    # Advance expected seq regardless; GapFill means admin-only gap
                                    self.expected_seq_num = new_seq
                                    logger.info(f"SequenceReset received (GapFill={gapfill}); expected_seq_num -> {self.expected_seq_num}")
                                    # SequenceReset is admin; do not forward to app layer
                                    continue

                                # Normal message sequence checks
                                if seq_num is not None:
                                    if seq_num > self.expected_seq_num:
                                        # Gap detected; request missing range and accept current message
                                        self._send_resend_request(self.expected_seq_num, seq_num - 1)
                                        # Advance expected to after this message; resends will arrive with lower seq and PossDupFlag
                                        self.expected_seq_num = seq_num + 1
                                        logger.warning(f"Inbound gap detected (got {seq_num}, expected {self.expected_seq_num - 1}); advanced expected to {self.expected_seq_num}")
                                    elif seq_num < self.expected_seq_num:
                                        # Duplicate or out-of-order; drop if marked poss dup
                                        if poss_dup == 'Y':
                                            logger.info(f"Dropping PossDup message seq={seq_num} (< expected {self.expected_seq_num})")
                                            continue
                                        else:
                                            logger.warning(f"Out-of-order message without PossDup seq={seq_num} (< expected {self.expected_seq_num})")
                                    else:
                                        # In-order
                                        self.expected_seq_num += 1

                            except Exception as seq_e:
                                logger.error(f"Sequence handling error: {seq_e}")

                            # Forward message through message handler
                            self.message_handler(parsed, self.session_type)
                            logger.debug(f"Processed message: {parsed.get('35', 'Unknown')}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                # Mark as disconnected to trigger supervisor
                self.connected = False
                self.authenticated = False
                break
                
        logger.info(f"{self.session_type} receiver thread stopped")
        
    def _parse_complete_message(self, raw_message: bytes) -> Optional[Dict[str, str]]:
        """Parse complete FIX message with proper validation."""
        try:
            # Convert to string and split by SOH
            message_str = raw_message.decode('utf-8', errors='ignore')
            fields = message_str.split('\x01')
            
            parsed = {}
            pairs = []
            for field in fields:
                if '=' in field:
                    tag, value = field.split('=', 1)
                    parsed[tag] = value
                    pairs.append((tag, value))
                    
            # Validate required fields
            if '8' not in parsed or '35' not in parsed:
                logger.warning("Invalid FIX message - missing required fields")
                return None
            # Attach ordered pairs for handlers that need group parsing (e.g., SecurityList)
            parsed['__pairs__'] = pairs
                
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
            # Join threads to avoid leaks
            try:
                if self.receiver_thread and self.receiver_thread.is_alive():
                    self.receiver_thread.join(timeout=2.0)
            except Exception:
                pass
            try:
                if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                    self.heartbeat_thread.join(timeout=2.0)
            except Exception:
                pass
            
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
        self.security_list_samples: List[Dict[str, str]] = []
        # Track cancel outcomes and last business reject text for diagnostics
        self.cancel_results: Dict[str, Dict[str, Any]] = {}
        self.last_business_reject: Optional[str] = None
        # Reconnection supervisor state
        self._reconnect_state: Dict[str, Dict[str, Any]] = {
            'quote': {'delay': 1.0, 'next': 0.0},
            'trade': {'delay': 1.0, 'next': 0.0},
        }
        self._supervisor_thread: Optional[threading.Thread] = None
        
    def _send_md_request(self, symbol: str, req_id: str, use_security_id: bool = False, include_idsource: bool = False) -> bool:
        """Build and send MarketDataRequest using 55 or 48 identification.
        When use_security_id is True, sends 48, and includes 22 only if include_idsource is True.
        """
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "V")  # MarketDataRequest
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "QUOTE")  # TargetSubID
            msg.append_pair(50, "QUOTE")  # SenderSubID
            msg.append_pair(262, req_id)  # MDReqID
            msg.append_pair(263, "1")     # Snapshot + Updates
            msg.append_pair(264, "0")     # Full Book
            # Related symbol group first (Mode A)
            msg.append_pair(146, "1")
            if use_security_id:
                try:
                    from src.operational.icmarkets_config import get_symbol_id
                    _sym = str(symbol)
                    if not _sym.isdigit():
                        _id = get_symbol_id(_sym)
                        if _id is not None:
                            _sym = str(_id)
                            logger.info(f"Mapped symbol {symbol} -> {_sym} for FIX tag 48 (MD)")
                    msg.append_pair(48, _sym)  # SecurityID
                    if include_idsource:
                        msg.append_pair(22, "8")  # Optional IDSource
                except Exception:
                    msg.append_pair(48, str(symbol))
                    if include_idsource:
                        msg.append_pair(22, "8")
            else:
                try:
                    from src.operational.icmarkets_config import get_symbol_id
                    _sym = str(symbol)
                    if not _sym.isdigit():
                        _id = get_symbol_id(_sym)
                        if _id is not None:
                            _sym = str(_id)
                            logger.info(f"Mapped symbol {symbol} -> {_sym} for FIX tag 55 (MD)")
                    msg.append_pair(55, _sym)  # Symbol numeric
                except Exception:
                    msg.append_pair(55, str(symbol))
            # Entry types
            msg.append_pair(267, "2")
            msg.append_pair(269, "0")
            msg.append_pair(269, "1")
            return self.price_connection.send_message_and_track(msg, req_id)
        except Exception as e:
            logger.error(f"Error building/sending MDReq: {e}")
            return False

    def _send_md_request_mode_b(self, symbol: str, req_id: str, use_security_id: bool = False) -> bool:
        """Alternative MDReq ordering (Mode B): entry types before related symbols."""
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "V")
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "QUOTE")
            msg.append_pair(50, "QUOTE")
            msg.append_pair(262, req_id)
            msg.append_pair(263, "1")
            msg.append_pair(264, "0")
            # Entry types first
            msg.append_pair(267, "2")
            msg.append_pair(269, "0")
            msg.append_pair(269, "1")
            # Then related symbols
            msg.append_pair(146, "1")
            if use_security_id:
                try:
                    from src.operational.icmarkets_config import get_symbol_id
                    _sym = str(symbol)
                    if not _sym.isdigit():
                        _id = get_symbol_id(_sym)
                        if _id is not None:
                            _sym = str(_id)
                            logger.info(f"Mapped symbol {symbol} -> {_sym} for 48 (MD, mode B)")
                    msg.append_pair(48, _sym)
                except Exception:
                    msg.append_pair(48, str(symbol))
            else:
                try:
                    from src.operational.icmarkets_config import get_symbol_id
                    _sym = str(symbol)
                    if not _sym.isdigit():
                        _id = get_symbol_id(_sym)
                        if _id is not None:
                            _sym = str(_id)
                            logger.info(f"Mapped symbol {symbol} -> {_sym} for 55 (MD, mode B)")
                    msg.append_pair(55, _sym)
                except Exception:
                    msg.append_pair(55, str(symbol))
            return self.price_connection.send_message_and_track(msg, req_id)
        except Exception as e:
            logger.error(f"Error building/sending MDReq (mode B): {e}")
            return False

    def _send_md_request_mode_c(self, symbol: str, req_id: str, use_security_id: bool = False) -> bool:
        """Minimal MDReq (Mode C): omit 265/267/269 to test venue-specific requirements."""
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "V")
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "QUOTE")
            msg.append_pair(50, "QUOTE")
            msg.append_pair(262, req_id)
            msg.append_pair(263, "1")
            msg.append_pair(264, "0")
            msg.append_pair(146, "1")
            if use_security_id:
                try:
                    from src.operational.icmarkets_config import get_symbol_id
                    _sym = str(symbol)
                    if not _sym.isdigit():
                        _id = get_symbol_id(_sym)
                        if _id is not None:
                            _sym = str(_id)
                    msg.append_pair(48, _sym)
                except Exception:
                    msg.append_pair(48, str(symbol))
            else:
                try:
                    from src.operational.icmarkets_config import get_symbol_id
                    _sym = str(symbol)
                    if not _sym.isdigit():
                        _id = get_symbol_id(_sym)
                        if _id is not None:
                            _sym = str(_id)
                    msg.append_pair(55, _sym)
                except Exception:
                    msg.append_pair(55, str(symbol))
            return self.price_connection.send_message_and_track(msg, req_id)
        except Exception as e:
            logger.error(f"Error building/sending MDReq (mode C): {e}")
            return False
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
                elif msg_type == 'y':  # SecurityList
                    self._handle_security_list(message)
                    
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
        """Handle Market Data Snapshot messages - parse repeating groups and populate book."""
        try:
            raw_symbol = message.get('55') or message.get('48')
            if not raw_symbol:
                logger.error("MarketDataSnapshot missing symbol identifier (55/48)")
                return

            mapped_symbol = raw_symbol
            try:
                if raw_symbol.isdigit():
                    from src.operational.icmarkets_config import get_symbol_name
                    name = get_symbol_name(int(raw_symbol))
                    if name:
                        mapped_symbol = name
            except Exception:
                pass

            if mapped_symbol not in self.order_books:
                self.order_books[mapped_symbol] = OrderBook(symbol=mapped_symbol)

            order_book = self.order_books[mapped_symbol]
            order_book.bids.clear()
            order_book.asks.clear()

            pairs = message.get('__pairs__') if isinstance(message.get('__pairs__'), list) else []
            entries: List[Dict[str, str]] = []
            current: Dict[str, str] = {}
            in_groups = False
            remaining = None
            for tag, value in pairs:
                if tag == '268':
                    try:
                        remaining = int(value)
                        in_groups = True
                        # reset any current
                        current = {}
                        continue
                    except Exception:
                        pass
                if not in_groups:
                    continue
                # Start a new group when a new 269 appears and current already has data
                if tag == '269' and current.get('269') is not None:
                    entries.append(current)
                    current = {}
                    if remaining is not None:
                        remaining -= 1
                # Collect known fields
                if tag in ('269','270','271','278','290','55','48'):
                    current[tag] = value
            # Push last
            if in_groups and current:
                entries.append(current)

            for e in entries:
                side = e.get('269')  # 0=bid,1=ask,2=trade
                px = e.get('270')
                sz = e.get('271')
                entry_id = e.get('278')
                position_no = e.get('290')
                if not side or not px:
                    continue
                price = float(px)
                size = float(sz) if sz else 0.0
                entry = MarketDataEntry(
                    symbol=mapped_symbol,
                    entry_type=side,
                    price=price,
                    size=size,
                    entry_id=entry_id,
                    position_no=int(position_no) if position_no and position_no.isdigit() else None,
                )
                if side == '0':
                    order_book.bids.append(entry)
                elif side == '1':
                    order_book.asks.append(entry)
                elif side == '2':
                    order_book.last_trade = entry

            # Sort the books
            order_book.bids.sort(key=lambda x: x.price, reverse=True)
            order_book.asks.sort(key=lambda x: x.price)
            order_book.last_update = datetime.utcnow()

            logger.info(f"✅ Market data snapshot processed for {mapped_symbol}: {len(order_book.bids)} bids, {len(order_book.asks)} asks")

            for callback in self.market_data_callbacks:
                try:
                    callback(mapped_symbol, order_book)
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
                # Start reconnection supervisor
                self._supervisor_thread = threading.Thread(target=self._supervise_sessions, daemon=True)
                self._supervisor_thread.start()
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
            # Map human-readable symbol to numeric id if available
            try:
                from src.operational.icmarkets_config import get_symbol_id
                _sym = str(symbol)
                if not _sym.isdigit():
                    _id = get_symbol_id(_sym)
                    if _id is not None:
                        _sym = str(_id)
                        logger.info(f"Mapped symbol {symbol} -> {_sym} for FIX tag 55")
                msg.append_pair(55, _sym)
            except Exception:
                msg.append_pair(55, str(symbol))
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
                # Attempt Mode A with 55
                success = self._send_md_request(symbol, req_id, use_security_id=False)
                if not success:
                    logger.error(f"❌ Failed to send market data request for {symbol}")
                    results[symbol] = False
                    continue
                    
                logger.info(f"📤 Market data request sent for {symbol}, waiting for broker response...")
                
                # CRITICAL: Wait for Market Data Snapshot or Reject
                start_time = time.time()
                subscription_confirmed = False
                
                while time.time() - start_time < timeout / 3:
                    # Check if we received market data for this symbol
                    if symbol in self.order_books:
                        order_book = self.order_books[symbol]
                        # Check if we have recent data
                        if (datetime.utcnow() - order_book.last_update).total_seconds() < timeout:
                            logger.info(f"✅ Market data subscription confirmed for {symbol}")
                            subscription_confirmed = True
                            break
                            
                    time.sleep(0.1)  # Small delay to avoid busy waiting
                # Fallback to SecurityID path (48 only, no 22)
                if not subscription_confirmed:
                    logger.info(f"Fallback MDReq (SecurityID) for {symbol}")
                    req_id2 = f"MD2_{symbol}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                    alt_ok = self._send_md_request(symbol, req_id2, use_security_id=True, include_idsource=False)
                    if alt_ok:
                        t2 = time.time()
                        while time.time() - t2 < timeout / 3:
                            if symbol in self.order_books:
                                order_book = self.order_books[symbol]
                                if (datetime.utcnow() - order_book.last_update).total_seconds() < timeout:
                                    logger.info(f"✅ Market data subscription confirmed (SecurityID) for {symbol}")
                                    subscription_confirmed = True
                                    break
                            time.sleep(0.1)
                # Final fallbacks
                if not subscription_confirmed:
                    logger.info(f"Final fallback MDReq (mode B, 55) for {symbol}")
                    req_id3 = f"MD3_{symbol}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                    alt2 = self._send_md_request_mode_b(symbol, req_id3, use_security_id=False)
                    if alt2:
                        t3 = time.time()
                        while time.time() - t3 < timeout / 3:
                            if symbol in self.order_books:
                                order_book = self.order_books[symbol]
                                if (datetime.utcnow() - order_book.last_update).total_seconds() < timeout:
                                    logger.info(f"✅ Market data subscription confirmed (mode B) for {symbol}")
                                    subscription_confirmed = True
                                    break
                            time.sleep(0.1)
                if not subscription_confirmed:
                    logger.info(f"Final fallback MDReq (mode C, minimal) for {symbol}")
                    req_id4 = f"MD4_{symbol}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                    alt3 = self._send_md_request_mode_c(symbol, req_id4, use_security_id=False)
                    if alt3:
                        t4 = time.time()
                        while time.time() - t4 < timeout / 3:
                            if symbol in self.order_books:
                                order_book = self.order_books[symbol]
                                if (datetime.utcnow() - order_book.last_update).total_seconds() < timeout:
                                    logger.info(f"✅ Market data subscription confirmed (mode C) for {symbol}")
                                    subscription_confirmed = True
                                    break
                            time.sleep(0.1)
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
        
    def send_order_status_request(self, cl_ord_id: str, side_hint: Optional[str] = None) -> bool:
        """Send Order Status Request (35=H) with minimal fields accepted by venue.
        - Includes 11 (OrigClOrdID) and 54 (Side). Omits 55 as venue rejects it.
        side_hint: '1' for Buy, '2' for Sell. If not provided, attempts to infer from tracked order.
        """
        if not self.trade_connection or not self.trade_connection.is_connected():
            logger.error("❌ Trade connection not available for status request")
            return False
        import simplefix
        try:
            if side_hint is None:
                ord_obj = self.orders.get(cl_ord_id)
                if ord_obj and ord_obj.side in ("1", "2"):
                    side_hint = ord_obj.side
            if side_hint not in ("1", "2"):
                # default to Buy if unknown; venue requires a side value
                side_hint = "1"
            stat_id = f"STAT_{cl_ord_id}"
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "H")
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "TRADE")
            msg.append_pair(50, "TRADE")
            msg.append_pair(11, cl_ord_id)
            msg.append_pair(54, side_hint)
            return self.trade_connection.send_message_and_track(msg, stat_id)
        except Exception as e:
            logger.error(f"Error sending OrderStatusRequest for {cl_ord_id}: {e}")
            return False

    def cancel_order_minimal(self,
                             orig_cl_ord_id: str,
                             order_id: Optional[str] = None,
                             wait_timeout: float = 5.0,
                             initial_delay: float = 0.3,
                             retry_on_order_not_found: bool = True) -> bool:
        """Send minimal OrderCancelRequest for an existing order and wait for cancel.
        - First attempt: 11/41 only per cTrader minimal schema
        - Retry: include 37 if broker responds ORDER_NOT_FOUND and we have order_id
        """
        if not self.trade_connection or not self.trade_connection.is_connected():
            logger.error("❌ Trade connection not available for cancel")
            return False

        try:
            # Gate on last known order state: only cancel if New or Pending New
            ord_obj = self.orders.get(orig_cl_ord_id)
            if ord_obj and ord_obj.status not in (OrderStatus.NEW, OrderStatus.PENDING_NEW):
                logger.info(f"Skip cancel for {orig_cl_ord_id} - not cancelable (status={ord_obj.status.value})")
                return False
            time.sleep(max(0.0, initial_delay))
            cncl_id_1 = f"CNCL_{orig_cl_ord_id}"
            msg1 = simplefix.FixMessage()
            msg1.append_pair(8, "FIX.4.4")
            msg1.append_pair(35, "F")
            msg1.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg1.append_pair(56, "cServer")
            msg1.append_pair(57, "TRADE")
            msg1.append_pair(50, "TRADE")
            msg1.append_pair(11, cncl_id_1)
            msg1.append_pair(41, orig_cl_ord_id)
            ok1 = self.trade_connection.send_message_and_track(msg1, cncl_id_1)
            if not ok1:
                logger.error(f"❌ Failed to send minimal cancel for {orig_cl_ord_id}")
                return False

            # Wait for cancel confirmation or business reject
            t0 = time.time()
            saw_order_not_found = False
            while time.time() - t0 < wait_timeout:
                order = self.orders.get(orig_cl_ord_id)
                if order and order.status == OrderStatus.CANCELED:
                    logger.info(f"✅ Order {orig_cl_ord_id} canceled")
                    return True
                if self.last_business_reject and "ORDER_NOT_FOUND" in str(self.last_business_reject):
                    saw_order_not_found = True
                    break
                time.sleep(0.1)

            # Conditional retry with 37
            if retry_on_order_not_found and saw_order_not_found and order_id:
                cncl_id_2 = f"CNCL2_{orig_cl_ord_id}"
                msg2 = simplefix.FixMessage()
                msg2.append_pair(8, "FIX.4.4")
                msg2.append_pair(35, "F")
                msg2.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
                msg2.append_pair(56, "cServer")
                msg2.append_pair(57, "TRADE")
                msg2.append_pair(50, "TRADE")
                msg2.append_pair(11, cncl_id_2)
                msg2.append_pair(41, orig_cl_ord_id)
                msg2.append_pair(37, order_id)
                ok2 = self.trade_connection.send_message_and_track(msg2, cncl_id_2)
                if not ok2:
                    logger.error(f"❌ Failed to send retry cancel (with 37) for {orig_cl_ord_id}")
                    return False
                t1 = time.time()
                while time.time() - t1 < wait_timeout:
                    order = self.orders.get(orig_cl_ord_id)
                    if order and order.status == OrderStatus.CANCELED:
                        logger.info(f"✅ Order {orig_cl_ord_id} canceled (retry with 37)")
                        return True
                    time.sleep(0.1)

            logger.error(f"⏰ Timeout waiting for cancel confirmation for {orig_cl_ord_id}")
            return False
        except Exception as e:
            logger.error(f"Error during cancel for {orig_cl_ord_id}: {e}")
            return False

    def cancel_all_tracked_orders(self,
                                  working_statuses: Optional[List[OrderStatus]] = None,
                                  per_order_delay: float = 0.2) -> Dict[str, bool]:
        """Cancel all currently tracked working orders.
        Returns mapping of cl_ord_id -> success flag.
        """
        if working_statuses is None:
            working_statuses = [
                OrderStatus.NEW,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.PENDING_NEW,
                OrderStatus.PENDING_CANCEL,
                OrderStatus.PENDING_REPLACE,
            ]
        results: Dict[str, bool] = {}
        # Snapshot to avoid mutation during iteration
        snapshot = list(self.orders.values())
        for oi in snapshot:
            if oi.status in working_statuses:
                success = self.cancel_order_minimal(
                    orig_cl_ord_id=oi.cl_ord_id,
                    order_id=oi.order_id,
                    initial_delay=per_order_delay,
                )
                results[oi.cl_ord_id] = success
                # small pacing between cancels
                time.sleep(per_order_delay)
        return results

    def _handle_order_cancel_reject(self, message: Dict[str, str]):
        """Handle OrderCancelReject messages."""
        try:
            cl_ord_id = message.get('11')
            text = message.get('58', 'No reason provided')
            ord_status = message.get('39')  # OrdStatus
            cancel_reject_reason = message.get('434')  # CxlRejReason
            logger.error(f"❌ Order cancel rejected for {cl_ord_id}: {text} (39={ord_status}, 434={cancel_reject_reason})")
        except Exception as e:
            logger.error(f"Error handling OrderCancelReject: {e}")
            
    def _handle_business_message_reject(self, message: Dict[str, str]):
        """Handle BusinessMessageReject messages."""
        try:
            ref_msg_type = message.get('372')
            text = message.get('58', 'No reason provided')
            logger.error(f"❌ Business message reject for message type {ref_msg_type}: {text}")
            self.last_business_reject = text
        except Exception as e:
            logger.error(f"Error handling BusinessMessageReject: {e}")
        
    def _handle_security_list(self, message: Dict[str, str]):
        """Handle SecurityList (MsgType=y) responses for symbol discovery."""
        try:
            # Minimal parsing: log availability and hint for MD identifiers
            num_sec = message.get('146') or message.get('393')  # count hint if available
            logger.info(f"Received SecurityList response (y); count hint={num_sec}")
            # Store sample for later analysis
            try:
                # Only store a small projection to avoid huge logs
                sample = {k: message.get(k) for k in ('55','48','22','207','460','393','146') if k in message}
                self.security_list_samples.append(sample)
                logger.info(f"SecurityList sample projection: {sample}")
            except Exception:
                pass
            # Heuristic: attempt to infer EURUSD numeric id from ordered pairs sequence
            pairs = message.get('__pairs__') if isinstance(message.get('__pairs__'), list) else []
            inferred_sym = None
            last_55 = None
            for tag, value in pairs:
                if tag == '55':
                    last_55 = value
                # Some venues echo full name in text; if we see EURUSD, bind recent 55
                if tag == '58' and value and 'EURUSD' in value.upper():
                    inferred_sym = last_55
                    break
            if inferred_sym and str(inferred_sym).isdigit():
                logger.info(f"Inferred EURUSD numeric id from SecurityList: {inferred_sym}")
            # Note: IC Markets demo often expects numeric Symbol(55) in MDReq; if rejected, try SecurityID(48) without 22.
        except Exception as e:
            logger.error(f"Error handling SecurityList: {e}")
            
    def _handle_market_data_incremental_refresh(self, message: Dict[str, str]):
        """Handle Market Data Incremental Refresh (35=X) with proper group parsing and updates."""
        try:
            msg_level_symbol = message.get('55') or message.get('48')
            mapped_symbol_level = msg_level_symbol
            try:
                if msg_level_symbol and msg_level_symbol.isdigit():
                    from src.operational.icmarkets_config import get_symbol_name
                    name = get_symbol_name(int(msg_level_symbol))
                    if name:
                        mapped_symbol_level = name
            except Exception:
                pass

            pairs = message.get('__pairs__') if isinstance(message.get('__pairs__'), list) else []
            entries: List[Dict[str, str]] = []
            current: Dict[str, str] = {}
            in_groups = False
            for tag, value in pairs:
                if tag == '268':
                    in_groups = True
                    current = {}
                    continue
                if not in_groups:
                    continue
                # New group on 279 (preferred) or 269 when 279 missing
                if (tag in ('279', '269')) and (current.get('279') is not None or current.get('269') is not None):
                    entries.append(current)
                    current = {}
                if tag in ('279','269','270','271','278','290','55','48'):
                    current[tag] = value
            if in_groups and current:
                entries.append(current)

            # Ensure order book exists when any entry resolves to a symbol
            def resolve_symbol(entry: Dict[str,str]) -> Optional[str]:
                s = entry.get('55') or entry.get('48') or mapped_symbol_level
                if not s:
                    return None
                try:
                    if s.isdigit():
                        from src.operational.icmarkets_config import get_symbol_name
                        name = get_symbol_name(int(s))
                        return name or s
                except Exception:
                    pass
                return s

            # Group-by symbol: IC Markets tends to send message-level symbol for X, but be robust
            symbol_to_book: Dict[str, OrderBook] = {}
            for e in entries:
                sym = resolve_symbol(e)
                if not sym:
                    # Skip entries without symbol context
                    continue
                if sym not in self.order_books:
                    self.order_books[sym] = OrderBook(symbol=sym)
                symbol_to_book[sym] = self.order_books[sym]

            for e in entries:
                sym = resolve_symbol(e)
                if not sym:
                    continue
                order_book = symbol_to_book[sym]
                action = e.get('279', '0')  # default New
                side = e.get('269')
                px = e.get('270')
                sz = e.get('271')
                entry_id = e.get('278')
                if side == '2':
                    # Trade
                    if px:
                        order_book.last_trade = MarketDataEntry(
                            symbol=sym,
                            entry_type=side,
                            price=float(px),
                            size=float(sz) if sz else 0.0,
                            entry_id=entry_id,
                        )
                    continue
                if not side or not px:
                    continue
                price = float(px)
                size = float(sz) if sz else 0.0
                if side == '0':
                    self._update_order_book_side(order_book.bids, action, price, size, sym, side, entry_id)
                elif side == '1':
                    self._update_order_book_side(order_book.asks, action, price, size, sym, side, entry_id)

            # Update timestamps and sort
            for ob in symbol_to_book.values():
                ob.last_update = datetime.utcnow()
                ob.bids.sort(key=lambda x: x.price, reverse=True)
                ob.asks.sort(key=lambda x: x.price)

            # Notify callbacks per message-level symbol if available, else per updated symbol
            notify_symbols = list(symbol_to_book.keys()) or ([mapped_symbol_level] if mapped_symbol_level else [])
            for sym in notify_symbols:
                ob = self.order_books.get(sym)
                if not ob:
                    continue
                for callback in self.market_data_callbacks:
                    try:
                        callback(sym, ob)
                    except Exception as e:
                        logger.error(f"Error in market data callback: {e}")

        except Exception as e:
            logger.error(f"Error handling MarketDataIncrementalRefresh: {e}")
            
    def _update_order_book_side(self, book_side: List[MarketDataEntry], update_action: str,
                                price: float, size: float, symbol: str, entry_type: str,
                                entry_id: Optional[str] = None):
        """Update order book side based on update action. Prefer entry_id matching, fallback to price."""
        try:
            if update_action == '0':  # New
                entry = MarketDataEntry(symbol=symbol, entry_type=entry_type, price=price, size=size, entry_id=entry_id)
                book_side.append(entry)
            elif update_action == '1':  # Change
                target = None
                if entry_id:
                    for e in book_side:
                        if e.entry_id == entry_id:
                            target = e
                            break
                if target is None:
                    for e in book_side:
                        if e.price == price:
                            target = e
                            break
                if target is not None:
                    target.size = size
                    target.entry_time = datetime.utcnow()
                else:
                    # If we cannot find it, treat as New to avoid losing state
                    book_side.append(MarketDataEntry(symbol=symbol, entry_type=entry_type, price=price, size=size, entry_id=entry_id))
            elif update_action == '2':  # Delete
                if entry_id:
                    book_side[:] = [e for e in book_side if e.entry_id != entry_id]
                else:
                    book_side[:] = [e for e in book_side if e.price != price]
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
        
        # Stop supervisor thread
        try:
            if self._supervisor_thread and self._supervisor_thread.is_alive():
                self._supervisor_thread.join(timeout=3.0)
        except Exception:
            pass

        try:
            if self.price_connection:
                self.price_connection.disconnect()
        except Exception:
            pass
        
        try:
            if self.trade_connection:
                self.trade_connection.disconnect()
        except Exception:
            pass
            
        logger.info("✅ Genuine FIX Manager stopped")

    def _is_session_connected(self, session: str) -> bool:
        if session == 'quote' and self.price_connection:
            return self.price_connection.is_connected()
        if session == 'trade' and self.trade_connection:
            return self.trade_connection.is_connected()
        return False

    def _attempt_reconnect(self, session: str) -> bool:
        try:
            if session == 'quote' and self.price_connection:
                ok = self.price_connection.connect()
            elif session == 'trade' and self.trade_connection:
                ok = self.trade_connection.connect()
            else:
                return False
            if ok:
                # Reset backoff on success
                self._reconnect_state[session]['delay'] = 1.0
                self._reconnect_state[session]['next'] = 0.0
                logger.info(f"🔁 Reconnected {session} session successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Reconnect attempt for {session} failed: {e}")
            return False

    def _supervisor_step(self) -> None:
        now = time.time()
        for session in ('quote', 'trade'):
            if not self.running:
                break
            if self._is_session_connected(session):
                continue
            state = self._reconnect_state[session]
            if now < state['next']:
                continue
            ok = self._attempt_reconnect(session)
            if not ok:
                # Exponential backoff up to 60s
                state['delay'] = min(60.0, state['delay'] * 2.0 if state['delay'] > 0 else 1.0)
                state['next'] = now + state['delay']
            else:
                state['delay'] = 1.0
                state['next'] = 0.0

    def _supervise_sessions(self) -> None:
        """Background loop to maintain sessions with exponential backoff reconnects."""
        while self.running:
            try:
                self._supervisor_step()
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"Supervisor error: {e}")
                time.sleep(2.0)

