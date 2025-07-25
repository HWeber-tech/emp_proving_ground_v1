"""
Robust IC Markets FIX Application
Production-ready implementation with error handling and session management
"""

import socket
import ssl
import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import simplefix
import json
import os
from concurrent.futures import ThreadPoolExecutor
import queue

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


class RobustMessageParser:
    """Robust FIX message parser with error handling."""
    
    @staticmethod
    def parse_fix_message(raw_message: bytes) -> Optional[Dict[str, str]]:
        """Parse FIX message with comprehensive error handling."""
        try:
            if not raw_message:
                return None
                
            # Use proper buffer-based parsing
            parser = simplefix.FixParser()
            parser.append_buffer(raw_message)
            message = parser.get_message()
            
            if not message:
                return None
                
            # Convert to dictionary
            parsed = {}
            for tag in [8, 9, 35, 49, 56, 34, 52, 262, 55, 270, 271, 11, 39, 150, 151]:
                value = message.get(tag)
                if value:
                    parsed[str(tag)] = value.decode() if isinstance(value, bytes) else str(value)
                    
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing FIX message: {e}")
            return None


class ConnectionRecoveryManager:
    """Manages connection recovery and health monitoring."""
    
    def __init__(self, max_retries: int = 5, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_attempts = 0
        self.last_success = None
        
    def should_retry(self) -> bool:
        """Determine if connection should be retried."""
        return self.connection_attempts < self.max_retries
        
    def get_retry_delay(self) -> float:
        """Calculate exponential backoff delay."""
        return self.retry_delay * (2 ** self.connection_attempts)
        
    def record_success(self):
        """Record successful connection."""
        self.connection_attempts = 0
        self.last_success = datetime.utcnow()
        
    def record_failure(self):
        """Record connection failure."""
        self.connection_attempts += 1


class ICMarketsRobustConnection:
    """Robust IC Markets connection with error handling and recovery."""
    
    def __init__(self, config, session_type: str):
        self.config = config
        self.session_type = session_type
        self.socket = None
        self.ssl_socket = None
        self.connected = False
        self.sequence_number = 1
        self.recovery_manager = ConnectionRecoveryManager()
        self.message_queue = queue.Queue()
        self.running = False
        self.receiver_thread = None
        self.heartbeat_thread = None
        self.last_heartbeat = None
        
    def connect(self) -> bool:
        """Connect with automatic retry and recovery."""
        while self.recovery_manager.should_retry():
            try:
                logger.info(f"Attempting {self.session_type} connection (attempt {self.recovery_manager.connection_attempts + 1})")
                
                # Create SSL context
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
                # Create socket
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10)
                
                # Wrap with SSL
                self.ssl_socket = context.wrap_socket(
                    self.socket, 
                    server_hostname=self.config._get_host()
                )
                
                # Connect
                self.ssl_socket.connect((self.config._get_host(), self.config._get_port(self.session_type)))
                
                # Send logon
                if self._send_logon():
                    self.connected = True
                    self.recovery_manager.record_success()
                    self.running = True
                    
                    # Start receiver thread
                    self.receiver_thread = threading.Thread(target=self._receive_messages, daemon=True)
                    self.receiver_thread.start()
                    
                    # Start heartbeat thread
                    self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
                    self.heartbeat_thread.start()
                    
                    logger.info(f"{self.session_type} connection established successfully")
                    return True
                    
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self.recovery_manager.record_failure()
                
                if self.recovery_manager.should_retry():
                    delay = self.recovery_manager.get_retry_delay()
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries exceeded for {self.session_type}")
                    return False
                    
        return False
        
    def _send_logon(self) -> bool:
        """Send logon message with proper formatting."""
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "A")
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "QUOTE" if self.session_type == "price" else "TRADE")
            msg.append_pair(34, self.sequence_number)
            msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
            msg.append_pair(98, 0)
            msg.append_pair(108, 30)
            msg.append_pair(553, self.config.account_number)
            msg.append_pair(554, self.config.password)
            
            message_str = msg.encode()
            self.ssl_socket.send(message_str)
            self.sequence_number += 1
            
            # Wait for response
            response = self.ssl_socket.recv(1024)
            if response and b"35=A" in response:
                self.last_heartbeat = datetime.utcnow()
                return True
                
        except Exception as e:
            logger.error(f"Logon failed: {e}")
            return False
            
    def _send_heartbeat(self):
        """Send heartbeat message."""
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "0")
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "QUOTE" if self.session_type == "price" else "TRADE")
            msg.append_pair(34, self.sequence_number)
            msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
            
            message_str = msg.encode()
            self.ssl_socket.send(message_str)
            self.sequence_number += 1
            self.last_heartbeat = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            self.connected = False
            
    def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self.running and self.connected:
            try:
                time.sleep(30)  # 30-second heartbeat
                if self.connected:
                    self._send_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                self.connected = False
                
    def _receive_messages(self):
        """Receive messages with error handling."""
        while self.running:
            try:
                if not self.connected:
                    break
                    
                data = self.ssl_socket.recv(4096)
                if data:
                    parsed = RobustMessageParser.parse_fix_message(data)
                    if parsed:
                        self.message_queue.put(parsed)
                        
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                self.connected = False
                break
                
    def send_message(self, message: simplefix.FixMessage) -> bool:
        """Send message with error handling."""
        try:
            if not self.connected:
                return False
                
            message_str = message.encode()
            self.ssl_socket.send(message_str)
            self.sequence_number += 1
            return True
            
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Graceful disconnection."""
        self.running = False
        self.connected = False
        
        if self.ssl_socket:
            try:
                # Send logout
                msg = simplefix.FixMessage()
                msg.append_pair(8, "FIX.4.4")
                msg.append_pair(35, "5")
                msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
                msg.append_pair(56, "cServer")
                msg.append_pair(57, "QUOTE" if self.session_type == "price" else "TRADE")
                msg.append_pair(34, self.sequence_number)
                msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
                
                self.ssl_socket.send(msg.encode())
            except:
                pass
                
            self.ssl_socket.close()
            
    def is_connected(self) -> bool:
        """Check connection status."""
        return self.connected and self.running
        
    def get_messages(self) -> List[Dict[str, str]]:
        """Get received messages."""
        messages = []
        try:
            while True:
                messages.append(self.message_queue.get_nowait())
        except queue.Empty:
            pass
        return messages


class ICMarketsRobustManager:
    """Robust IC Markets manager with full automation."""
    
    def __init__(self, config):
        self.config = config
        self.price_connection = None
        self.trade_connection = None
        self.market_data: Dict[str, MarketDataEntry] = {}
        self.orders: Dict[str, OrderStatus] = {}
        self.running = False
        
    def start(self) -> bool:
        """Start all connections with full automation."""
        try:
            self.config.validate_config()
            
            # Create connections
            self.price_connection = ICMarketsRobustConnection(self.config, "price")
            self.trade_connection = ICMarketsRobustConnection(self.config, "trade")
            
            # Start connections
            price_success = self.price_connection.connect()
            trade_success = self.trade_connection.connect()
            
            if price_success and trade_success:
                self.running = True
                logger.info("IC Markets robust connections established")
                return True
            else:
                logger.error("Failed to establish robust connections")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start robust manager: {e}")
            return False
            
    def subscribe_market_data(self, symbols: List[str]) -> bool:
        """Subscribe to market data with error handling."""
        if not self.price_connection or not self.price_connection.is_connected():
            return False
            
        try:
            for symbol in symbols:
                msg = simplefix.FixMessage()
                msg.append_pair(8, "FIX.4.4")
                msg.append_pair(35, "V")
                msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
                msg.append_pair(56, "cServer")
                msg.append_pair(57, "QUOTE")
                msg.append_pair(34, 1)
                msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
                msg.append_pair(262, f"MD_{symbol}_{int(time.time())}")
                msg.append_pair(263, "1")
                msg.append_pair(264, "0")
                msg.append_pair(146, "1")
                msg.append_pair(55, symbol)
                
                self.price_connection.send_message(msg)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {e}")
            return False
            
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        """Place market order with error handling."""
        if not self.trade_connection or not self.trade_connection.is_connected():
            return None
            
        try:
            cl_ord_id = f"ORDER_{int(time.time() * 1000)}"
            
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "D")
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "TRADE")
            msg.append_pair(34, 1)
            msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
            msg.append_pair(11, cl_ord_id)
            msg.append_pair(55, symbol)
            msg.append_pair(54, "1" if side.upper() == "BUY" else "2")
            msg.append_pair(38, str(quantity))
            msg.append_pair(40, "1")
            msg.append_pair(60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
            
            if self.trade_connection.send_message(msg):
                return cl_ord_id
                
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None
            
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'price_connected': self.price_connection.is_connected() if self.price_connection else False,
            'trade_connected': self.trade_connection.is_connected() if self.trade_connection else False,
            'running': self.running,
            'market_data_count': len(self.market_data),
            'order_count': len(self.orders)
        }
        
    def stop(self):
        """Stop all connections gracefully."""
        self.running = False
        
        if self.price_connection:
            self.price_connection.disconnect()
            
        if self.trade_connection:
            self.trade_connection.disconnect()
            
        logger.info("IC Markets robust manager stopped")
