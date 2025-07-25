"""
FIX Application Module
Handles FIX protocol communication with IC Markets cTrader
"""

import logging
import simplefix
from typing import Dict, Any, Optional
from datetime import datetime

log = logging.getLogger(__name__)


class FIXApplication:
    """
    The core FIX Application for handling session events with IC Markets cTrader.
    
    This class manages the FIX protocol communication, handling logon, logout,
    and message processing for both price and trade connections.
    """
    
    def __init__(self, session_config: Dict[str, str], connection_type: str):
        """
        Initialize the FIX application.
        
        Args:
            session_config: Configuration containing credentials and settings
            connection_type: Either 'price' or 'trade'
        """
        self.session_config = session_config
        self.connection_type = connection_type
        self.connected = False
        self.message_queue = []
        self.last_heartbeat = None
        
    def on_connect(self):
        """Called when connection is established."""
        log.info(f"FIX {self.connection_type} connection established")
        
    def on_disconnect(self):
        """Called when connection is lost."""
        log.info(f"FIX {self.connection_type} connection lost")
        self.connected = False
        
    def on_logon(self):
        """Called when logon is successful."""
        log.info(f"SUCCESSFUL LOGON: FIX {self.connection_type} session")
        self.connected = True
        self.last_heartbeat = datetime.utcnow()
        
    def on_logout(self):
        """Called when logout occurs."""
        log.info(f"Session logged out: FIX {self.connection_type}")
        self.connected = False
        
    def on_message(self, message: simplefix.FixMessage):
        """
        Process incoming FIX messages.
        
        Args:
            message: The received FIX message
        """
        try:
            msg_type = message.get(35)
            if msg_type:
                msg_type_str = msg_type.decode() if isinstance(msg_type, bytes) else str(msg_type)
                
                if msg_type_str == "0":  # Heartbeat
                    self.last_heartbeat = datetime.utcnow()
                    log.debug(f"Heartbeat received for {self.connection_type}")
                elif msg_type_str == "A":  # Logon
                    self.on_logon()
                elif msg_type_str == "5":  # Logout
                    self.on_logout()
                elif msg_type_str == "W":  # MarketDataSnapshotFullRefresh
                    log.debug(f"Market data snapshot received for {self.connection_type}")
                    self.message_queue.append(message)
                elif msg_type_str == "X":  # MarketDataIncrementalRefresh
                    log.debug(f"Market data incremental received for {self.connection_type}")
                    self.message_queue.append(message)
                elif msg_type_str == "Y":  # MarketDataRequestReject
                    log.warning(f"Market data request rejected for {self.connection_type}")
                    self.message_queue.append(message)
                elif msg_type_str == "8":  # ExecutionReport
                    log.debug(f"Execution report received for {self.connection_type}")
                    self.message_queue.append(message)
                else:
                    log.debug(f"Received message type {msg_type_str} for {self.connection_type}")
                    
        except Exception as e:
            log.error(f"Error processing message for {self.connection_type}: {e}")
            
    def send_message(self, message: simplefix.FixMessage) -> bool:
        """
        Send a FIX message.
        
        Args:
            message: The FIX message to send
            
        Returns:
            True if message was queued for sending, False otherwise
        """
        try:
            # In a real implementation, this would send via the FIX connection
            log.debug(f"Queuing message for {self.connection_type}: {message}")
            return True
        except Exception as e:
            log.error(f"Error sending message for {self.connection_type}: {e}")
            return False
            
    def get_next_message(self) -> Optional[simplefix.FixMessage]:
        """
        Get the next message from the queue.
        
        Returns:
            The next FIX message or None if queue is empty
        """
        if self.message_queue:
            return self.message_queue.pop(0)
        return None
        
    def is_connected(self) -> bool:
        """Check if the session is connected."""
        return self.connected
        
    def set_message_queue(self, queue):
        """Set the external message queue for thread-safe communication."""
        self.message_queue = queue
        log.info(f"Message queue set for {self.connection_type} connection")
        
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            'connected': self.connected,
            'connection_type': self.connection_type,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'message_queue_size': len(self.message_queue) if hasattr(self, 'message_queue') else 0
        }
