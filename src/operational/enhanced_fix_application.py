"""
Enhanced FIX Application with Sensory Organ Integration
Handles FIX protocol communication and routes market data to sensory organs
"""

import logging
import time
import socket
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

try:
    import quickfix as fix
    HAS_QUICKFIX = True
except ImportError:
    HAS_QUICKFIX = False
    import simplefix

log = logging.getLogger(__name__)


class EnhancedFIXApplication:
    """
    Enhanced FIX application that supports sensory organ integration.
    
    This application handles FIX protocol communication and routes market data
    messages to the FIXSensoryOrgan for processing.
    """
    
    def __init__(self, session_config: Dict[str, str], session_type: str, sensory_organ=None, broker_interface=None):
        """
        Initialize enhanced FIX application
        
        Args:
            session_config: Dictionary containing session configuration
            session_type: Either 'price' or 'trade'
            sensory_organ: FIXSensoryOrgan instance for market data processing
            broker_interface: FIXBrokerInterface instance for trade execution
        """
        self.session_config = session_config
        self.session_type = session_type
        self.sensory_organ = sensory_organ
        self.broker_interface = broker_interface
        self.session_id = None
        self.connected = False
        self.message_sequence = 1
        
        # Callbacks for message processing
        self.on_market_data_snapshot: Optional[Callable] = None
        self.on_market_data_incremental: Optional[Callable] = None
        self.on_market_data_reject: Optional[Callable] = None
        self.on_execution_report: Optional[Callable] = None
        
        # For simplefix implementation
        self.parser = simplefix.FixParser()
        self.fix_connection = None
        self.receive_thread = None
        self.running = False
        
    def set_sensory_organ(self, sensory_organ):
        """Set the sensory organ for market data processing."""
        self.sensory_organ = sensory_organ
        
    def start(self, host: str, port: int) -> bool:
        """
        Start FIX connection
        
        Args:
            host: FIX server host
            port: FIX server port
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            log.info(f"Starting {self.session_type} FIX connection to {host}:{port}")
            
            # Create socket connection
            self.fix_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.fix_connection.settimeout(30)
            self.fix_connection.connect((host, port))
            
            # Send Logon message
            self._send_logon()
            
            # Start receive thread
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_messages)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            self.connected = True
            log.info(f"SUCCESSFUL LOGON: {self.session_type} session")
            return True
            
        except Exception as e:
            log.error(f"Failed to start {self.session_type} FIX connection: {e}")
            return False
    
    def stop(self):
        """Stop FIX connection"""
        self.running = False
        self.connected = False
        
        if self.fix_connection:
            try:
                self._send_logout()
                self.fix_connection.close()
            except:
                pass
            self.fix_connection = None
            
        if self.receive_thread:
            self.receive_thread.join(timeout=5)
            
        log.info(f"Session Logged Out: {self.session_type}")
    
    def send_message(self, msg: simplefix.FixMessage) -> bool:
        """Send a FIX message."""
        if self.fix_connection and self.connected:
            try:
                data = msg.encode()
                self.fix_connection.send(data)
                log.debug(f"Sending {self.session_type}: {msg}")
                return True
            except Exception as e:
                log.error(f"Failed to send {self.session_type} message: {e}")
                return False
        return False
    
    def _send_logon(self):
        """Send FIX Logon message"""
        msg = simplefix.FixMessage()
        msg.append_pair(8, "FIX.4.4")  # BeginString
        msg.append_pair(35, "A")  # MsgType = Logon
        msg.append_pair(49, self.session_config['SenderCompID'])  # SenderCompID
        msg.append_pair(56, "CSERVER")  # TargetCompID
        msg.append_pair(34, self.message_sequence)  # MsgSeqNum
        msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))  # SendingTime
        msg.append_pair(98, 0)  # EncryptMethod = None
        msg.append_pair(108, 30)  # HeartBtInt = 30 seconds
        msg.append_pair(553, self.session_config['Username'])  # Username
        msg.append_pair(554, self.session_config['Password'])  # Password
        
        if self.session_type == 'price':
            msg.append_pair(57, "QUOTE")  # TargetSubID
        else:
            msg.append_pair(57, "TRADE")  # TargetSubID
            
        self.send_message(msg)
        self.message_sequence += 1
        
    def _send_logout(self):
        """Send FIX Logout message"""
        msg = simplefix.FixMessage()
        msg.append_pair(8, "FIX.4.4")
        msg.append_pair(35, "5")  # MsgType = Logout
        msg.append_pair(49, self.session_config['SenderCompID'])
        msg.append_pair(56, "CSERVER")
        msg.append_pair(34, self.message_sequence)
        msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))
        
        if self.session_type == 'price':
            msg.append_pair(57, "QUOTE")
        else:
            msg.append_pair(57, "TRADE")
            
        self.send_message(msg)
        self.message_sequence += 1
    
    def _receive_messages(self):
        """Receive and process FIX messages"""
        buffer = b''
        
        while self.running:
            try:
                data = self.fix_connection.recv(4096)
                if not data:
                    break
                    
                buffer += data
                
                # Process complete messages
                while b'\x01' in buffer:
                    msg_end = buffer.find(b'\x01') + 1
                    msg_data = buffer[:msg_end]
                    buffer = buffer[msg_end:]
                    
                    try:
                        msg = simplefix.FixMessage()
                        msg.decode(msg_data)
                        self._process_message(msg)
                    except Exception as e:
                        log.error(f"Error processing {self.session_type} message: {e}")
                        
            except socket.timeout:
                continue
            except Exception as e:
                log.error(f"Error receiving {self.session_type} messages: {e}")
                break
    
    def _process_message(self, msg: simplefix.FixMessage):
        """Process incoming FIX message"""
        msg_type = msg.get(35)
        
        if msg_type == b"A":  # Logon
            log.info(f"Received Logon confirmation for {self.session_type}")
        elif msg_type == b"0":  # Heartbeat
            log.debug(f"Received Heartbeat for {self.session_type}")
        elif msg_type == b"5":  # Logout
            log.info(f"Received Logout for {self.session_type}")
        elif msg_type == b"W":  # MarketDataSnapshotFullRefresh
            log.debug(f"Received MarketDataSnapshotFullRefresh for {self.session_type}")
            if self.sensory_organ:
                asyncio.create_task(self.sensory_organ.process_market_data_snapshot(msg))
        elif msg_type == b"X":  # MarketDataIncrementalRefresh
            log.debug(f"Received MarketDataIncrementalRefresh for {self.session_type}")
            if self.sensory_organ:
                asyncio.create_task(self.sensory_organ.process_market_data_incremental(msg))
        elif msg_type == b"Y":  # MarketDataRequestReject
            log.warning(f"Received MarketDataRequestReject for {self.session_type}")
            if self.sensory_organ:
                asyncio.create_task(self.sensory_organ.process_market_data_reject(msg))
        elif msg_type == b"8":  # Execution Report
            log.info(f"Received Execution Report for {self.session_type}: {msg}")
        else:
            log.info(f"Received {self.session_type} message type {msg_type}: {msg}")
    
    def send_heartbeat(self):
        """Send heartbeat message"""
        if self.connected:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "0")  # MsgType = Heartbeat
            msg.append_pair(49, self.session_config['SenderCompID'])
            msg.append_pair(56, "CSERVER")
            msg.append_pair(34, self.message_sequence)
            msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))
            
            if self.session_type == 'price':
                msg.append_pair(57, "QUOTE")
            else:
                msg.append_pair(57, "TRADE")
                
            self.send_message(msg)
            self.message_sequence += 1
    
    def is_connected(self) -> bool:
        """Check if session is connected"""
        return self.connected


# QuickFIX compatibility layer for when quickfix is available
if HAS_QUICKFIX:
    class QuickFIXApplication(fix.Application):
        """QuickFIX Application wrapper for compatibility"""
        
        def __init__(self, session_config: Dict[str, str]):
            super().__init__()
            self.session_config = session_config
            self.session_id = None
            
        def onCreate(self, sessionID):
            self.session_id = sessionID
            log.info(f"FIX Session Created: {sessionID}")
            
        def onLogon(self, sessionID):
            log.info(f"SUCCESSFUL LOGON: {sessionID}")
            
        def onLogout(self, sessionID):
            log.info(f"Session Logged Out: {sessionID}")
            
        def toAdmin(self, message, sessionID):
            msg_type = message.getHeader().getField(fix.MsgType())
            if msg_type == fix.MsgType_Logon:
                message.setField(fix.Username(self.session_config['Username']))
                message.setField(fix.Password(self.session_config['Password']))
            log.debug(f"Sending Admin: {message}")
            
        def fromAdmin(self, message, sessionID):
            log.debug(f"Received Admin: {message}")
            
        def toApp(self, message, sessionID):
            log.debug(f"Sending App: {message}")
            
        def fromApp(self, message, sessionID):
            log.info(f"Received App: {message}")
