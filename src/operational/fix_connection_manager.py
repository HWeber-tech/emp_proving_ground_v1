"""
FIX Connection Manager
Manages the lifecycle of Price and Trade FIX sessions
"""

import logging
import asyncio
import threading
import time
from typing import Dict, Any
import simplefix

from src.operational.fix_application import FIXApplication
from src.governance.system_config import SystemConfig

log = logging.getLogger(__name__)


class FIXConnectionManager:
    """
    Manages the lifecycle of Price and Trade FIX sessions.
    
    This class handles the initialization, starting, and stopping of FIX sessions
    for both price and trade connections to IC Markets cTrader.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the FIX connection manager.
        
        Args:
            config: System configuration containing FIX credentials
        """
        self.config = config
        self.applications: Dict[str, FIXApplication] = {}
        self.connections: Dict[str, Any] = {}
        self.running = False
        
    def start_sessions(self):
        """
        Start both price and trade FIX sessions.
        """
        log.info("Starting FIX sessions...")
        
        # Start price session
        self._start_session("price")
        
        # Start trade session
        self._start_session("trade")
        
        log.info("Both FIX sessions initiated. Check logs for logon confirmation.")
        
    def _start_session(self, session_type: str):
        """
        Start a specific FIX session.
        
        Args:
            session_type: Either 'price' or 'trade'
        """
        log.info(f"Starting {session_type} FIX session...")
        
        # Get credentials based on session type
        if session_type == "price":
            session_config = {
                'SenderCompID': self.config.fix_price_sender_comp_id,
                'Username': self.config.fix_price_username,
                'Password': self.config.fix_price_password,
                'TargetCompID': 'CSERVER',
                'TargetSubID': 'QUOTE',
                'SocketConnectHost': 'demo-uk-eqx-01.p.c-trader.com',
                'SocketConnectPort': '5211'
            }
        else:
            session_config = {
                'SenderCompID': self.config.fix_trade_sender_comp_id,
                'Username': self.config.fix_trade_username,
                'Password': self.config.fix_trade_password,
                'TargetCompID': 'CSERVER',
                'TargetSubID': 'TRADE',
                'SocketConnectHost': 'demo-uk-eqx-01.p.c-trader.com',
                'SocketConnectPort': '5212'
            }
        
        # Create FIX application
        app = FIXApplication(session_config, session_type)
        self.applications[session_type] = app
        
        # Create connection (simulated for now)
        connection = self._create_connection(session_config, app)
        self.connections[session_type] = connection
        
        # Start connection in background
        thread = threading.Thread(
            target=self._run_connection,
            args=(session_type, connection, app),
            daemon=True
        )
        thread.start()
        
    def _create_connection(self, config: Dict[str, str], app: FIXApplication):
        """
        Create a FIX connection (simulated for now).
        
        Args:
            config: Session configuration
            app: FIX application instance
            
        Returns:
            Connection object
        """
        return {
            'config': config,
            'app': app,
            'connected': False,
            'socket': None
        }
        
    def _run_connection(self, session_type: str, connection: Dict[str, Any], app: FIXApplication):
        """
        Run the FIX connection (simulated for now).
        
        Args:
            session_type: Type of connection
            connection: Connection configuration
            app: FIX application instance
        """
        try:
            config = connection['config']
            host = config['SocketConnectHost']
            port = int(config['SocketConnectPort'])
            
            log.info(f"Connecting to {host}:{port} for {session_type}...")
            
            # Simulate connection process
            time.sleep(2)  # Simulate connection delay
            
            # Simulate successful connection
            app.on_connect()
            
            # Simulate logon
            logon_msg = simplefix.FixMessage()
            logon_msg.append_pair(35, "A")  # Logon
            logon_msg.append_pair(49, config['SenderCompID'])
            logon_msg.append_pair(56, config['TargetCompID'])
            logon_msg.append_pair(98, "0")  # EncryptMethod = NONE
            logon_msg.append_pair(108, "30")  # HeartBtInt = 30
            logon_msg.append_pair(553, config['Username'])
            logon_msg.append_pair(554, config['Password'])
            
            log.info(f"Sending logon for {session_type}...")
            time.sleep(1)  # Simulate logon delay
            
            # Simulate successful logon
            app.on_logon()
            
            # Keep connection alive
            while self.running:
                time.sleep(30)  # Heartbeat interval
                if app.is_connected():
                    heartbeat_msg = simplefix.FixMessage()
                    heartbeat_msg.append_pair(35, "0")  # Heartbeat
                    log.debug(f"Sending heartbeat for {session_type}")
                    
        except Exception as e:
            log.error(f"Error in {session_type} connection: {e}")
            app.on_disconnect()
            
    def stop_sessions(self):
        """
        Stop all FIX sessions.
        """
        log.info("Stopping FIX sessions...")
        self.running = False
        
        for session_type, app in self.applications.items():
            log.info(f"Stopping {session_type} FIX session...")
            app.on_logout()
            
        log.info("All FIX sessions stopped.")
        
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get status of all connections.
        
        Returns:
            Dictionary containing connection status for each session type
        """
        status = {}
        for session_type, app in self.applications.items():
            status[session_type] = app.get_connection_status()
        return status
        
    def is_connected(self, session_type: str) -> bool:
        """
        Check if a specific session is connected.
        
        Args:
            session_type: Either 'price' or 'trade'
            
        Returns:
            True if connected, False otherwise
        """
        app = self.applications.get(session_type)
        return app.is_connected() if app else False
        
    def get_application(self, session_type: str) -> FIXApplication:
        """
        Get the FIX application for a specific session type.
        
        Args:
            session_type: Either 'price' or 'trade'
            
        Returns:
            FIXApplication instance
        """
        return self.applications.get(session_type)
