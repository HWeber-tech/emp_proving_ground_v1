"""
FIX Connection Manager
Manages the lifecycle of Price and Trade FIX sessions
"""

import logging
import time
import threading
from typing import Dict, Any, Optional
from src.operational.fix_application import FIXApplication
from src.governance.system_config import SystemConfig

log = logging.getLogger(__name__)


class FIXConnectionManager:
    """Manages the lifecycle of Price and Trade FIX sessions"""

    def __init__(self, config: SystemConfig):
        """
        Initialize FIX Connection Manager
        
        Args:
            config: System configuration instance
        """
        self.config = config
        self.price_app: Optional[FIXApplication] = None
        self.trade_app: Optional[FIXApplication] = None
        self.heartbeat_thread = None
        self.running = False
        
    def start_sessions(self) -> bool:
        """
        Start both Price and Trade FIX sessions
        
        Returns:
            True if both sessions started successfully, False otherwise
        """
        log.info("Starting FIX sessions...")
        
        # Start Price session
        price_success = self._start_price_session()
        if not price_success:
            log.error("Failed to start Price FIX session")
            return False
            
        # Start Trade session
        trade_success = self._start_trade_session()
        if not trade_success:
            log.error("Failed to start Trade FIX session")
            self.stop_sessions()
            return False
            
        # Start heartbeat thread
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
        log.info("Both FIX sessions initiated successfully")
        return True
    
    def _start_price_session(self) -> bool:
        """Start Price FIX session"""
        log.info("Starting Price FIX session...")
        
        # Prepare price session configuration
        price_config = {
            'SenderCompID': self.config.fix_price_sender_comp_id,
            'Username': self.config.fix_price_username,
            'Password': self.config.fix_price_password
        }
        
        # Validate configuration
        if not all(price_config.values()):
            log.error("Missing Price FIX credentials in configuration")
            return False
            
        # Create and start price application
        self.price_app = FIXApplication(price_config, 'price')
        return self.price_app.start('demo-uk-eqx-01.p.c-trader.com', 5211)
    
    def _start_trade_session(self) -> bool:
        """Start Trade FIX session"""
        log.info("Starting Trade FIX session...")
        
        # Prepare trade session configuration
        trade_config = {
            'SenderCompID': self.config.fix_trade_sender_comp_id,
            'Username': self.config.fix_trade_username,
            'Password': self.config.fix_trade_password
        }
        
        # Validate configuration
        if not all(trade_config.values()):
            log.error("Missing Trade FIX credentials in configuration")
            return False
            
        # Create and start trade application
        self.trade_app = FIXApplication(trade_config, 'trade')
        return self.trade_app.start('demo-uk-eqx-01.p.c-trader.com', 5212)
    
    def stop_sessions(self):
        """Stop all FIX sessions"""
        log.info("Stopping FIX sessions...")
        
        self.running = False
        
        # Stop heartbeat thread
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
            self.heartbeat_thread = None
            
        # Stop price session
        if self.price_app:
            self.price_app.stop()
            self.price_app = None
            
        # Stop trade session
        if self.trade_app:
            self.trade_app.stop()
            self.trade_app = None
            
        log.info("All FIX sessions stopped")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to keep connections alive"""
        while self.running:
            try:
                if self.price_app and self.price_app.is_connected():
                    self.price_app.send_heartbeat()
                    
                if self.trade_app and self.trade_app.is_connected():
                    self.trade_app.send_heartbeat()
                    
                time.sleep(30)  # Send heartbeat every 30 seconds
                
            except Exception as e:
                log.error(f"Error in heartbeat loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def get_connection_status(self) -> Dict[str, bool]:
        """
        Get current connection status
        
        Returns:
            Dictionary with connection status for each session
        """
        return {
            'price_connected': self.price_app.is_connected() if self.price_app else False,
            'trade_connected': self.trade_app.is_connected() if self.trade_app else False
        }
    
    def is_ready(self) -> bool:
        """
        Check if both sessions are connected and ready
        
        Returns:
            True if both sessions are connected, False otherwise
        """
        status = self.get_connection_status()
        return status['price_connected'] and status['trade_connected']
    
    def wait_for_connections(self, timeout: int = 30) -> bool:
        """
        Wait for both sessions to connect
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if both sessions connected within timeout, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_ready():
                return True
            time.sleep(1)
            
        return False
