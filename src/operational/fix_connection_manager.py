"""
FIX Connection Manager for IC Markets
Manages FIX connections and sessions
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import simplefix

from config.fix.icmarkets_config import ICMarketsConfig

logger = logging.getLogger(__name__)


class FIXApplication:
    """Base FIX application for handling messages."""
    
    def __init__(self, session_type: str):
        self.session_type = session_type
        self.message_queue = None
        self.connected = False
        
    def set_message_queue(self, queue):
        """Set the message queue for async communication."""
        self.message_queue = queue
        
    def on_message(self, message):
        """Handle incoming FIX messages."""
        if self.message_queue:
            # Put message in queue for async processing
            asyncio.create_task(self.message_queue.put(message))
            
    def on_connect(self):
        """Handle connection establishment."""
        self.connected = True
        logger.info(f"{self.session_type} session connected")
        
    def on_disconnect(self):
        """Handle connection loss."""
        self.connected = False
        logger.info(f"{self.session_type} session disconnected")


class FIXInitiator:
    """FIX initiator for sending messages."""
    
    def __init__(self, config: ICMarketsConfig, session_type: str):
        self.config = config
        self.session_type = session_type
        self.connected = False
        
    def send_message(self, message: simplefix.FixMessage):
        """Send a FIX message."""
        if self.connected:
            logger.info(f"Sending {self.session_type} message: {message}")
            # In a real implementation, this would send via socket
            return True
        else:
            logger.error(f"Cannot send message - {self.session_type} not connected")
            return False
            
    def connect(self):
        """Connect to the FIX server."""
        self.connected = True
        logger.info(f"{self.session_type} initiator connected")
        
    def disconnect(self):
        """Disconnect from the FIX server."""
        self.connected = False
        logger.info(f"{self.session_type} initiator disconnected")


class FIXConnectionManager:
    """Manages FIX connections and sessions."""
    
    def __init__(self, config: ICMarketsConfig):
        self.config = config
        self.applications = {}
        self.initiators = {}
        self.running = False
        
    def start_sessions(self):
        """Start all FIX sessions."""
        try:
            logger.info("Starting FIX sessions...")
            
            # Create price session
            self.applications['price'] = FIXApplication('price')
            self.initiators['price'] = FIXInitiator(self.config, 'price')
            
            # Create trade session
            self.applications['trade'] = FIXApplication('trade')
            self.initiators['trade'] = FIXInitiator(self.config, 'trade')
            
            # Connect sessions
            for session_type in ['price', 'trade']:
                self.initiators[session_type].connect()
                self.applications[session_type].on_connect()
                
            self.running = True
            logger.info("All FIX sessions started successfully")
            
        except Exception as e:
            logger.error(f"Error starting FIX sessions: {e}")
            raise
            
    def stop_sessions(self):
        """Stop all FIX sessions."""
        try:
            logger.info("Stopping FIX sessions...")
            
            for session_type in ['price', 'trade']:
                if session_type in self.initiators:
                    self.initiators[session_type].disconnect()
                if session_type in self.applications:
                    self.applications[session_type].on_disconnect()
                    
            self.running = False
            logger.info("All FIX sessions stopped")
            
        except Exception as e:
            logger.error(f"Error stopping FIX sessions: {e}")
            
    def get_application(self, session_type: str) -> Optional[FIXApplication]:
        """Get application for a session type."""
        return self.applications.get(session_type)
        
    def get_initiator(self, session_type: str) -> Optional[FIXInitiator]:
        """Get initiator for a session type."""
        return self.initiators.get(session_type)
        
    def is_running(self) -> bool:
        """Check if sessions are running."""
        return self.running
        
    def get_status(self) -> Dict[str, Any]:
        """Get connection status."""
        return {
            "running": self.running,
            "sessions": {
                session_type: {
                    "application": app.connected if app else False,
                    "initiator": init.connected if init else False
                }
                for session_type, (app, init) in {
                    'price': (self.applications.get('price'), self.initiators.get('price')),
                    'trade': (self.applications.get('trade'), self.initiators.get('trade'))
                }.items()
            }
        }
