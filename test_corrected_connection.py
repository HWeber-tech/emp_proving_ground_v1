#!/usr/bin/env python3
"""
Test Corrected IC Markets FIX Connection
Using correct host, port, and configuration from documentation
"""

import sys
import time
import logging
import socket
import ssl
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.operational.corrected_fix_config import CorrectedFIXConfig
import simplefix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorrectedFIXConnection:
    """Test connection using corrected configuration."""
    
    def __init__(self, config_dict, session_type):
        self.config = config_dict
        self.session_type = session_type
        self.socket = None
        self.seq_num = 1
        self.connected = False
        self.authenticated = False
        
    def connect(self):
        """Connect to IC Markets using corrected configuration."""
        try:
            logger.info(f"Connecting to {self.session_type} session...")
            logger.info(f"Host: {self.config['host']}")
            logger.info(f"Port: {self.config['port']}")
            logger.info(f"SSL: {self.config['use_ssl']}")
            
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Apply SSL if required
            if self.config['use_ssl']:
                context = ssl.create_default_context()
                self.socket = context.wrap_socket(self.socket, server_hostname=self.config['host'])
                logger.info("SSL context applied")
            
            # Connect
            self.socket.connect((self.config['host'], self.config['port']))
            self.connected = True
            logger.info(f"✅ Connected to {self.config['host']}:{self.config['port']}")
            
            # Send logon
            return self.send_logon()
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.connected = False
            return False
            
    def send_logon(self):
        """Send logon message with corrected format."""
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, self.config['fix_version'])  # BeginString
            msg.append_pair(35, 'A')  # MsgType = Logon
            msg.append_pair(49, self.config['sender_comp_id'])  # SenderCompID
            msg.append_pair(56, self.config['target_comp_id'])  # TargetCompID
            msg.append_pair(57, self.config['target_sub_id'])   # TargetSubID
            msg.append_pair(50, self.config['sender_sub_id'])   # SenderSubID
            msg.append_pair(34, self.seq_num)  # MsgSeqNum
            msg.append_pair(52, time.strftime('%Y%m%d-%H:%M:%S', time.gmtime()))  # SendingTime
            msg.append_pair(98, 0)  # EncryptMethod = None
            msg.append_pair(108, self.config['heartbeat_interval'])  # HeartBtInt
            msg.append_pair(553, self.config['username'])  # Username
            msg.append_pair(554, self.config['password'])  # Password
            
            # Send message
            message_bytes = msg.encode()
            logger.info(f"Sending logon message: {message_bytes}")
            self.socket.sendall(message_bytes)
            self.seq_num += 1
            
            # Wait for response
            response = self.socket.recv(1024)
            logger.info(f"Received logon response: {response}")
            
            # Parse response
            response_str = response.decode('utf-8', errors='ignore')
            logger.info(f"Logon response (readable): {response_str.replace(chr(1), '|')}")
            
            # Check if logon was accepted
            if b'35=A' in response:  # Logon response
                logger.info("✅ Logon accepted by server")
                self.authenticated = True
                return True
            else:
                logger.error("❌ Logon rejected or unexpected response")
                return False
                
        except Exception as e:
            logger.error(f"❌ Logon failed: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from server."""
        try:
            if self.socket:
                self.socket.close()
                logger.info(f"{self.session_type} session disconnected")
            self.connected = False
            self.authenticated = False
        except Exception as e:
            logger.error(f"Disconnect error: {e}")


def test_corrected_connection():
    """Test connection with corrected configuration."""
    print("🧪 TESTING CORRECTED IC MARKETS FIX CONNECTION")
    print("=" * 60)
    
    try:
        # Create corrected configuration
        config = CorrectedFIXConfig(environment="demo", account_number="9533708")
        
        # Note: In real implementation, you would set the actual FIX API password
        # For testing, we'll use a placeholder
        config.set_fix_api_password("WNSE5822")  # This should be the actual FIX API password
        
        print(f"📋 Configuration: {config}")
        print(f"📋 Price Host: {config.price_host}:{config.price_port}")
        print(f"📋 Trade Host: {config.trade_host}:{config.trade_port}")
        print(f"📋 SenderCompID: {config.sender_comp_id}")
        print(f"📋 TargetCompID: {config.target_comp_id}")
        
        # Test price connection
        print(f"\n" + "="*50)
        print("TEST 1: PRICE CONNECTION WITH CORRECTED CONFIG")
        print("="*50)
        
        price_config = config.get_price_config()
        price_connection = CorrectedFIXConnection(price_config, "price")
        
        price_success = price_connection.connect()
        if price_success:
            print("✅ Price connection successful")
            time.sleep(2)
            price_connection.disconnect()
        else:
            print("❌ Price connection failed")
            
        # Test trade connection
        print(f"\n" + "="*50)
        print("TEST 2: TRADE CONNECTION WITH CORRECTED CONFIG")
        print("="*50)
        
        trade_config = config.get_trade_config()
        trade_connection = CorrectedFIXConnection(trade_config, "trade")
        
        trade_success = trade_connection.connect()
        if trade_success:
            print("✅ Trade connection successful")
            time.sleep(2)
            trade_connection.disconnect()
        else:
            print("❌ Trade connection failed")
            
        # Summary
        print(f"\n" + "="*50)
        print("CONNECTION TEST SUMMARY")
        print("="*50)
        
        print(f"Price Connection: {'✅ Success' if price_success else '❌ Failed'}")
        print(f"Trade Connection: {'✅ Success' if trade_success else '❌ Failed'}")
        
        if price_success and trade_success:
            print("🎉 Both connections successful with corrected configuration!")
        elif price_success or trade_success:
            print("⚠️ Partial success - one connection working")
        else:
            print("❌ Both connections failed - configuration may need further adjustment")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_corrected_connection()

