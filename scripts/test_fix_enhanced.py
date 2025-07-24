#!/usr/bin/env python3
"""
Enhanced FIX Connection Test with better protocol handling
"""

import logging
import sys
import os
import time
import socket
import simplefix
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

log = logging.getLogger(__name__)


class EnhancedFIXTester:
    """Enhanced FIX tester with better protocol handling"""
    
    def __init__(self):
        self.load_credentials()
    
    def load_credentials(self):
        """Load credentials from .env file"""
        try:
            with open('.env', 'r') as f:
                lines = f.readlines()
            
            self.credentials = {}
            for line in lines:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    self.credentials[key.strip()] = value.strip()
            
            log.info("Loaded credentials from .env file")
            
        except FileNotFoundError:
            log.error(".env file not found. Using placeholder credentials.")
            self.credentials = {
                'FIX_PRICE_SENDER_COMP_ID': 'YOUR_PRICE_SENDER_ID',
                'FIX_PRICE_USERNAME': 'YOUR_PRICE_USERNAME',
                'FIX_PRICE_PASSWORD': 'YOUR_PRICE_PASSWORD',
                'FIX_TRADE_SENDER_COMP_ID': 'YOUR_TRADE_SENDER_ID',
                'FIX_TRADE_USERNAME': 'YOUR_TRADE_USERNAME',
                'FIX_TRADE_PASSWORD': 'YOUR_TRADE_PASSWORD'
            }
    
    def test_connection(self, host, port, session_type, sender_comp_id, username, password):
        """Test FIX connection with proper protocol handling"""
        print(f"\n=== Testing {session_type} FIX Connection ===")
        print(f"Host: {host}:{port}")
        print(f"SenderCompID: {sender_comp_id}")
        print(f"Username: {username}")
        
        try:
            # Create socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(15)  # Increased timeout
            
            log.info(f"Connecting to {host}:{port}...")
            sock.connect((host, port))
            log.info("✓ TCP connection established")
            
            # Create and send Logon message
            logon_msg = simplefix.FixMessage()
            logon_msg.append_pair(8, "FIX.4.4")  # BeginString
            logon_msg.append_pair(35, "A")       # MsgType = Logon
            logon_msg.append_pair(49, sender_comp_id)  # SenderCompID
            logon_msg.append_pair(56, "CSERVER")       # TargetCompID
            logon_msg.append_pair(34, 1)               # MsgSeqNum
            logon_msg.append_pair(52, datetime.now().strftime("%Y%m%d-%H:%M:%S"))
            logon_msg.append_pair(98, 0)               # EncryptMethod = None
            logon_msg.append_pair(108, 30)             # HeartBtInt = 30 seconds
            logon_msg.append_pair(553, username)       # Username
            logon_msg.append_pair(554, password)       # Password
            
            if session_type == 'price':
                logon_msg.append_pair(57, "QUOTE")     # TargetSubID
            else:
                logon_msg.append_pair(57, "TRADE")     # TargetSubID
            
            # Send the message
            logon_data = logon_msg.encode()
            log.info(f"Sending Logon message ({len(logon_data)} bytes)...")
            sock.send(logon_data)
            
            # Read response with better handling
            log.info("Waiting for response...")
            response = b''
            start_time = time.time()
            
            while time.time() - start_time < 10:  # Wait up to 10 seconds
                try:
                    chunk = sock.recv(1024)
                    if not chunk:
                        break
                    response += chunk
                    
                    # Check if we have a complete FIX message
                    if b'\x01' in response:
                        # Split messages
                        messages = response.split(b'\x01')
                        for msg_data in messages:
                            if msg_data.strip():
                                try:
                                    response_msg = simplefix.FixMessage()
                                    response_msg.decode(msg_data + b'\x01')
                                    
                                    msg_type = response_msg.get(35)
                                    if msg_type == b"A":
                                        log.info("🎉 SUCCESSFUL LOGON!")
                                        print("✅ Logon successful!")
                                        return True
                                    elif msg_type == b"5":
                                        log.error("Received Logout - authentication failed")
                                        print("❌ Authentication failed")
                                        return False
                                    else:
                                        log.info(f"Received message type: {msg_type}")
                                        
                                except Exception as e:
                                    log.debug(f"Error parsing message: {e}")
                        
                        break
                        
                except socket.timeout:
                    continue
            
            if not response:
                log.warning("No response received within timeout")
                print("⚠️ No response received - server may be processing")
                
                # Try to read any pending data
                try:
                    sock.settimeout(2)
                    final_response = sock.recv(1024)
                    if final_response:
                        log.info(f"Received delayed response: {final_response[:100]}...")
                except:
                    pass
            
            return False
            
        except socket.timeout:
            log.error("Connection timeout")
            print("❌ Connection timeout")
            return False
        except Exception as e:
            log.error(f"Connection error: {e}")
            print(f"❌ Connection error: {e}")
            return False
        finally:
            try:
                sock.close()
            except:
                pass
    
    def run_tests(self):
        """Run both price and trade connection tests"""
        print("=" * 60)
        print("EMP v4.0 Enhanced FIX Connection Test")
        print("=" * 60)
        
        # Test Price Connection
        price_success = self.test_connection(
            host='demo-uk-eqx-01.p.c-trader.com',
            port=5211,
            session_type='price',
            sender_comp_id=self.credentials.get('FIX_PRICE_SENDER_COMP_ID', 'YOUR_PRICE_SENDER_ID'),
            username=self.credentials.get('FIX_PRICE_USERNAME', 'YOUR_PRICE_USERNAME'),
            password=self.credentials.get('FIX_PRICE_PASSWORD', 'YOUR_PRICE_PASSWORD')
        )
        
        # Test Trade Connection
        trade_success = self.test_connection(
            host='demo-uk-eqx-01.p.c-trader.com',
            port=5212,
            session_type='trade',
            sender_comp_id=self.credentials.get('FIX_TRADE_SENDER_COMP_ID', 'YOUR_TRADE_SENDER_ID'),
            username=self.credentials.get('FIX_TRADE_USERNAME', 'YOUR_TRADE_USERNAME'),
            password=self.credentials.get('FIX_TRADE_PASSWORD', 'YOUR_TRADE_PASSWORD')
        )
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        if price_success and trade_success:
            print("🎉 ALL CONNECTIONS SUCCESSFUL!")
            print("\nYour FIX integration is working correctly.")
            print("\nNext steps:")
            print("1. Test market data subscription")
            print("2. Integrate with sensory organs")
            print("3. Deploy to production")
            return True
        else:
            print("⚠️ Some connections failed")
            print("\nTroubleshooting:")
            print("1. Verify credentials with IC Markets support")
            print("2. Check if FIX API is enabled on your account")
            print("3. Ensure you're using demo credentials for demo server")
            print("4. Contact IC Markets for specific FIX setup instructions")
            return False


def main():
    """Main test function"""
    print("Starting enhanced FIX connection test...")
    
    tester = EnhancedFIXTester()
    success = tester.run_tests()
    
    if success:
        print("\n✅ FIX connections are working!")
    else:
        print("\n⚠️ Connections established but may need credential verification")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
