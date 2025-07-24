#!/usr/bin/env python3
"""
Simple FIX Connection Test - Standalone version
Tests basic FIX connection without complex dependencies
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


class SimpleFIXTester:
    """Simple FIX connection tester without dependencies"""
    
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
        """Test a single FIX connection"""
        print(f"\n=== Testing {session_type} FIX Connection ===")
        print(f"Host: {host}:{port}")
        print(f"SenderCompID: {sender_comp_id}")
        print(f"Username: {username}")
        
        try:
            # Create socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            
            log.info(f"Connecting to {host}:{port}...")
            sock.connect((host, port))
            log.info("✓ TCP connection established")
            
            # Send Logon message
            logon_msg = simplefix.FixMessage()
            logon_msg.append_pair(8, "FIX.4.4")
            logon_msg.append_pair(35, "A")
            logon_msg.append_pair(49, sender_comp_id)
            logon_msg.append_pair(56, "CSERVER")
            logon_msg.append_pair(34, 1)
            logon_msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))
            logon_msg.append_pair(98, 0)
            logon_msg.append_pair(108, 30)
            logon_msg.append_pair(553, username)
            logon_msg.append_pair(554, password)
            
            if session_type == 'price':
                logon_msg.append_pair(57, "QUOTE")
            else:
                logon_msg.append_pair(57, "TRADE")
            
            log.info("Sending Logon message...")
            sock.send(logon_msg.encode())
            
            # Wait for response
            log.info("Waiting for Logon response...")
            response = sock.recv(4096)
            
            if response:
                log.info("✓ Received response from server")
                response_msg = simplefix.FixMessage()
                response_msg.decode(response)
                
                msg_type = response_msg.get(35)
                if msg_type == b"A":
                    log.info("🎉 SUCCESSFUL LOGON!")
                    print("✅ Connection successful!")
                    return True
                else:
                    log.warning(f"Received message type: {msg_type}")
                    print("⚠️ Unexpected response type")
                    return False
            else:
                log.error("No response received")
                print("❌ No response from server")
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
        print("EMP v4.0 FIX Connection Test")
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
            return True
        else:
            print("⚠️ Some connections failed")
            print("\nCheck your credentials and network connectivity")
            return False


def main():
    """Main test function"""
    print("Starting FIX connection test...")
    
    tester = SimpleFIXTester()
    success = tester.run_tests()
    
    if success:
        print("\n✅ Ready to proceed with market data testing!")
    else:
        print("\n❌ Please check your credentials and try again")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
