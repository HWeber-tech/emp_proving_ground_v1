#!/usr/bin/env python3
"""
FIX Diagnostic Tool - Comprehensive troubleshooting
"""

import logging
import socket
import sys
import time
from datetime import datetime

import simplefix

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

log = logging.getLogger(__name__)


class FIXDiagnostic:
    """Comprehensive FIX diagnostic tool"""
    
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
            log.error(".env file not found")
            self.credentials = {}
    
    def test_tcp_connectivity(self, host, port):
        """Test basic TCP connectivity"""
        print(f"\n=== Testing TCP Connectivity ===")
        print(f"Host: {host}:{port}")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            
            start_time = time.time()
            result = sock.connect_ex((host, port))
            connect_time = time.time() - start_time
            
            if result == 0:
                print(f"✅ TCP connection successful ({connect_time:.2f}s)")
                sock.close()
                return True
            else:
                print(f"❌ TCP connection failed: {result}")
                return False
                
        except socket.timeout:
            print("❌ TCP connection timeout")
            return False
        except Exception as e:
            print(f"❌ TCP connection error: {e}")
            return False
    
    def test_fix_handshake(self, host, port, session_type, sender_comp_id, username, password):
        """Test FIX handshake with detailed logging"""
        print(f"\n=== Testing FIX Handshake ({session_type}) ===")
        print(f"Host: {host}:{port}")
        print(f"SenderCompID: {sender_comp_id}")
        print(f"Username: {username}")
        print(f"Password: {'*' * len(password)}")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            
            log.info("Connecting...")
            sock.connect((host, port))
            log.info("✓ TCP connection established")
            
            # Build Logon message
            logon_msg = simplefix.FixMessage()
            logon_msg.append_pair(8, "FIX.4.4")
            logon_msg.append_pair(35, "A")
            logon_msg.append_pair(49, sender_comp_id)
            logon_msg.append_pair(56, "CSERVER")
            logon_msg.append_pair(34, 1)
            logon_msg.append_pair(52, datetime.now().strftime("%Y%m%d-%H:%M:%S"))
            logon_msg.append_pair(98, 0)
            logon_msg.append_pair(108, 30)
            logon_msg.append_pair(553, username)
            logon_msg.append_pair(554, password)
            
            if session_type == 'price':
                logon_msg.append_pair(57, "QUOTE")
            else:
                logon_msg.append_pair(57, "TRADE")
            
            # Log the raw message
            raw_msg = logon_msg.encode()
            log.info(f"Raw Logon message: {raw_msg}")
            log.info(f"Message length: {len(raw_msg)} bytes")
            
            # Send message
            log.info("Sending Logon...")
            sock.send(raw_msg)
            
            # Read response
            log.info("Waiting for response...")
            response = b''
            start_time = time.time()
            
            while time.time() - start_time < 15:
                try:
                    chunk = sock.recv(1024)
                    if chunk:
                        response += chunk
                        log.info(f"Received {len(chunk)} bytes: {chunk}")
                        
                        # Try to parse response
                        if b'\x01' in response:
                            messages = response.split(b'\x01')
                            for msg_data in messages:
                                if msg_data.strip():
                                    try:
                                        response_msg = simplefix.FixMessage()
                                        response_msg.decode(msg_data + b'\x01')
                                        
                                        msg_type = response_msg.get(35)
                                        log.info(f"Parsed message type: {msg_type}")
                                        
                                        if msg_type == b"A":
                                            print("🎉 SUCCESSFUL LOGON!")
                                            return True
                                        elif msg_type == b"5":
                                            print("❌ Logout received - authentication failed")
                                            return False
                                        else:
                                            print(f"⚠️ Unexpected message type: {msg_type}")
                                            
                                    except Exception as e:
                                        log.debug(f"Parse error: {e}")
                            break
                    else:
                        break
                except socket.timeout:
                    continue
            
            if not response:
                print("⚠️ No response received")
                print("This typically indicates:")
                print("1. FIX API not enabled on your account")
                print("2. Credentials need verification")
                print("3. Server requires specific setup")
                
            return False
            
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
        finally:
            try:
                sock.close()
            except:
                pass
    
    def run_diagnostic(self):
        """Run comprehensive diagnostic"""
        print("=" * 70)
        print("EMP v4.0 FIX Diagnostic Tool")
        print("=" * 70)
        
        # Show credentials
        print("\n=== Current Credentials ===")
        for key, value in self.credentials.items():
            if 'FIX' in key:
                if 'PASSWORD' in key:
                    print(f"{key}: {'*' * len(value)}")
                else:
                    print(f"{key}: {value}")
        
        # Test connectivity
        price_tcp = self.test_tcp_connectivity('demo-uk-eqx-01.p.c-trader.com', 5211)
        trade_tcp = self.test_tcp_connectivity('demo-uk-eqx-01.p.c-trader.com', 5212)
        
        # Test FIX handshake
        price_fix = self.test_fix_handshake(
            'demo-uk-eqx-01.p.c-trader.com', 5211, 'price',
            self.credentials.get('FIX_PRICE_SENDER_COMP_ID', ''),
            self.credentials.get('FIX_PRICE_USERNAME', ''),
            self.credentials.get('FIX_PRICE_PASSWORD', '')
        )
        
        trade_fix = self.test_fix_handshake(
            'demo-uk-eqx-01.p.c-trader.com', 5212, 'trade',
            self.credentials.get('FIX_TRADE_SENDER_COMP_ID', ''),
            self.credentials.get('FIX_TRADE_USERNAME', ''),
            self.credentials.get('FIX_TRADE_PASSWORD', '')
        )
        
        # Summary
        print("\n" + "=" * 70)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 70)
        
        print(f"TCP Price Connection: {'✅' if price_tcp else '❌'}")
        print(f"TCP Trade Connection: {'✅' if trade_tcp else '❌'}")
        print(f"FIX Price Handshake: {'✅' if price_fix else '❌'}")
        print(f"FIX Trade Handshake: {'✅' if trade_fix else '❌'}")
        
        print("\n📋 RECOMMENDATIONS:")
        print("1. Contact IC Markets support to verify:")
        print("   - FIX API is enabled on account 9533708")
        print("   - Username 'heinzweber.23' is correct")
        print("   - SenderCompID 'demo.icmarkets.9533708' is correct")
        print("   - Demo server configuration")
        print("2. Check if you need to request FIX API access")
        print("3. Verify if different credentials are needed for demo vs live")
        
        return price_tcp and trade_tcp and price_fix and trade_fix


def main():
    """Main diagnostic function"""
    print("Starting FIX diagnostic...")
    
    diagnostic = FIXDiagnostic()
    success = diagnostic.run_diagnostic()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Tests completed - contact IC Markets for credential verification")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
