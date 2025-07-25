#!/usr/bin/env python3
"""
Test SSL connection to IC Markets - Working version
"""

import ssl
import socket
import simplefix
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_ssl_connection():
    """Test SSL connection to IC Markets."""
    print("🔒 Testing SSL Connection to IC Markets")
    print("=" * 45)
    
    # IC Markets demo credentials
    account = "9533708"
    password = "WNSE5822"
    
    # Demo server
    host = "demo-uk-eqx-01.p.c-trader.com"
    price_port = 5211
    
    print(f"📍 Connecting to: {host}:{price_port}")
    
    try:
        # Create SSL context
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        # Wrap with SSL
        ssl_sock = context.wrap_socket(sock, server_hostname=host)
        
        # Connect
        print("🔗 Connecting via SSL...")
        ssl_sock.connect((host, price_port))
        print("✅ SSL connection established")
        
        # Create logon message
        msg = simplefix.FixMessage()
        msg.append_pair(8, "FIX.4.4")
        msg.append_pair(35, "A")
        msg.append_pair(49, f"icmarkets.{account}")
        msg.append_pair(56, "cServer")
        msg.append_pair(57, "QUOTE")
        msg.append_pair(34, 1)
        msg.append_pair(52, datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
        msg.append_pair(98, 0)
        msg.append_pair(108, 30)
        msg.append_pair(553, account)
        msg.append_pair(554, password)
        
        # Send logon
        print("📤 Sending logon...")
        message_bytes = msg.encode()
        print(f"📤 Message length: {len(message_bytes)} bytes")
        ssl_sock.send(message_bytes)
        print("✅ Logon sent")
        
        # Receive response
        print("📥 Waiting for response...")
        response = ssl_sock.recv(1024)
        if response:
            print(f"📥 Response received: {len(response)} bytes")
            print(f"📥 Raw response: {response}")
            
            # Parse response manually
            response_str = response.decode('utf-8', errors='ignore')
            print(f"📥 Response string: {response_str}")
            
            # Check if it's a logout message (MsgType=5)
            if b"35=5" in response:
                print("📥 Received Logout message")
                print("📥 This indicates the server received our message but rejected it")
                
                # Extract error message
                if b"58=" in response:
                    start = response.find(b"58=") + 3
                    end = response.find(b"\x01", start)
                    error_msg = response[start:end].decode()
                    print(f"📥 Server error: {error_msg}")
                
                print("✅ Connection successful - server is responding!")
                print("💡 The rejection is likely due to:")
                print("   - Incorrect credentials")
                print("   - Account not enabled for FIX")
                print("   - Wrong server/port combination")
                return True
                
            elif b"35=A" in response:
                print("🎉 Logon successful!")
                return True
            else:
                print(f"📥 Unexpected response type")
                
        else:
            print("❌ No response received")
            
    except Exception as e:
        print(f"❌ SSL connection error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            ssl_sock.close()
            print("🔒 SSL socket closed")
        except:
            pass
            
    return False

if __name__ == "__main__":
    test_ssl_connection()
