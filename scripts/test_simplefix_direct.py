#!/usr/bin/env python3
"""
Direct test for IC Markets SimpleFIX connection
"""

import socket
import simplefix
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_direct_connection():
    """Test direct connection to IC Markets."""
    print("🔌 Testing Direct IC Markets Connection")
    print("=" * 40)
    
    # IC Markets demo credentials
    account = "9533708"
    password = "WNSE5822"
    
    # Demo server
    host = "demo-uk-eqx-01.p.c-trader.com"
    price_port = 5211
    trade_port = 5212
    
    print(f"📍 Connecting to: {host}:{price_port}")
    
    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        # Connect
        print("🔗 Connecting...")
        sock.connect((host, price_port))
        print("✅ Socket connected")
        
        # Create logon message
        msg = simplefix.FixMessage()
        msg.append_pair(8, "FIX.4.4")
        msg.append_pair(35, "A")
        msg.append_pair(49, f"icmarkets.{account}")
        msg.append_pair(56, "cServer")
        msg.append_pair(57, "QUOTE")
        msg.append_pair(34, 1)
        msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
        msg.append_pair(98, 0)
        msg.append_pair(108, 30)
        msg.append_pair(553, account)
        msg.append_pair(554, password)
        
        # Send logon
        print("📤 Sending logon...")
        message_bytes = msg.encode()
        print(f"📤 Message: {message_bytes}")
        sock.send(message_bytes)
        print("✅ Logon sent")
        
        # Receive response
        print("📥 Waiting for response...")
        response = sock.recv(1024)
        if response:
            print(f"📥 Raw response: {response}")
            try:
                response_str = response.decode('utf-8', errors='ignore')
                print(f"📥 Response string: {response_str}")
                
                parser = simplefix.FixParser()
                response_msg = parser.get_message(response_str)
                print(f"📥 Parsed response: {response_msg}")
                
                msg_type = response_msg.get(35)
                print(f"📥 Message type: {msg_type}")
                
                if msg_type == b'A':
                    print("🎉 Logon successful!")
                    return True
                else:
                    print(f"❌ Unexpected message type: {msg_type}")
                    
            except Exception as e:
                print(f"❌ Error parsing response: {e}")
        else:
            print("❌ No response received")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
        
    finally:
        try:
            sock.close()
            print("🔒 Socket closed")
        except:
            pass
            
    return False

if __name__ == "__main__":
    test_direct_connection()
