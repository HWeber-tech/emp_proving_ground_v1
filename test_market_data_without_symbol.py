#!/usr/bin/env python3
"""
Test MarketDataRequest without Symbol field
Based on broker feedback: "Tag not defined for this message type, field=55"
"""

import sys
import time
import logging
import uuid
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.operational.icmarkets_api import GenuineFIXConnection
from src.operational.working_fix_config import WorkingFIXConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_market_data_without_symbol():
    """Test MarketDataRequest without Symbol field based on broker rejection."""
    print("🧪 TESTING MARKET DATA REQUEST WITHOUT SYMBOL FIELD")
    print("=" * 60)
    
    try:
        # Create configuration
        config = WorkingFIXConfig(environment="demo", account_number="9533708")
        print(f"📋 Configuration: {config.account_number}")
        
        # Create price connection
        def message_handler(message, session_type):
            msg_type = message.get('35')
            print(f"📨 Received {session_type} message type: {msg_type}")
            print(f"📋 Raw message: {message}")
            
        connection = GenuineFIXConnection(config, "quote", message_handler)
        
        # Connect
        print(f"\n🚀 Connecting to price session...")
        if not connection.connect():
            print("❌ Connection failed")
            return
            
        print("✅ Connected and authenticated")
        
        # Test 1: MarketDataRequest without Symbol field
        print(f"\n" + "="*50)
        print("TEST 1: MARKET DATA REQUEST WITHOUT SYMBOL FIELD")
        print("="*50)
        
        import simplefix
        
        msg = simplefix.FixMessage()
        msg.append_pair(8, "FIX.4.4")
        msg.append_pair(35, "V")  # MarketDataRequest
        msg.append_pair(49, f"demo.icmarkets.{config.account_number}")
        msg.append_pair(56, "cServer")
        msg.append_pair(57, "QUOTE")  # TargetSubID
        msg.append_pair(50, "QUOTE")  # SenderSubID
        msg.append_pair(262, f"MD_TEST_{int(time.time())}")  # MDReqID
        msg.append_pair(263, "1")  # SubscriptionRequestType = Snapshot + Updates
        msg.append_pair(264, "0")  # MarketDepth = Full Book
        msg.append_pair(265, "1")  # MDUpdateType = Incremental
        msg.append_pair(267, "2")  # NoMDEntryTypes
        msg.append_pair(269, "0")  # MDEntryType = Bid
        msg.append_pair(269, "1")  # MDEntryType = Offer
        # NOTE: Deliberately omitting Symbol field (tag 55)
        
        print("📤 Sending MarketDataRequest without Symbol field...")
        success = connection.send_message_and_track(msg, "TEST1")
        print(f"📤 Message sent: {success}")
        
        # Wait for response
        time.sleep(5)
        
        # Test 2: MarketDataRequest with different structure
        print(f"\n" + "="*50)
        print("TEST 2: MARKET DATA REQUEST WITH MINIMAL FIELDS")
        print("="*50)
        
        msg2 = simplefix.FixMessage()
        msg2.append_pair(8, "FIX.4.4")
        msg2.append_pair(35, "V")  # MarketDataRequest
        msg2.append_pair(49, f"demo.icmarkets.{config.account_number}")
        msg2.append_pair(56, "cServer")
        msg2.append_pair(57, "QUOTE")  # TargetSubID
        msg2.append_pair(50, "QUOTE")  # SenderSubID
        msg2.append_pair(262, f"MD_MIN_{int(time.time())}")  # MDReqID
        msg2.append_pair(263, "1")  # SubscriptionRequestType
        msg2.append_pair(264, "0")  # MarketDepth
        
        print("📤 Sending minimal MarketDataRequest...")
        success2 = connection.send_message_and_track(msg2, "TEST2")
        print(f"📤 Message sent: {success2}")
        
        # Wait for response
        time.sleep(5)
        
        # Test 3: Try with SecurityID instead of Symbol
        print(f"\n" + "="*50)
        print("TEST 3: MARKET DATA REQUEST WITH SECURITY ID")
        print("="*50)
        
        msg3 = simplefix.FixMessage()
        msg3.append_pair(8, "FIX.4.4")
        msg3.append_pair(35, "V")  # MarketDataRequest
        msg3.append_pair(49, f"demo.icmarkets.{config.account_number}")
        msg3.append_pair(56, "cServer")
        msg3.append_pair(57, "QUOTE")  # TargetSubID
        msg3.append_pair(50, "QUOTE")  # SenderSubID
        msg3.append_pair(262, f"MD_SEC_{int(time.time())}")  # MDReqID
        msg3.append_pair(263, "1")  # SubscriptionRequestType
        msg3.append_pair(264, "0")  # MarketDepth
        msg3.append_pair(267, "2")  # NoMDEntryTypes
        msg3.append_pair(269, "0")  # MDEntryType = Bid
        msg3.append_pair(269, "1")  # MDEntryType = Offer
        msg3.append_pair(146, "1")  # NoRelatedSym
        msg3.append_pair(48, "1023")  # SecurityID instead of Symbol
        
        print("📤 Sending MarketDataRequest with SecurityID...")
        success3 = connection.send_message_and_track(msg3, "TEST3")
        print(f"📤 Message sent: {success3}")
        
        # Wait for response
        time.sleep(5)
        
        # Test 4: Try with custom field 1007 (from SecurityList)
        print(f"\n" + "="*50)
        print("TEST 4: MARKET DATA REQUEST WITH CUSTOM FIELD 1007")
        print("="*50)
        
        msg4 = simplefix.FixMessage()
        msg4.append_pair(8, "FIX.4.4")
        msg4.append_pair(35, "V")  # MarketDataRequest
        msg4.append_pair(49, f"demo.icmarkets.{config.account_number}")
        msg4.append_pair(56, "cServer")
        msg4.append_pair(57, "QUOTE")  # TargetSubID
        msg4.append_pair(50, "QUOTE")  # SenderSubID
        msg4.append_pair(262, f"MD_1007_{int(time.time())}")  # MDReqID
        msg4.append_pair(263, "1")  # SubscriptionRequestType
        msg4.append_pair(264, "0")  # MarketDepth
        msg4.append_pair(267, "2")  # NoMDEntryTypes
        msg4.append_pair(269, "0")  # MDEntryType = Bid
        msg4.append_pair(269, "1")  # MDEntryType = Offer
        msg4.append_pair(146, "1")  # NoRelatedSym
        msg4.append_pair(1007, "EURRUB")  # Custom field from SecurityList
        
        print("📤 Sending MarketDataRequest with custom field 1007...")
        success4 = connection.send_message_and_track(msg4, "TEST4")
        print(f"📤 Message sent: {success4}")
        
        # Wait for response
        time.sleep(5)
        
        print(f"\n" + "="*50)
        print("FINAL WAIT FOR RESPONSES")
        print("="*50)
        
        # Wait for any delayed responses
        print("⏳ Waiting for broker responses...")
        time.sleep(10)
        
        # Disconnect
        connection.disconnect()
        print(f"\n✅ Test completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_market_data_without_symbol()

