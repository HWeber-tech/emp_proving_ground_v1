#!/usr/bin/env python3
"""
Final FIX API Test - Complete Implementation
Combining account-specific host from screenshot with TargetSubID fix
"""

import sys
import time
import logging
import socket
import ssl
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.operational.icmarkets_config import ICMarketsConfig, get_symbol_name
import simplefix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalFIXTester:
    """Final FIX API tester with complete functionality."""
    
    def __init__(self, config_dict, session_type):
        self.config = config_dict
        self.session_type = session_type
        self.socket = None
        self.seq_num = 1
        self.connected = False
        self.authenticated = False
        
    def connect_and_authenticate(self):
        """Connect and authenticate with final configuration."""
        try:
            logger.info(f"Connecting to {self.session_type} session...")
            logger.info(f"Host: {self.config['host']}:{self.config['port']}")
            logger.info(f"SSL: {self.config['use_ssl']}")
            
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Apply SSL
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
            return False
            
    def send_logon(self):
        """Send logon message."""
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, self.config['fix_version'])
            msg.append_pair(35, 'A')  # Logon
            msg.append_pair(49, self.config['sender_comp_id'])
            msg.append_pair(56, self.config['target_comp_id'])
            msg.append_pair(57, self.config['target_sub_id'])  # Our fix: QUOTE for both
            msg.append_pair(50, self.config['sender_sub_id'])  # QUOTE or TRADE
            msg.append_pair(34, self.seq_num)
            msg.append_pair(52, time.strftime('%Y%m%d-%H:%M:%S', time.gmtime()))
            msg.append_pair(98, 0)  # No encryption
            msg.append_pair(108, self.config['heartbeat_interval'])
            msg.append_pair(141, 'Y')  # ResetSeqNumFlag
            msg.append_pair(553, self.config['username'])
            msg.append_pair(554, self.config['password'])
            
            logger.info(f"Sending logon message: {msg.encode()}")
            self.socket.sendall(msg.encode())
            self.seq_num += 1
            
            # Wait for response
            response = self.socket.recv(1024)
            response_str = response.decode('utf-8', errors='ignore').replace(chr(1), '|')
            logger.info(f"Logon response: {response_str}")
            
            if b'35=A' in response:
                logger.info("✅ Authentication successful")
                self.authenticated = True
                return True
            else:
                logger.error("❌ Authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Logon failed: {e}")
            return False
            
    def test_market_data(self, symbol_id):
        """Test market data subscription."""
        if not self.authenticated:
            return False
            
        try:
            logger.info(f"Testing market data for symbol {symbol_id} ({get_symbol_name(symbol_id)})")
            
            msg = simplefix.FixMessage()
            msg.append_pair(8, self.config['fix_version'])
            msg.append_pair(35, 'V')  # MarketDataRequest
            msg.append_pair(49, self.config['sender_comp_id'])
            msg.append_pair(56, self.config['target_comp_id'])
            msg.append_pair(57, self.config['target_sub_id'])
            msg.append_pair(50, self.config['sender_sub_id'])
            msg.append_pair(34, self.seq_num)
            msg.append_pair(52, time.strftime('%Y%m%d-%H:%M:%S', time.gmtime()))
            
            # Market data request fields
            msg.append_pair(262, f"MD_REQ_{int(time.time())}")  # MDReqID
            msg.append_pair(263, "1")  # SubscriptionRequestType
            msg.append_pair(264, "0")  # MarketDepth
            msg.append_pair(265, "1")  # MDUpdateType
            msg.append_pair(267, "2")  # NoMDEntryTypes
            msg.append_pair(269, "0")  # Bid
            msg.append_pair(269, "1")  # Ask
            msg.append_pair(146, "1")  # NoRelatedSym
            msg.append_pair(55, str(symbol_id))  # Symbol as numeric ID
            
            self.socket.sendall(msg.encode())
            self.seq_num += 1
            
            # Wait for response
            time.sleep(2)
            response = self.socket.recv(2048)
            response_str = response.decode('utf-8', errors='ignore').replace(chr(1), '|')
            logger.info(f"Market data response: {response_str[:200]}...")
            
            if b'35=W' in response:  # Market Data Snapshot
                logger.info(f"✅ Market data received for symbol {symbol_id}")
                return True
            elif b'35=3' in response or b'35=Y' in response:  # Reject
                logger.error(f"❌ Market data rejected for symbol {symbol_id}")
                return False
            else:
                logger.info(f"Unexpected response for symbol {symbol_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Market data test failed: {e}")
            return False
            
    def test_order_placement(self, symbol_id, side, quantity):
        """Test order placement with proper session."""
        if not self.authenticated:
            return False
            
        try:
            logger.info(f"Testing order: {side} {quantity} units of symbol {symbol_id} ({get_symbol_name(symbol_id)})")
            
            msg = simplefix.FixMessage()
            msg.append_pair(8, self.config['fix_version'])
            msg.append_pair(35, 'D')  # NewOrderSingle
            msg.append_pair(49, self.config['sender_comp_id'])
            msg.append_pair(56, self.config['target_comp_id'])
            msg.append_pair(57, self.config['target_sub_id'])  # QUOTE for both sessions
            msg.append_pair(50, self.config['sender_sub_id'])  # TRADE for trade session
            msg.append_pair(34, self.seq_num)
            msg.append_pair(52, time.strftime('%Y%m%d-%H:%M:%S', time.gmtime()))
            
            # Order fields
            order_id = f"ORDER_{int(time.time())}"
            msg.append_pair(11, order_id)  # ClOrdID
            msg.append_pair(55, str(symbol_id))  # Symbol as numeric ID
            msg.append_pair(54, "1" if side.upper() == "BUY" else "2")  # Side
            msg.append_pair(38, str(quantity))  # OrderQty
            msg.append_pair(40, "1")  # OrdType = Market
            msg.append_pair(59, "0")  # TimeInForce = Day
            msg.append_pair(60, time.strftime('%Y%m%d-%H:%M:%S', time.gmtime()))  # TransactTime
            
            logger.info(f"Sending order: {msg.encode()}")
            self.socket.sendall(msg.encode())
            self.seq_num += 1
            
            # Wait for ExecutionReport
            logger.info("Waiting for ExecutionReport...")
            time.sleep(5)
            
            response = self.socket.recv(2048)
            response_str = response.decode('utf-8', errors='ignore').replace(chr(1), '|')
            logger.info(f"Order response: {response_str}")
            
            if b'35=8' in response:  # ExecutionReport
                logger.info(f"✅ ExecutionReport received for order {order_id}")
                # Parse execution details
                if b'39=2' in response:  # OrdStatus = Filled
                    logger.info(f"🎉 ORDER FILLED! Order {order_id} executed successfully")
                    return True
                elif b'39=0' in response:  # OrdStatus = New
                    logger.info(f"📋 Order {order_id} accepted (New status)")
                    return True
                else:
                    logger.info(f"📊 Order {order_id} status update received")
                    return True
            elif b'35=j' in response:  # BusinessMessageReject
                logger.error(f"❌ Business reject for order {order_id}")
                return False
            else:
                logger.info(f"Unexpected response for order {order_id}: {response_str[:100]}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Order placement failed: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from server."""
        try:
            if self.socket:
                self.socket.close()
                logger.info(f"{self.session_type} session disconnected")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")


def test_final_fix_implementation():
    """Test complete FIX API implementation with final configuration."""
    print("🚀 FINAL FIX API IMPLEMENTATION TEST")
    print("=" * 60)
    print("Combining account-specific host + TargetSubID fix")
    print("=" * 60)
    
    try:
        # Create final configuration
        config = ICMarketsConfig(environment="demo", account_number="9533708")
        config.set_fix_api_password("WNSE5822")
        
        print(f"📋 Final Configuration:")
        print(f"   Price Host: {config.price_host}:{config.price_port}")
        print(f"   Trade Host: {config.trade_host}:{config.trade_port}")
        print(f"   SenderCompID: {config.sender_comp_id}")
        print(f"   TargetCompID: {config.target_comp_id}")
        print(f"   SSL: {config.use_ssl}")
        
        # Test price session
        print(f"\n" + "="*50)
        print("STEP 1: PRICE SESSION TEST")
        print("="*50)
        
        price_config = config.get_price_config()
        price_tester = FinalFIXTester(price_config, "price")
        
        price_auth = price_tester.connect_and_authenticate()
        print(f"Price Authentication: {'✅ Success' if price_auth else '❌ Failed'}")
        
        if price_auth:
            # Test market data for multiple symbols
            md_results = []
            for symbol_id in [1, 2, 3]:  # EURUSD, GBPUSD, USDJPY
                result = price_tester.test_market_data(symbol_id)
                md_results.append(result)
                print(f"Market Data Symbol {symbol_id}: {'✅ Success' if result else '❌ Failed'}")
                time.sleep(1)
                
        price_tester.disconnect()
        
        # Test trade session
        print(f"\n" + "="*50)
        print("STEP 2: TRADE SESSION TEST")
        print("="*50)
        
        trade_config = config.get_trade_config()
        trade_tester = FinalFIXTester(trade_config, "trade")
        
        trade_auth = trade_tester.connect_and_authenticate()
        print(f"Trade Authentication: {'✅ Success' if trade_auth else '❌ Failed'}")
        
        order_success = False
        if trade_auth:
            # Test order placement
            order_success = trade_tester.test_order_placement(1, "BUY", 1000)  # Buy 1000 units EURUSD
            print(f"Order Placement: {'✅ Success' if order_success else '❌ Failed'}")
            
        trade_tester.disconnect()
        
        # Final summary
        print(f"\n" + "="*50)
        print("FINAL IMPLEMENTATION RESULTS")
        print("="*50)
        
        print(f"Price Authentication: {'✅ Success' if price_auth else '❌ Failed'}")
        print(f"Trade Authentication: {'✅ Success' if trade_auth else '❌ Failed'}")
        if price_auth:
            print(f"Market Data (Multi-Symbol): {'✅ Success' if all(md_results) else '⚠️ Partial'}")
        print(f"Order Execution: {'✅ Success' if order_success else '❌ Failed'}")
        
        if price_auth and trade_auth and order_success:
            print("\n🎉 COMPLETE SUCCESS - FIX API FULLY FUNCTIONAL!")
            print("✅ All components working: Authentication, Market Data, Order Execution")
        elif price_auth and trade_auth:
            print("\n⚠️ PARTIAL SUCCESS - Authentication working, order execution needs verification")
        else:
            print("\n❌ IMPLEMENTATION INCOMPLETE - Check configuration and permissions")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_final_fix_implementation()

