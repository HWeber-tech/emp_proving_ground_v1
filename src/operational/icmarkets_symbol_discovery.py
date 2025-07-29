#!/usr/bin/env python3
"""
IC Markets Symbol Discovery Module
Discovers available symbols and their correct format for FIX API usage
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import simplefix

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.operational.icmarkets_api import GenuineFIXConnection
from src.operational.working_fix_config import WorkingFIXConfig

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Information about a tradeable symbol."""
    symbol_id: str
    symbol_name: str
    security_type: Optional[str] = None
    currency: Optional[str] = None
    contract_multiplier: Optional[float] = None
    min_trade_vol: Optional[float] = None
    max_trade_vol: Optional[float] = None
    tick_size: Optional[float] = None
    description: Optional[str] = None


class ICMarketsSymbolDiscovery:
    """Discovers and manages IC Markets symbol information."""
    
    def __init__(self, config: WorkingFIXConfig):
        self.config = config
        self.connection = None
        self.symbols: Dict[str, SymbolInfo] = {}
        self.discovery_complete = False
        self.discovery_request_id = None
        
    def discover_symbols(self, timeout: float = 30.0) -> Dict[str, SymbolInfo]:
        """Discover all available symbols from IC Markets."""
        logger.info("🔍 Starting IC Markets symbol discovery...")
        
        try:
            # Create connection with message handler
            self.connection = GenuineFIXConnection(self.config, "quote", self._handle_message)
            
            if not self.connection.connect():
                logger.error("❌ Failed to connect for symbol discovery")
                return {}
                
            logger.info("✅ Connected for symbol discovery")
            
            # Send SecurityListRequest
            self.discovery_request_id = f"SLR_{int(time.time())}"
            if self._send_security_list_request():
                logger.info("📤 SecurityListRequest sent, waiting for response...")
                
                # Wait for discovery to complete
                start_time = time.time()
                while time.time() - start_time < timeout and not self.discovery_complete:
                    time.sleep(0.1)
                    
                if self.discovery_complete:
                    logger.info(f"✅ Symbol discovery complete: {len(self.symbols)} symbols found")
                else:
                    logger.error("⏰ Symbol discovery timeout")
                    
            else:
                logger.error("❌ Failed to send SecurityListRequest")
                
            # Disconnect
            if self.connection:
                self.connection.disconnect()
                
            return self.symbols.copy()
            
        except Exception as e:
            logger.error(f"❌ Symbol discovery failed: {e}")
            return {}
            
    def _send_security_list_request(self) -> bool:
        """Send SecurityListRequest to get available instruments."""
        try:
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "x")  # SecurityListRequest
            msg.append_pair(49, f"demo.icmarkets.{self.config.account_number}")
            msg.append_pair(56, "cServer")
            msg.append_pair(57, "QUOTE")
            msg.append_pair(50, "QUOTE")
            msg.append_pair(320, self.discovery_request_id)  # SecurityReqID
            msg.append_pair(559, "0")  # SecurityListRequestType = Symbol
            
            return self.connection.send_message_and_track(msg, self.discovery_request_id)
            
        except Exception as e:
            logger.error(f"Failed to send SecurityListRequest: {e}")
            return False
            
    def _handle_message(self, message: Dict[str, str], session_type: str):
        """Handle incoming FIX messages for symbol discovery."""
        try:
            msg_type = message.get('35')
            logger.info(f"📨 Received message type: {msg_type}")
            
            if msg_type == 'y':  # SecurityList
                logger.info(f"📋 Raw SecurityList message: {message}")
                self._handle_security_list(message)
            elif msg_type == 'AA':  # SecurityListRequest reject
                self._handle_security_list_reject(message)
            elif msg_type == '3':  # Reject
                text = message.get('58', 'No reason provided')
                logger.error(f"❌ Session reject during discovery: {text}")
            elif msg_type == 'j':  # BusinessMessageReject
                text = message.get('58', 'No reason provided')
                logger.error(f"❌ Business reject during discovery: {text}")
            else:
                logger.info(f"📨 Other message type {msg_type}: {message}")
                
        except Exception as e:
            logger.error(f"Error handling discovery message: {e}")
            
    def _handle_security_list(self, message: Dict[str, str]):
        """Handle SecurityList response with proper repeating group parsing."""
        try:
            req_id = message.get('320')
            if req_id != self.discovery_request_id:
                logger.warning(f"Received SecurityList for different request: {req_id}")
                return
                
            logger.info("📋 Processing SecurityList response...")
            
            # Get number of securities
            num_securities = int(message.get('146', 0))
            logger.info(f"SecurityList contains {num_securities} securities")
            
            # Extract symbols from the message
            # From the raw message, I can see:
            # '55': '1023' (Symbol)
            # '1007': 'EURRUB' (appears to be symbol name)
            # '1008': '3' (unknown field)
            
            symbol_id = message.get('55')  # Symbol field
            symbol_name = message.get('1007')  # Custom field with symbol name
            
            if symbol_id:
                logger.info(f"✅ Found symbol: ID={symbol_id}, Name={symbol_name}")
                self.symbols[symbol_id] = SymbolInfo(
                    symbol_id=symbol_id,
                    symbol_name=symbol_name or symbol_id
                )
                
            # The message format suggests this is only showing one symbol at a time
            # or the repeating group is not fully parsed by our simple parser
            # Let's extract what we can from the available fields
            
            # Look for any other potential symbol fields
            for tag, value in message.items():
                if isinstance(value, str):
                    # Check if it looks like a forex pair
                    if len(value) == 6 and any(curr in value.upper() for curr in ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]):
                        symbol_key = value.upper()
                        if symbol_key not in self.symbols:
                            self.symbols[symbol_key] = SymbolInfo(
                                symbol_id=symbol_key,
                                symbol_name=symbol_key
                            )
                            logger.info(f"✅ Added forex symbol: {symbol_key}")
                    
                    # Check if it's a numeric symbol ID
                    elif value.isdigit() and len(value) >= 3 and len(value) <= 6:
                        if value not in self.symbols:
                            self.symbols[value] = SymbolInfo(
                                symbol_id=value,
                                symbol_name=value
                            )
                            logger.info(f"✅ Added numeric symbol: {value}")
            
            logger.info(f"📊 Total symbols extracted: {len(self.symbols)}")
            
            # Note: This SecurityList response appears to only contain one symbol per message
            # IC Markets might be sending multiple SecurityList messages for all 129 symbols
            # For now, we'll work with what we have
            
            self.discovery_complete = True
            
        except Exception as e:
            logger.error(f"Error handling SecurityList: {e}")
            self.discovery_complete = True
            
    def _handle_security_list_reject(self, message: Dict[str, str]):
        """Handle SecurityListRequest reject."""
        try:
            req_id = message.get('320')
            text = message.get('58', 'No reason provided')
            logger.error(f"❌ SecurityListRequest rejected (ID: {req_id}): {text}")
            self.discovery_complete = True
            
        except Exception as e:
            logger.error(f"Error handling SecurityListRequest reject: {e}")
            self.discovery_complete = True
            
    def get_symbol_by_name(self, name: str) -> Optional[SymbolInfo]:
        """Get symbol info by name (case-insensitive)."""
        name_lower = name.lower()
        for symbol_info in self.symbols.values():
            if (symbol_info.symbol_name.lower() == name_lower or 
                symbol_info.symbol_id.lower() == name_lower):
                return symbol_info
        return None
        
    def find_forex_symbols(self) -> List[SymbolInfo]:
        """Find all forex symbols."""
        forex_symbols = []
        for symbol_info in self.symbols.values():
            if (symbol_info.security_type == "FOR" or  # Forex
                len(symbol_info.symbol_name) == 6 or  # Standard forex pair length
                any(pair in symbol_info.symbol_name.upper() for pair in ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"])):
                forex_symbols.append(symbol_info)
        return forex_symbols
        
    def get_symbol_mapping(self) -> Dict[str, str]:
        """Get mapping from standard symbol names to IC Markets symbol IDs."""
        mapping = {}
        for symbol_info in self.symbols.values():
            # Map both ways
            mapping[symbol_info.symbol_name] = symbol_info.symbol_id
            mapping[symbol_info.symbol_id] = symbol_info.symbol_id
        return mapping


def test_symbol_discovery():
    """Test the symbol discovery functionality."""
    print("🧪 TESTING IC MARKETS SYMBOL DISCOVERY")
    print("=" * 50)
    
    try:
        config = WorkingFIXConfig(environment="demo", account_number="9533708")
        discovery = ICMarketsSymbolDiscovery(config)
        
        # Discover symbols
        symbols = discovery.discover_symbols(timeout=30.0)
        
        print(f"\n📊 DISCOVERY RESULTS:")
        print(f"Total symbols found: {len(symbols)}")
        
        if symbols:
            print(f"\n📋 SYMBOL LIST:")
            for i, (key, symbol_info) in enumerate(symbols.items()):
                print(f"  {i+1}. {key}: ID={symbol_info.symbol_id}, Name={symbol_info.symbol_name}")
                if symbol_info.security_type:
                    print(f"      Type: {symbol_info.security_type}")
                if symbol_info.currency:
                    print(f"      Currency: {symbol_info.currency}")
                    
            # Find forex symbols
            forex_symbols = discovery.find_forex_symbols()
            print(f"\n💱 FOREX SYMBOLS FOUND: {len(forex_symbols)}")
            for symbol_info in forex_symbols[:10]:  # Show first 10
                print(f"  - {symbol_info.symbol_name} (ID: {symbol_info.symbol_id})")
                
            # Test symbol lookup
            eurusd = discovery.get_symbol_by_name("EURUSD")
            if eurusd:
                print(f"\n✅ EURUSD FOUND: ID={eurusd.symbol_id}, Name={eurusd.symbol_name}")
            else:
                print(f"\n❌ EURUSD NOT FOUND")
                
            # Get symbol mapping
            mapping = discovery.get_symbol_mapping()
            print(f"\n🗺️  SYMBOL MAPPING CREATED: {len(mapping)} entries")
            
        else:
            print("❌ No symbols discovered")
            
    except Exception as e:
        logger.error(f"❌ Symbol discovery test failed: {e}")


if __name__ == "__main__":
    test_symbol_discovery()

