#!/usr/bin/env python3
"""
FIX Connection Verification Script
Tests the connection to IC Markets cTrader FIX gateways
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.governance.system_config import SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager

# Configure logging to see the output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/fix_verification.log')
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """
    Main verification function for FIX connection testing.
    """
    print("=" * 60)
    print("EMP v4.0 FIX Connection Verification")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Load configuration
        config = SystemConfig()
        
        # Check if FIX credentials are configured
        if not all([
            config.fix_price_sender_comp_id,
            config.fix_price_username,
            config.fix_price_password,
            config.fix_trade_sender_comp_id,
            config.fix_trade_username,
            config.fix_trade_password
        ]):
            print("❌ FIX credentials not fully configured!")
            print("Please update your .env file with:")
            print("  FIX_PRICE_SENDER_COMP_ID=your_value")
            print("  FIX_PRICE_USERNAME=your_value")
            print("  FIX_PRICE_PASSWORD=your_value")
            print("  FIX_TRADE_SENDER_COMP_ID=your_value")
            print("  FIX_TRADE_USERNAME=your_value")
            print("  FIX_TRADE_PASSWORD=your_value")
            return False
        
        print("✅ FIX credentials found in configuration")
        print()
        
        # Create connection manager
        manager = FIXConnectionManager(config)
        
        # Start FIX sessions
        print("🚀 Starting FIX sessions...")
        manager.start_sessions()
        
        # Wait for connections to establish
        print("⏳ Waiting for connections to establish...")
        print("Look for 'SUCCESSFUL LOGON' messages in the logs...")
        
        # Monitor for 60 seconds
        for i in range(60):
            await asyncio.sleep(1)
            if i % 10 == 0:
                print(f"⏱️  {60 - i} seconds remaining...")
        
        # Stop sessions
        print("🛑 Stopping FIX sessions...")
        manager.stop_sessions()
        
        print()
        print("=" * 60)
        print("Verification Complete!")
        print("=" * 60)
        print("Check the logs above for:")
        print("✅ 'SUCCESSFUL LOGON' messages for both price and trade sessions")
        print("✅ Heartbeat messages being exchanged")
        print("✅ No connection errors")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Verification interrupted by user")
        sys.exit(1)
