#!/usr/bin/env python3
"""
FIX Connection Verification Script
Tests the connection to IC Markets cTrader FIX gateways
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

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
        # Load .env first
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

        # Load configuration
        config = SystemConfig()
        
        # Check required credentials
        if not (config.account_number and config.password):
            print("FIX credentials not configured! Set ICMARKETS_ACCOUNT and ICMARKETS_PASSWORD in .env")
            return False

        print("Credentials present for account:", "****" + str(config.account_number)[-4:])
        print()
        
        # Create connection manager
        manager = FIXConnectionManager(config)
        
        # Start FIX sessions
        print("Starting FIX sessions...")
        manager.start_sessions()
        
        # Wait for connections to establish
        print("Waiting for connections to establish...")
        print("Look for 'SUCCESSFUL LOGON' messages in the logs...")
        
        # Monitor for 60 seconds
        for i in range(60):
            await asyncio.sleep(1)
            if i % 10 == 0:
                print(f"{60 - i} seconds remaining...")
        
        # Stop sessions
        print("Stopping FIX sessions...")
        manager.stop_sessions()
        
        print()
        print("=" * 60)
        print("Verification Complete!")
        print("=" * 60)
        print("Check the logs above for:")
        print("Look for: 'SUCCESSFUL LOGON' messages for both price and trade sessions")
        print("Look for: Heartbeat messages being exchanged")
        print("Look for: No connection errors")
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        print(f"Verification failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
        sys.exit(1)
