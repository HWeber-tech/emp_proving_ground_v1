#!/usr/bin/env python3
"""
Simple FIX Connection Verification Script
Tests basic FIX connection to IC Markets cTrader
"""

import logging
import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.governance.system_config import SystemConfig
from src.operational.enhanced_fix_application import EnhancedFIXApplication

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

log = logging.getLogger(__name__)


def main():
    """
    Simple verification script for FIX connections.
    This script tests basic connectivity to IC Markets cTrader FIX gateways.
    """
    print("=" * 60)
    print("EMP v4.0 FIX Connection Verification")
    print("=" * 60)
    
    try:
        config = SystemConfig()
        
        # Test Price Connection
        print("\n1. Testing Price FIX Connection...")
        price_config = {
            'SenderCompID': config.fix_price_sender_comp_id or 'YOUR_PRICE_SENDER_ID',
            'Username': config.fix_price_username or 'YOUR_PRICE_USERNAME',
            'Password': config.fix_price_password or 'YOUR_PRICE_PASSWORD'
        }
        
        price_app = EnhancedFIXApplication(
            session_config=price_config,
            session_type='price'
        )
        
        price_success = price_app.start(
            host='demo-uk-eqx-01.p.c-trader.com',
            port=5211
        )
        
        if price_success:
            print("✓ Price FIX connection successful")
        else:
            print("✗ Price FIX connection failed")
        
        # Test Trade Connection
        print("\n2. Testing Trade FIX Connection...")
        trade_config = {
            'SenderCompID': config.fix_trade_sender_comp_id or 'YOUR_TRADE_SENDER_ID',
            'Username': config.fix_trade_username or 'YOUR_TRADE_USERNAME',
            'Password': config.fix_trade_password or 'YOUR_TRADE_PASSWORD'
        }
        
        trade_app = EnhancedFIXApplication(
            session_config=trade_config,
            session_type='trade'
        )
        
        trade_success = trade_app.start(
            host='demo-uk-eqx-01.p.c-trader.com',
            port=5212
        )
        
        if trade_success:
            print("✓ Trade FIX connection successful")
        else:
            print("✗ Trade FIX connection failed")
        
        # Keep connections alive for observation
        if price_success or trade_success:
            print("\n3. Observing connections for 30 seconds...")
            print("Look for 'SUCCESSFUL LOGON' messages in the logs above.")
            print("Press Ctrl+C to exit early.")
            
            try:
                time.sleep(30)
            except KeyboardInterrupt:
                print("\nExiting early...")
        
        # Cleanup
        print("\n4. Cleaning up...")
        if price_app:
            price_app.stop()
        if trade_app:
            trade_app.stop()
        
        # Summary
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        
        if price_success and trade_success:
            print("🎉 Both FIX connections established successfully!")
            print("\nNext steps:")
            print("1. Update your .env file with actual FIX credentials")
            print("2. Run: python scripts/verify_fix_sensory_integration.py")
            print("3. Test market data subscription")
            return True
        else:
            print("⚠ Some connections failed")
            print("\nTroubleshooting:")
            print("1. Check your .env file for FIX credentials")
            print("2. Verify network connectivity to demo-uk-eqx-01.p.c-trader.com")
            print("3. Ensure ports 5211 (price) and 5212 (trade) are accessible")
            return False
            
    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
        return False
    except Exception as e:
        print(f"\nVerification failed: {e}")
        return False


if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)
