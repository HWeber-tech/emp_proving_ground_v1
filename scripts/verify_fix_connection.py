#!/usr/bin/env python3
"""
FIX Connection Verification Script
Tests the complete FIX protocol integration with IC Markets cTrader
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.governance.system_config import SystemConfig
from src.operational.enhanced_fix_application import EnhancedFIXApplication

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/fix/connection_test.log')
    ]
)

log = logging.getLogger(__name__)


class FIXConnectionTester:
    """Comprehensive FIX connection tester for IC Markets cTrader"""
    
    def __init__(self):
        self.config = SystemConfig()
        self.price_session = None
        self.trade_session = None
        
    def test_price_connection(self) -> bool:
        """Test price FIX connection (port 5211)"""
        log.info("Testing price FIX connection...")
        
        try:
            # Check credentials
            if not self.config.fix_price_sender_comp_id:
                log.error("Price FIX credentials not configured")
                return False
                
            session_config = {
                'SenderCompID': self.config.fix_price_sender_comp_id,
                'Username': self.config.fix_price_username,
                'Password': self.config.fix_price_password
            }
            
            # Create price session
            self.price_session = EnhancedFIXApplication(
                session_config=session_config,
                session_type='price'
            )
            
            # Connect to demo server
            success = self.price_session.start(
                host='demo-uk-eqx-01.p.c-trader.com',
                port=5211
            )
            
            if success:
                log.info("✓ Price FIX connection successful")
                return True
            else:
                log.error("✗ Price FIX connection failed")
                return False
                
        except Exception as e:
            log.error(f"Price FIX connection error: {e}")
            return False
    
    def test_trade_connection(self) -> bool:
        """Test trade FIX connection (port 5212)"""
        log.info("Testing trade FIX connection...")
        
        try:
            # Check credentials
            if not self.config.fix_trade_sender_comp_id:
                log.error("Trade FIX credentials not configured")
                return False
                
            session_config = {
                'SenderCompID': self.config.fix_trade_sender_comp_id,
                'Username': self.config.fix_trade_username,
                'Password': self.config.fix_trade_password
            }
            
            # Create trade session
            self.trade_session = EnhancedFIXApplication(
                session_config=session_config,
                session_type='trade'
            )
            
            # Connect to demo server
            success = self.trade_session.start(
                host='demo-uk-eqx-01.p.c-trader.com',
                port=5212
            )
            
            if success:
                log.info("✓ Trade FIX connection successful")
                return True
            else:
                log.error("✗ Trade FIX connection failed")
                return False
                
        except Exception as e:
            log.error(f"Trade FIX connection error: {e}")
            return False
    
    def run_comprehensive_test(self) -> bool:
        """Run comprehensive FIX connection test"""
        log.info("=" * 60)
        log.info("EMP v4.0 FIX Connection Verification")
        log.info("=" * 60)
        
        results = {
            'price_connection': False,
            'trade_connection': False
        }
        
        # Test price connection
        results['price_connection'] = self.test_price_connection()
        
        # Test trade connection
        results['trade_connection'] = self.test_trade_connection()
        
        # Summary
        log.info("=" * 60)
        log.info("FIX Connection Test Results:")
        log.info(f"Price Connection: {'✓ PASS' if results['price_connection'] else '✗ FAIL'}")
        log.info(f"Trade Connection: {'✓ PASS' if results['trade_connection'] else '✗ FAIL'}")
        
        all_passed = all(results.values())
        if all_passed:
            log.info("🎉 All FIX connections verified successfully!")
        else:
            log.error("❌ Some FIX connections failed. Check logs for details.")
        
        return all_passed
    
    def cleanup(self):
        """Clean up connections"""
        if self.price_session:
            self.price_session.stop()
        if self.trade_session:
            self.trade_session.stop()
    
    def print_setup_instructions(self):
        """Print setup instructions for users"""
        print("\n" + "=" * 60)
        print("FIX CONNECTION SETUP INSTRUCTIONS")
        print("=" * 60)
        print("\n1. Copy .env.example to .env:")
        print("   cp .env.example .env")
        print("\n2. Update your FIX credentials in .env:")
        print("   FIX_PRICE_SENDER_COMP_ID=your_price_sender_id")
        print("   FIX_PRICE_USERNAME=your_price_username")
        print("   FIX_PRICE_PASSWORD=your_price_password")
        print("   FIX_TRADE_SENDER_COMP_ID=your_trade_sender_id")
        print("   FIX_TRADE_USERNAME=your_trade_username")
        print("   FIX_TRADE_PASSWORD=your_trade_password")
        print("\n3. Run the verification:")
        print("   python scripts/verify_fix_connection.py")
        print("\n4. Check logs in logs/fix/ for detailed information")
        print("=" * 60)


def main():
    """Main test runner"""
    tester = FIXConnectionTester()
    
    # Check if credentials are configured
    config = SystemConfig()
    credentials_configured = any([
        config.fix_price_sender_comp_id,
        config.fix_price_username,
        config.fix_price_password,
        config.fix_trade_sender_comp_id,
        config.fix_trade_username,
        config.fix_trade_password
    ])
    
    if not credentials_configured:
        print("\n❌ FIX credentials not configured!")
        tester.print_setup_instructions()
        return
    
    # Run comprehensive test
    success = tester.run_comprehensive_test()
    
    # Wait for connections to establish
    print("\n⏳ Waiting 30 seconds for connections to establish...")
    time.sleep(30)
    
    # Cleanup
    tester.cleanup()
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    import time
    main()
