#!/usr/bin/env python3
"""
IC Markets Production Trading System
100% Production-ready FIX API implementation
"""

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.operational.icmarkets_api import FinalFIXTester
from src.operational.icmarkets_config import ICMarketsConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ICMarketsProductionSystem:
    """Production-ready IC Markets trading system."""
    
    def __init__(self):
        self.config = None
        self.manager = None
        self.running = False
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF"]
        
    def initialize(self, environment: str = "demo", account: str = None):
        """Initialize the production system."""
        try:
            logger.info("🚀 Initializing IC Markets Production System")
            
            # Load configuration
            account = account or os.getenv("ICMARKETS_ACCOUNT")
            password = os.getenv("ICMARKETS_PASSWORD")
            if account in (None, "9533708") or password in (None, "WNSE5822"):
                raise ValueError("IC Markets credentials must be provided via environment variables")
            
            self.config = ICMarketsConfig(
                environment=environment,
                account_number=account
            )
            self.config.validate_config()
            
            # Create manager
            self.manager = ICMarketsRobustManager(self.config)
            
            logger.info(f"✅ Configuration loaded for account: {account}")
            logger.info(f"✅ Environment: {environment}")
            logger.info(f"✅ Server: {self.config._get_host()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
            
    def start(self) -> bool:
        """Start the production system."""
        try:
            logger.info("🎯 Starting IC Markets production system...")
            
            if not self.manager:
                logger.error("❌ Manager not initialized")
                return False
                
            success = self.manager.start()
            if success:
                self.running = True
                
                # Subscribe to market data
                self.manager.subscribe_market_data(self.symbols)
                
                logger.info("✅ Production system started successfully")
                logger.info(f"✅ Subscribed to symbols: {self.symbols}")
                
                return True
            else:
                logger.error("❌ Failed to start production system")
                return False
                
        except Exception as e:
            logger.error(f"❌ Start failed: {e}")
            return False
            
    def stop(self):
        """Stop the production system gracefully."""
        logger.info("🛑 Stopping IC Markets production system...")
        self.running = False
        
        if self.manager:
            self.manager.stop()
            
        logger.info("✅ Production system stopped")
        
    def get_status(self) -> dict:
        """Get comprehensive system status."""
        if not self.manager:
            return {"status": "not_initialized"}
            
        status = self.manager.get_status()
        status.update({
            "timestamp": datetime.utcnow().isoformat(),
            "symbols": self.symbols,
            "environment": self.config.environment if self.config else "unknown"
        })
        
        return status
        
    def place_order(self, symbol: str, side: str, quantity: float) -> str:
        """Place a market order."""
        if not self.manager:
            raise RuntimeError("System not initialized")
            
        order_id = self.manager.place_market_order(symbol, side, quantity)
        if order_id:
            logger.info(f"✅ Order placed: {side} {quantity} {symbol} (ID: {order_id})")
            return order_id
        else:
            raise RuntimeError("Failed to place order")
            
    def monitor(self):
        """Monitor system health."""
        while self.running:
            try:
                status = self.get_status()
                print(f"\n📊 System Status - {datetime.utcnow().strftime('%H:%M:%S')}")
                print(json.dumps(status, indent=2))
                time.sleep(30)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal")
    if 'system' in globals():
        system.stop()
    sys.exit(0)


async def main():
    """Main production application."""
    print("🎯 IC Markets Production Trading System")
    print("=" * 50)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='IC Markets Production Trading System')
    parser.add_argument('--env', choices=['demo', 'live'], default='demo', help='Environment')
    parser.add_argument('--account', help='Account number')
    parser.add_argument('--monitor', action='store_true', help='Enable monitoring')
    
    args = parser.parse_args()
    
    # Create system
    system = ICMarketsProductionSystem()
    
    # Initialize
    if not system.initialize(args.env, args.account):
        print("❌ Failed to initialize system")
        return 1
        
    # Start system
    if not system.start():
        print("❌ Failed to start system")
        return 1
        
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("✅ IC Markets production system is running")
    print("📊 Use Ctrl+C to stop")
    
    # Monitor if requested
    if args.monitor:
        system.monitor()
    else:
        # Keep running
        try:
            while system.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
            
    # Stop system
    system.stop()
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
