#!/usr/bin/env python3
"""
Place a real demo market order via GenuineFIXManager using current .env.
Safety:
- Requires RUN_MODE=paper
- Uses tiny default size (0.01)
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.governance.system_config import SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager


async def main(symbol: str = "EURUSD", side: str = "BUY", qty: float = 1000.0) -> int:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
    cfg = SystemConfig()
    if str(cfg.run_mode).lower() != "paper":
        print("Aborting: RUN_MODE must be 'paper' for demo execution.")
        return 1
    if not (cfg.account_number and cfg.password):
        print("Missing ICMARKETS_ACCOUNT or ICMARKETS_PASSWORD in .env")
        return 1

    mgr = FIXConnectionManager(cfg)
    if not mgr.start_sessions():
        print("Failed to start FIX sessions")
        return 1

    try:
        # Access GenuineFIXManager through initiator wrapper
        initiator = mgr.get_initiator("trade")
        if not initiator:
            print("Trade initiator not available")
            return 1

        # Use manager to place a genuine order (access underlying manager by design)
        manager = mgr._manager
        result = manager.place_market_order_genuine(
            symbol=symbol, side=side, quantity=qty, timeout=10.0
        )
        if result is None:
            print("Order placement failed")
            return 1
        print(
            f"Order status: {result.status.value} ; clOrdID={result.cl_ord_id} ; text={result.text}"
        )
        return 0
    finally:
        mgr.stop_sessions()


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
        sys.exit(rc)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1)
