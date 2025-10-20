#!/usr/bin/env python3
"""
Send OrderMassCancelRequest (q) to cancel all open orders on demo (paper) via FIX.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.env_loader import load_dotenv_if_available, resolve_env_file

from src.governance.system_config import SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager


async def main() -> int:
    _env_path, loaded = load_dotenv_if_available()
    cfg = SystemConfig()
    if str(cfg.run_mode).lower() != "paper":
        print("Aborting: RUN_MODE must be 'paper' for demo execution.")
        return 1
    if not (cfg.account_number and cfg.password):
        message = "Missing ICMARKETS_ACCOUNT or ICMARKETS_PASSWORD"
        if not loaded:
            message += f" in {resolve_env_file()}"
        print(message)
        return 1

    mgr = FIXConnectionManager(cfg)
    if not mgr.start_sessions():
        print("Failed to start FIX sessions")
        return 1

    try:
        manager = mgr._manager
        # Use session-tracked cancel for venues that don't support 35=q
        results = manager.cancel_all_tracked_orders()
        if not results:
            print("No tracked working orders to cancel.")
        else:
            ok = sum(1 for v in results.values() if v)
            total = len(results)
            print(f"Canceled {ok}/{total} tracked orders.")
        await asyncio.sleep(0.5)
        return 0
    finally:
        await asyncio.sleep(0.5)
        mgr.stop_sessions()


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
        sys.exit(rc)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1)
