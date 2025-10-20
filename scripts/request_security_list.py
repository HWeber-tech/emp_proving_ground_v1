#!/usr/bin/env python3
"""
Request Security List (35=x) to discover valid symbol identifiers for MD subscription.
"""

import asyncio
import os
import sys

import simplefix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.env_loader import load_dotenv_if_available, resolve_env_file

from src.governance.system_config import SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager


async def main() -> int:
    _env_path, loaded = load_dotenv_if_available()
    cfg = SystemConfig()
    mgr = FIXConnectionManager(cfg)
    if not mgr.start_sessions():
        print("Failed to start FIX sessions")
        return 1
    try:
        # Build SecurityListRequest to quote session
        req_id = "SEC_LIST_0"
        msg = simplefix.FixMessage()
        msg.append_pair(8, "FIX.4.4")
        msg.append_pair(35, "x")  # SecurityListRequest
        if not cfg.account_number:
            account_error = "Missing ICMARKETS_ACCOUNT for security list request"
            if not loaded:
                account_error += f" in {resolve_env_file()}"
            print(account_error)
            return 1

        msg.append_pair(49, f"demo.icmarkets.{cfg.account_number}")
        msg.append_pair(56, "cServer")
        msg.append_pair(57, "QUOTE")
        msg.append_pair(50, "QUOTE")
        msg.append_pair(320, req_id)  # SecurityReqID
        msg.append_pair(559, "0")  # SecurityListRequestType = 0 (symbol)

        ok = mgr._manager.price_connection.send_message_and_track(msg, req_id)
        print("SecurityListRequest sent:", ok)
        await asyncio.sleep(3)
        # Print discovered sample projection (if captured)
        samples = (
            mgr._manager.security_list_samples
            if hasattr(mgr._manager, "security_list_samples")
            else []
        )
        if samples:
            print("SecurityList first-sample projection:", samples[0])
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
