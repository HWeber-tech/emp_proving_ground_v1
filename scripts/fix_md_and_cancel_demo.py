#!/usr/bin/env python3
"""
Subscribe to FIX market data for EURUSD and place + cancel a demo order (paper mode).
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.env_loader import load_dotenv_if_available, resolve_env_file

from src.governance.system_config import SystemConfig
from src.operational.fix_connection_manager import FIXConnectionManager


async def main(symbol: str = "EURUSD", side: str = "BUY", qty: float = 1000.0) -> int:
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
        manager = mgr._manager  # underlying GenuineFIXManager

        # Subscribe market data (will use MDReq builder with fallback)
        sub = manager.subscribe_market_data_genuine([symbol], timeout=6.0)
        print("MD subscribe result:", sub)

        # Place a resting limit order far from market to test cancel reliably
        # Build a synthetic price: for BUY limit, set a very low price; for SELL limit, set very high
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        import simplefix

        # Use direct message for limit order (OrdType=2)
        now_utc = _dt.now(_tz.utc)
        cl_id = f"LIM_{int(now_utc.timestamp() * 1000)}"
        price = 0.5 if side.upper() == "BUY" else 2.0  # far from FX spot to avoid fill
        msg = simplefix.FixMessage()
        msg.append_pair(8, "FIX.4.4")
        msg.append_pair(35, "D")  # NewOrderSingle
        msg.append_pair(49, f"demo.icmarkets.{cfg.account_number}")
        msg.append_pair(56, "cServer")
        msg.append_pair(57, "TRADE")
        msg.append_pair(50, "TRADE")
        msg.append_pair(11, cl_id)
        # Symbol numeric (EURUSD assumed 1)
        msg.append_pair(55, "1")
        msg.append_pair(54, "1" if side.upper() == "BUY" else "2")
        msg.append_pair(38, str(qty))
        msg.append_pair(40, "2")  # Limit
        msg.append_pair(44, str(price))  # Price
        msg.append_pair(59, "0")  # Day
        msg.append_pair(60, now_utc.strftime("%Y%m%d-%H:%M:%S.%f")[:-3])
        ok = manager.trade_connection.send_message_and_track(msg, cl_id)
        if not ok:
            print("Failed to send limit order")
            return 1
        print("Limit order sent:", cl_id, "price=", price)

        # Emulate a minimal NEW wait (demo may need to register)
        await asyncio.sleep(1.0)

        # Wrap order object stub for downstream cancel code
        class _Order:
            def __init__(self, cl, oid=None):
                self.cl_ord_id = cl
                self.order_id = oid
                self.status = type("S", (), {"value": "0"})()  # New

        order = _Order(cl_id)
        if not order:
            print("Order failed")
            return 1
        print("Order status:", order.status.value, order.cl_ord_id)

        # Optional: Order Status Request gate (belt-and-suspenders)
        # Send via manager helper (omits 55, includes 54)
        manager.send_order_status_request(order.cl_ord_id, "1" if side.upper() == "BUY" else "2")
        await asyncio.sleep(0.3)

        # Attempt cancel if NEW or PARTIAL (delay briefly)
        if order.status.value in {"0", "A", "1"}:  # New, PendingNew, Partial
            # minimal settle for demo; reduce if not needed
            await asyncio.sleep(0.3)
            import simplefix

            # First attempt: 11/41 only
            cncl_id_1 = f"CNCL_{order.cl_ord_id}"
            msg1 = simplefix.FixMessage()
            msg1.append_pair(8, "FIX.4.4")
            msg1.append_pair(35, "F")
            msg1.append_pair(49, f"demo.icmarkets.{cfg.account_number}")
            msg1.append_pair(56, "cServer")
            msg1.append_pair(57, "TRADE")
            msg1.append_pair(50, "TRADE")
            msg1.append_pair(11, cncl_id_1)
            msg1.append_pair(41, order.cl_ord_id)
            ok1 = manager.trade_connection.send_message_and_track(msg1, cncl_id_1)
            print("Cancel #1 (11/41) sent:", ok1)
            # brief wait for response
            await asyncio.sleep(0.7)
            # Retry once with OrderID only if last business reject indicates ORDER_NOT_FOUND
            rej = getattr(manager, "last_business_reject", None)
            if rej and "ORDER_NOT_FOUND" in str(rej):
                if getattr(order, "order_id", None):
                    cncl_id_2 = f"CNCL2_{order.cl_ord_id}"
                    msg2 = simplefix.FixMessage()
                    msg2.append_pair(8, "FIX.4.4")
                    msg2.append_pair(35, "F")
                    msg2.append_pair(49, f"demo.icmarkets.{cfg.account_number}")
                    msg2.append_pair(56, "cServer")
                    msg2.append_pair(57, "TRADE")
                    msg2.append_pair(50, "TRADE")
                    msg2.append_pair(11, cncl_id_2)
                    msg2.append_pair(41, order.cl_ord_id)
                    msg2.append_pair(37, order.order_id)
                    ok2 = manager.trade_connection.send_message_and_track(msg2, cncl_id_2)
                    print("Cancel #2 (with 37) sent:", ok2)
                    await asyncio.sleep(0.7)

        return 0
    finally:
        # keep session alive a bit longer to receive cancel responses
        await asyncio.sleep(1.5)
        mgr.stop_sessions()


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
        sys.exit(rc)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1)
