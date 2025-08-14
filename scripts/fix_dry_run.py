#!/usr/bin/env python3

import asyncio
import signal
from datetime import datetime

from src.operational.fix_connection_manager import FIXConnectionManager
from src.operational.metrics import start_metrics_server


async def main() -> int:
    # Prefer real FIX by default; set EMP_USE_MOCK_FIX=1 to force the simulator
    start_metrics_server()

    class Cfg:
        environment = "test"
        account_number = "000"
        password = "x"
        use_mock_fix = True

    mgr = FIXConnectionManager(Cfg())
    if not mgr.start_sessions():
        print("Failed to start sessions")
        return 1

    price_q = asyncio.Queue()
    trade_q = asyncio.Queue()
    mgr.get_application("price").set_message_queue(price_q)
    mgr.get_application("trade").set_message_queue(trade_q)

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    print("FIX dry-run started. Press Ctrl+C to stop.")
    last = datetime.utcnow()
    while not stop.is_set():
        try:
            msg = await asyncio.wait_for(price_q.get(), timeout=0.25)
            now = datetime.utcnow()
            print(f"[MD] {now.isoformat()} type={msg.get(35)} entries={len(msg.get('entries', []))}")
            last = now
        except Exception:
            await asyncio.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


