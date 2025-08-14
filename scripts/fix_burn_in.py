#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import threading
import time

from src.operational.icmarkets_api import GenuineFIXManager
from src.operational.icmarkets_config import ICMarketsConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FIX reconnect burn-in tester")
    p.add_argument("--cycles", type=int, default=10, help="Number of start/stop cycles")
    p.add_argument("--md-symbol", type=str, default="EURUSD", help="Optional symbol to subscribe for MD per cycle")
    p.add_argument("--md", action="store_true", help="Subscribe to market data each cycle")
    p.add_argument("--sleep", type=float, default=2.0, help="Seconds to keep sessions up per cycle")
    p.add_argument("--between", type=float, default=1.0, help="Seconds between cycles")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("burn_in")

    account = os.environ.get("ACCOUNT_NUMBER") or os.environ.get("ICMARKETS_ACCOUNT")
    password = os.environ.get("FIX_PASSWORD") or os.environ.get("ICMARKETS_PASSWORD")
    if not account or not password:
        log.error("Missing ACCOUNT_NUMBER/ICMARKETS_ACCOUNT or FIX_PASSWORD/ICMARKETS_PASSWORD env vars")
        return 2

    base_threads = len(threading.enumerate())
    failures = 0
    thread_leaks = 0

    for i in range(1, args.cycles + 1):
        t0 = time.time()
        try:
            cfg = ICMarketsConfig(environment="demo", account_number=account)
            cfg.password = password
            mgr = GenuineFIXManager(cfg)
            if not mgr.start():
                failures += 1
                log.error(f"Cycle {i}: start failed")
                continue
            if args.md:
                try:
                    sub = mgr.subscribe_market_data_genuine([args.md_symbol], timeout=5.0)
                    log.info(f"Cycle {i}: MD subscribe result: {sub}")
                except Exception as e:
                    log.warning(f"Cycle {i}: MD subscribe error: {e}")
            time.sleep(max(0.0, args.sleep))
        except Exception as e:
            failures += 1
            log.error(f"Cycle {i}: exception: {e}")
        finally:
            try:
                mgr.stop()
            except Exception:
                pass
            time.sleep(max(0.0, args.between))
            # Thread leak check (best-effort)
            threads_now = len(threading.enumerate())
            if threads_now > base_threads + 2:  # allow small fluctuations
                thread_leaks += 1
                log.warning(f"Cycle {i}: potential thread leak (baseline={base_threads}, now={threads_now})")
            dt = time.time() - t0
            log.info(f"Cycle {i} completed in {dt:.2f}s")

    log.info(f"Burn-in done: cycles={args.cycles}, failures={failures}, thread_leaks={thread_leaks}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


