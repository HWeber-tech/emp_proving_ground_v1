#!/usr/bin/env python3

import argparse
import json
import logging
import os
from datetime import datetime

from src.operational.md_capture import MarketDataReplayer
from src.sensory.dimensions.microstructure import RollingMicrostructure
from src.data_foundation.persist.parquet_writer import write_events_parquet


def parse_args():
    p = argparse.ArgumentParser(description="Offline backtest report generator")
    p.add_argument("--file", required=True)
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--speed", type=float, default=100.0)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--out-dir", default="reports/backtests")
    p.add_argument("--parquet", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("backtest")
    r = RollingMicrostructure(window=50)
    pos = 0
    pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    feats = []

    def on_event(sym, ob):
        nonlocal pos, pnl, peak, max_dd
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        f = r.update(bids, asks)
        f["timestamp"] = datetime.utcnow().isoformat()
        f["symbol"] = sym
        # Simple paper PnL accounting using mid
        mid = f.get("mid", 0.0)
        micro = f.get("microprice", 0.0)
        if mid and micro:
            # naive rule: if micro > mid → long, else short
            target = 1 if (micro - mid) > 0 else -1
            if target != pos:
                pnl += pos * mid
                pos = target
                pnl -= pos * mid
                peak = max(peak, pnl)
                max_dd = min(max_dd, pnl - peak)
            f["pos"] = pos
            f["pnl"] = pnl
        feats.append(f)

    emitted = MarketDataReplayer(args.file, speed=args.speed).replay(on_event, max_events=args.limit)
    os.makedirs(args.out_dir, exist_ok=True)
    report = {
        "file": args.file,
        "symbol": args.symbol,
        "events": emitted,
        "pnl": pnl,
        "max_dd": max_dd,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    if args.parquet and feats:
        write_events_parquet(feats, os.path.join(args.out_dir, "features"), partition=args.symbol)
    log.info(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


