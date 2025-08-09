#!/usr/bin/env python3

import argparse
import json
import logging
from datetime import datetime

from src.operational.md_capture import MarketDataReplayer
from src.sensory.dimensions.microstructure import RollingMicrostructure


def parse_args():
    p = argparse.ArgumentParser(description="Offline paper microstructure strategy")
    p.add_argument("--file", required=True)
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--speed", type=float, default=50.0)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--threshold", type=float, default=0.2, help="mean_obi threshold")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("paper")
    r = RollingMicrostructure(window=50)
    pos = 0
    cash_pnl = 0.0

    def on_event(sym, ob):
        nonlocal pos, cash_pnl
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        feat = r.update(bids, asks)
        mid = feat.get("mid", 0.0)
        if mid == 0:
            return
        # Simple rule: if mean_obi > threshold go long, < -threshold go short, else flatten
        mean_obi = feat.get("mean_obi", 0.0)
        target = 0
        if mean_obi > args.threshold:
            target = 1
        elif mean_obi < -args.threshold:
            target = -1
        if target != pos:
            # Exit/Enter at mid for accounting
            cash_pnl += pos * (mid)
            pos = target
            cash_pnl -= pos * (mid)
            log.info(json.dumps({"ts": datetime.utcnow().isoformat(), "mid": mid, "mean_obi": mean_obi, "pos": pos, "pnl": cash_pnl}))

    emitted = MarketDataReplayer(args.file, speed=args.speed).replay(on_event, max_events=args.limit)
    log.info(f"Done. Emitted {emitted} events. PnL={cash_pnl:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


