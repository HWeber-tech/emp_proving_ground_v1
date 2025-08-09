#!/usr/bin/env python3

import argparse
import json
import logging
import os
from datetime import datetime

from src.data_foundation.replay.multidim_replayer import MultiDimReplayer
from src.sensory.dimensions.microstructure import RollingMicrostructure
from src.data_foundation.persist.parquet_writer import write_events_parquet
from src.sensory.dimensions.why.macro_signal import macro_proximity_signal


def parse_args():
    p = argparse.ArgumentParser(description="Offline backtest report generator")
    p.add_argument("--file", required=True)
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--macro-file", default="", help="Optional macro JSONL file to merge")
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
    # Macro tracking
    last_macro_ts = None
    next_macros = []
    last_macro_minutes = None
    next_macro_minutes = None
    currencies = {args.symbol[:3], args.symbol[3:6]} if len(args.symbol) >= 6 else set()

    def on_md_event(e: dict):
        nonlocal pos, pnl, peak, max_dd
        ob = {"bids": e.get("bids", []), "asks": e.get("asks", [])}
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        f = r.update(bids, asks)
        f["timestamp"] = e.get("timestamp", datetime.utcnow().isoformat())
        f["symbol"] = e.get("symbol", args.symbol)
        # Macro proximity
        if last_macro_ts:
            try:
                md_ts = datetime.fromisoformat(f["timestamp"]).timestamp()
                last_macro_minutes = (md_ts - last_macro_ts) / 60.0
                f["mins_since_macro"] = last_macro_minutes
                # minutes to next
                upcoming = [x for x in next_macros if x >= md_ts]
                next_macro_minutes = ((upcoming[0] - md_ts) / 60.0) if upcoming else None
                f["mins_to_next_macro"] = next_macro_minutes
            except Exception:
                f["mins_since_macro"] = None
                f["mins_to_next_macro"] = None
        else:
            f["mins_since_macro"] = None
            f["mins_to_next_macro"] = None
        f["next_macro_count"] = len(next_macros)
        # WHY macro proximity signal
        why_sig, why_conf = macro_proximity_signal(last_macro_minutes, next_macro_minutes)
        f["why_macro_signal"] = why_sig
        f["why_macro_confidence"] = why_conf
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

    def on_macro_event(e: dict):
        nonlocal last_macro_ts, next_macros
        try:
            cur = str(e.get("currency") or "").upper()
            if currencies and cur and cur not in currencies:
                return
            ts = datetime.fromisoformat(e.get("timestamp")).timestamp()
            last_macro_ts = ts
            # keep a short-term queue of upcoming macro events (count only)
            now = datetime.utcnow().timestamp()
            # purge stale
            next_macros = [x for x in next_macros if x > now]
            # add if in future
            if ts > now:
                next_macros.append(ts)
        except Exception:
            return

    emitted = MultiDimReplayer(md_path=args.file, macro_path=args.macro_file or None).replay(
        on_md=on_md_event, on_macro=on_macro_event, limit=args.limit
    )
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


