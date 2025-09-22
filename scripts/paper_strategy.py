#!/usr/bin/env python3

import argparse
import json
import logging
from datetime import datetime

from src.data_foundation.config.vol_config import load_vol_config
from src.operational.md_capture import MarketDataReplayer
from src.sensory.dimensions.microstructure import RollingMicrostructure
from src.sensory.dimensions.what.volatility_engine import vol_signal
from src.sensory.dimensions.why.macro_signal import macro_proximity_signal


def parse_args():
    p = argparse.ArgumentParser(description="Offline paper microstructure strategy")
    p.add_argument("--file", required=True)
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--speed", type=float, default=50.0)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--obi-thresh", type=float, default=0.2, help="mean_obi threshold")
    p.add_argument(
        "--rev-thresh", type=float, default=0.00005, help="mid reversion absolute threshold"
    )
    p.add_argument(
        "--micro-diff-thresh", type=float, default=0.00002, help="microprice-mid absolute threshold"
    )
    p.add_argument("--csv", default="", help="Optional CSV output path for features/trades")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("paper")
    r = RollingMicrostructure(window=50)
    pos = 0
    cash_pnl = 0.0
    trades = 0
    peak = 0.0
    max_dd = 0.0
    # Macro and vol state
    last_macro_ts = None
    next_macros = []
    last_macro_minutes = None
    next_macro_minutes = None
    vol_cfg = load_vol_config()
    daily_returns = []
    rv_window = []
    last_mid = None
    csv_fh = None
    if args.csv:
        import os

        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        csv_fh = open(args.csv, "w", encoding="utf-8")
        csv_fh.write("ts,mid,microprice,spread,mean_obi,mid_reversion,pos,pnl\n")

    def on_event(sym, ob):
        nonlocal pos, cash_pnl
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        feat = r.update(bids, asks)
        mid = feat.get("mid", 0.0)
        micro = feat.get("microprice", 0.0)
        if not mid or not micro:
            return
        # Update return windows for vol estimation
        nonlocal last_mid
        if last_mid:
            r_ret = (mid - last_mid) / last_mid
            daily_returns.append(r_ret)
            rv_window.append(r_ret)
            if len(rv_window) > max(1, int(60 / max(1, vol_cfg.bar_interval_minutes))):
                rv_window.pop(0)
        last_mid = mid
        # Compute macro proximity and gate trading if inside window
        mins_since = None
        mins_to_next = None
        if last_macro_ts:
            try:
                now_ts = datetime.utcnow().timestamp()
                mins_since = (now_ts - last_macro_ts) / 60.0
                future = [x for x in next_macros if x >= now_ts]
                mins_to_next = ((future[0] - now_ts) / 60.0) if future else None
            except Exception:
                pass
        why_sig, why_conf = macro_proximity_signal(mins_since, mins_to_next)
        macro_avoid = why_conf <= 0.3  # avoid trading near macro
        # Vol regime
        try:
            vs = vol_signal(
                sym,
                datetime.utcnow().isoformat(),
                rv_window[-12:] if len(rv_window) >= 12 else rv_window,
                daily_returns[-500:],
            )
            regime = vs.regime
        except Exception:
            regime = "normal"
        mean_obi = feat.get("mean_obi", 0.0)
        rev = feat.get("mid_reversion", 0.0)
        micro_diff = micro - mid
        # Regime-based rules
        target = 0
        if not macro_avoid:
            if regime == "calm":
                # Reversion: fade micro diff if small OBI
                if abs(micro_diff) > args.micro_diff_thresh and abs(mean_obi) < args.obi_thresh:
                    target = -1 if micro_diff > 0 else 1
            elif regime == "storm":
                # Breakout: follow diff with strong OBI
                if abs(mean_obi) > args.obi_thresh:
                    target = 1 if micro_diff > 0 else -1
            else:
                # Normal: combined rule from before
                if mean_obi > args.obi_thresh and micro_diff > args.micro_diff_thresh:
                    target = 1
                elif mean_obi < -args.obi_thresh and micro_diff < -args.micro_diff_thresh:
                    target = -1
        # Reversion override: if strong opposite reversion signal, flatten
        if (
            abs(rev) > args.rev_thresh
            and target != 0
            and ((target == 1 and rev < 0) or (target == -1 and rev > 0))
        ):
            target = 0

        if target != pos:
            # Exit/Enter at mid for accounting
            cash_pnl += pos * (mid)
            pos = target
            cash_pnl -= pos * (mid)
            # track equity curve
            nonlocal peak, max_dd, trades
            trades += 1
            peak = max(peak, cash_pnl)
            max_dd = min(max_dd, cash_pnl - peak)
            log.info(
                json.dumps(
                    {
                        "ts": datetime.utcnow().isoformat(),
                        "mid": mid,
                        "micro": micro,
                        "mean_obi": mean_obi,
                        "rev": rev,
                        "pos": pos,
                        "pnl": cash_pnl,
                        "max_dd": max_dd,
                    }
                )
            )

        if csv_fh:
            csv_fh.write(
                f"{datetime.utcnow().isoformat()},{mid},{micro},{feat.get('spread', 0.0)},{mean_obi},{rev},{pos},{cash_pnl}\n"
            )

    emitted = MarketDataReplayer(args.file, speed=args.speed).replay(
        on_event, max_events=args.limit
    )
    if csv_fh:
        csv_fh.close()
    log.info(
        f"Done. Emitted {emitted} events. Trades={trades} PnL={cash_pnl:.6f} MaxDD={max_dd:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
