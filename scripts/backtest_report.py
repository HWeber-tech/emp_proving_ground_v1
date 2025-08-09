#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Ensure project root on path when executed as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_foundation.replay.multidim_replayer import MultiDimReplayer
from src.sensory.dimensions.microstructure import RollingMicrostructure
from src.data_foundation.persist.parquet_writer import write_events_parquet
from src.sensory.dimensions.why.macro_signal import macro_proximity_signal
from src.sensory.dimensions.what.volatility_engine import vol_signal
from src.data_foundation.config.vol_config import load_vol_config
from src.sensory.dimensions.why.yield_signal import YieldSlopeTracker
from src.data_foundation.config.why_config import load_why_config


def parse_args():
    p = argparse.ArgumentParser(description="Offline backtest report generator")
    p.add_argument("--file", required=True)
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--macro-file", default="", help="Optional macro JSONL file to merge")
    p.add_argument("--yields-file", default="", help="Optional yields JSONL file to merge (for slope)")
    # WHY overrides
    p.add_argument("--why-weight-macro", type=float, default=None, help="Override WHY macro weight")
    p.add_argument("--why-weight-yields", type=float, default=None, help="Override WHY yields weight")
    p.add_argument("--disable-why-macro", action="store_true", help="Disable macro proximity in WHY")
    p.add_argument("--disable-why-yields", action="store_true", help="Disable yield features in WHY")
    p.add_argument("--speed", type=float, default=100.0)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--out-dir", default="docs/reports/backtests")
    p.add_argument("--parquet", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("backtest")
    r = RollingMicrostructure(window=50)
    ytracker = YieldSlopeTracker()
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
    # Volatility and WHY config
    vol_cfg = load_vol_config()
    why_cfg = load_why_config()
    # CLI overrides for WHY config
    if args.why_weight_macro is not None:
        try:
            why_cfg.weight_macro = float(args.why_weight_macro)
        except Exception:
            pass
    if args.why_weight_yields is not None:
        try:
            why_cfg.weight_yields = float(args.why_weight_yields)
        except Exception:
            pass
    if args.disable_why_macro:
        why_cfg.enable_macro_proximity = False
    if args.disable_why_yields:
        why_cfg.enable_yields = False
    daily_returns = []
    rv_window = []
    last_mid = None

    def on_md_event(e: dict):
        nonlocal pos, pnl, peak, max_dd, last_macro_minutes, next_macro_minutes, last_mid
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
        macro_sig, macro_conf = (0.0, 0.0)
        if why_cfg.enable_macro_proximity:
            macro_sig, macro_conf = macro_proximity_signal(last_macro_minutes, next_macro_minutes)
        f["why_macro_signal"] = macro_sig
        f["why_macro_confidence"] = macro_conf
        # WHY yield-curve features
        y_sig, y_conf = (0.0, 0.0)
        if why_cfg.enable_yields:
            y_sig, y_conf = ytracker.signal()
            f["why_yield_slope_2s10s"] = ytracker.slope_2s10s()
            f["why_yield_slope_5s30s"] = ytracker.slope_5s30s()
            f["why_yield_curvature_2_10_30"] = ytracker.curvature_2_10_30()
            f["why_yield_parallel_shift"] = ytracker.parallel_shift()
        f["why_yield_signal"] = y_sig
        f["why_yield_confidence"] = y_conf
        # WHAT microstructure signal (simple heuristic)
        what_sig = 0.0
        what_conf = 0.5
        mid = f.get("mid", 0.0)
        micro = f.get("microprice", 0.0)
        if last_mid and mid:
            r_ret = (mid - last_mid) / last_mid
            daily_returns.append(r_ret)
            rv_window.append(r_ret)
            # keep ~60m window for 5m bars
            max_rv_len = max(1, int(60 / max(1, vol_cfg.bar_interval_minutes) * (vol_cfg.bar_interval_minutes / vol_cfg.bar_interval_minutes)))
            if len(rv_window) > max_rv_len:
                rv_window.pop(0)
        last_mid = mid if mid else last_mid
        if mid and micro:
            diff = micro - mid
            what_sig = 1.0 if diff > 0 else -1.0
            # confidence scales with absolute diff and imbalance
            what_conf = min(1.0, max(0.1, abs(diff) * 1e5 + abs(f.get("top_imbalance", 0.0))))
        f["what_signal"] = what_sig
        f["what_confidence"] = what_conf
        # Volatility signal (Tier-0)
        try:
            vs = vol_signal(args.symbol, f["timestamp"], rv_window[-12:] if len(rv_window) >= 12 else rv_window, daily_returns[-500:])
            f["sigma_ann"] = vs.sigma_ann
            f["regime"] = vs.regime
            f["sizing_multiplier"] = vs.sizing_multiplier
        except Exception:
            f["sigma_ann"] = None
            f["regime"] = "unknown"
            f["sizing_multiplier"] = 0.5
        # Composite signal (confidence-weighted)
        # WHY composite with weights
        why_w_total = (why_cfg.weight_macro if macro_conf > 0 else 0.0) + (why_cfg.weight_yields if y_conf > 0 else 0.0)
        why_comp = 0.0
        if why_w_total > 0:
            why_comp = (
                (macro_sig * macro_conf * why_cfg.weight_macro) +
                (y_sig * y_conf * why_cfg.weight_yields)
            ) / why_w_total
        f["why_composite_signal"] = why_comp
        # Final composite across WHAT and WHY
        total_w = what_conf + (macro_conf + y_conf)
        comp_num = (what_sig * what_conf) + (macro_sig * macro_conf) + (y_sig * y_conf)
        comp = (comp_num / total_w) if total_w > 0 else 0.0
        f["composite_signal"] = comp
        # Simple paper PnL accounting using mid with regime gate option
        if mid and micro:
            # naive rule: if micro > mid → long, else short
            target = 1 if (micro - mid) > 0 else -1
            # Apply regime gate: block entries in blocked regime
            try:
                if vol_cfg.use_regime_gate and f.get("regime") == getattr(vol_cfg, "block_regime", "storm"):
                    target = 0
            except Exception:
                pass
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

    def on_yield_event(e: dict):
        tenor = str(e.get("tenor") or "").upper()
        val = e.get("value")
        if not tenor or val is None:
            return
        try:
            ytracker.update(tenor, float(val))
        except Exception:
            return

    ypath = args.yields_file or None
    emitted = MultiDimReplayer(md_path=args.file, macro_path=args.macro_file or None, yields_path=ypath).replay(
        on_md=on_md_event, on_macro=on_macro_event, on_yield=on_yield_event, limit=args.limit
    )
    os.makedirs(args.out_dir, exist_ok=True)
    # Aggregate simple stats
    regimes = {"calm": 0, "normal": 0, "storm": 0}
    for f in feats:
        reg = f.get("regime")
        if reg in regimes:
            regimes[reg] += 1
    report = {
        "file": args.file,
        "symbol": args.symbol,
        "events": emitted,
        "pnl": pnl,
        "max_dd": max_dd,
        "timestamp": datetime.utcnow().isoformat(),
        "regimes": regimes,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    if args.parquet and feats:
        write_events_parquet(feats, os.path.join(args.out_dir, "features"), partition=args.symbol)
    # Write Markdown summary
    md = []
    md.append(f"# Backtest Summary\n\n")
    md.append(f"- **Symbol**: {args.symbol}\n")
    md.append(f"- **Events**: {emitted}\n")
    md.append(f"- **PnL**: {pnl:.6f}\n")
    md.append(f"- **Max DD**: {max_dd:.6f}\n")
    md.append(f"- **Regimes**: calm={regimes['calm']}, normal={regimes['normal']}, storm={regimes['storm']}\n")
    # WHY feature summary
    if feats:
        def mean_key(k: str):
            vals = [f[k] for f in feats if f.get(k) is not None]
            return (sum(vals) / len(vals)) if vals else None
        why_avg = mean_key("why_composite_signal")
        s21 = mean_key("why_yield_slope_2s10s")
        s530 = mean_key("why_yield_slope_5s30s")
        curv = mean_key("why_yield_curvature_2_10_30")
        pshift = mean_key("why_yield_parallel_shift")
        md.append("\n## WHY Features\n\n")
        md.append(f"- **WHY composite (avg)**: {why_avg:.6f}\n" if why_avg is not None else "")
        md.append(f"- **2s10s slope (avg)**: {s21:.6f}\n" if s21 is not None else "")
        md.append(f"- **5s30s slope (avg)**: {s530:.6f}\n" if s530 is not None else "")
        md.append(f"- **2-10-30 curvature (avg)**: {curv:.6f}\n" if curv is not None else "")
        md.append(f"- **Parallel shift proxy (avg)**: {pshift:.6f}\n" if pshift is not None else "")
    with open(os.path.join(args.out_dir, "BACKTEST_SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(md))
    log.info(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


