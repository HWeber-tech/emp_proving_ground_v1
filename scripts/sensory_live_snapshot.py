#!/usr/bin/env python3

import argparse
import asyncio
import random
from datetime import datetime

from src.operational.metrics import start_metrics_server
from src.sensory.core.base import InstrumentMeta, MarketData
from src.sensory.integration.sensory_cortex import MasterOrchestrator


def parse_args():
    p = argparse.ArgumentParser(description="Live-like sensory snapshot (mock data)")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between updates")
    p.add_argument("--steps", type=int, default=60, help="Number of iterations (-1 for infinite)")
    p.add_argument("--metrics", action="store_true", help="Start Prometheus metrics server")
    return p.parse_args()


def _synthetic_md(symbol: str, prev_close: float | None) -> MarketData:
    now = datetime.utcnow()
    base = prev_close if prev_close else 1.1000
    # small drift and noise
    rnd = random.gauss(0.0, 0.0002)
    close = max(0.2, base + rnd)
    high = max(close, base) + abs(random.gauss(0.0, 0.0001))
    low = min(close, base) - abs(random.gauss(0.0, 0.0001))
    bid = close - 0.00005
    ask = close + 0.00005
    vol = abs(random.gauss(10000, 3000))
    return MarketData(
        symbol=symbol,
        timestamp=now,
        open=base,
        high=high,
        low=low,
        close=close,
        volume=vol,
        bid=bid,
        ask=ask,
        source="mock",
    )


async def main() -> int:
    args = parse_args()
    if args.metrics:
        start_metrics_server()

    instrument_meta = InstrumentMeta(
        symbol=args.symbol, pip_size=0.0001, lot_size=100000, timezone="UTC"
    )
    orchestrator = MasterOrchestrator(instrument_meta)

    prev_close = None
    n = 0
    print("Starting sensory live snapshot (mock)... Ctrl+C to stop.")
    try:
        while args.steps < 0 or n < args.steps:
            md = _synthetic_md(args.symbol, prev_close)
            prev_close = md.close
            result = await orchestrator.update(md)
            # Render concise snapshot
            ts = md.timestamp.isoformat()
            sig = f"{result.signal_strength:.3f}"
            conf = f"{result.confidence:.3f}"
            reg = getattr(result.regime, "value", str(result.regime))
            why_contrib = result.dimensional_contributions.get("WHY", 0.0)
            what_contrib = result.dimensional_contributions.get("WHAT", 0.0)
            print(
                f"[{ts}] {args.symbol} sig={sig} conf={conf} reg={reg} WHY={why_contrib:.3f} WHAT={what_contrib:.3f}"
            )
            await asyncio.sleep(max(0.0, args.interval))
            n += 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
