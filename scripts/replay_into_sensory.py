#!/usr/bin/env python3
# ruff: noqa: I001

import argparse
import json
import logging
from datetime import datetime

from src.operational.md_capture import MarketDataReplayer
from src.sensory.integration.sensory_cortex import MasterOrchestrator
from src.sensory.organs.dimensions.base_organ import InstrumentMeta, MarketData
from src.trading.order_management.order_book.snapshot import (
    OrderBookLevel,
    OrderBookSnapshot,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay captured MD into sensory cortex")
    p.add_argument("--file", required=True, help="Path to capture.jsonl")
    p.add_argument("--symbol", default="EURUSD")
    p.add_argument("--speed", type=float, default=10.0)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("replay")

    im = InstrumentMeta(symbol=args.symbol, pip_size=0.00001, lot_size=1000)
    mo = MasterOrchestrator(im)

    async def process(symbol: str, ob_dict):
        now = datetime.utcnow()
        md = MarketData(
            symbol=symbol,
            timestamp=now,
            open=0,
            high=0,
            low=0,
            close=0,
            volume=0,
            bid=ob_dict["bids"][0][0] if ob_dict["bids"] else 0.0,
            ask=ob_dict["asks"][0][0] if ob_dict["asks"] else 0.0,
        )
        ob = OrderBookSnapshot(
            symbol=symbol,
            timestamp=now,
            bids=[OrderBookLevel(price=p, volume=v) for p, v in ob_dict["bids"]],
            asks=[OrderBookLevel(price=p, volume=v) for p, v in ob_dict["asks"]],
        )
        res = await mo.update(md, order_book=ob)
        log.info(
            json.dumps(
                {
                    "signal": res.signal_strength,
                    "confidence": res.confidence,
                    "regime": res.regime.value,
                    "what_ms_keys": [
                        k
                        for k in mo.engines["WHAT"].last_reading.context.keys()
                        if k.startswith("ms_")
                    ]
                    if hasattr(mo.engines["WHAT"], "last_reading")
                    else [],
                }
            )
        )

    count = 0

    def cb(symbol, ob):
        nonlocal count
        count += 1
        if count > args.limit:
            return
        import asyncio

        asyncio.get_event_loop().run_until_complete(process(symbol, ob))

    emitted = MarketDataReplayer(args.file, speed=args.speed).replay(cb, max_events=args.limit)
    log.info(f"Replayed {emitted} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
