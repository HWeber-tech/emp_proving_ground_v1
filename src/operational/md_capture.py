"""
Market Data capture and replay utilities.

Recorder attaches to GenuineFIXManager market data callbacks and writes JSONL
snapshots. Replayer reads JSONL and replays to a supplied callback with a
speed factor.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


def _serialize_order_book(symbol: str, order_book) -> Dict[str, Any]:
    def side(entries) -> List[Tuple[float, float]]:
        return [(float(e.price), float(e.size)) for e in entries]
    return {
        "t": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "bids": side(getattr(order_book, "bids", [])[:10]),
        "asks": side(getattr(order_book, "asks", [])[:10]),
    }


class MarketDataRecorder:
    def __init__(self, out_path: str = "data/md_capture/capture.jsonl") -> None:
        self.out_path = out_path
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self._fh = open(self.out_path, "a", encoding="utf-8")

    def attach_to_manager(self, manager) -> None:
        def on_md(symbol: str, order_book) -> None:
            rec = _serialize_order_book(symbol, order_book)
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()
        manager.add_market_data_callback(on_md)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class MarketDataReplayer:
    def __init__(self, in_path: str, speed: float = 1.0) -> None:
        self.in_path = in_path
        self.speed = max(0.0, speed)

    def replay(self, callback: Callable[[str, Dict[str, Any]], None], max_events: Optional[int] = None,
               feature_writer: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> int:
        """Replay captured MD to callback(symbol, order_book_like_dict).

        Returns number of events emitted.
        """
        if not os.path.exists(self.in_path):
            return 0
        emitted = 0
        prev_ts: Optional[float] = None
        with open(self.in_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    symbol = rec.get("symbol")
                    # Simple sleep based on recorded time deltas
                    ts = datetime.fromisoformat(rec.get("t")).timestamp() if rec.get("t") else None
                    if ts is not None and prev_ts is not None and self.speed > 0:
                        delay = max(0.0, (ts - prev_ts) / self.speed)
                        if delay > 0:
                            time.sleep(delay)
                    prev_ts = ts
                    # Convert record into lightweight order_book-like dict
                    ob = {
                        "symbol": symbol,
                        "bids": rec.get("bids", []),
                        "asks": rec.get("asks", []),
                        "last_update": rec.get("t"),
                    }
                    callback(symbol, ob)
                    if feature_writer:
                        try:
                            feature_writer(symbol, ob)
                        except Exception:
                            pass
                    emitted += 1
                    if max_events and emitted >= max_events:
                        break
                except Exception:
                    continue
        return emitted


