from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class MarketTick:
    symbol: str
    price: float
    volume: int
    timestamp: datetime


class YahooFinanceDataSource:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def get_market_data(self, symbol: str) -> List[MarketTick]:
        # Placeholder implementation for compatibility; real ingest lives in
        # src/data_foundation/ingest/yahoo_ingest.py
        return []


