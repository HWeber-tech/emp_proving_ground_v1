"""
Core models shim (sensory-independent)
=====================================

This module provides minimal, standalone core representations that are
safe for use across layers without importing from higher layers (e.g. sensory).
It intentionally avoids any imports from src.sensory.* to satisfy layered architecture.

Provided:
- MarketRegime (Enum)
- DimensionalReading (with legacy .value alias for signal_strength)
- MarketData (wrapper that accepts legacy constructor fields and normalizes)
- InstrumentMeta (placeholder metadata container)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MarketRegime(Enum):
    UNKNOWN = "unknown"
    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    CONSOLIDATING = "consolidating"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    EXHAUSTED = "exhausted"


@dataclass
class InstrumentMeta:
    """Lightweight instrument metadata placeholder."""
    symbol: str = "UNKNOWN"
    tick_size: float = 0.0
    lot_size: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionalReading:
    """
    Minimal core representation of a dimensional reading.

    Notes:
    - Provides a legacy `.value` property that aliases to `signal_strength`.
    """
    dimension: str
    signal_strength: float
    confidence: float = 0.0
    regime: MarketRegime = MarketRegime.UNKNOWN
    context: Dict[str, Any] = field(default_factory=dict)
    data_quality: float = 1.0
    processing_time_ms: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def value(self) -> float:
        """Legacy alias for signal_strength."""
        return float(self.signal_strength)


class MarketData:
    """
    Core market data wrapper that accepts legacy constructor arguments
    and normalizes them to a consistent shape.

    Supported legacy fields:
      - symbol: str
      - timestamp: datetime
      - bid: float
      - ask: float
      - volume: float
      - volatility: float (ignored; accepted for compatibility)
      - price: float (used if bid/ask missing to infer OHLC)
      - open/high/low/close: optional; inferred when absent
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Symbol
        self.symbol: str = str(kwargs.pop("symbol", "UNKNOWN"))

        # Time
        ts_raw: Any = kwargs.pop("timestamp", datetime.utcnow())
        self.timestamp: datetime = ts_raw if isinstance(ts_raw, datetime) else datetime.utcnow()

        # Prices and helpers
        def _to_float(x: Optional[Union[float, int, str]], default: float = 0.0) -> float:
            if x is None:
                return default
            try:
                return float(x)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        val_price: Optional[Union[float, int, str]] = kwargs.pop("price", None)
        val_bid: Optional[Union[float, int, str]] = kwargs.pop("bid", None)
        val_ask: Optional[Union[float, int, str]] = kwargs.pop("ask", None)

        self.bid: float = _to_float(val_bid)
        self.ask: float = _to_float(val_ask, default=self.bid if self.bid else 0.0)

        # Close inferred from mid or provided fields
        close_in: Optional[Union[float, int, str]] = kwargs.pop("close", None)
        self.close: float = _to_float(close_in)
        if self.close == 0.0:
            if self.bid or self.ask:
                self.close = (self.bid + self.ask) / 2.0
            else:
                self.close = _to_float(val_price, default=0.0)

        # OHLC fallbacks
        open_in: Optional[Union[float, int, str]] = kwargs.pop("open", None)
        high_in: Optional[Union[float, int, str]] = kwargs.pop("high", None)
        low_in: Optional[Union[float, int, str]] = kwargs.pop("low", None)

        self.open: float = _to_float(open_in, default=self.close)
        self.high: float = _to_float(high_in, default=max(self.open, self.close))
        self.low: float = _to_float(low_in, default=min(self.open, self.close))

        # Volume
        self.volume: float = _to_float(kwargs.pop("volume", None), default=0.0)

        # Accept and ignore legacy extras
        kwargs.pop("volatility", None)  # accepted but not used

        # Retain any additional fields as passthrough (non-breaking)
        for k, v in kwargs.items():
            try:
                setattr(self, k, v)
            except Exception:
                # Ignore non-assignable extras
                pass


__all__ = ["DimensionalReading", "MarketData", "MarketRegime", "InstrumentMeta"]