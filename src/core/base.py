"""
Core models shim (sensory-independent)
=====================================

This module provides minimal, standalone core representations that are
safe for use across layers without importing from higher layers (e.g. sensory).
It intentionally avoids any imports from src.sensory.* to satisfy layered architecture.

Provided:
- MarketRegime (Enum) [with legacy aliases]
- DimensionalReading (with legacy .value alias for signal_strength)
- SensoryReading (dataclass used by organs; matches call sites)
- MarketData (wrapper that accepts legacy constructor fields and normalizes)
- InstrumentMeta (placeholder metadata container)
- SensoryOrgan (Protocol; very permissive to avoid false positives)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, TypeAlias, Union, cast, runtime_checkable


class MarketRegime(Enum):
    UNKNOWN = "unknown"
    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    CONSOLIDATING = "consolidating"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    EXHAUSTED = "exhausted"
    # Legacy aliases used across modules
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


@dataclass
class InstrumentMeta:
    """Lightweight instrument metadata placeholder."""

    symbol: str = "UNKNOWN"
    tick_size: float = 0.0
    lot_size: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)


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
    context: dict[str, object] = field(default_factory=dict)
    data_quality: float = 1.0
    processing_time_ms: float = 0.0
    evidence: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def value(self) -> float:
        """Legacy alias for signal_strength."""
        return float(self.signal_strength)


@dataclass
class SensoryReading:
    """
    Canonical sensory reading used by organs.* modules.

    Matches construction pattern:
        SensoryReading(organ_name=..., timestamp=..., data={...}, metadata={...})
    """

    organ_name: str
    timestamp: datetime
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


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
      - volatility: float (accepted and stored for compatibility)
      - price: float (used if bid/ask missing to infer OHLC)
      - open/high/low/close: optional; inferred when absent
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        # Symbol
        self.symbol: str = str(kwargs.pop("symbol", "UNKNOWN"))

        # Time
        ts_raw: object = kwargs.pop("timestamp", datetime.utcnow())
        self.timestamp: datetime = ts_raw if isinstance(ts_raw, datetime) else datetime.utcnow()

        # Prices and helpers
        def _to_float(x: Optional[Union[float, int, str]], default: float = 0.0) -> float:
            if x is None:
                return default
            try:
                return float(x)
            except (TypeError, ValueError):
                return default

        val_price = cast(Optional[Union[float, int, str]], kwargs.pop("price", None))
        val_bid = cast(Optional[Union[float, int, str]], kwargs.pop("bid", None))
        val_ask = cast(Optional[Union[float, int, str]], kwargs.pop("ask", None))

        self.bid: float = _to_float(val_bid)
        self.ask: float = _to_float(val_ask, default=self.bid if self.bid else 0.0)

        # Close inferred from mid or provided fields
        close_in = cast(Optional[Union[float, int, str]], kwargs.pop("close", None))
        self.close: float = _to_float(close_in)
        if self.close == 0.0:
            if self.bid or self.ask:
                self.close = (self.bid + self.ask) / 2.0
            else:
                self.close = _to_float(val_price, default=0.0)

        # OHLC fallbacks
        open_in = cast(Optional[Union[float, int, str]], kwargs.pop("open", None))
        high_in = cast(Optional[Union[float, int, str]], kwargs.pop("high", None))
        low_in = cast(Optional[Union[float, int, str]], kwargs.pop("low", None))

        self.open: float = _to_float(open_in, default=self.close)
        self.high: float = _to_float(high_in, default=max(self.open, self.close))
        self.low: float = _to_float(low_in, default=min(self.open, self.close))

        # Volume
        self.volume: float = _to_float(
            cast(Optional[Union[float, int, str]], kwargs.pop("volume", None)), default=0.0
        )

        # Volatility (accepted and stored for compatibility)
        self.volatility: float = _to_float(
            cast(Optional[Union[float, int, str]], kwargs.pop("volatility", None)), default=0.0
        )

        # Retain any additional fields as passthrough (non-breaking)
        for k, v in kwargs.items():
            try:
                setattr(self, k, v)
            except Exception:
                # Ignore non-assignable extras
                pass

    @property
    def mid_price(self) -> float:
        try:
            if self.bid or self.ask:
                return (float(self.bid) + float(self.ask)) / 2.0
        except Exception:
            pass
        try:
            return float(self.close)
        except Exception:
            return 0.0

    @property
    def spread(self) -> float:
        try:
            return max(0.0, float(self.ask) - float(self.bid))
        except Exception:
            return 0.0


@runtime_checkable
class SensoryOrgan(Protocol):
    """
    Minimal sensory organ Protocol used by organs.* modules without creating a hard dependency.

    Very permissive signatures to avoid false-positive override errors across mixed legacy/new code.
    """

    name: str
    config: dict[str, object]

    def calibrate(self) -> bool: ...
    def perceive(self, market_data: Any) -> Any: ...
    def reset(self) -> None: ...


# Backwards-compatible alias retained for legacy use (not used by organs now)
LegacySensoryReading: TypeAlias = DimensionalReading

__all__ = [
    "DimensionalReading",
    "SensoryReading",
    "MarketData",
    "MarketRegime",
    "InstrumentMeta",
    "SensoryOrgan",
    "LegacySensoryReading",
]
