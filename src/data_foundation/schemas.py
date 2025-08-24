from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MarketDataEvent(BaseModel):
    """Canonical market data event for WHAT dimension (L1/L2 snapshot)."""

    symbol: str
    timestamp: datetime
    # Best levels
    bid: float = 0.0
    ask: float = 0.0
    # Up to N levels for storage (optional)
    bids_px: Optional[list[float]] = None
    bids_sz: Optional[list[float]] = None
    asks_px: Optional[list[float]] = None
    asks_sz: Optional[list[float]] = None
    source: str = Field(default="unknown")


class MacroEvent(BaseModel):
    """Canonical macro/economic event for WHY dimension."""

    timestamp: datetime
    calendar: str
    event: str
    currency: Optional[str] = None
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    importance: Optional[str] = None
    source: str = Field(default="unknown")


class SessionEvent(BaseModel):
    """Session markers for WHEN dimension (e.g., London open/close)."""

    timestamp: datetime
    session: str
    action: str  # open/close
    source: str = Field(default="derived")


class VolSignal(BaseModel):
    """Canonical volatility signal for sizing and regime selection."""

    symbol: str
    t: datetime
    sigma_ann: float
    var95_1d: float
    regime: str
    sizing_multiplier: float
    stop_mult: float
    quality: float


class YieldEvent(BaseModel):
    """Yield curve point for WHY signals."""

    timestamp: datetime
    curve: str = Field(default="UST")
    tenor: str
    value: float
    source: str = Field(default="openbb")
