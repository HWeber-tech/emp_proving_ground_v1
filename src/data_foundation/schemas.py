from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from pydantic import BaseModel, Field, validator


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
    category: Optional[str] = None
    related_symbols: tuple[str, ...] = Field(default_factory=tuple)
    causal_links: tuple[str, ...] = Field(default_factory=tuple)

    @validator("category", pre=True, always=True)
    def _validate_category(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    @validator("related_symbols", pre=True, always=True)
    def _validate_related_symbols(
        cls, value: object
    ) -> tuple[str, ...]:  # noqa: D401 - short normaliser
        """Normalise related symbol payloads into a tuple."""

        return cls._normalise_sequence(value, upper=True)

    @validator("causal_links", pre=True, always=True)
    def _validate_causal_links(
        cls, value: object
    ) -> tuple[str, ...]:  # noqa: D401 - short normaliser
        """Normalise causal link payloads into a tuple."""

        return cls._normalise_sequence(value, upper=False)

    @staticmethod
    def _normalise_sequence(value: object, *, upper: bool) -> tuple[str, ...]:
        if value is None:
            return tuple()
        items: Sequence[object]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return tuple()
            items = (text,)
        elif isinstance(value, Sequence):
            items = value
        else:
            items = (value,)

        normalised: list[str] = []
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            if upper:
                text = text.upper()
            normalised.append(text)

        if not normalised:
            return tuple()

        # Preserve order while removing duplicates
        seen: dict[str, None] = {}
        for entry in normalised:
            if entry not in seen:
                seen[entry] = None
        return tuple(seen)


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
