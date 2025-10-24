from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from pydantic import BaseModel, Field, validator


def _normalise_symbol(value: str) -> str:
    text = value.strip().upper()
    if not text:
        raise ValueError("symbol must not be empty")
    return text


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


class TradeTick(BaseModel):
    """Normalised trade tick used by the microstructure archive."""

    symbol: str
    timestamp: datetime
    price: float
    size: float | None = None
    sequence: int | None = None
    trade_id: str | None = None
    venue: str | None = None
    liquidity_side: str | None = None
    conditions: tuple[str, ...] = Field(default_factory=tuple)
    source: str = Field(default="unknown")

    _normalise_symbol_field = validator("symbol", allow_reuse=True)(_normalise_symbol)

    @validator("price")
    def _price_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("price must be positive")
        return value

    @validator("size")
    def _size_non_negative(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("size must be non-negative")
        return value

    @validator("sequence")
    def _sequence_non_negative(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("sequence must be non-negative")
        return value

    @validator("venue")
    def _normalise_venue(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @validator("liquidity_side")
    def _normalise_liquidity_side(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip().lower()
        return text or None

    @validator("conditions", pre=True, always=True)
    def _normalise_conditions(cls, value: object) -> tuple[str, ...]:
        return MacroEvent._normalise_sequence(value, upper=True)


class QuoteTick(BaseModel):
    """Top-of-book quote snapshot (best bid/ask) aligned with tick storage."""

    symbol: str
    timestamp: datetime
    bid_price: float | None = None
    bid_size: float | None = None
    ask_price: float | None = None
    ask_size: float | None = None
    sequence: int | None = None
    mid_price: float | None = None
    spread_bps: float | None = None
    venue: str | None = None
    source: str = Field(default="unknown")

    _normalise_symbol_field = validator("symbol", allow_reuse=True)(_normalise_symbol)

    @validator("sequence")
    def _sequence_non_negative(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("sequence must be non-negative")
        return value

    @validator("bid_price", "ask_price")
    def _price_non_negative(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("price must be positive")
        return value

    @validator("bid_size", "ask_size")
    def _size_non_negative(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("size must be non-negative")
        return value

    @validator("mid_price", pre=True, always=True)
    def _derive_mid(cls, value: object, values: dict[str, object]) -> float | None:
        if value is not None:
            return float(value)
        bid = values.get("bid_price")
        ask = values.get("ask_price")
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
            return (float(bid) + float(ask)) / 2
        return None

    @validator("spread_bps", pre=True, always=True)
    def _derive_spread(cls, value: object, values: dict[str, object]) -> float | None:
        if value is not None:
            return float(value)
        bid = values.get("bid_price")
        ask = values.get("ask_price")
        mid = values.get("mid_price")
        if (
            isinstance(bid, (int, float))
            and isinstance(ask, (int, float))
            and isinstance(mid, (int, float))
            and mid != 0
        ):
            return ((float(ask) - float(bid)) / float(mid)) * 10_000
        return None

    @validator("venue")
    def _normalise_venue(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @validator("source")
    def _source_not_empty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("source must not be empty")
        return text

    @validator("bid_price", always=True)
    def _at_least_one_side(
        cls, value: float | None, values: dict[str, object]
    ) -> float | None:
        ask_price = values.get("ask_price")
        if value is None and ask_price is None:
            raise ValueError("quotes must provide at least one side")
        return value


class OrderBookLevel(BaseModel):
    """Single consolidated level of the order book."""

    level: int
    bid_price: float | None = None
    bid_size: float | None = None
    ask_price: float | None = None
    ask_size: float | None = None
    imbalance: float | None = None

    @validator("level")
    def _level_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("level must be >= 1")
        return value

    @validator("bid_price", "ask_price")
    def _price_non_negative(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("price must be positive")
        return value

    @validator("bid_size", "ask_size")
    def _size_non_negative(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("size must be non-negative")
        return value

    @validator("imbalance", pre=True, always=True)
    def _derive_imbalance(cls, value: object, values: dict[str, object]) -> float | None:
        if value is not None:
            return float(value)
        bid = values.get("bid_size")
        ask = values.get("ask_size")
        if isinstance(bid, (int, float)) or isinstance(ask, (int, float)):
            bid_size = max(float(bid or 0.0), 0.0)
            ask_size = max(float(ask or 0.0), 0.0)
            total = bid_size + ask_size
            if total == 0:
                return None
            return (bid_size - ask_size) / total
        return None


class OrderBookSnapshot(BaseModel):
    """Aggregated order book snapshot across multiple levels."""

    symbol: str
    timestamp: datetime
    sequence: int | None = None
    levels: tuple[OrderBookLevel, ...]
    venue: str | None = None
    source: str = Field(default="unknown")

    _normalise_symbol_field = validator("symbol", allow_reuse=True)(_normalise_symbol)

    @validator("sequence")
    def _sequence_non_negative(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0:
            raise ValueError("sequence must be non-negative")
        return value

    @validator("levels", pre=True, always=True)
    def _coerce_levels(cls, value: object) -> tuple[OrderBookLevel, ...]:
        if value is None:
            raise ValueError("levels must not be empty")
        if isinstance(value, OrderBookLevel):
            return (value,)
        if isinstance(value, Sequence):
            entries: list[OrderBookLevel] = []
            for item in value:
                if isinstance(item, OrderBookLevel):
                    entries.append(item)
                else:
                    entries.append(OrderBookLevel.parse_obj(item))
            if not entries:
                raise ValueError("levels must not be empty")
            return tuple(entries)
        raise TypeError("levels must be a sequence of OrderBookLevel entries")

    @validator("levels")
    def _validate_levels(cls, value: tuple[OrderBookLevel, ...]) -> tuple[OrderBookLevel, ...]:
        seen: set[int] = set()
        ordered = sorted(value, key=lambda level: level.level)
        for entry in ordered:
            if entry.level in seen:
                raise ValueError("duplicate order book levels are not allowed")
            seen.add(entry.level)
        return tuple(ordered)

    @validator("venue")
    def _normalise_venue(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None


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
