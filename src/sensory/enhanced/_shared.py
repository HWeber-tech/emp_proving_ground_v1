from __future__ import annotations

from collections.abc import Awaitable, Generator, Mapping
from datetime import datetime
from typing import Any

from src.core.base import DimensionalReading, MarketData


def ensure_market_data(payload: object | None) -> MarketData:
    """Return a :class:`MarketData` instance for heterogeneous payloads."""
    if isinstance(payload, MarketData):
        return payload
    if isinstance(payload, Mapping):
        candidate: dict[str, Any] = {}
        for key, value in payload.items():
            try:
                candidate[str(key)] = value
            except Exception:
                continue
        try:
            return MarketData(**candidate)
        except TypeError:
            return MarketData()
    return MarketData()


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp *value* to the inclusive ``[lower, upper]`` interval."""
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def safe_timestamp(data: MarketData) -> datetime:
    """Extract a timestamp from ``data`` falling back to ``datetime.utcnow``."""
    ts = getattr(data, "timestamp", None)
    if isinstance(ts, datetime):
        return ts
    return datetime.utcnow()


class ReadingAdapter(dict[str, Any], Awaitable[DimensionalReading]):
    """Hybrid container exposing dict semantics and awaitable access."""

    __slots__ = ("_reading",)

    def __init__(self, reading: DimensionalReading, payload: Mapping[str, Any]) -> None:
        super().__init__(payload)
        self._reading = reading

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._reading, name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __await__(self) -> Generator[Any, None, DimensionalReading]:
        async def _coro() -> DimensionalReading:
            return self._reading

        return _coro().__await__()

    def __hash__(self) -> int:  # type: ignore[override]
        """Make adapters hashable for use in asyncio.gather task sets."""
        return id(self)

    @property
    def reading(self) -> DimensionalReading:
        return self._reading


def build_legacy_payload(
    reading: DimensionalReading,
    *,
    source: str,
    extras: Mapping[str, Any] | None = None,
) -> ReadingAdapter:
    payload: dict[str, Any] = {
        "signal": float(getattr(reading, "signal_strength", 0.0)),
        "confidence": float(getattr(reading, "confidence", 0.0)),
        "meta": {"source": source},
        "context": dict(getattr(reading, "context", {}) or {}),
    }
    if extras:
        payload.update(extras)
    return ReadingAdapter(reading, payload)
