"""Historical replay connector for the market data fabric."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.core.base import MarketData

from .market_data_fabric import ConnectorResult


def _normalise_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def _coerce_market_data(symbol: str, payload: object) -> MarketData:
    if isinstance(payload, MarketData):
        return payload
    if isinstance(payload, Mapping):
        data: MutableMapping[str, object] = dict(payload)
        data.setdefault("symbol", symbol)
        if "timestamp" in data:
            try:
                data["timestamp"] = _normalise_timestamp(data["timestamp"])
            except TypeError:
                data.pop("timestamp")
        return MarketData(**data)
    raise TypeError(f"Unsupported historical payload: {type(payload)!r}")


@dataclass
class HistoricalReplayConnector:
    """Serve deterministic market data ticks from an in-memory sequence."""

    name: str = "historical_replay"
    priority: int = 100
    forward_fill: bool = True
    _series: dict[str, list[MarketData]] = field(default_factory=dict, init=False)
    _indices: dict[str, int] = field(default_factory=dict, init=False)

    def __init__(
        self,
        bars_by_symbol: Mapping[str, Sequence[object]] | Mapping[str, Iterable[object]],
        *,
        name: str = "historical_replay",
        priority: int = 100,
        forward_fill: bool = True,
    ) -> None:
        self.name = name
        self.priority = priority
        self.forward_fill = forward_fill
        self._series = {}
        self._indices = {}

        for symbol, items in bars_by_symbol.items():
            normalised: list[MarketData] = []
            for payload in items:
                normalised.append(_coerce_market_data(symbol, payload))
            normalised.sort(key=lambda md: md.timestamp)
            self._series[symbol] = normalised
            self._indices[symbol] = 0

    async def fetch(
        self, symbol: str, *, as_of: datetime | None = None
    ) -> ConnectorResult:
        series = self._series.get(symbol)
        if not series:
            return None

        if as_of is not None:
            candidate = None
            for item in series:
                if item.timestamp <= as_of:
                    candidate = item
                else:
                    break
            if candidate is not None:
                return candidate
            return series[0] if self.forward_fill else None

        index = self._indices.get(symbol, 0)
        if index >= len(series):
            return series[-1] if self.forward_fill and series else None

        result = series[index]
        self._indices[symbol] = index + 1
        return result


__all__ = ["HistoricalReplayConnector"]
