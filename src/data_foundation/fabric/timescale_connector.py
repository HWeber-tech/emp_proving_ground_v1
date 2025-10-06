"""Timescale-backed connectors that feed the market data fabric."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, cast

import pandas as pd

from src.core.base import MarketData

try:  # pragma: no cover - Python 3.11+ provides datetime.UTC
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    UTC = timezone.utc  # type: ignore[assignment]

from ..cache import TimescaleQueryCache
from ..persist.timescale_reader import TimescaleQueryResult, TimescaleReader
from ..services.macro_events import TimescaleMacroEventService
from .market_data_fabric import ConnectorResult

logger = logging.getLogger(__name__)


def _to_datetime(value: object) -> datetime | None:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return cast(datetime, value.to_pydatetime())
    if isinstance(value, datetime):
        return value
    return None


def _fallback_timestamp(candidate: datetime | None) -> datetime:
    if candidate is None:
        return datetime.now(tz=UTC).replace(tzinfo=None)
    return candidate


class _TimescaleConnectorBase:
    """Shared behaviours for Timescale-backed connectors."""

    def __init__(
        self,
        reader: TimescaleReader,
        *,
        name: str,
        priority: int,
        cache: TimescaleQueryCache | None = None,
        macro_service: TimescaleMacroEventService | None = None,
    ) -> None:
        self._reader = reader
        self._cache = cache
        self._macro_service = macro_service
        self.name = name
        self.priority = priority
        self._logger = logging.getLogger(f"{__name__}.{name}")

    async def _run(self, fn: Callable[[], ConnectorResult]) -> ConnectorResult:
        try:
            return await asyncio.to_thread(fn)
        except Exception:
            self._logger.exception("Timescale connector query failed")
            return None

    def _fetch_daily_bars(
        self,
        *,
        symbols: list[str],
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
    ) -> TimescaleQueryResult:
        if self._cache is not None and self._cache.enabled:
            return self._cache.fetch_daily_bars(symbols=symbols, start=start, end=end, limit=limit)
        return self._reader.fetch_daily_bars(symbols=symbols, start=start, end=end, limit=limit)

    def _fetch_intraday_trades(
        self,
        *,
        symbols: list[str],
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
    ) -> TimescaleQueryResult:
        if self._cache is not None and self._cache.enabled:
            return self._cache.fetch_intraday_trades(
                symbols=symbols, start=start, end=end, limit=limit
            )
        return self._reader.fetch_intraday_trades(
            symbols=symbols, start=start, end=end, limit=limit
        )

    def _macro_enrichment(self, symbol: str, as_of: datetime | None) -> dict[str, object]:
        if self._macro_service is None:
            return {}
        try:
            result = self._macro_service.calculate_macro_bias(symbol, as_of=as_of)
        except Exception:
            self._logger.debug("Macro enrichment failed for %s", symbol, exc_info=True)
            return {}

        enrichment: dict[str, object] = {
            "macro_bias": result.bias,
            "macro_confidence": result.confidence,
        }
        if result.metadata:
            enrichment["macro_metadata"] = dict(result.metadata)
        if result.events_analyzed:
            enrichment["macro_events"] = [event.as_dict() for event in result.events_analyzed]
        return enrichment


class TimescaleDailyBarConnector(_TimescaleConnectorBase):
    """Serve the latest Timescale daily bar for a symbol."""

    def __init__(
        self,
        reader: TimescaleReader,
        *,
        name: str = "timescale_daily",
        priority: int = 40,
        cache: TimescaleQueryCache | None = None,
        macro_service: TimescaleMacroEventService | None = None,
    ) -> None:
        super().__init__(
            reader,
            name=name,
            priority=priority,
            cache=cache,
            macro_service=macro_service,
        )

    async def fetch(self, symbol: str, *, as_of: datetime | None = None) -> ConnectorResult:
        def _query() -> ConnectorResult:
            result = self._fetch_daily_bars(symbols=[symbol], start=None, end=as_of, limit=None)
            if result.frame.empty:
                return None
            row = result.frame.iloc[-1]
            timestamp = _fallback_timestamp(_to_datetime(row.get("ts")))
            ingested_at = _to_datetime(row.get("ingested_at"))
            attributes: dict[str, object] = {
                "symbol": row.get("symbol", symbol),
                "timestamp": timestamp,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
                "adj_close": row.get("adj_close"),
                "source": row.get("source"),
                "ingested_at": ingested_at,
            }
            attributes.update(self._macro_enrichment(symbol, timestamp))
            return MarketData(**attributes)

        payload = await self._run(_query)
        if payload is None:
            self._logger.debug("No Timescale daily bar available for %s", symbol)
        return payload


class TimescaleIntradayTradeConnector(_TimescaleConnectorBase):
    """Serve the most recent intraday trade for a symbol."""

    def __init__(
        self,
        reader: TimescaleReader,
        *,
        name: str = "timescale_intraday",
        priority: int = 30,
        cache: TimescaleQueryCache | None = None,
        macro_service: TimescaleMacroEventService | None = None,
    ) -> None:
        super().__init__(
            reader,
            name=name,
            priority=priority,
            cache=cache,
            macro_service=macro_service,
        )

    async def fetch(self, symbol: str, *, as_of: datetime | None = None) -> ConnectorResult:
        def _query() -> ConnectorResult:
            result = self._fetch_intraday_trades(
                symbols=[symbol], start=None, end=as_of, limit=None
            )
            if result.frame.empty:
                return None
            row = result.frame.iloc[-1]
            timestamp = _fallback_timestamp(_to_datetime(row.get("ts")))
            ingested_at = _to_datetime(row.get("ingested_at"))
            attributes: dict[str, object] = {
                "symbol": row.get("symbol", symbol),
                "timestamp": timestamp,
                "price": row.get("price"),
                "volume": row.get("size"),
                "source": row.get("source"),
                "exchange": row.get("exchange"),
                "conditions": row.get("conditions"),
                "ingested_at": ingested_at,
            }
            attributes.update(self._macro_enrichment(symbol, timestamp))
            return MarketData(**attributes)

        payload = await self._run(_query)
        if payload is None:
            self._logger.debug("No Timescale intraday trade available for %s", symbol)
        return payload


__all__ = [
    "TimescaleDailyBarConnector",
    "TimescaleIntradayTradeConnector",
]
