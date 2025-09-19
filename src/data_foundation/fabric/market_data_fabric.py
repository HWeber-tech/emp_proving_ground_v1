from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol

from src.core.base import MarketData

__all__ = [
    "ConnectorResult",
    "MarketDataConnector",
    "CallableConnector",
    "MarketDataFabric",
]

logger = logging.getLogger(__name__)

ConnectorResult = Optional[MarketData | Mapping[str, Any]]


class MarketDataConnector(Protocol):
    """Protocol describing the fabric connector contract."""

    name: str
    priority: int

    async def fetch(
        self, symbol: str, *, as_of: datetime | None = None
    ) -> ConnectorResult: ...


@dataclass
class CallableConnector:
    """Adapter that turns plain callables into :class:`MarketDataConnector` objects."""

    name: str
    func: Callable[
        [str, datetime | None], ConnectorResult | Awaitable[ConnectorResult]
    ] | Callable[[str], ConnectorResult | Awaitable[ConnectorResult]]
    priority: int = 100
    timeout: float | None = None

    async def fetch(
        self, symbol: str, *, as_of: datetime | None = None
    ) -> ConnectorResult:
        try:
            result = self.func(symbol, as_of)  # type: ignore[arg-type]
        except TypeError:
            result = self.func(symbol)  # type: ignore[misc]
        if inspect.isawaitable(result):
            return await result
        return result


@dataclass
class _CacheEntry:
    data: MarketData
    source: str
    fetched_at: datetime

    @property
    def age(self) -> timedelta:
        return datetime.now(timezone.utc) - self.fetched_at


@dataclass
class FabricTelemetry:
    total_requests: int = 0
    cache_hits: int = 0
    connector_success: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    connector_failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    connector_latency_ms: Dict[str, float] = field(default_factory=dict)


class MarketDataFabric:
    """Asynchronous market data fabric with connector failover and caching."""

    def __init__(
        self,
        connectors: Mapping[str, MarketDataConnector] | None = None,
        *,
        cache_ttl: timedelta | None = None,
        default_timeout: float = 2.0,
    ) -> None:
        self._connectors: Dict[str, MarketDataConnector] = {}
        if connectors:
            for name, connector in connectors.items():
                self.register_connector(connector)
        self.cache_ttl = cache_ttl if cache_ttl is not None else timedelta(seconds=10)
        self.default_timeout = default_timeout
        self._cache: Dict[str, _CacheEntry] = {}
        self.telemetry = FabricTelemetry()

    def register_connector(self, connector: MarketDataConnector) -> None:
        if connector.name in self._connectors:
            logger.warning("Replacing existing connector %s", connector.name)
        self._connectors[connector.name] = connector

    @property
    def connectors(self) -> Dict[str, MarketDataConnector]:
        return dict(self._connectors)

    async def fetch_latest(
        self,
        symbol: str,
        *,
        as_of: datetime | None = None,
        timeout: float | None = None,
        allow_stale: bool = True,
        use_cache: bool = True,
    ) -> MarketData:
        now = datetime.now(timezone.utc)
        self.telemetry.total_requests += 1
        entry = self._cache.get(symbol)
        if (
            use_cache
            and entry is not None
            and now - entry.fetched_at <= self.cache_ttl
        ):
            self.telemetry.cache_hits += 1
            return entry.data

        ordered = sorted(self._connectors.values(), key=lambda c: getattr(c, "priority", 100))
        last_error: Exception | None = None

        for connector in ordered:
            started = datetime.now(timezone.utc)
            conn_timeout = timeout if timeout is not None else getattr(connector, "timeout", None)
            if conn_timeout is None:
                conn_timeout = self.default_timeout

            try:
                result = await asyncio.wait_for(
                    connector.fetch(symbol, as_of=as_of), timeout=conn_timeout
                )
            except asyncio.TimeoutError as exc:
                self.telemetry.connector_failures[connector.name] += 1
                last_error = exc
                logger.warning(
                    "Connector %s timed out for %s after %.2fs",
                    connector.name,
                    symbol,
                    conn_timeout,
                )
                continue
            except Exception as exc:  # pragma: no cover - resilience guard
                self.telemetry.connector_failures[connector.name] += 1
                last_error = exc
                logger.warning("Connector %s failed for %s: %s", connector.name, symbol, exc)
                continue

            latency = (datetime.now(timezone.utc) - started).total_seconds() * 1000.0
            self.telemetry.connector_latency_ms[connector.name] = latency

            market_data = self._coerce_market_data(result, symbol)
            if market_data is None:
                self.telemetry.connector_failures[connector.name] += 1
                continue

            self.telemetry.connector_success[connector.name] += 1
            self._cache[symbol] = _CacheEntry(
                data=market_data,
                source=connector.name,
                fetched_at=datetime.now(timezone.utc),
            )
            return market_data

        if entry and allow_stale:
            logger.warning(
                "All connectors failed for %s, returning stale data fetched %.1fs ago",
                symbol,
                entry.age.total_seconds(),
            )
            return entry.data

        raise RuntimeError(
            f"No market data available for {symbol}: {last_error!r}" if last_error else f"No data for {symbol}"
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        cache_info = {
            symbol: {
                "source": entry.source,
                "age_seconds": round(entry.age.total_seconds(), 3),
            }
            for symbol, entry in self._cache.items()
        }
        return {
            "connectors": sorted(self._connectors.keys()),
            "telemetry": {
                "total_requests": self.telemetry.total_requests,
                "cache_hits": self.telemetry.cache_hits,
                "success": dict(self.telemetry.connector_success),
                "failures": dict(self.telemetry.connector_failures),
                "latency_ms": dict(self.telemetry.connector_latency_ms),
            },
            "cache": cache_info,
        }

    def invalidate_cache(self, symbol: str | None = None) -> None:
        """Remove cached entries for ``symbol`` or clear the entire cache."""

        if symbol is None:
            self._cache.clear()
            return
        self._cache.pop(symbol, None)

    @staticmethod
    def _coerce_market_data(
        result: ConnectorResult, symbol: str
    ) -> MarketData | None:
        if result is None:
            return None
        if isinstance(result, MarketData):
            return result
        if isinstance(result, Mapping):
            payload = dict(result)
            payload.setdefault("symbol", symbol)
            if "timestamp" not in payload:
                payload["timestamp"] = datetime.now(timezone.utc)
            return MarketData(**payload)
        logger.debug("Unsupported connector payload type %s", type(result))
        return None
