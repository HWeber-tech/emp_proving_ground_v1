"""
Core Market Data Port (Protocol)
===============================

Defines a minimal gateway interface for fetching market data without
binding domain code to specific data providers or packages.

Concrete adapters should live in higher layers (e.g., src/data_integration or src/sensory)
and must not be imported from here.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class MarketDataGateway(Protocol):
    """
    Abstract gateway for retrieving market data.

    Implementations may return pandas.DataFrame or another tabular type.
    """

    def fetch_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Any:
        """Synchronous fetch suitable for CLI/tests."""
        ...

    async def get_market_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Any:
        """Async fetch suitable for orchestrated flows."""
        ...


class NoOpMarketDataGateway:
    """Default gateway that returns no data. Safe placeholder for DI."""

    def fetch_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Any:
        return None

    async def get_market_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Any:
        return None


def is_market_data_gateway(obj: object) -> bool:
    """Runtime check helper for duck-typed implementations."""
    try:
        return isinstance(obj, MarketDataGateway)
    except Exception:
        return False