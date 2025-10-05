"""Market data gateway backed by the hardened Yahoo ingest helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import pandas as pd

from src.core.market_data import MarketDataGateway

from .yahoo_ingest import fetch_price_history

logger = logging.getLogger(__name__)


class YahooMarketDataGateway(MarketDataGateway):
    """Thin adapter that normalises Yahoo Finance history retrieval."""

    def __init__(self) -> None:
        self._logger = logger.getChild(self.__class__.__name__)

    def fetch_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        try:
            frame = fetch_price_history(
                symbol,
                period=period,
                interval=interval or "1d",
                start=start,
                end=end,
            )
        except ValueError as exc:
            self._logger.warning("Yahoo history fetch rejected for %s: %s", symbol, exc)
            return None
        except Exception:
            self._logger.exception("Unexpected Yahoo history failure for %s", symbol)
            raise

        return frame if not frame.empty else None

    async def get_market_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        return await asyncio.to_thread(
            self.fetch_data,
            symbol,
            period,
            interval,
            start,
            end,
        )


__all__ = ["YahooMarketDataGateway"]
