"""
YFinanceGateway - MarketDataGateway adapter
===========================================

Concrete adapter implementing the core MarketDataGateway Protocol using yfinance.
This keeps validation and other domains decoupled from sensory organs and specific providers.

Returns pandas DataFrames with standardized columns:
- timestamp, open, high, low, close, volume, symbol
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from src.core.market_data import MarketDataGateway


logger = logging.getLogger(__name__)


class YFinanceGateway(MarketDataGateway):
    """MarketDataGateway implementation backed by yfinance."""

    def fetch_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Any:
        try:
            yf_symbol = self._to_yahoo_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            # Fetch into an object-typed temporary to avoid Any pollution
            df_obj: object
            if period is not None or (start is None and end is None):
                # Period-based request
                df_obj = ticker.history(period=period or "1d", interval=interval or "1h")
            else:
                # Start/end range request
                df_obj = ticker.history(start=start, end=end, interval=interval or "1h")

            if df_obj is None:
                return None
            if not isinstance(df_obj, pd.DataFrame):
                return None

            # Safe to treat as DataFrame after guards
            df: pd.DataFrame = df_obj.reset_index()

            if df.empty:
                return None

            # Normalize column names
            rename_map = {
                "Datetime": "timestamp",
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "close",
                "Volume": "volume",
            }
            df = df.rename(columns=rename_map)

            # Ensure timestamp column exists and is datetime
            if "timestamp" not in df.columns:
                # Some intervals may use 'Date'
                if "Date" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["Date"])
                else:
                    # Fallback: no timestamp available
                    return None
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Add symbol column for convenience
            df["symbol"] = symbol
            return df
        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            logger.warning(
                "Failed to fetch yfinance data for %s: %s", symbol, exc, exc_info=False
            )
            return None
        except Exception as exc:  # pragma: no cover - unexpected safeguard
            logger.exception(
                "Unexpected error while fetching yfinance data for %s", symbol
            )
            raise

    async def get_market_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Any:
        # Simple async wrapper; in a real system use an executor or async HTTP client
        return self.fetch_data(symbol, period=period, interval=interval, start=start, end=end)

    def _to_yahoo_symbol(self, symbol: str) -> str:
        """Map common forex symbols like 'EURUSD' to Yahoo '=X' form."""
        s = symbol.strip().upper()
        if len(s) == 6 and s[:3].isalpha() and s[3:6].isalpha():
            return f"{s}=X"
        return s
