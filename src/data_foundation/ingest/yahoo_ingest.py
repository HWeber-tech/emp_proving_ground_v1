#!/usr/bin/env python3
"""Canonical Yahoo Finance ingest helpers used by tier-0 pipelines."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yfinance as yf

from src.core.types import JSONObject


def fetch_daily_bars(symbols: list[str], days: int = 60) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = cast(
            pd.DataFrame,
            yf.download(
                sym,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
            ),
        )
        if not df.empty:
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            )
            df["symbol"] = sym
            df = df.reset_index().rename(columns={"Date": "date"})
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_intraday_trades(symbols: list[str], days: int = 2, interval: str = "1m") -> pd.DataFrame:
    """Fetch intraday bars from Yahoo and reshape them into trade-like records."""

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = cast(
            pd.DataFrame,
            yf.download(
                sym,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
            ),
        )
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"Datetime": "timestamp"})
        df = df.rename(columns={"Close": "price", "Volume": "size"})
        df["symbol"] = sym
        df["exchange"] = "YAHOO"
        df["conditions"] = "HISTORICAL"
        frames.append(df[["timestamp", "symbol", "price", "size", "exchange", "conditions"]].copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


_VALID_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9_.=\-]{1,32}$")
_ALLOWED_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "45m",
    "1h",
    "90m",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}
_INTERVAL_ALIASES = {
    "60m": "1h",
    "1hour": "1h",
    "1hr": "1h",
}
_ALLOWED_PERIODS = {
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
}
_PERIOD_ALIASES = {
    "1day": "1d",
    "1week": "1wk",
    "1month": "1mo",
    "3month": "3mo",
    "6month": "6mo",
    "1year": "1y",
}


def _sanitize_table_name(name: str | None, default: str = "daily_bars") -> str:
    """Return a safe table identifier comprised of alphanumerics/underscores only."""

    if name and _VALID_IDENTIFIER.fullmatch(name):
        return name
    return default


def _sanitize_symbol(symbol: str) -> str:
    candidate = symbol.strip()
    if not candidate or not _SYMBOL_PATTERN.fullmatch(candidate):
        raise ValueError(f"Invalid symbol: {symbol!r}")
    return candidate.upper()


def _normalize_interval(interval: str | None) -> str:
    raw = (interval or "1d").strip().lower()
    raw = _INTERVAL_ALIASES.get(raw, raw)
    if raw not in _ALLOWED_INTERVALS:
        raise ValueError(f"Unsupported Yahoo Finance interval: {interval!r}")
    return raw


def _normalize_period(period: str | None) -> str | None:
    if period is None:
        return None
    raw = period.strip().lower()
    raw = _PERIOD_ALIASES.get(raw, raw)
    if raw not in _ALLOWED_PERIODS:
        raise ValueError(f"Unsupported Yahoo Finance period: {period!r}")
    return raw


def _coerce_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value.strip())
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid ISO timestamp: {value!r}") from exc


def fetch_price_history(
    symbol: str,
    *,
    interval: str = "1d",
    period: str | None = None,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> pd.DataFrame:
    """Fetch normalized price history from Yahoo Finance for a symbol.

    The function sanitizes symbols/intervals/periods and returns a dataframe with
    canonical column names (timestamp/open/high/low/close/adj_close/volume/symbol).
    """

    normalized_symbol = _sanitize_symbol(symbol)
    normalized_interval = _normalize_interval(interval)
    normalized_period = _normalize_period(period)

    if normalized_period and (start is not None or end is not None):
        raise ValueError("Provide either period or start/end window, not both.")

    kwargs: dict[str, Any] = {
        "interval": normalized_interval,
        "progress": False,
    }
    if normalized_period:
        kwargs["period"] = normalized_period
    if start is not None:
        kwargs["start"] = _coerce_datetime(start)
    if end is not None:
        kwargs["end"] = _coerce_datetime(end)

    frame = cast(pd.DataFrame, yf.download(normalized_symbol, **kwargs))
    if frame.empty:
        return pd.DataFrame()

    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    frame = frame.reset_index()
    if "Date" in frame.columns:
        frame = frame.rename(columns={"Date": "timestamp"})
    elif "Datetime" in frame.columns:
        frame = frame.rename(columns={"Datetime": "timestamp"})
    elif "timestamp" not in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame.index)

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    frame["symbol"] = normalized_symbol

    preferred_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "symbol",
    ]
    columns = [col for col in preferred_columns if col in frame.columns]
    extras = [col for col in frame.columns if col not in columns]
    return frame[columns + extras]


def store_duckdb(df: pd.DataFrame, db_path: Path, table: str = "daily_bars") -> None:
    try:
        import duckdb
    except Exception:
        # Fallback to CSV if duckdb not available
        csv_path = db_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return

    safe_table = _sanitize_table_name(table)
    escape_identifier = getattr(duckdb, "escape_identifier", None)
    quoted_table = (
        escape_identifier(safe_table)
        if callable(escape_identifier)
        else safe_table
    )

    connection = cast(Any, duckdb.connect(str(db_path)))
    try:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {quoted_table} (
                date TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume DOUBLE,
                symbol VARCHAR
            )
            """
        )

        unique_symbols = list(dict.fromkeys(df["symbol"].tolist()))
        if unique_symbols:
            delete_statement = f"DELETE FROM {quoted_table} WHERE symbol = ?"
            for symbol in unique_symbols:
                connection.execute(delete_statement, [symbol])

        connection.register("tmp_df", df)
        connection.execute(f"INSERT INTO {quoted_table} SELECT * FROM tmp_df")
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier-0 Yahoo ingest")
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols")
    parser.add_argument("--db", type=str, default="data/tier0.duckdb", help="DuckDB path")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    df = fetch_daily_bars(symbols)
    if df.empty:
        print("No data fetched")
        return 1
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    store_duckdb(df, Path(args.db))
    print(f"Stored {len(df)} rows for {len(symbols)} symbols to {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
