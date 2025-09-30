#!/usr/bin/env python3
"""
Tier-0 Yahoo ingest: fetch daily bars for symbols and persist to DuckDB (if available) or CSV.
"""

from __future__ import annotations

import argparse
import contextlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yfinance as yf


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


def _sanitise_identifier(identifier: str, default: str) -> str:
    candidate = identifier or default
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
        return candidate
    return default


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier}"'


def store_duckdb(df: pd.DataFrame, db_path: Path, table: str = "daily_bars") -> None:
    try:
        import duckdb
    except ModuleNotFoundError:
        # Fallback to CSV if duckdb not available
        csv_path = db_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return

    safe_table = _sanitise_identifier(table, "daily_bars")
    quoted_table = _quote_identifier(safe_table)
    with contextlib.closing(cast(Any, duckdb.connect(str(db_path)))) as con:
        con.execute(
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
        symbols = list(df["symbol"].unique())
        placeholders = ",".join(["?"] * len(symbols))
        if symbols:
            con.execute(
                f"DELETE FROM {quoted_table} WHERE symbol IN ({placeholders})",
                symbols,
            )
        con.register("tmp_df", df)
        con.execute(f"INSERT INTO {quoted_table} SELECT * FROM tmp_df")


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
