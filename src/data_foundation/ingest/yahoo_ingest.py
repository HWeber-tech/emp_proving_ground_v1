#!/usr/bin/env python3
"""
Tier-0 Yahoo ingest: fetch daily bars for symbols and persist to DuckDB (if available) or CSV.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import re

import pandas as pd
import yfinance as yf


def fetch_daily_bars(symbols: List[str], days: int = 60) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    frames = []
    for sym in symbols:
        df = yf.download(sym, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", progress=False)
        if not df.empty:
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            })
            df["symbol"] = sym
            df = df.reset_index().rename(columns={"Date": "date"})
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def store_duckdb(df: pd.DataFrame, db_path: Path, table: str = "daily_bars") -> None:
    try:
        import duckdb
    except Exception:
        # Fallback to CSV if duckdb not available
        csv_path = db_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return

    con = duckdb.connect(str(db_path))
    # Bandit B608: parameterized query to avoid SQL injection (sanitize identifier)
    safe_table = table if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table or "daily_bars") else "daily_bars"
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {safe_table} (
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
    # Bandit B608: parameterized query to avoid SQL injection
    con.execute(f"DELETE FROM {safe_table} WHERE symbol IN ({','.join(['?']*len(df['symbol'].unique()))})", list(df['symbol'].unique()))
    con.register("tmp_df", df)
    # Bandit B608: parameterized query to avoid SQL injection (identifier sanitized)
    con.execute(f"INSERT INTO {safe_table} SELECT * FROM tmp_df")
    con.close()


def main():
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


