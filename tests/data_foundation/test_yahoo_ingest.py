from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.data_foundation.ingest.yahoo_ingest import store_duckdb


def _build_frame(symbol: str, price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [datetime(2024, 1, 1)],
            "open": [price],
            "high": [price + 1],
            "low": [price - 1],
            "close": [price + 0.5],
            "adj_close": [price + 0.25],
            "volume": [1_000.0],
            "symbol": [symbol],
        }
    )


def test_store_duckdb_sanitises_and_quotes_table(tmp_path) -> None:
    duckdb = pytest.importorskip("duckdb")

    db_path = tmp_path / "tier0.duckdb"
    safe_df = _build_frame("SAFE", 10.0)
    store_duckdb(safe_df, db_path, table="bars_v1")

    unsafe_df = _build_frame("UNSAFE", 20.0)
    store_duckdb(unsafe_df, db_path, table="bars_v1; DROP TABLE strategies; --")

    with duckdb.connect(str(db_path)) as con:
        safe_count = con.execute('SELECT COUNT(*) FROM "bars_v1" WHERE symbol = ?', ("SAFE",)).fetchone()[0]
        fallback_count = con.execute('SELECT COUNT(*) FROM "daily_bars" WHERE symbol = ?', ("UNSAFE",)).fetchone()[0]

    assert safe_count == len(safe_df)
    assert fallback_count == len(unsafe_df)


def test_store_duckdb_replaces_existing_symbol_rows(tmp_path) -> None:
    duckdb = pytest.importorskip("duckdb")

    db_path = tmp_path / "tier0.duckdb"
    initial_df = pd.concat([_build_frame("AAA", 11.0), _build_frame("BBB", 12.0)], ignore_index=True)
    store_duckdb(initial_df, db_path)

    replacement_df = _build_frame("AAA", 15.0)
    store_duckdb(replacement_df, db_path)

    with duckdb.connect(str(db_path)) as con:
        rows = con.execute('SELECT symbol, close FROM "daily_bars" ORDER BY symbol').fetchall()

    assert rows == [("AAA", 15.5)]
