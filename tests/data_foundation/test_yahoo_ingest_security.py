from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import testing as pd_testing

from src.data_foundation.ingest.yahoo_ingest import store_duckdb


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self.registered: list[tuple[str, pd.DataFrame]] = []
        self.closed = False

    def execute(self, query: str, parameters: Any | None = None) -> "_FakeConnection":
        self.executed.append((query, parameters))
        return self

    def register(self, name: str, frame: pd.DataFrame) -> None:
        # Store a copy to mimic duckdb semantics without aliasing tests
        self.registered.append((name, frame.copy()))

    def close(self) -> None:
        self.closed = True


def test_store_duckdb_sanitizes_and_parameterizes(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "adj_close": [1.1, 2.1],
            "volume": [100, 200],
            "symbol": ["AAPL", "MSFT"],
        }
    )

    fake_connection = _FakeConnection()

    def fake_connect(path: str) -> _FakeConnection:
        assert path.endswith("test.duckdb")
        return fake_connection

    fake_duckdb = types.SimpleNamespace(
        connect=fake_connect,
        escape_identifier=lambda identifier: f'"{identifier}"',
    )

    monkeypatch.setitem(sys.modules, "duckdb", fake_duckdb)

    store_duckdb(df, Path("test.duckdb"), table="daily bars; drop table")

    # Validate table name sanitization
    create_stmt, _ = fake_connection.executed[0]
    assert 'CREATE TABLE IF NOT EXISTS "daily_bars"' in create_stmt

    # Delete statements should use parameter binding and the sanitized identifier
    delete_calls = [call for call in fake_connection.executed if "DELETE FROM" in call[0]]
    assert len(delete_calls) == 2
    for statement, params in delete_calls:
        assert statement == 'DELETE FROM "daily_bars" WHERE symbol = ?'
        assert params in (["AAPL"], ["MSFT"])

    # Insert uses sanitized identifier and registration occurs once
    insert_stmt, insert_params = fake_connection.executed[-1]
    assert insert_stmt == 'INSERT INTO "daily_bars" SELECT * FROM tmp_df'
    assert insert_params is None
    assert len(fake_connection.registered) == 1
    name, registered_df = fake_connection.registered[0]
    assert name == "tmp_df"
    pd_testing.assert_frame_equal(registered_df, df)
    assert fake_connection.closed is True
