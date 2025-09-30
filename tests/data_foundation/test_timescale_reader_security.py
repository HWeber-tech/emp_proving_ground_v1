from __future__ import annotations

# Security regression tests for TimescaleReader identifier sanitation.

import pytest

from src.data_foundation.persist.timescale_reader import _ensure_safe_identifiers


@pytest.mark.parametrize(
    "columns",
    [
        ["ts; DROP TABLE market_data_daily_bars"],
        ["ts", "symbol", "price", "size", "ts"],
    ],
)
def test_identifier_guard_rejects_suspicious_columns(columns: list[str]) -> None:
    with pytest.raises(ValueError):
        _ensure_safe_identifiers(
            schema="market_data",
            table="daily_bars",
            columns=columns,
            timestamp_column="ts",
            symbol_column="symbol",
        )


@pytest.mark.parametrize(
    "schema,table",
    [
        ("market-data", "daily_bars"),
        ("market_data", "daily bars"),
    ],
)
def test_identifier_guard_rejects_invalid_schema_or_table(schema: str, table: str) -> None:
    with pytest.raises(ValueError):
        _ensure_safe_identifiers(
            schema=schema,
            table=table,
            columns=["ts", "symbol"],
            timestamp_column="ts",
            symbol_column="symbol",
        )


@pytest.mark.parametrize(
    "timestamp_column,symbol_column",
    [
        ("ts;DROP", "symbol"),
        ("ts", "symbol name"),
    ],
)
def test_identifier_guard_rejects_invalid_column_aliases(
    timestamp_column: str, symbol_column: str
) -> None:
    with pytest.raises(ValueError):
        _ensure_safe_identifiers(
            schema="market_data",
            table="daily_bars",
            columns=["ts", "symbol"],
            timestamp_column=timestamp_column,
            symbol_column=symbol_column,
        )
