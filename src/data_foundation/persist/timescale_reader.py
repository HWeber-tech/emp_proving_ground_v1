"""Query helpers for the Timescale-backed institutional ingest slice.

The roadmap tasks us with proving how the new Timescale schema will serve
up data to downstream sensors and risk modules.  The ingest pipeline already
populates hypertables; this module provides a tiny read layer that keeps those
tables accessible to the rest of the runtime while CI (or notebooks) exercise
the queries end-to-end.

These helpers purposely operate on plain ``pandas`` frames so the existing
analytics stack – which still expects bootstrap ``DataFrame`` inputs – can
reuse them without a broader refactor.  They translate common lookup patterns
required by the WHY/WHAT/WHEN sensors (daily bars, intraday trades) and the
risk/compliance modules (macro events) while remaining dialect agnostic so the
SQLite fallback used in tests behaves like a Timescale deployment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
import re
from typing import Sequence, cast

import pandas as pd
from sqlalchemy.engine import Engine


def _table_name(schema: str, table: str, dialect: str) -> str:
    """Mirror the naming convention used by :mod:`timescale` migrations."""

    if dialect == "postgresql":
        return f"{schema}.{table}"
    return f"{schema}_{table}"


def _coerce_timestamp(value: pd.Series) -> pd.Series:
    if value.empty:
        return value
    coerced = pd.to_datetime(value, utc=True, errors="coerce")
    return coerced


def _normalise_symbols(symbols: Sequence[str] | None) -> tuple[str, ...]:
    if not symbols:
        return tuple()
    return tuple(sorted({symbol.strip() for symbol in symbols if symbol}))


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(value: str, *, label: str) -> str:
    """Ensure SQL identifiers do not contain unsafe characters."""

    if not value:
        raise ValueError(f"{label} identifier cannot be empty")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{label} identifier '{value}' contains unsafe characters")
    return value


def _ensure_safe_identifiers(
    *,
    schema: str,
    table: str,
    columns: Sequence[str],
    timestamp_column: str,
    symbol_column: str,
) -> tuple[tuple[str, ...], str, str]:
    """Validate schema, table, and column identifiers before building SQL."""

    _validate_identifier(schema, label="schema")
    _validate_identifier(table, label="table")

    safe_columns = tuple(
        _validate_identifier(column, label="column") for column in columns
    )
    if not safe_columns:
        raise ValueError("column list cannot be empty")
    if len(set(safe_columns)) != len(safe_columns):
        raise ValueError("column list contains duplicates")

    safe_timestamp = _validate_identifier(timestamp_column, label="timestamp column")
    safe_symbol = _validate_identifier(symbol_column, label="symbol column")

    return safe_columns, safe_timestamp, safe_symbol


@dataclass(frozen=True)
class TimescaleQueryResult:
    """Structured response describing a Timescale query."""

    dimension: str
    frame: pd.DataFrame
    symbols: tuple[str, ...]
    start_ts: datetime | None
    end_ts: datetime | None
    max_ingested_at: datetime | None

    @property
    def rowcount(self) -> int:
        return int(self.frame.shape[0])

    def freshness_age_seconds(self, *, reference: datetime | None = None) -> float | None:
        if self.max_ingested_at is None:
            return None
        reference_dt = reference or datetime.now(tz=UTC)
        candidate = self.max_ingested_at
        if candidate.tzinfo is None:
            candidate = candidate.replace(tzinfo=UTC)
        return (reference_dt.astimezone(UTC) - candidate.astimezone(UTC)).total_seconds()


class TimescaleReader:
    """Read Timescale-backed market data into familiar pandas structures."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._logger = logging.getLogger(f"{__name__}.TimescaleReader")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_daily_bars(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> TimescaleQueryResult:
        columns = [
            "ts",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "source",
            "ingested_at",
        ]
        return self._fetch(
            schema="market_data",
            table="daily_bars",
            columns=columns,
            timestamp_column="ts",
            symbols=symbols,
            start=start,
            end=end,
            limit=limit,
            dimension="daily_bars",
        )

    def fetch_intraday_trades(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> TimescaleQueryResult:
        columns = [
            "ts",
            "symbol",
            "price",
            "size",
            "exchange",
            "conditions",
            "source",
            "ingested_at",
        ]
        return self._fetch(
            schema="market_data",
            table="intraday_trades",
            columns=columns,
            timestamp_column="ts",
            symbols=symbols,
            start=start,
            end=end,
            limit=limit,
            dimension="intraday_trades",
        )

    def fetch_macro_events(
        self,
        *,
        calendars: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> TimescaleQueryResult:
        columns = [
            "ts",
            "calendar",
            "event_name",
            "currency",
            "importance",
            "actual",
            "forecast",
            "previous",
            "source",
            "ingested_at",
        ]
        return self._fetch(
            schema="macro_data",
            table="events",
            columns=columns,
            timestamp_column="ts",
            symbols=calendars,
            symbol_column="calendar",
            start=start,
            end=end,
            limit=limit,
            dimension="macro_events",
        )

    def latest_daily_bar(self, symbol: str) -> pd.Series | None:
        result = self.fetch_daily_bars(symbols=[symbol])
        if result.frame.empty:
            return None
        return result.frame.iloc[-1]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch(
        self,
        *,
        schema: str,
        table: str,
        columns: Sequence[str],
        timestamp_column: str,
        symbols: Sequence[str] | None,
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
        dimension: str,
        symbol_column: str = "symbol",
    ) -> TimescaleQueryResult:
        safe_columns, safe_timestamp_column, safe_symbol_column = _ensure_safe_identifiers(
            schema=schema,
            table=table,
            columns=columns,
            timestamp_column=timestamp_column,
            symbol_column=symbol_column,
        )
        normalized_symbols = _normalise_symbols(symbols)
        with self._engine.begin() as conn:
            dialect = conn.dialect.name
            table_name = _table_name(schema, table, dialect)
            self._logger.debug(
                "TimescaleReader fetch %s table=%s symbols=%s start=%s end=%s limit=%s dialect=%s",
                dimension,
                table_name,
                normalized_symbols or "ALL",
                start,
                end,
                limit,
                dialect,
            )
            sql = self._build_query(
                columns=safe_columns,
                table_name=table_name,
                timestamp_column=safe_timestamp_column,
                symbol_column=safe_symbol_column,
                symbols=normalized_symbols,
                start=start,
                end=end,
                limit=limit,
                dialect=dialect,
            )
            params = self._build_params(
                symbols=normalized_symbols,
                start=start,
                end=end,
                limit=limit,
                dialect=dialect,
            )
            frame = pd.read_sql_query(sql, conn, params=params)

        if frame.empty:
            empty_frame = pd.DataFrame(columns=safe_columns)
            return TimescaleQueryResult(
                dimension=dimension,
                frame=empty_frame,
                symbols=normalized_symbols,
                start_ts=None,
                end_ts=None,
                max_ingested_at=None,
            )

        if safe_timestamp_column in frame:
            frame[safe_timestamp_column] = _coerce_timestamp(frame[safe_timestamp_column])
        if "ingested_at" in frame:
            frame["ingested_at"] = _coerce_timestamp(frame["ingested_at"])

        frame = frame.sort_values(safe_timestamp_column)

        if safe_symbol_column in frame:
            series = frame[safe_symbol_column].dropna().astype(str)
            symbols_present = _normalise_symbols(series.tolist())
        else:
            symbols_present = tuple()

        start_ts = (
            self._to_datetime(frame[safe_timestamp_column].min())
            if safe_timestamp_column in frame
            else None
        )
        end_ts = (
            self._to_datetime(frame[safe_timestamp_column].max())
            if safe_timestamp_column in frame
            else None
        )
        max_ingested_at = (
            self._to_datetime(frame["ingested_at"].max())
            if "ingested_at" in frame and not frame["ingested_at"].isna().all()
            else None
        )

        return TimescaleQueryResult(
            dimension=dimension,
            frame=frame.reset_index(drop=True),
            symbols=symbols_present or normalized_symbols,
            start_ts=start_ts,
            end_ts=end_ts,
            max_ingested_at=max_ingested_at,
        )

    def _build_query(
        self,
        *,
        columns: Sequence[str],
        table_name: str,
        timestamp_column: str,
        symbol_column: str,
        symbols: Sequence[str],
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
        dialect: str,
    ) -> str:
        selected = ", ".join(columns)
        query = [f"SELECT {selected} FROM {table_name} WHERE 1=1"]

        if start is not None:
            query.append(f"AND {timestamp_column} >= :start_ts")
        if end is not None:
            query.append(f"AND {timestamp_column} <= :end_ts")

        if symbols:
            if dialect == "postgresql":
                query.append(f"AND {symbol_column} = ANY(:symbols)")
            else:
                placeholders = ", ".join(f":symbol_{idx}" for idx in range(len(symbols)))
                query.append(f"AND {symbol_column} IN ({placeholders})")

        query.append(f"ORDER BY {timestamp_column} ASC")
        if limit is not None:
            query.append("LIMIT :limit")

        return " ".join(query)

    def _build_params(
        self,
        *,
        symbols: Sequence[str],
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
        dialect: str,
    ) -> dict[str, object]:
        params: dict[str, object] = {}
        start_bound = self._coerce_bound(start, dialect)
        end_bound = self._coerce_bound(end, dialect)
        if start_bound is not None:
            params["start_ts"] = start_bound
        if end_bound is not None:
            params["end_ts"] = end_bound
        if limit is not None:
            params["limit"] = int(limit)

        if symbols:
            if dialect == "postgresql":
                params["symbols"] = list(symbols)
            else:
                params.update({f"symbol_{idx}": symbol for idx, symbol in enumerate(symbols)})

        return params

    @staticmethod
    def _to_datetime(value: object) -> datetime | None:
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            return cast(datetime, value.to_pydatetime())
        if isinstance(value, datetime):
            return value
        return None

    @staticmethod
    def _coerce_bound(bound: object, dialect: str) -> datetime | None:
        if isinstance(bound, pd.Timestamp):
            if pd.isna(bound):
                return None
            bound = bound.to_pydatetime()
        if isinstance(bound, datetime):
            if dialect != "postgresql" and bound.tzinfo is not None:
                return bound.astimezone(UTC).replace(tzinfo=None)
            return bound
        return None
