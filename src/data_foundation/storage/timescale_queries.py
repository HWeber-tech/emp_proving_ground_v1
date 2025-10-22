"""Query interface for historical tick data stored in TimescaleDB.

The roadmap calls for a dedicated access layer that can execute the
high-frequency queries exercised by the sensory cortex without forcing the test
suite to depend on a running Timescale daemon.  This module fulfils that need
by building lightweight SQLAlchemy statements that run against both the
Timescale deployment and the SQLite fallback used in CI.  A tiny LRU cache is
included so repeated ad-hoc lookups (for example inside notebooks) avoid
hitting the database when parameters stay the same.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping, Sequence

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.engine import Engine
from sqlalchemy.sql import ColumnElement, column, table

__all__ = [
    "QueryCacheStats",
    "TimescaleTickQueryResult",
    "TimescaleQueryInterface",
]


def _normalise_symbols(symbols: Sequence[str] | None) -> tuple[str, ...]:
    if not symbols:
        return tuple()
    return tuple(sorted({symbol.strip() for symbol in symbols if symbol}))


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _resolve_table_name(*, schema: str, table_name: str, dialect_name: str) -> str:
    if dialect_name == "postgresql":
        return f"{schema}.{table_name}"
    return f"{schema}_{table_name}"


@dataclass(frozen=True)
class QueryCacheStats:
    """Expose cache hit/miss telemetry for observability."""

    hits: int
    misses: int


@dataclass(frozen=True)
class TimescaleTickQueryResult:
    """Structured response representing a tick query."""

    dimension: str
    frame: pd.DataFrame
    symbols: tuple[str, ...]
    window_start: datetime | None
    window_end: datetime | None
    max_ingested_at: datetime | None

    @property
    def rowcount(self) -> int:
        return int(self.frame.shape[0])

    def copy(self) -> "TimescaleTickQueryResult":
        return TimescaleTickQueryResult(
            dimension=self.dimension,
            frame=self.frame.copy(deep=True),
            symbols=self.symbols,
            window_start=self.window_start,
            window_end=self.window_end,
            max_ingested_at=self.max_ingested_at,
        )


class _QueryCache:
    """Very small LRU cache suitable for repeated lookups in tests or notebooks."""

    def __init__(self, *, maxsize: int = 32) -> None:
        self._data: OrderedDict[tuple[str, ...], TimescaleTickQueryResult] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, key: tuple[str, ...]) -> TimescaleTickQueryResult | None:
        result = self._data.get(key)
        if result is None:
            self._misses += 1
            return None
        self._hits += 1
        self._data.move_to_end(key)
        return result.copy()

    def set(self, key: tuple[str, ...], value: TimescaleTickQueryResult) -> None:
        self._data[key] = value.copy()
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    @property
    def stats(self) -> QueryCacheStats:
        return QueryCacheStats(hits=self._hits, misses=self._misses)


class TimescaleQueryInterface:
    """Facade for querying historical tick data from Timescale-backed storage."""

    def __init__(
        self,
        engine: Engine,
        *,
        schema: str = "market_data",
        trade_table: str = "ticks",
        quote_table: str = "quotes",
        book_table: str = "order_book",
        cache_size: int = 32,
    ) -> None:
        self._engine = engine
        self._schema = schema
        self._trade_table = trade_table
        self._quote_table = quote_table
        self._book_table = book_table
        self._dialect = engine.dialect.name
        self._cache = _QueryCache(maxsize=cache_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cache_stats(self) -> QueryCacheStats:
        return self._cache.stats

    def fetch_trades(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        resample: str | None = None,
    ) -> TimescaleTickQueryResult:
        return self._execute_query(
            dimension="ticks",
            schema=self._schema,
            table_name=self._trade_table,
            symbols=_normalise_symbols(symbols),
            start=_as_utc(start),
            end=_as_utc(end),
            limit=limit,
            resample=resample,
            default_columns=("ts", "symbol", "price", "size", "ingested_at"),
        )

    def fetch_quotes(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        resample: str | None = None,
    ) -> TimescaleTickQueryResult:
        return self._execute_query(
            dimension="quotes",
            schema=self._schema,
            table_name=self._quote_table,
            symbols=_normalise_symbols(symbols),
            start=_as_utc(start),
            end=_as_utc(end),
            limit=limit,
            resample=resample,
            default_columns=(
                "ts",
                "symbol",
                "bid_price",
                "bid_size",
                "ask_price",
                "ask_size",
                "ingested_at",
            ),
        )

    def fetch_order_book(
        self,
        *,
        symbols: Sequence[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> TimescaleTickQueryResult:
        return self._execute_query(
            dimension="order_book",
            schema=self._schema,
            table_name=self._book_table,
            symbols=_normalise_symbols(symbols),
            start=_as_utc(start),
            end=_as_utc(end),
            limit=limit,
            resample=None,
            default_columns=(
                "ts",
                "symbol",
                "bid_price",
                "bid_size",
                "ask_price",
                "ask_size",
                "level",
                "ingested_at",
            ),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute_query(
        self,
        *,
        dimension: str,
        schema: str,
        table_name: str,
        symbols: tuple[str, ...],
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
        resample: str | None,
        default_columns: Sequence[str],
    ) -> TimescaleTickQueryResult:
        cache_key = self._cache_key(
            dimension=dimension,
            schema=schema,
            table_name=table_name,
            symbols=symbols,
            start=start,
            end=end,
            limit=limit,
            resample=resample,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw_frame = self._run_select(
            schema=schema,
            table_name=table_name,
            columns=default_columns,
            symbols=symbols,
            start=start,
            end=end,
            limit=limit,
        )
        frame = raw_frame

        if resample and not frame.empty:
            frame = self._resample(frame, resample)

        result = TimescaleTickQueryResult(
            dimension=dimension,
            frame=frame,
            symbols=symbols,
            window_start=start,
            window_end=end,
            max_ingested_at=self._max_ingested_at(raw_frame),
        )
        self._cache.set(cache_key, result)
        return result

    def _cache_key(
        self,
        *,
        dimension: str,
        schema: str,
        table_name: str,
        symbols: tuple[str, ...],
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
        resample: str | None,
    ) -> tuple[str, ...]:
        start_key = start.isoformat() if start else ""
        end_key = end.isoformat() if end else ""
        limit_key = str(limit) if limit is not None else ""
        resample_key = resample or ""
        return (
            dimension,
            schema,
            table_name,
            *symbols,
            start_key,
            end_key,
            limit_key,
            resample_key,
        )

    def _run_select(
        self,
        *,
        schema: str,
        table_name: str,
        columns: Sequence[str],
        symbols: tuple[str, ...],
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
    ) -> pd.DataFrame:
        safe_columns = [column(name) for name in columns]
        resolved_name = _resolve_table_name(
            schema=schema, table_name=table_name, dialect_name=self._dialect
        )
        tick_table = table(resolved_name, *safe_columns)

        filters: list[ColumnElement[bool]] = []
        if symbols:
            filters.append(tick_table.c.symbol.in_(symbols))
        if start is not None:
            filters.append(tick_table.c.ts >= start)
        if end is not None:
            filters.append(tick_table.c.ts <= end)

        stmt = select(*safe_columns)
        if filters:
            stmt = stmt.where(and_(*filters))
        stmt = stmt.order_by(tick_table.c.ts.asc())
        if limit is not None:
            stmt = stmt.limit(limit)

        with self._engine.connect() as conn:
            result = conn.execute(stmt)
            frame = pd.DataFrame(result.fetchall(), columns=result.keys())

        for column_name in ("ts", "ingested_at"):
            if column_name in frame.columns:
                frame[column_name] = pd.to_datetime(frame[column_name], utc=True, errors="coerce")
        return frame

    def _resample(self, frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        if "ts" not in frame.columns:
            return frame
        if frame.empty:
            return frame
        index = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
        grouped = frame.assign(ts=index).set_index("ts")
        aggregations: Mapping[str, str] = {}
        for column_name in frame.columns:
            if column_name in {"ts", "symbol"}:
                continue
            if column_name.endswith("size"):
                aggregations[column_name] = "sum"
            elif column_name.endswith("price") or column_name in {"price", "bid_price", "ask_price"}:
                aggregations[column_name] = "mean"
            elif column_name == "ingested_at":
                aggregations[column_name] = "max"
        resampled = (
            grouped.groupby("symbol")
            .resample(rule)
            .agg(aggregations)
            .dropna(how="all")
            .reset_index()
        )
        resampled.rename(columns={"level_1": "ts"}, inplace=True)
        if "ingested_at" in resampled.columns:
            resampled["ingested_at"] = pd.to_datetime(
                resampled["ingested_at"], utc=True, errors="coerce"
            )
        return resampled

    def _max_ingested_at(self, frame: pd.DataFrame) -> datetime | None:
        if "ingested_at" not in frame.columns or frame.empty:
            return None
        series = pd.to_datetime(frame["ingested_at"], utc=True, errors="coerce")
        if series.isna().all():
            return None
        return series.max().to_pydatetime(warn=False)


