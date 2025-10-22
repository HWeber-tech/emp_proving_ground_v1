"""Async ingestion adapter for TimescaleDB-backed market data."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Sequence

import pandas as pd
from sqlalchemy.engine import Engine

from ..persist.timescale import TimescaleIngestResult, TimescaleIngestor

logger = logging.getLogger(__name__)


Payload = pd.DataFrame | Sequence[dict[str, object]] | Iterable[dict[str, object]]
IngestorFactory = Callable[[Engine, int], TimescaleIngestor]


def _as_dataframe(payload: Payload) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload.copy()
    if isinstance(payload, Sequence):
        if not payload:
            return pd.DataFrame()
        return pd.DataFrame(payload)
    materialised = list(payload)
    if not materialised:
        return pd.DataFrame()
    return pd.DataFrame(materialised)


def _chunk_frame(frame: pd.DataFrame, *, batch_size: int) -> Iterable[pd.DataFrame]:
    if frame.empty:
        return []
    size = max(batch_size, 1)
    total = len(frame.index)
    for start in range(0, total, size):
        yield frame.iloc[start : start + size].copy()


def _merge_symbols(results: Sequence[TimescaleIngestResult]) -> tuple[str, ...]:
    symbols: set[str] = set()
    for result in results:
        symbols.update(result.symbols)
    return tuple(sorted(symbols))


def _min_ts(values: Iterable[datetime | None]) -> datetime | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return min(filtered)


def _max_ts(values: Iterable[datetime | None]) -> datetime | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def _min_freshness(values: Iterable[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return min(filtered)


@dataclass(frozen=True)
class TimescaleAdapterResult:
    """Aggregated telemetry from a Timescale adapter ingestion run."""

    dimension: str
    batches: int
    rows_written: int
    symbols: tuple[str, ...]
    start_ts: datetime | None
    end_ts: datetime | None
    total_duration_seconds: float
    freshness_seconds: float | None
    errors: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def empty(cls, *, dimension: str) -> "TimescaleAdapterResult":
        return cls(
            dimension=dimension,
            batches=0,
            rows_written=0,
            symbols=tuple(),
            start_ts=None,
            end_ts=None,
            total_duration_seconds=0.0,
            freshness_seconds=None,
            errors=tuple(),
        )

    @property
    def ok(self) -> bool:
        return not self.errors


class TimescaleAdapter:
    """High-throughput async writer built on top of :class:`TimescaleIngestor`."""

    def __init__(
        self,
        engine: Engine,
        *,
        batch_size: int = 1000,
        chunk_size: int = 500,
        default_source: str = "stream",
        ingestor_factory: IngestorFactory | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        factory = ingestor_factory or (lambda eng, chunk: TimescaleIngestor(eng, chunk_size=chunk))

        self._engine = engine
        self._ingestor = factory(engine, chunk_size)
        self._batch_size = batch_size
        self._default_source = default_source
        self._logger = logging.getLogger(f"{__name__}.TimescaleAdapter")

    @property
    def ingestor(self) -> TimescaleIngestor:
        return self._ingestor

    async def ingest_intraday_trades(self, payload: Payload, *, source: str | None = None) -> TimescaleAdapterResult:
        frame = _as_dataframe(payload)
        return await self._ingest(
            frame,
            dimension="intraday_trades",
            upsert_fn=self._ingestor.upsert_intraday_trades,
            source=source or self._default_source,
        )

    async def ingest_daily_bars(
        self, payload: Payload, *, source: str | None = None
    ) -> TimescaleAdapterResult:
        frame = _as_dataframe(payload)
        return await self._ingest(
            frame,
            dimension="daily_bars",
            upsert_fn=self._ingestor.upsert_daily_bars,
            source=source or self._default_source,
        )

    async def ingest_macro_events(self, payload: Payload, *, source: str | None = None) -> TimescaleAdapterResult:
        frame = _as_dataframe(payload)
        return await self._ingest(
            frame,
            dimension="macro_events",
            upsert_fn=self._ingestor.upsert_macro_events,
            source=source or self._default_source,
        )

    async def _ingest(
        self,
        frame: pd.DataFrame,
        *,
        dimension: str,
        upsert_fn: Callable[..., TimescaleIngestResult],
        source: str,
    ) -> TimescaleAdapterResult:
        if frame.empty:
            return TimescaleAdapterResult.empty(dimension=dimension)

        results: list[TimescaleIngestResult] = []
        errors: list[str] = []
        batches = 0
        for chunk in _chunk_frame(frame, batch_size=self._batch_size):
            batches += 1
            try:
                result = await asyncio.to_thread(upsert_fn, chunk, source=source)
            except Exception as exc:  # pragma: no cover - defensive guard
                self._logger.exception(
                    "Timescale adapter failed for %s batch %s", dimension, batches
                )
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            results.append(result)

        if not results:
            return TimescaleAdapterResult(
                dimension=dimension,
                batches=batches,
                rows_written=0,
                symbols=tuple(),
                start_ts=None,
                end_ts=None,
                total_duration_seconds=0.0,
                freshness_seconds=None,
                errors=tuple(errors),
            )

        rows_written = sum(result.rows_written for result in results)
        symbols = _merge_symbols(results)
        start_ts = _min_ts(result.start_ts for result in results)
        end_ts = _max_ts(result.end_ts for result in results)
        total_duration = sum(result.ingest_duration_seconds for result in results)
        freshness = _min_freshness(result.freshness_seconds for result in results)

        return TimescaleAdapterResult(
            dimension=dimension,
            batches=batches,
            rows_written=rows_written,
            symbols=symbols,
            start_ts=start_ts,
            end_ts=end_ts,
            total_duration_seconds=total_duration,
            freshness_seconds=freshness,
            errors=tuple(errors),
        )


__all__ = ["TimescaleAdapter", "TimescaleAdapterResult"]
