"""Regression coverage for the Timescale ingest orchestrator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackboneOrchestrator,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleIngestResult
from src.data_foundation.schemas import MacroEvent


class _FailingSettings:
    """Settings stub that fails if the engine is constructed."""

    def create_engine(self) -> Any:  # pragma: no cover - defensive helper
        raise AssertionError("Engine should not be created for an empty plan")


class _RecordingSettings:
    """Settings stub that records engine lifecycle interactions."""

    def __init__(self) -> None:
        self.engine = _RecordingEngine()
        self.creations = 0

    def create_engine(self) -> "_RecordingEngine":
        self.creations += 1
        return self.engine


class _RecordingEngine:
    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


class _RecordingMigrator:
    def __init__(self, engine: _RecordingEngine) -> None:  # pragma: no cover - trivial glue
        self.engine = engine
        self.applied = False

    def apply(self) -> None:
        self.applied = True


@dataclass(slots=True)
class _RecordingIngestor:
    engine: _RecordingEngine
    daily_frames: list[tuple[pd.DataFrame, str]] | None = None
    intraday_frames: list[tuple[pd.DataFrame, str]] | None = None
    macro_frames: list[tuple[pd.DataFrame, str]] | None = None

    def __post_init__(self) -> None:
        self.daily_frames = []
        self.intraday_frames = []
        self.macro_frames = []

    def upsert_daily_bars(self, frame: pd.DataFrame, *, source: str) -> TimescaleIngestResult:
        assert isinstance(frame, pd.DataFrame)
        self.daily_frames.append((frame, source))
        return TimescaleIngestResult(
            rows_written=int(len(frame)),
            symbols=tuple(frame.get("symbol", [])),
            start_ts=datetime(2024, 1, 1, tzinfo=UTC),
            end_ts=datetime(2024, 1, 5, tzinfo=UTC),
            ingest_duration_seconds=0.5,
            freshness_seconds=60.0,
            dimension="daily_bars",
            source=source,
        )

    def upsert_intraday_trades(
        self, frame: pd.DataFrame, *, source: str
    ) -> TimescaleIngestResult:
        assert isinstance(frame, pd.DataFrame)
        self.intraday_frames.append((frame, source))
        return TimescaleIngestResult(
            rows_written=int(len(frame)),
            symbols=tuple(frame.get("symbol", [])),
            start_ts=datetime(2024, 1, 2, tzinfo=UTC),
            end_ts=datetime(2024, 1, 2, 23, 59, tzinfo=UTC),
            ingest_duration_seconds=0.75,
            freshness_seconds=15.0,
            dimension="intraday_trades",
            source=source,
        )

    def upsert_macro_events(self, frame: pd.DataFrame, *, source: str) -> TimescaleIngestResult:
        assert isinstance(frame, pd.DataFrame)
        self.macro_frames.append((frame, source))
        return TimescaleIngestResult(
            rows_written=int(len(frame)),
            symbols=tuple(frame.get("event", [])),
            start_ts=datetime(2024, 1, 3, tzinfo=UTC),
            end_ts=datetime(2024, 1, 4, tzinfo=UTC),
            ingest_duration_seconds=0.25,
            freshness_seconds=None,
            dimension="macro_events",
            source=source,
        )


class _RecordingPublisher:
    def __init__(self) -> None:
        self.published: list[tuple[TimescaleIngestResult, dict[str, Any]]] = []

    def publish(
        self, result: TimescaleIngestResult, *, metadata: dict[str, Any] | None = None
    ) -> None:
        self.published.append((result, dict(metadata or {})))


def test_orchestrator_ignores_empty_plan(caplog: pytest.LogCaptureFixture) -> None:
    orchestrator = TimescaleBackboneOrchestrator(_FailingSettings())

    with caplog.at_level("INFO"):
        results = orchestrator.run(plan=TimescaleBackbonePlan())

    assert results == {}
    assert any("nothing to execute" in message for message in caplog.messages)


def test_orchestrator_runs_all_slices_and_publishes_metadata() -> None:
    settings = _RecordingSettings()
    publisher = _RecordingPublisher()

    orchestrator = TimescaleBackboneOrchestrator(
        settings,
        migrator_cls=_RecordingMigrator,
        ingestor_cls=_RecordingIngestor,
        event_publisher=publisher,
    )

    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["AAPL", " MSFT "], lookback_days=3, source="yahoo"),
        intraday=IntradayTradeIngestPlan(
            symbols=["ES=F", " NQ=F"],
            lookback_days=1,
            interval="5m",
            source="yahoo",
        ),
        macro=MacroEventIngestPlan(
            events=[
                MacroEvent(
                    timestamp=datetime(2024, 1, 3, tzinfo=UTC),
                    calendar="FOMC",
                    event="Rate Decision",
                ),
                {
                    "timestamp": datetime(2024, 1, 4, tzinfo=UTC),
                    "calendar": "ECB",
                    "event": "Press Conference",
                },
            ],
            source="fred",
        ),
    )

    daily_calls: list[tuple[list[str], int]] = []
    intraday_calls: list[tuple[list[str], int, str]] = []

    def fake_fetch_daily(symbols: list[str], lookback: int) -> pd.DataFrame:
        daily_calls.append((symbols, lookback))
        return pd.DataFrame(
            {
                "symbol": symbols,
                "open": [1.0] * len(symbols),
                "close": [1.5] * len(symbols),
            }
        )

    def fake_fetch_intraday(symbols: list[str], lookback: int, interval: str) -> pd.DataFrame:
        intraday_calls.append((symbols, lookback, interval))
        return pd.DataFrame(
            {
                "symbol": symbols,
                "price": [100.0 + index for index, _ in enumerate(symbols)],
            }
        )

    results = orchestrator.run(
        plan=plan,
        fetch_daily=fake_fetch_daily,
        fetch_intraday=fake_fetch_intraday,
    )

    assert set(results.keys()) == {"daily_bars", "intraday_trades", "macro_events"}
    assert settings.creations == 1
    assert settings.engine.disposed is True

    # Validate fetcher calls received normalised symbols and lookback metadata.
    assert daily_calls == [(["AAPL", "MSFT"], 3)]
    assert intraday_calls == [(["ES=F", "NQ=F"], 1, "5m")]

    # Publisher captured metadata for each ingest slice with enrichment details.
    assert len(publisher.published) == 3
    metadata_dimensions = {metadata["plan"] for _, metadata in publisher.published}
    assert metadata_dimensions == {"daily_bars", "intraday_trades", "macro_events"}
    for result, metadata in publisher.published:
        assert metadata["source"] == result.source
        assert metadata["requested_symbols"] if metadata["plan"] != "macro_events" else True


def test_orchestrator_disposes_engine_when_intraday_fetcher_missing() -> None:
    settings = _RecordingSettings()
    migrator_cls = _RecordingMigrator
    ingestor_cls = _RecordingIngestor

    orchestrator = TimescaleBackboneOrchestrator(
        settings,
        migrator_cls=migrator_cls,
        ingestor_cls=ingestor_cls,
    )

    plan = TimescaleBackbonePlan(
        intraday=IntradayTradeIngestPlan(symbols=["ES=F"], lookback_days=1, interval="1m"),
    )

    with pytest.raises(ValueError):
        orchestrator.run(plan=plan, fetch_intraday=None)

    assert settings.engine.disposed is True


def test_orchestrator_fetches_macro_window_when_events_missing() -> None:
    settings = _RecordingSettings()
    publisher = _RecordingPublisher()

    orchestrator = TimescaleBackboneOrchestrator(
        settings,
        migrator_cls=_RecordingMigrator,
        ingestor_cls=_RecordingIngestor,
        event_publisher=publisher,
    )

    plan = TimescaleBackbonePlan(
        macro=MacroEventIngestPlan(start="2024-01-01", end="2024-01-02", source="fred"),
    )

    def fake_fetch_macro(start: str, end: str) -> Sequence[MacroEvent]:
        assert start == "2024-01-01"
        assert end == "2024-01-02"
        return [
            MacroEvent(
                timestamp=datetime(2024, 1, 1, 13, tzinfo=UTC),
                calendar="FOMC",
                event="Minutes",
            )
        ]

    results = orchestrator.run(plan=plan, fetch_macro=fake_fetch_macro)

    assert set(results.keys()) == {"macro_events"}
    (macro_result,) = results.values()
    assert macro_result.dimension == "macro_events"
    assert publisher.published  # metadata should be emitted
    _, metadata = publisher.published[0]
    assert metadata["fetched_via_window"] is True
    assert metadata["fetched_events"] == 1
    assert settings.engine.disposed is True

