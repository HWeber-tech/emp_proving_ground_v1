from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Mapping, Sequence
from uuid import uuid4

import pandas as pd
import pytest
from sqlalchemy import create_engine

from main import _execute_timescale_ingest
from src.core.event_bus import EventBus
from src.data_foundation.cache.redis_cache import RedisConnectionSettings
from src.data_foundation.ingest.configuration import (
    InstitutionalIngestConfig,
    TimescaleIngestRecoverySettings,
)
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.batch.spark_export import (
    SparkExportFormat,
    SparkExportJob,
    SparkExportPlan,
    SparkExportStatus,
)
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestJournal,
    TimescaleIngestResult,
    TimescaleIngestRunRecord,
    TimescaleIngestor,
    TimescaleMigrator,
)
from src.data_foundation.streaming.kafka_stream import KafkaConnectionSettings
from src.operations.data_backbone import BackboneStatus
from src.operations.professional_readiness import ProfessionalReadinessStatus


def test_ingest_journal_round_trip(tmp_path) -> None:
    db_path = tmp_path / "journal.db"
    engine = create_engine(f"sqlite:///{db_path}")
    TimescaleMigrator(engine).apply()

    journal = TimescaleIngestJournal(engine)
    record = TimescaleIngestRunRecord(
        run_id=str(uuid4()),
        dimension="daily_bars",
        status="ok",
        rows_written=10,
        freshness_seconds=30.0,
        ingest_duration_seconds=1.5,
        executed_at=datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
        source="yahoo",
        symbols=("EURUSD",),
        metadata={"message": "healthy", "overall_status": "ok"},
    )

    journal.record([record])
    fetched = journal.fetch_recent()

    assert len(fetched) == 1
    stored = fetched[0]
    assert stored.run_id == record.run_id
    assert stored.dimension == "daily_bars"
    assert stored.status == "ok"
    assert stored.rows_written == 10
    assert stored.symbols == ("EURUSD",)
    assert stored.metadata["overall_status"] == "ok"
    summary = stored.as_dict()
    assert summary["run_id"] == record.run_id
    assert summary["dimension"] == "daily_bars"
    assert summary["symbols"] == ["EURUSD"]


@pytest.mark.asyncio()
async def test_execute_timescale_ingest_records_journal(tmp_path) -> None:
    db_path = tmp_path / "integration.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()
    engine.dispose()

    class _StubOrchestrator:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def run(self, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            assert plan.daily is not None
            return {
                "daily_bars": TimescaleIngestResult(
                    rows_written=5,
                    symbols=("EURUSD",),
                    start_ts=datetime(2024, 1, 1, tzinfo=UTC),
                    end_ts=datetime(2024, 1, 2, tzinfo=UTC),
                    ingest_duration_seconds=1.0,
                    freshness_seconds=45.0,
                    dimension="daily_bars",
                    source="yahoo",
                )
            }

    ingest_config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=["EURUSD"]),
        ),
        timescale_settings=settings,
        kafka_settings=KafkaConnectionSettings.from_mapping({}),
        redis_settings=RedisConnectionSettings(),
        metadata={"test": True},
        schedule=None,
    )

    bus = EventBus()

    succeeded, backup_snapshot = await _execute_timescale_ingest(
        ingest_config=ingest_config,
        event_bus=bus,
        publisher=None,
        kafka_health_publisher=None,
        kafka_metrics_publisher=None,
        kafka_quality_publisher=None,
        fallback=None,
        orchestrator_cls=_StubOrchestrator,
    )
    assert succeeded is True
    assert backup_snapshot is not None

    journal_engine = settings.create_engine()
    journal = TimescaleIngestJournal(journal_engine)
    entries = journal.fetch_recent()
    journal_engine.dispose()

    assert entries
    entry = entries[0]
    assert entry.dimension == "daily_bars"
    assert entry.status == "ok"
    assert entry.rows_written == 5
    assert entry.symbols == ("EURUSD",)
    assert entry.metadata.get("plan", {}).get("test") is True
    assert entry.metadata.get("overall_status") == "ok"


@pytest.mark.asyncio()
async def test_execute_timescale_ingest_publishes_kafka_metrics(tmp_path) -> None:
    db_path = tmp_path / "metrics.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()
    engine.dispose()

    class _StubOrchestrator:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def run(self, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            assert plan.daily is not None
            return {
                "daily_bars": TimescaleIngestResult(
                    rows_written=60,
                    symbols=("EURUSD",),
                    start_ts=datetime(2024, 1, 1, tzinfo=UTC),
                    end_ts=datetime(2024, 1, 2, tzinfo=UTC),
                    ingest_duration_seconds=0.5,
                    freshness_seconds=30.0,
                    dimension="daily_bars",
                    source="yahoo",
                )
            }

    class _StubMetricsPublisher:
        def __init__(self) -> None:
            self.calls: list[tuple[object, Mapping[str, object] | None]] = []

        def publish(
            self,
            snapshot,
            *,
            metadata: Mapping[str, object] | None = None,
        ) -> None:
            self.calls.append((snapshot, metadata))

    class _StubQualityPublisher:
        def __init__(self) -> None:
            self.calls: list[tuple[object, Mapping[str, object] | None]] = []

        def publish(
            self,
            report,
            *,
            metadata: Mapping[str, object] | None = None,
        ) -> None:
            self.calls.append((report, metadata))

    ingest_config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=["EURUSD"]),
        ),
        timescale_settings=settings,
        kafka_settings=KafkaConnectionSettings.from_mapping({}),
        redis_settings=RedisConnectionSettings(),
        metadata={"slice": "daily"},
        schedule=None,
    )

    bus = EventBus()
    metrics_publisher = _StubMetricsPublisher()
    quality_publisher = _StubQualityPublisher()

    succeeded, backup_snapshot = await _execute_timescale_ingest(
        ingest_config=ingest_config,
        event_bus=bus,
        publisher=None,
        kafka_health_publisher=None,
        kafka_metrics_publisher=metrics_publisher,
        kafka_quality_publisher=quality_publisher,
        fallback=None,
        orchestrator_cls=_StubOrchestrator,
    )

    assert succeeded is True
    assert backup_snapshot is not None
    assert metrics_publisher.calls
    snapshot, metadata = metrics_publisher.calls[0]
    for key, value in ingest_config.metadata.items():
        assert metadata[key] == value
    validation_meta = metadata["validation"]
    assert validation_meta["status"] in {"ok", "warn"}
    assert getattr(snapshot, "total_rows")() == 60
    assert quality_publisher.calls
    report, quality_metadata = quality_publisher.calls[0]
    for key, value in ingest_config.metadata.items():
        assert quality_metadata[key] == value
    quality_validation = quality_metadata["validation"]
    assert quality_validation["status"] in {"ok", "warn"}
    assert getattr(report, "status").value == "ok"


@pytest.mark.asyncio()
async def test_execute_timescale_ingest_runs_recovery(tmp_path) -> None:
    db_path = tmp_path / "recovery.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()
    engine.dispose()

    class _RecoveryOrchestrator:
        def __init__(self, *_: object, **__: object) -> None:
            self.calls = 0
            self.plans: list[TimescaleBackbonePlan] = []

        def run(self, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            self.calls += 1
            self.plans.append(plan)
            assert plan.daily is not None
            if self.calls == 1:
                return {
                    "daily_bars": TimescaleIngestResult(
                        rows_written=5,
                        symbols=("EURUSD",),
                        start_ts=datetime(2024, 1, 1, tzinfo=UTC),
                        end_ts=datetime(2024, 1, 2, tzinfo=UTC),
                        ingest_duration_seconds=1.0,
                        freshness_seconds=120.0,
                        dimension="daily_bars",
                        source="yahoo",
                    )
                }
            if self.calls == 2:
                assert plan.daily.symbols == ["GBPUSD"]
                return {
                    "daily_bars": TimescaleIngestResult(
                        rows_written=3,
                        symbols=("GBPUSD",),
                        start_ts=datetime(2024, 1, 1, tzinfo=UTC),
                        end_ts=datetime(2024, 1, 2, tzinfo=UTC),
                        ingest_duration_seconds=0.5,
                        freshness_seconds=90.0,
                        dimension="daily_bars",
                        source="yahoo",
                    )
                }
            pytest.fail("Unexpected additional recovery attempt")

    ingest_config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=["EURUSD", "GBPUSD"]),
        ),
        timescale_settings=settings,
        kafka_settings=KafkaConnectionSettings.from_mapping({}),
        redis_settings=RedisConnectionSettings(),
        metadata={},
        schedule=None,
        recovery=TimescaleIngestRecoverySettings(
            enabled=True,
            max_attempts=2,
            lookback_multiplier=2.0,
        ),
    )

    bus = EventBus()

    succeeded, backup_snapshot = await _execute_timescale_ingest(
        ingest_config=ingest_config,
        event_bus=bus,
        publisher=None,
        kafka_health_publisher=None,
        kafka_metrics_publisher=None,
        kafka_quality_publisher=None,
        fallback=None,
        orchestrator_cls=_RecoveryOrchestrator,
    )

    assert succeeded is True
    assert backup_snapshot is not None

    journal_engine = settings.create_engine()
    journal = TimescaleIngestJournal(journal_engine)
    entries = journal.fetch_recent()
    journal_engine.dispose()

    assert entries
    entry = entries[0]
    assert entry.rows_written == 8
    assert entry.symbols == ("EURUSD", "GBPUSD")
    recovery_meta = entry.metadata.get("plan", {}).get("recovery", {})
    assert recovery_meta.get("attempts") == 1
    steps = recovery_meta.get("steps") or []
    assert steps and steps[0]["health_status"] == "ok"
    assert steps[0]["dimensions"] == ["daily_bars"]


def test_ingest_journal_fetch_latest(tmp_path) -> None:
    db_path = tmp_path / "journal_latest.db"
    engine = create_engine(f"sqlite:///{db_path}")
    TimescaleMigrator(engine).apply()

    journal = TimescaleIngestJournal(engine)
    base_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    older = TimescaleIngestRunRecord(
        run_id="older",
        dimension="daily_bars",
        status="ok",
        rows_written=5,
        freshness_seconds=60.0,
        ingest_duration_seconds=2.0,
        executed_at=base_time,
        source="yahoo",
        symbols=("EURUSD",),
    )
    newer = TimescaleIngestRunRecord(
        run_id="newer",
        dimension="daily_bars",
        status="warn",
        rows_written=4,
        freshness_seconds=120.0,
        ingest_duration_seconds=3.0,
        executed_at=base_time.replace(hour=13),
        source="yahoo",
        symbols=("EURUSD",),
    )
    macro = TimescaleIngestRunRecord(
        run_id="macro",
        dimension="macro_events",
        status="ok",
        rows_written=2,
        freshness_seconds=None,
        ingest_duration_seconds=1.2,
        executed_at=base_time.replace(hour=11),
        source="fred",
        symbols=("USD",),
    )

    journal.record([older, macro, newer])

    latest_all = journal.fetch_latest_by_dimension()
    assert latest_all["daily_bars"].run_id == "newer"
    assert latest_all["macro_events"].run_id == "macro"

    latest_filtered = journal.fetch_latest_by_dimension(["macro_events"])
    assert set(latest_filtered) == {"macro_events"}
    assert latest_filtered["macro_events"].run_id == "macro"


@pytest.mark.asyncio()
async def test_execute_timescale_ingest_short_circuits_when_validation_fails(
    monkeypatch,
) -> None:
    settings = TimescaleConnectionSettings.from_mapping({})

    class _StubOrchestrator:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def run(self, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            assert plan.daily is not None
            return {
                "daily_bars": TimescaleIngestResult(
                    rows_written=3,
                    symbols=("EURUSD",),
                    start_ts=datetime(2024, 1, 1, tzinfo=UTC),
                    end_ts=datetime(2024, 1, 2, tzinfo=UTC),
                    ingest_duration_seconds=0.5,
                    freshness_seconds=120.0,
                    dimension="daily_bars",
                    source="yahoo",
                )
            }

    ingest_config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=["EURUSD"]),
        ),
        timescale_settings=settings,
        kafka_settings=KafkaConnectionSettings.from_mapping({}),
        redis_settings=RedisConnectionSettings(),
        metadata={"plan": {"daily_bars": {"symbols": ["EURUSD"], "lookback_days": 60}}},
        schedule=None,
    )

    bus = EventBus()
    fallback_called = False

    async def _fallback() -> None:
        nonlocal fallback_called
        fallback_called = True

    validation_published: list[object] = []
    readiness_published: list[object] = []
    professional_published: list[object] = []
    validation_recorded: list[object] = []
    readiness_recorded: list[object] = []
    professional_recorded: list[object] = []

    monkeypatch.setattr(
        "src.runtime.runtime_builder._publish_data_backbone_validation",
        lambda event_bus, snapshot: validation_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder._publish_data_backbone_readiness",
        lambda event_bus, snapshot: readiness_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder._publish_professional_readiness",
        lambda event_bus, snapshot: professional_published.append(snapshot),
    )

    succeeded, backup_snapshot = await _execute_timescale_ingest(
        ingest_config=ingest_config,
        event_bus=bus,
        publisher=None,
        kafka_health_publisher=None,
        kafka_metrics_publisher=None,
        kafka_quality_publisher=None,
        fallback=_fallback,
        orchestrator_cls=_StubOrchestrator,
        record_backbone_validation_snapshot=validation_recorded.append,
        record_backbone_snapshot=readiness_recorded.append,
        record_professional_snapshot=professional_recorded.append,
    )

    assert succeeded is False
    assert backup_snapshot is None
    assert fallback_called is True

    assert validation_published and validation_recorded
    published_validation = validation_published[0]
    assert published_validation.status is BackboneStatus.fail

    assert readiness_published and readiness_recorded
    readiness_snapshot = readiness_recorded[0]
    assert readiness_snapshot.status is BackboneStatus.fail
    component_names = {component.name for component in readiness_snapshot.components}
    assert "validation" in component_names

    assert professional_published and professional_recorded
    professional_snapshot = professional_recorded[0]
    assert professional_snapshot.status is ProfessionalReadinessStatus.fail


@pytest.mark.asyncio()
async def test_execute_timescale_ingest_runs_spark_export(tmp_path) -> None:
    db_path = tmp_path / "spark_export.db"
    settings = TimescaleConnectionSettings(url=f"sqlite:///{db_path}")
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()
    ingestor = TimescaleIngestor(engine)

    frame = pd.DataFrame(
        [
            {
                "date": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "open": 1.0,
                "high": 1.05,
                "low": 0.98,
                "close": 1.02,
                "adj_close": 1.01,
                "volume": 900,
                "symbol": "EURUSD",
            }
        ]
    )
    ingestor.upsert_daily_bars(frame)
    engine.dispose()

    class _StubOrchestrator:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def run(self, plan: TimescaleBackbonePlan) -> dict[str, TimescaleIngestResult]:
            return {
                "daily_bars": TimescaleIngestResult(
                    rows_written=1,
                    symbols=("EURUSD",),
                    start_ts=datetime(2024, 1, 2, tzinfo=UTC),
                    end_ts=datetime(2024, 1, 2, tzinfo=UTC),
                    ingest_duration_seconds=0.2,
                    freshness_seconds=30.0,
                    dimension="daily_bars",
                    source="yahoo",
                )
            }

    spark_plan = SparkExportPlan(
        root_path=tmp_path / "spark",
        format=SparkExportFormat.csv,
        jobs=(
            SparkExportJob(
                dimension="daily_bars",
                symbols=("EURUSD",),
                filename="daily.csv",
            ),
        ),
    )

    ingest_config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(daily=DailyBarIngestPlan(symbols=["EURUSD"])),
        timescale_settings=settings,
        kafka_settings=KafkaConnectionSettings.from_mapping({}),
        redis_settings=RedisConnectionSettings(),
        metadata={},
        schedule=None,
        spark_export=spark_plan,
    )

    bus = EventBus()
    spark_snapshots: list[SparkExportStatus] = []

    succeeded, backup_snapshot = await _execute_timescale_ingest(
        ingest_config=ingest_config,
        event_bus=bus,
        publisher=None,
        kafka_health_publisher=None,
        kafka_metrics_publisher=None,
        kafka_quality_publisher=None,
        fallback=None,
        orchestrator_cls=_StubOrchestrator,
        record_spark_snapshot=lambda snapshot: spark_snapshots.append(snapshot.status),
    )

    assert succeeded is True
    assert backup_snapshot is not None
    assert spark_snapshots and spark_snapshots[0] is SparkExportStatus.ok
    export_path = tmp_path / "spark" / "daily_bars" / "daily.csv"
    assert export_path.exists()


@pytest.mark.asyncio()
async def test_execute_timescale_ingest_preserves_external_manager(tmp_path) -> None:
    class _StubManager:
        def __init__(self) -> None:
            self.shutdown_called = False
            self.ingest_invocations = 0
            self._base = datetime(2024, 1, 3, tzinfo=UTC)

        def cache_metrics(self, *, reset: bool = False) -> Mapping[str, int | str]:
            return {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "namespace": "stub",
            }

        def ingest_market_slice(self, *, symbols, **_: object) -> Mapping[str, TimescaleIngestResult]:
            self.ingest_invocations += 1
            upper_symbols = tuple(str(symbol).upper() for symbol in symbols)
            start_ts = self._base - timedelta(days=1)
            return {
                "daily_bars": TimescaleIngestResult(
                    rows_written=2,
                    symbols=upper_symbols,
                    start_ts=start_ts,
                    end_ts=self._base,
                    ingest_duration_seconds=0.1,
                    freshness_seconds=15.0,
                    dimension="daily_bars",
                    source="stub",
                )
            }

        def fetch_data(
            self,
            symbol: str,
            *,
            period: str | None = None,
            interval: str | None = None,
            start: str | datetime | None = None,
            end: str | datetime | None = None,
        ) -> pd.DataFrame:
            _ = (period, start, end)
            frame = pd.DataFrame(
                [
                    {
                        "timestamp": self._base - timedelta(days=1),
                        "symbol": symbol.upper(),
                        "close": 1.1,
                        "price": 1.1,
                        "volume": 100,
                    },
                    {
                        "timestamp": self._base,
                        "symbol": symbol.upper(),
                        "close": 1.2,
                        "price": 1.2,
                        "volume": 150,
                    },
                ]
            )
            if interval and interval.lower() not in {"1d", "daily"}:
                frame["price"] = frame["price"] + 0.01
            return frame

        def fetch_macro_events(
            self,
            *,
            calendars: Sequence[str] | None = None,
            start: str | datetime | None = None,
            end: str | datetime | None = None,
            limit: int | None = None,
        ) -> pd.DataFrame:
            _ = (calendars, start, end, limit)
            return pd.DataFrame(
                [
                    {
                        "timestamp": self._base,
                        "calendar": "ECB",
                        "event_name": "Rate Decision",
                    }
                ]
            )

        async def shutdown(self) -> None:
            self.shutdown_called = True

    manager = _StubManager()

    ingest_config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=TimescaleBackbonePlan(
            daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=2),
        ),
        timescale_settings=TimescaleConnectionSettings(url=f"sqlite:///{tmp_path / 'stub.db'}"),
        kafka_settings=KafkaConnectionSettings.from_mapping({}),
        redis_settings=RedisConnectionSettings(),
        metadata={},
        schedule=None,
    )

    bus = EventBus()

    succeeded, backup_snapshot = await _execute_timescale_ingest(
        ingest_config=ingest_config,
        event_bus=bus,
        publisher=None,
        kafka_health_publisher=None,
        kafka_metrics_publisher=None,
        kafka_quality_publisher=None,
        fallback=None,
        data_manager=manager,
    )

    assert succeeded is True
    assert backup_snapshot is not None
    assert manager.ingest_invocations == 1
    assert manager.shutdown_called is False

    post_frame = manager.fetch_data("EURUSD", interval="1d")
    assert not post_frame.empty
