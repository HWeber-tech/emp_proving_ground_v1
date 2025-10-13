from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.data_foundation.cache.redis_cache import (
    InMemoryRedis,
    ManagedRedisCache,
    RedisCachePolicy,
    RedisConnectionSettings,
)
from src.data_foundation.ingest.configuration import InstitutionalIngestConfig
from src.data_foundation.ingest.failover import IngestFailoverDecision
from src.data_foundation.ingest.health import (
    IngestHealthStatus,
    evaluate_ingest_health,
)
from src.data_foundation.ingest.metrics import summarise_ingest_metrics
from src.data_foundation.ingest.quality import evaluate_ingest_quality
from src.data_foundation.ingest.recovery import IngestRecoveryRecommendation
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.data_foundation.pipelines.operational_backbone import (
    OperationalBackbonePipeline,
    OperationalIngestRequest,
)
from src.data_foundation.streaming.in_memory_broker import InMemoryKafkaBroker
from src.data_foundation.streaming.kafka_stream import (
    KafkaConnectionSettings,
    KafkaIngestEventConsumer,
    KafkaIngestEventPublisher,
)
from src.data_integration.real_data_integration import RealDataManager
from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.operations.backup import BackupReadinessSnapshot, BackupStatus
from src.operations.data_backbone import (
    BackboneRuntimeContext,
    BackboneStatus,
    evaluate_data_backbone_readiness,
)
from src.operations.retention import (
    DataRetentionSnapshot,
    RetentionComponentSnapshot,
    RetentionStatus,
)
from src.data_foundation.batch.spark_export import (
    SparkExportFormat,
    SparkExportJobResult,
    SparkExportSnapshot,
    SparkExportStatus,
)
from src.operations.spark_stress import (
    SparkStressCycleResult,
    SparkStressSnapshot,
    SparkStressStatus,
)


try:  # pragma: no cover - optional dependency mirrors production Redis wiring
    import fakeredis
except Exception:  # pragma: no cover
    fakeredis = None  # type: ignore[assignment]


def _managed_cache(policy: RedisCachePolicy | None = None) -> ManagedRedisCache:
    policy = policy or RedisCachePolicy.institutional_defaults()
    if fakeredis is not None:
        client = fakeredis.FakeRedis()
    else:
        client = InMemoryRedis()
    return ManagedRedisCache(client, policy)


def _flush_cache(cache: ManagedRedisCache) -> None:
    flush = getattr(cache.raw_client, "flushall", None)
    if callable(flush):  # pragma: no branch - optional dependency cleanup
        flush()


def _daily_frame(base: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": base,
                "symbol": "EURUSD",
                "open": 1.10,
                "high": 1.13,
                "low": 1.09,
                "close": 1.125,
                "adj_close": 1.12,
                "volume": 1500,
            },
            {
                "date": base - pd.Timedelta(days=1),
                "symbol": "EURUSD",
                "open": 1.11,
                "high": 1.14,
                "low": 1.10,
                "close": 1.13,
                "adj_close": 1.13,
                "volume": 1400,
            },
        ]
    )


def _intraday_frame(base: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": base - pd.Timedelta(minutes=2),
                "symbol": "EURUSD",
                "price": 1.123,
                "size": 640,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
            {
                "timestamp": base - pd.Timedelta(minutes=1),
                "symbol": "EURUSD",
                "price": 1.126,
                "size": 720,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
        ]
    )


@pytest.mark.asyncio()
async def test_evaluate_data_backbone_readiness_combines_signals(tmp_path) -> None:
    settings = TimescaleConnectionSettings(url=f"sqlite:///{tmp_path / 'readiness.db'}")
    broker = InMemoryKafkaBroker()

    cache = _managed_cache(RedisCachePolicy.institutional_defaults())
    publisher = KafkaIngestEventPublisher(
        broker.create_producer(),
        topic_map={"daily_bars": "telemetry.ingest", "intraday_trades": "telemetry.ingest"},
    )
    manager = RealDataManager(
        timescale_settings=settings,
        ingest_publisher=publisher,
        managed_cache=cache,
    )

    event_bus = EventBus()

    def consumer_factory() -> KafkaIngestEventConsumer:
        consumer = broker.create_consumer()
        return KafkaIngestEventConsumer(
            consumer,
            topics=("telemetry.ingest",),
            event_bus=event_bus,
            poll_timeout=0.05,
            publish_consumer_lag=False,
        )

    pipeline = OperationalBackbonePipeline(
        manager=manager,
        event_bus=event_bus,
        kafka_consumer_factory=consumer_factory,
        sensory_organ=RealSensoryOrgan(),
        event_topics=("telemetry.ingest",),
    )

    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=("EURUSD",), lookback_days=2),
        intraday=IntradayTradeIngestPlan(symbols=("EURUSD",), lookback_days=1, interval="1d"),
    )

    base = datetime.now(tz=UTC)
    request = OperationalIngestRequest(
        symbols=("eurusd",),
        daily_lookback_days=2,
        intraday_lookback_days=1,
        intraday_interval="1d",
    )

    try:
        result = await pipeline.execute(
            request,
            fetch_daily=lambda symbols, lookback: _daily_frame(base),
            fetch_intraday=lambda symbols, lookback, interval: _intraday_frame(base),
        )
    finally:
        await pipeline.shutdown()
        _flush_cache(cache)

    ingest_results = result.ingest_results
    health_report = evaluate_ingest_health(ingest_results, plan=plan)
    quality_report = evaluate_ingest_quality(ingest_results, plan=plan)
    metrics_snapshot = summarise_ingest_metrics(ingest_results)

    generated = datetime.now(tz=UTC)
    failover_decision = IngestFailoverDecision(
        should_failover=False,
        status=IngestHealthStatus.ok,
        reason=None,
        generated_at=generated,
        triggered_dimensions=tuple(),
        optional_triggers=tuple(),
        planned_dimensions=tuple(ingest_results.keys()),
        metadata={},
    )
    recovery_recommendation = IngestRecoveryRecommendation(plan=TimescaleBackbonePlan())
    backup_snapshot = BackupReadinessSnapshot(
        service="timescale",
        generated_at=generated,
        status=BackupStatus.ok,
        latest_backup_at=generated,
        next_backup_due_at=generated,
        retention_days=7,
        issues=tuple(),
    )
    retention_snapshot = DataRetentionSnapshot(
        status=RetentionStatus.ok,
        generated_at=generated,
        components=(
            RetentionComponentSnapshot(
                name="market_data.daily_bars",
                status=RetentionStatus.ok,
                summary="retention window 60d",
                metadata={"observed_days": 60},
            ),
        ),
        metadata={"policies": 1},
    )

    spark_snapshot = SparkExportSnapshot(
        generated_at=generated,
        status=SparkExportStatus.ok,
        format=SparkExportFormat.csv,
        root_path="/tmp/spark",
        jobs=(
            SparkExportJobResult(
                dimension="daily_bars",
                status=SparkExportStatus.ok,
                rows=ingest_results["daily_bars"].rows_written,
                paths=("daily.parquet",),
                issues=tuple(),
                metadata={"symbols": ingest_results["daily_bars"].symbols},
            ),
        ),
        metadata={"publish_telemetry": True},
    )

    spark_stress_snapshot = SparkStressSnapshot(
        label="resilience",
        status=SparkStressStatus.ok,
        generated_at=generated,
        cycles=(
            SparkStressCycleResult(
                cycle=1,
                status=SparkStressStatus.ok,
                export_status=SparkExportStatus.ok,
                duration_seconds=0.4,
                issues=tuple(),
                metadata={"job_count": 1},
            ),
        ),
        metadata={"cycles": 1},
    )

    connectivity = manager.connectivity_report()
    context = BackboneRuntimeContext(
        redis_expected=False,
        redis_configured=connectivity.redis,
        redis_namespace=cache.policy.namespace,
        redis_backing=type(cache.raw_client).__name__,
        kafka_expected=True,
        kafka_configured=connectivity.kafka,
        kafka_topics=("telemetry.ingest",),
        kafka_publishers=("events",),
        scheduler_enabled=False,
        scheduler_state=None,
    )

    ingest_config = InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=plan,
        timescale_settings=settings,
        kafka_settings=KafkaConnectionSettings(bootstrap_servers="inmemory"),
        redis_settings=RedisConnectionSettings(),
        metadata={
            "plan": {
                "daily_bars": {"symbols": ["EURUSD"], "lookback_days": 2},
                "intraday_trades": {"symbols": ["EURUSD"], "lookback_days": 1, "interval": "1d"},
            },
            "kafka_configured": connectivity.kafka,
            "kafka_topics": ["telemetry.ingest"],
        },
    )

    task_snapshots = (
        {
            "name": "ingest.kafka.bridge",
            "state": "running",
            "created_at": generated.isoformat(),
            "metadata": {"namespace": "tests"},
        },
    )

    snapshot = evaluate_data_backbone_readiness(
        ingest_config=ingest_config,
        health_report=health_report,
        quality_report=quality_report,
        metrics_snapshot=metrics_snapshot,
        failover_decision=failover_decision,
        recovery_recommendation=recovery_recommendation,
        backup_snapshot=backup_snapshot,
        retention_snapshot=retention_snapshot,
        context=context,
        metadata={"test": True},
        spark_snapshot=spark_snapshot,
        spark_stress_snapshot=spark_stress_snapshot,
        task_snapshots=task_snapshots,
    )

    assert snapshot.status is BackboneStatus.ok
    components = {component.name: component for component in snapshot.components}
    assert components["plan"].status is BackboneStatus.ok
    assert components["ingest_health"].status is BackboneStatus.ok
    kafka_metadata = components["kafka_streaming"].metadata
    assert kafka_metadata["topics"] == ["telemetry.ingest"]
    assert "retention" in components
    retention_component = components["retention"]
    assert retention_component.status is BackboneStatus.ok
    assert retention_component.metadata["status"] == RetentionStatus.ok.value
    if "redis_cache" in components:
        assert components["redis_cache"].metadata.get("namespace") == cache.policy.namespace
    else:
        assert connectivity.redis is False

    await manager.shutdown()
    cache.metrics(reset=True)
