from datetime import UTC, datetime

from src.data_foundation.batch.spark_export import (
    SparkExportFormat,
    SparkExportJobResult,
    SparkExportSnapshot,
    SparkExportStatus,
)
from src.data_foundation.cache.redis_cache import RedisConnectionSettings
from src.data_foundation.ingest.configuration import InstitutionalIngestConfig
from src.data_foundation.ingest.failover import IngestFailoverDecision
from src.data_foundation.ingest.health import (
    IngestHealthCheck,
    IngestHealthReport,
    IngestHealthStatus,
)
from src.data_foundation.ingest.metrics import IngestDimensionMetrics, IngestMetricsSnapshot
from src.data_foundation.ingest.quality import (
    IngestQualityCheck,
    IngestQualityReport,
    IngestQualityStatus,
)
from src.data_foundation.ingest.recovery import IngestRecoveryRecommendation
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.ingest.scheduler import IngestSchedulerState
from src.data_foundation.ingest.scheduler_telemetry import build_scheduler_snapshot
from src.data_foundation.persist.timescale import TimescaleConnectionSettings
from src.operations.backup import BackupReadinessSnapshot, BackupStatus
from src.operations.data_backbone import (
    BackboneRuntimeContext,
    BackboneStatus,
    DataBackboneValidationSnapshot,
    evaluate_data_backbone_readiness,
    evaluate_data_backbone_validation,
)
from src.operations.spark_stress import (
    SparkStressCycleResult,
    SparkStressSnapshot,
    SparkStressStatus,
)


def _sample_ingest_config() -> InstitutionalIngestConfig:
    plan = TimescaleBackbonePlan(daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=5))
    metadata = {
        "plan": {"daily_bars": {"symbols": ["EURUSD"], "lookback_days": 5}},
        "kafka_configured": True,
        "kafka_topics": ["telemetry.ingest.events"],
    }
    return InstitutionalIngestConfig(
        should_run=True,
        reason=None,
        plan=plan,
        metadata=metadata,
        redis_settings=RedisConnectionSettings(),
    )


def test_evaluate_data_backbone_readiness_combines_signals() -> None:
    config = _sample_ingest_config()
    generated = datetime(2025, 1, 1, tzinfo=UTC)
    health_report = IngestHealthReport(
        status=IngestHealthStatus.ok,
        generated_at=generated,
        checks=(
            IngestHealthCheck(
                dimension="daily_bars",
                status=IngestHealthStatus.ok,
                message="healthy",
                rows_written=10,
                freshness_seconds=30.0,
            ),
        ),
    )
    quality_report = IngestQualityReport(
        status=IngestQualityStatus.ok,
        score=0.98,
        generated_at=generated,
        checks=(
            IngestQualityCheck(
                dimension="daily_bars",
                status=IngestQualityStatus.ok,
                score=0.98,
                observed_rows=10,
            ),
        ),
    )
    metrics_snapshot = IngestMetricsSnapshot(
        generated_at=generated,
        dimensions=(
            IngestDimensionMetrics(
                dimension="daily_bars",
                rows=10,
                symbols=("EURUSD",),
                ingest_duration_seconds=1.2,
                freshness_seconds=42.0,
            ),
        ),
    )
    failover_decision = IngestFailoverDecision(
        should_failover=False,
        status=IngestHealthStatus.ok,
        reason=None,
        generated_at=generated,
        triggered_dimensions=tuple(),
        optional_triggers=tuple(),
        planned_dimensions=("daily_bars",),
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
    scheduler_state = IngestSchedulerState(
        running=False,
        last_started_at=generated,
        last_completed_at=generated,
        consecutive_failures=1,
        next_run_at=None,
        interval_seconds=60.0,
        jitter_seconds=5.0,
        max_failures=3,
    )
    context = BackboneRuntimeContext(
        redis_expected=True,
        redis_configured=False,
        redis_backing="in-memory",
        kafka_expected=True,
        kafka_configured=True,
        kafka_topics=("telemetry.ingest.events",),
        kafka_publishers=("events", "metrics"),
        scheduler_enabled=True,
        scheduler_state=scheduler_state,
    )

    spark_snapshot = SparkExportSnapshot(
        generated_at=generated,
        status=SparkExportStatus.warn,
        format=SparkExportFormat.csv,
        root_path="/tmp/spark",
        jobs=(
            SparkExportJobResult(
                dimension="daily_bars",
                status=SparkExportStatus.warn,
                rows=0,
                paths=("daily.csv",),
                issues=("no_rows_returned",),
                metadata={"symbols": ["EURUSD"]},
            ),
        ),
        metadata={"publish_telemetry": True},
    )

    spark_stress_snapshot = SparkStressSnapshot(
        label="resilience",
        status=SparkStressStatus.warn,
        generated_at=generated,
        cycles=(
            SparkStressCycleResult(
                cycle=1,
                status=SparkStressStatus.warn,
                export_status=SparkExportStatus.warn,
                duration_seconds=0.8,
                issues=("duration_exceeded_warn_threshold",),
                metadata={"job_count": 1},
            ),
        ),
        metadata={"cycles": 1},
    )

    snapshot = evaluate_data_backbone_readiness(
        ingest_config=config,
        health_report=health_report,
        quality_report=quality_report,
        metrics_snapshot=metrics_snapshot,
        failover_decision=failover_decision,
        recovery_recommendation=recovery_recommendation,
        backup_snapshot=backup_snapshot,
        context=context,
        metadata={"test": True},
        spark_snapshot=spark_snapshot,
        spark_stress_snapshot=spark_stress_snapshot,
    )

    assert snapshot.status is BackboneStatus.warn
    component_names = {component.name: component for component in snapshot.components}
    assert component_names["plan"].status is BackboneStatus.ok
    assert component_names["ingest_health"].status is BackboneStatus.ok
    assert component_names["redis_cache"].status is BackboneStatus.warn
    assert component_names["kafka_streaming"].status is BackboneStatus.ok
    assert "scheduler" in component_names
    scheduler_component = component_names["scheduler"]
    assert scheduler_component.metadata.get("enabled") is True
    assert scheduler_component.metadata.get("running") is False
    assert scheduler_component.metadata.get("interval_seconds") == 60.0
    assert "spark_exports" in component_names
    spark_component = component_names["spark_exports"]
    assert spark_component.status is BackboneStatus.warn
    assert spark_component.metadata.get("format") == "csv"
    assert spark_component.metadata.get("job_count") == 1
    assert "spark_stress" in component_names
    stress_component = component_names["spark_stress"]
    assert stress_component.status is BackboneStatus.warn
    assert stress_component.metadata.get("cycle_count") == 1
    markdown = snapshot.to_markdown()
    assert "DATA_BACKBONE" not in markdown  # ensure formatting uses table header
    assert "redis_cache" in markdown


def test_evaluate_data_backbone_validation_flags_missing_services() -> None:
    config = _sample_ingest_config()
    context = BackboneRuntimeContext(
        redis_expected=True,
        redis_configured=False,
        kafka_expected=True,
        kafka_configured=False,
        kafka_topics=tuple(),
        kafka_publishers=tuple(),
        scheduler_enabled=True,
    )
    scheduler_snapshot = build_scheduler_snapshot(
        enabled=True,
        schedule=config.schedule,
        state=None,
    )

    snapshot = evaluate_data_backbone_validation(
        ingest_config=config,
        context=context,
        scheduler_snapshot=scheduler_snapshot,
    )

    assert isinstance(snapshot, DataBackboneValidationSnapshot)
    assert snapshot.status is BackboneStatus.fail
    check_map = {check.name: check for check in snapshot.checks}
    assert check_map["plan"].status is BackboneStatus.ok
    assert check_map["timescale_connection"].status is BackboneStatus.fail
    assert check_map["redis"].status is BackboneStatus.fail
    assert check_map["kafka"].status is BackboneStatus.fail
    assert check_map["scheduler"].status is BackboneStatus.fail


def test_evaluate_data_backbone_validation_passes_when_configured() -> None:
    config = _sample_ingest_config()
    config = InstitutionalIngestConfig(
        should_run=config.should_run,
        reason=config.reason,
        plan=config.plan,
        timescale_settings=TimescaleConnectionSettings(
            url="postgresql://example", application_name="test"
        ),
        kafka_settings=config.kafka_settings,
        redis_settings=config.redis_settings,
        metadata=config.metadata,
        schedule=config.schedule,
        recovery=config.recovery,
        operational_alert_routes=config.operational_alert_routes,
        backup=config.backup,
    )
    context = BackboneRuntimeContext(
        redis_expected=True,
        redis_configured=True,
        redis_namespace="emp:cache",
        redis_backing="ManagedRedisCache",
        kafka_expected=True,
        kafka_configured=True,
        kafka_topics=("telemetry.ingest.events",),
        kafka_publishers=("events", "metrics"),
        scheduler_enabled=False,
    )

    snapshot = evaluate_data_backbone_validation(
        ingest_config=config,
        context=context,
        scheduler_snapshot=None,
    )

    assert snapshot.status is BackboneStatus.ok
    check_map = {check.name: check for check in snapshot.checks}
    assert check_map["timescale_connection"].status is BackboneStatus.ok
    assert check_map["redis"].status is BackboneStatus.ok
    assert check_map["kafka"].status is BackboneStatus.ok
    assert "scheduler" not in check_map


def test_data_backbone_readiness_surfaces_failover_and_recovery() -> None:
    config = _sample_ingest_config()
    generated = datetime(2025, 2, 1, 12, 30, tzinfo=UTC)

    failover_decision = IngestFailoverDecision(
        should_failover=False,
        status=IngestHealthStatus.warn,
        reason=None,
        generated_at=generated,
        triggered_dimensions=tuple(),
        optional_triggers=("macro_events",),
        planned_dimensions=("daily_bars", "macro_events"),
        metadata={"latency_seconds": 420.0},
    )

    recovery_plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD"], lookback_days=3)
    )
    recovery_recommendation = IngestRecoveryRecommendation(
        plan=recovery_plan,
        reasons={"daily_bars": "freshness degraded"},
        missing_symbols={"daily_bars": ("EURUSD",)},
    )

    snapshot = evaluate_data_backbone_readiness(
        ingest_config=config,
        failover_decision=failover_decision,
        recovery_recommendation=recovery_recommendation,
    )

    components = {component.name: component for component in snapshot.components}
    failover_component = components["failover"]
    assert failover_component.status is BackboneStatus.warn
    assert failover_component.summary == "optional slices degraded"
    assert failover_component.metadata["optional_triggers"] == ["macro_events"]
    assert failover_component.metadata["triggered"] == []

    recovery_component = components["recovery"]
    assert recovery_component.status is BackboneStatus.warn
    assert "plan" in recovery_component.metadata
    plan_metadata = recovery_component.metadata["plan"]
    assert plan_metadata["daily_bars"]["lookback_days"] == 3
    assert plan_metadata["daily_bars"]["symbols"] == ["EURUSD"]
