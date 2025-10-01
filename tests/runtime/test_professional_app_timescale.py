from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data_foundation.batch.spark_export import (
    SparkExportFormat,
    SparkExportSnapshot,
    SparkExportStatus,
)

from src.compliance.workflow import evaluate_compliance_workflows
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestJournal,
    TimescaleIngestRunRecord,
    TimescaleIngestor,
    TimescaleIngestResult,
    TimescaleMigrator,
)
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackbonePlan,
)
from src.data_foundation.ingest.scheduler_telemetry import (
    IngestSchedulerSnapshot,
    IngestSchedulerStatus,
)
from src.governance.system_config import (
    ConnectionProtocol,
    DataBackboneMode,
    EmpTier,
    SystemConfig,
)
from src.operations.backup import BackupReadinessSnapshot, BackupStatus
from src.operations.data_backbone import (
    BackboneComponentSnapshot,
    BackboneStatus,
    DataBackboneReadinessSnapshot,
    DataBackboneValidationSnapshot,
)
from src.operations.retention import (
    DataRetentionSnapshot,
    RetentionComponentSnapshot,
    RetentionStatus,
)
from src.operations.system_validation import (
    SystemValidationCheck,
    SystemValidationSnapshot,
    SystemValidationStatus,
    evaluate_system_validation,
)
from src.operations.professional_readiness import (
    ProfessionalReadinessComponent,
    ProfessionalReadinessSnapshot,
    ProfessionalReadinessStatus,
)
from src.operations.security import (
    SecurityControlEvaluation,
    SecurityPostureSnapshot,
    SecurityStatus,
)
from src.operations.cache_health import evaluate_cache_health
from src.operations.event_bus_health import evaluate_event_bus_health
from src.operations.failover_drill import execute_failover_drill
from src.operations.cross_region_failover import (
    CrossRegionComponent,
    CrossRegionFailoverSnapshot,
    CrossRegionStatus,
)
from src.operations.kafka_readiness import (
    KafkaReadinessComponent,
    KafkaReadinessSnapshot,
    KafkaReadinessStatus,
)
from src.operations.execution import ExecutionPolicy, ExecutionState, evaluate_execution_readiness
from src.operations.sensory_drift import (
    DriftSeverity,
    SensoryDimensionDrift,
    SensoryDriftSnapshot,
)
from src.operations.spark_stress import (
    SparkStressCycleResult,
    SparkStressSnapshot,
    SparkStressStatus,
)
from src.operations.evolution_experiments import (
    ExperimentMetrics,
    ExperimentStatus,
    EvolutionExperimentSnapshot,
)
from src.operations.evolution_tuning import (
    EvolutionTuningRecommendation,
    EvolutionTuningSnapshot,
    EvolutionTuningStatus,
    EvolutionTuningSummary,
)
from src.operations.strategy_performance import evaluate_strategy_performance
from src.operations.incident_response import (
    IncidentResponsePolicy,
    IncidentResponseState,
    evaluate_incident_response,
)
from src.operations.roi import RoiStatus, RoiTelemetrySnapshot
from src.operations.ingest_trends import (
    IngestDimensionTrend,
    IngestTrendSnapshot,
    IngestTrendStatus,
)
from src.runtime import (
    BootstrapRuntime,
    build_professional_predator_app,
    build_professional_runtime_application,
)


@pytest.mark.asyncio()
async def test_professional_app_wires_timescale_connectors(tmp_path) -> None:
    db_path = tmp_path / "runtime_timescale.db"
    url = f"sqlite:///{db_path}"
    settings = TimescaleConnectionSettings(url=url)
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()

    ingestor = TimescaleIngestor(engine)
    daily_frame = pd.DataFrame(
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
    ingestor.upsert_daily_bars(daily_frame)

    trade_frame = pd.DataFrame(
        [
            {
                "timestamp": datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
                "symbol": "EURUSD",
                "price": 1.021,
                "size": 750,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
            {
                "timestamp": datetime(2024, 1, 2, 12, 1, tzinfo=timezone.utc),
                "symbol": "EURUSD",
                "price": 1.024,
                "size": 820,
                "exchange": "TEST",
                "conditions": "SYNTH",
            },
        ]
    )
    ingestor.upsert_intraday_trades(trade_frame)
    engine.dispose()

    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": url,
            "BOOTSTRAP_SYMBOLS": "EURUSD",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    assert isinstance(app.sensory_organ, BootstrapRuntime)

    fabric_connectors = app.sensory_organ.fabric.connectors
    assert "timescale_intraday" in fabric_connectors
    assert "timescale_daily" in fabric_connectors
    assert "historical_replay" in fabric_connectors

    market_data = await app.sensory_organ.fabric.fetch_latest(
        "EURUSD", allow_stale=False, use_cache=False
    )
    assert pytest.approx(market_data.close, rel=1e-6) == 1.024
    assert pytest.approx(market_data.volume, rel=1e-6) == 820

    await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_tracks_event_bus_worker() -> None:
    cfg = SystemConfig()

    app = await build_professional_predator_app(config=cfg)
    try:
        await app.event_bus.start()
        await asyncio.sleep(0)
        details = app.task_supervisor.describe()
        assert any(
            isinstance(entry, dict) and entry.get("metadata", {}).get("task") == "worker"
            for entry in details
        )
    finally:
        await app.event_bus.stop()
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_registers_kafka_consumer_bridge(monkeypatch, tmp_path) -> None:
    class _StubConsumer:
        def __init__(self) -> None:
            self.stop_event: asyncio.Event | None = None
            self.closed = False
            self.summary_calls: int = 0

        async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
            self.stop_event = stop_event
            if stop_event is None:
                await asyncio.sleep(0.01)
            else:
                await stop_event.wait()

        def close(self) -> None:
            self.closed = True

        def summary(self) -> str:
            self.summary_calls += 1
            return "stub-consumer"

    stub = _StubConsumer()

    def _fake_create_consumer(*args, **kwargs):
        return stub

    monkeypatch.setattr(
        "src.runtime.predator_app.create_ingest_event_consumer",
        _fake_create_consumer,
    )

    db_path = tmp_path / "bridge.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
            "KAFKA_BROKERS": "localhost:9092",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        assert any(task.get_name() == "kafka-ingest-bridge" for task in app.active_background_tasks)
        await asyncio.sleep(0)  # allow background task to start
        summary = app.summary()
        details = summary.get("background_task_details")
        assert details and any(item["name"] == "kafka-ingest-bridge" for item in details)
    finally:
        await app.shutdown()

    assert stub.closed is True
    assert stub.stop_event is not None
    assert stub.stop_event.is_set()


@pytest.mark.asyncio()
async def test_professional_runtime_summary_includes_configuration_audit(tmp_path) -> None:
    db_path = tmp_path / "config-audit.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        tier=EmpTier.tier_1,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
            "KAFKA_BROKERS": "broker:9092",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        build_professional_runtime_application(
            app,
            skip_ingest=True,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )

        summary = app.summary()
        assert "configuration_audit" in summary
        audit_block = summary["configuration_audit"]
        assert (
            audit_block["snapshot"]["current_config"]["data_backbone_mode"]
            == DataBackboneMode.institutional.value
        )
        assert "KAFKA_BROKERS" in audit_block["markdown"]
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_strategy_performance(tmp_path) -> None:
    db_path = tmp_path / "strategy_performance_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        events = [
            {
                "strategy_id": "alpha",
                "status": "executed",
                "confidence": 0.82,
                "notional": 12_000.0,
                "timestamp": datetime(2024, 1, 4, tzinfo=UTC).isoformat(),
            },
            {
                "strategy_id": "alpha",
                "status": "rejected",
                "metadata": {"reason": "risk_limit"},
                "timestamp": datetime(2024, 1, 4, 0, 5, tzinfo=UTC).isoformat(),
            },
        ]
        roi_snapshot = RoiTelemetrySnapshot(
            status=RoiStatus.tracking,
            generated_at=datetime(2024, 1, 4, tzinfo=UTC),
            initial_capital=100_000.0,
            current_equity=101_500.0,
            gross_pnl=2_000.0,
            net_pnl=1_500.0,
            infrastructure_cost=300.0,
            fees=200.0,
            days_active=12.0,
            executed_trades=8,
            total_notional=95_000.0,
            roi=0.015,
            annualised_roi=0.45,
            gross_roi=0.02,
            gross_annualised_roi=0.52,
            breakeven_daily_return=0.001,
            target_annual_roi=0.4,
        )
        snapshot = evaluate_strategy_performance(events, roi_snapshot=roi_snapshot)
        app.record_strategy_performance_snapshot(snapshot)

        summary = app.summary()
        section = summary.get("strategy_performance")
        assert section is not None
        assert section["snapshot"]["totals"]["total_events"] == 2
        assert section["snapshot"]["totals"]["roi_status"] == RoiStatus.tracking.value
        assert "Strategy alpha" in section["markdown"]
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_ingest_journal(tmp_path) -> None:
    db_path = tmp_path / "journal_summary.db"
    url = f"sqlite:///{db_path}"
    settings = TimescaleConnectionSettings(url=url)
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()

    journal = TimescaleIngestJournal(engine)
    journal.record(
        [
            TimescaleIngestRunRecord(
                run_id="run-1",
                dimension="daily_bars",
                status="ok",
                rows_written=3,
                freshness_seconds=45.0,
                ingest_duration_seconds=1.5,
                executed_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
                source="yahoo",
                symbols=("EURUSD",),
                metadata={"overall_status": "ok"},
            )
        ]
    )
    engine.dispose()

    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": url,
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        summary = app.summary()
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_scheduler_snapshot(tmp_path) -> None:
    db_path = tmp_path / "scheduler_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        snapshot = IngestSchedulerSnapshot(
            status=IngestSchedulerStatus.ok,
            generated_at=now,
            enabled=True,
            running=True,
            consecutive_failures=0,
            interval_seconds=60.0,
            jitter_seconds=5.0,
            max_failures=3,
            last_started_at=now - timedelta(seconds=5),
            last_completed_at=now - timedelta(seconds=4),
            last_success_at=now - timedelta(seconds=4),
            next_run_at=now + timedelta(seconds=60),
        )
        app.record_scheduler_snapshot(snapshot)
        summary = app.summary()
        scheduler_section = summary.get("ingest_scheduler")
        assert scheduler_section is not None
        assert scheduler_section["snapshot"]["status"] == "ok"
        assert "markdown" in scheduler_section
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_ingest_trends(tmp_path) -> None:
    db_path = tmp_path / "trend_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        dimension_trend = IngestDimensionTrend(
            dimension="daily_bars",
            status=IngestTrendStatus.ok,
            statuses=(IngestTrendStatus.ok,),
            rows_written=(120,),
            freshness_seconds=(45.0,),
            issues=(),
            metadata={"recent_runs": []},
        )
        snapshot = IngestTrendSnapshot(
            generated_at=datetime(2024, 1, 2, tzinfo=UTC),
            status=IngestTrendStatus.ok,
            lookback=5,
            dimensions=(dimension_trend,),
            issues=(),
            metadata={"example": True},
        )
        app.record_ingest_trend_snapshot(snapshot)

        summary = app.summary()
        trend_section = summary.get("ingest_trends")
        assert trend_section is not None
        assert trend_section["snapshot"]["status"] == "ok"
        assert "markdown" in trend_section
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_kyc_monitor(tmp_path) -> None:
    db_path = tmp_path / "kyc_runtime.db"
    url = f"sqlite:///{db_path}"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": url,
            "KYC_MONITOR_ENABLED": "true",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        assert app.kyc_monitor is not None
        app.kyc_monitor.evaluate_case(  # type: ignore[union-attr]
            {
                "case_id": "case-77",
                "entity_id": "client-kyc",
                "entity_type": "individual",
                "risk_score": 55.0,
                "checklist": [{"item_id": "id", "name": "ID", "status": "COMPLETE"}],
                "last_reviewed_at": datetime(2025, 2, 1, tzinfo=UTC),
                "review_frequency_days": 180,
            }
        )
        summary = app.summary()
    finally:
        await app.shutdown()

    kyc_summary = summary.get("kyc")
    assert kyc_summary is not None
    assert kyc_summary["last_snapshot"]["case_id"] == "case-77"


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_backup_section(tmp_path) -> None:
    db_path = tmp_path / "backup_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
        },
    )

    snapshot = BackupReadinessSnapshot(
        service="timescale",
        generated_at=datetime(2024, 1, 5, tzinfo=UTC),
        status=BackupStatus.warn,
        latest_backup_at=datetime(2024, 1, 4, tzinfo=UTC),
        next_backup_due_at=datetime(2024, 1, 5, 12, 0, tzinfo=UTC),
        retention_days=14,
        issues=("restore drill overdue",),
        metadata={"policy": {"providers": ["s3"]}},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        app.record_backup_snapshot(snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    backup_summary = summary.get("backups")
    assert backup_summary is not None
    assert backup_summary["snapshot"]["status"] == "warn"
    assert "restore drill overdue" in backup_summary["markdown"]


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_security_section(tmp_path) -> None:
    db_path = tmp_path / "security_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        control = SecurityControlEvaluation(
            control="mfa",
            status=SecurityStatus.passed,
            summary="coverage within target",
        )
        snapshot = SecurityPostureSnapshot(
            service="emp_platform",
            generated_at=datetime(2025, 1, 6, tzinfo=UTC),
            status=SecurityStatus.passed,
            controls=(control,),
            metadata={"controls_evaluated": 1},
        )
        app.record_security_snapshot(snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    security = summary.get("security")
    assert security is not None
    assert security["snapshot"]["service"] == "emp_platform"
    assert "mfa" in security["markdown"].lower()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_incident_response(tmp_path) -> None:
    db_path = tmp_path / "incident_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        policy = IncidentResponsePolicy(
            required_runbooks=("redis_outage",),
            training_interval_days=30,
            drill_interval_days=45,
            minimum_primary_responders=1,
            minimum_secondary_responders=1,
            postmortem_sla_hours=24.0,
            maximum_open_incidents=2,
            require_chatops=True,
        )
        state = IncidentResponseState(
            available_runbooks=("redis_outage", "kafka_lag"),
            training_age_days=10.0,
            drill_age_days=20.0,
            primary_oncall=("alice",),
            secondary_oncall=("bob",),
            open_incidents=tuple(),
            postmortem_backlog_hours=6.0,
            chatops_ready=True,
        )
        snapshot = evaluate_incident_response(
            policy,
            state,
            service="emp_incidents",
            now=datetime(2025, 1, 7, tzinfo=UTC),
        )
        app.record_incident_response_snapshot(snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    incident_section = summary.get("incident_response")
    assert incident_section is not None
    assert incident_section["snapshot"]["status"] == "ok"
    markdown = incident_section.get("markdown", "").lower()
    assert "incident response" in markdown
    assert "primary responders" in markdown


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_cache_section(tmp_path) -> None:
    db_path = tmp_path / "cache_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        snapshot = evaluate_cache_health(
            configured=True,
            expected=True,
            namespace="emp:cache",
            backing="RedisClient",
            metrics={"hits": 40, "misses": 5},
            policy={"ttl_seconds": 900, "max_keys": 256},
        )
        app.record_cache_snapshot(snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    cache = summary.get("cache")
    assert cache is not None
    assert cache["snapshot"]["namespace"] == "emp:cache"
    assert "ttl seconds" in cache["markdown"].lower()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_event_bus_section(tmp_path) -> None:
    db_path = tmp_path / "event_bus_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        snapshot = evaluate_event_bus_health(app.event_bus, expected=False)
        app.record_event_bus_snapshot(snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    event_bus = summary.get("event_bus")
    assert event_bus is not None
    assert event_bus["snapshot"]["expected"] is False
    assert "event bus health" in event_bus["markdown"].lower()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_system_validation(tmp_path) -> None:
    db_path = tmp_path / "system_validation_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        snapshot = evaluate_system_validation(
            {
                "timestamp": datetime(2025, 1, 5, tzinfo=UTC).isoformat(),
                "total_checks": 2,
                "results": {"core": True, "integration": False},
                "summary": {"status": "PARTIAL", "message": "integration pending"},
            }
        )
        app.record_system_validation_snapshot(snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    block = summary.get("system_validation")
    assert block is not None
    assert block["snapshot"]["failed_checks"] == 1
    assert "integration pending" in block["markdown"].lower()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_operational_readiness(tmp_path) -> None:
    db_path = tmp_path / "operational_readiness.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        system_snapshot = SystemValidationSnapshot(
            status=SystemValidationStatus.warn,
            generated_at=datetime(2025, 1, 5, tzinfo=UTC),
            total_checks=3,
            passed_checks=2,
            failed_checks=1,
            success_rate=2.0 / 3.0,
            checks=(
                SystemValidationCheck(name="timescale", passed=False, message="lag detected"),
                SystemValidationCheck(name="redis", passed=True),
                SystemValidationCheck(name="kafka", passed=True),
            ),
            metadata={"validator": "ops_guardian"},
        )

        policy = IncidentResponsePolicy(
            required_runbooks=("timescale_outage",),
            training_interval_days=30,
            drill_interval_days=45,
            minimum_primary_responders=1,
            minimum_secondary_responders=1,
            postmortem_sla_hours=24.0,
            maximum_open_incidents=0,
            require_chatops=True,
        )
        state = IncidentResponseState(
            available_runbooks=tuple(),
            training_age_days=90.0,
            drill_age_days=50.0,
            primary_oncall=tuple(),
            secondary_oncall=tuple(),
            open_incidents=("INC-204",),
            postmortem_backlog_hours=48.0,
            chatops_ready=False,
        )

        incident_snapshot = evaluate_incident_response(
            policy,
            state,
            service="emp_incidents",
            now=datetime(2025, 1, 5, 12, tzinfo=UTC),
        )

        app.record_system_validation_snapshot(system_snapshot)
        app.record_incident_response_snapshot(incident_snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    readiness = summary.get("operational_readiness")
    assert readiness is not None
    assert readiness["snapshot"]["status"] == "fail"
    markdown = readiness.get("markdown", "").lower()
    assert "operational readiness" in markdown
    assert "incident_response" in markdown


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_execution_section(tmp_path) -> None:
    db_path = tmp_path / "execution_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        snapshot = evaluate_execution_readiness(
            ExecutionPolicy(),
            ExecutionState(
                orders_submitted=6,
                orders_executed=6,
                avg_latency_ms=110.0,
                max_latency_ms=210.0,
                connection_healthy=True,
                drop_copy_active=True,
            ),
            metadata={"window": "smoke"},
            service="paper-stack",
        )
        app.record_execution_snapshot(snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    execution = summary.get("execution")
    assert execution is not None
    assert execution["snapshot"]["service"] == "paper-stack"
    assert "execution" in execution["markdown"].lower()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_execution_journal(tmp_path) -> None:
    db_path = tmp_path / "execution_journal.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
            "EXECUTION_JOURNAL_ENABLED": "true",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        snapshot = evaluate_execution_readiness(
            ExecutionPolicy(),
            ExecutionState(
                orders_submitted=5,
                orders_executed=4,
                orders_failed=1,
                pending_orders=1,
                avg_latency_ms=90.0,
                max_latency_ms=210.0,
                connection_healthy=True,
                drop_copy_active=True,
            ),
            metadata={"window": "journal"},
            service="institutional",
        )
        app.record_execution_snapshot(snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    journal_block = summary.get("execution_journal")
    assert journal_block is not None
    assert journal_block["recent"][0]["service"] == "institutional"


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_data_backbone(tmp_path) -> None:
    db_path = tmp_path / "backbone_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        validation_snapshot = DataBackboneValidationSnapshot(
            status=BackboneStatus.warn,
            generated_at=datetime(2024, 1, 5, tzinfo=UTC),
            checks=(
                BackboneComponentSnapshot(
                    name="redis",
                    status=BackboneStatus.warn,
                    summary="fallback",
                ),
            ),
            metadata={"redis_namespace": "emp:cache"},
        )
        app.record_data_backbone_validation_snapshot(validation_snapshot)
        snapshot = DataBackboneReadinessSnapshot(
            status=BackboneStatus.ok,
            generated_at=datetime(2024, 1, 6, tzinfo=UTC),
            components=(
                BackboneComponentSnapshot(
                    name="plan",
                    status=BackboneStatus.ok,
                    summary="configured",
                ),
            ),
            metadata={},
        )
        app.record_data_backbone_snapshot(snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    backbone_summary = summary.get("data_backbone")
    assert backbone_summary is not None
    assert backbone_summary["snapshot"]["status"] == "ok"
    component_names = {
        component["name"] for component in backbone_summary["snapshot"]["components"]
    }
    assert "plan" in component_names
    assert "markdown" in backbone_summary

    validation_summary = summary.get("data_backbone_validation")
    assert validation_summary is not None
    assert validation_summary["snapshot"]["status"] == "warn"
    assert validation_summary["snapshot"]["checks"][0]["name"] == "redis"


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_data_retention(tmp_path) -> None:
    db_path = tmp_path / "retention_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        retention_snapshot = DataRetentionSnapshot(
            status=RetentionStatus.ok,
            generated_at=datetime(2024, 1, 6, tzinfo=UTC),
            components=(
                RetentionComponentSnapshot(
                    name="daily_bars",
                    status=RetentionStatus.ok,
                    summary="coverage 370d",
                ),
            ),
        )
        app.record_data_retention_snapshot(retention_snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    retention_summary = summary.get("data_retention")
    assert retention_summary is not None
    assert retention_summary["snapshot"]["status"] == "ok"
    assert "coverage" in retention_summary["markdown"]


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_professional_readiness(tmp_path) -> None:
    db_path = tmp_path / "professional_readiness.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        snapshot = ProfessionalReadinessSnapshot(
            status=ProfessionalReadinessStatus.warn,
            generated_at=datetime(2024, 1, 7, tzinfo=UTC),
            components=(
                ProfessionalReadinessComponent(
                    name="backups",
                    status=ProfessionalReadinessStatus.warn,
                    summary="restore drill overdue",
                ),
            ),
            metadata={"notes": "test"},
        )
        app.record_professional_readiness_snapshot(snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    readiness = summary.get("professional_readiness")
    assert readiness is not None
    assert readiness["snapshot"]["status"] == "warn"
    assert "restore drill overdue" in readiness["markdown"]


@pytest.mark.asyncio()
async def test_professional_app_compliance_monitor_surfaces_snapshot() -> None:
    cfg = SystemConfig()

    app = await build_professional_predator_app(config=cfg)
    try:
        monitor = app.compliance_monitor
        assert monitor is not None

        report = SimpleNamespace(
            event_id="evt-100",
            symbol="EURUSD",
            side="BUY",
            quantity=10_000,
            price=1.2,
            timestamp=datetime(2025, 3, 5, 12, 0, tzinfo=timezone.utc),
        )
        await monitor.on_execution_report(report)

        summary = app.summary()
        compliance = summary.get("compliance")
        assert compliance is not None
        assert compliance["last_snapshot"]["symbol"] == "EURUSD"
        assert compliance["policy"]["report_channel"].startswith("telemetry.compliance")
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_records_compliance_workflow_snapshot() -> None:
    cfg = SystemConfig()

    app = await build_professional_predator_app(config=cfg)
    try:
        workflow_snapshot = evaluate_compliance_workflows(
            trade_summary={
                "policy": {"policy_name": "unit"},
                "last_snapshot": {"status": "pass", "checks": []},
                "history": [],
                "daily_totals": {"EURUSD": {"notional": 1000.0, "trades": 1}},
                "journal": {"last_entry": {"trade_id": "evt"}},
            },
            kyc_summary={
                "last_snapshot": {
                    "status": "APPROVED",
                    "risk_rating": "LOW",
                    "outstanding_items": [],
                    "watchlist_hits": [],
                    "alerts": [],
                },
                "recent": [],
                "open_cases": 0,
                "escalations": 0,
                "journal": {"last_entry": {"case_id": "case"}},
            },
        )
        app.record_compliance_workflow_snapshot(workflow_snapshot)
        summary = app.summary()
    finally:
        await app.shutdown()

    workflows = summary.get("compliance_workflows")
    assert workflows is not None
    assert workflows["snapshot"]["status"] == workflow_snapshot.status.value
    assert "Compliance Workflows" in workflows.get("markdown", "")


@pytest.mark.asyncio()
async def test_professional_app_enables_compliance_journal(tmp_path) -> None:
    db_path = tmp_path / "runtime_compliance.db"
    extras = {
        "TIMESCALEDB_URL": f"sqlite:///{db_path}",
        "COMPLIANCE_JOURNAL_ENABLED": "true",
        "BOOTSTRAP_SYMBOLS": "EURUSD",
    }

    settings = TimescaleConnectionSettings.from_mapping(extras)
    engine = settings.create_engine()
    TimescaleMigrator(engine).apply()
    engine.dispose()

    cfg = SystemConfig(
        data_backbone_mode=DataBackboneMode.institutional,
        extras=extras,
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        monitor = app.compliance_monitor
        assert monitor is not None

        report = SimpleNamespace(
            event_id="evt-journal-runtime",
            symbol="EURUSD",
            side="BUY",
            quantity=5_000,
            price=1.1,
            timestamp=datetime(2025, 4, 15, 9, 45, tzinfo=timezone.utc),
            status="FILLED",
        )

        await monitor.on_execution_report(report)

        summary = app.summary()
        compliance = summary.get("compliance")
        assert compliance is not None
        journal_block = compliance.get("journal")
        assert journal_block is not None
        last_entry = journal_block.get("last_entry")
        assert last_entry is not None
        assert last_entry["trade_id"] == "evt-journal-runtime"
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_spark_exports(tmp_path) -> None:
    db_path = tmp_path / "spark_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        snapshot = SparkExportSnapshot(
            generated_at=datetime(2024, 1, 5, tzinfo=UTC),
            status=SparkExportStatus.ok,
            format=SparkExportFormat.csv,
            root_path="/tmp/spark",
            jobs=tuple(),
            metadata={"job_count": 0},
        )
        app.record_spark_export_snapshot(snapshot)
        summary = app.summary()
        assert "spark_exports" in summary
        payload = summary["spark_exports"]
        assert payload["snapshot"]["status"] == "ok"
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_spark_stress(tmp_path) -> None:
    db_path = tmp_path / "spark_stress_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        snapshot = SparkStressSnapshot(
            label="drill",
            status=SparkStressStatus.warn,
            generated_at=datetime(2024, 1, 6, tzinfo=UTC),
            cycles=(
                SparkStressCycleResult(
                    cycle=1,
                    status=SparkStressStatus.warn,
                    export_status=SparkExportStatus.warn,
                    duration_seconds=0.9,
                    issues=("duration_exceeded_warn_threshold",),
                    metadata={"job_count": 1},
                ),
            ),
            metadata={"cycles": 1},
        )
        app.record_spark_stress_snapshot(snapshot)
        summary = app.summary()
        assert "spark_stress" in summary
        payload = summary["spark_stress"]
        assert payload["snapshot"]["label"] == "drill"
        assert "markdown" in payload
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_failover_drill(tmp_path) -> None:
    db_path = tmp_path / "failover_drill_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        plan = TimescaleBackbonePlan(daily=DailyBarIngestPlan(symbols=["EURUSD"]))
        ingest_result = TimescaleIngestResult(
            6,
            ("EURUSD",),
            datetime(2024, 1, 4, tzinfo=UTC),
            datetime(2024, 1, 4, 21, 0, tzinfo=UTC),
            1.0,
            25.0,
            "daily_bars",
            "yahoo",
        )
        snapshot = await execute_failover_drill(
            plan=plan,
            results={"daily_bars": ingest_result},
            fail_dimensions=("daily_bars",),
            scenario="summary-drill",
            fallback=None,
        )
        app.record_failover_drill_snapshot(snapshot)

        summary = app.summary()
        section = summary.get("failover_drill")
        assert section is not None
        assert section["snapshot"]["scenario"] == "summary-drill"
        assert "markdown" in section
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_cross_region(tmp_path) -> None:
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        snapshot = CrossRegionFailoverSnapshot(
            status=CrossRegionStatus.warn,
            generated_at=datetime(2024, 1, 8, tzinfo=UTC),
            primary_region="eu-west",
            replica_region="us-east",
            components=(
                CrossRegionComponent(
                    name="replica:daily_bars",
                    status=CrossRegionStatus.warn,
                    summary="lag 90s > warn threshold",
                    metadata={"lag_seconds": 90.0},
                ),
            ),
            metadata={"settings": {"primary_region": "eu-west"}},
        )
        app.record_cross_region_snapshot(snapshot)

        summary = app.summary()
        section = summary.get("cross_region_failover")
        assert section is not None
        assert section["snapshot"]["primary_region"] == "eu-west"
        assert "markdown" in section
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_kafka_readiness(tmp_path) -> None:
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        kafka_snapshot = KafkaReadinessSnapshot(
            status=KafkaReadinessStatus.warn,
            generated_at=datetime(2024, 1, 1, tzinfo=UTC),
            brokers="Kafka(bootstrap=localhost:9092, security=PLAINTEXT, acks=all, client_id=emp-institutional-ingest)",
            components=(
                KafkaReadinessComponent(
                    name="connection",
                    status=KafkaReadinessStatus.warn,
                    summary="test",
                ),
            ),
            metadata={"settings": {"enabled": True}},
        )
        app.record_kafka_readiness_snapshot(kafka_snapshot)

        summary = app.summary()
        section = summary.get("kafka_readiness")
        assert section is not None
        assert section["snapshot"]["status"] == "warn"
        assert "markdown" in section
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_sensory_drift(tmp_path) -> None:
    db_path = tmp_path / "sensory_drift_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        dimension = SensoryDimensionDrift(
            name="why",
            current_signal=0.42,
            baseline_signal=0.10,
            delta=0.32,
            current_confidence=0.76,
            baseline_confidence=0.70,
            confidence_delta=0.06,
            severity=DriftSeverity.warn,
            samples=3,
        )
        snapshot = SensoryDriftSnapshot(
            generated_at=datetime(2024, 1, 4, tzinfo=UTC),
            status=DriftSeverity.warn,
            dimensions={"why": dimension},
            sample_window=3,
            metadata={"entries": 3},
        )
        app.record_sensory_drift_snapshot(snapshot)

        summary = app.summary()
        drift_section = summary.get("sensory_drift")
        assert drift_section is not None
        assert drift_section["snapshot"]["status"] == DriftSeverity.warn.value
        assert "markdown" in drift_section
        assert "why" in drift_section["markdown"]
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_evolution_experiments(tmp_path) -> None:
    db_path = tmp_path / "evolution_experiment_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        metrics = ExperimentMetrics(
            total_events=4,
            executed=2,
            rejected=1,
            failed=1,
            execution_rate=0.5,
            rejection_rate=0.25,
            failure_rate=0.25,
            avg_confidence=0.74,
            avg_notional=15_000.0,
            roi_status=ExperimentStatus.warn.value,
            roi=-0.02,
            net_pnl=-500.0,
            metadata={"ingest_success": True},
        )
        snapshot = EvolutionExperimentSnapshot(
            generated_at=datetime(2024, 1, 4, tzinfo=UTC),
            status=ExperimentStatus.warn,
            metrics=metrics,
            rejection_reasons={"low_confidence": 1},
            metadata={"ingest_success": True},
        )
        app.record_evolution_experiment_snapshot(snapshot)

        summary = app.summary()
        section = summary.get("evolution_experiments")
        assert section is not None
        assert section["snapshot"]["status"] == ExperimentStatus.warn.value
        assert "markdown" in section
        assert "Execution rate" in section["markdown"]
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_evolution_tuning(tmp_path) -> None:
    db_path = tmp_path / "evolution_tuning_summary.db"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        extras={"TIMESCALEDB_URL": f"sqlite:///{db_path}"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        summary_block = EvolutionTuningSummary(
            total_recommendations=1,
            action_counts={"disable_strategy": 1},
            execution_rate=0.32,
            roi=-0.08,
            roi_status="at_risk",
            metadata={"ingest_success": True},
        )
        recommendation = EvolutionTuningRecommendation(
            strategy_id="trend_follow",
            action="disable_strategy",
            rationale="Failure rate 50% exceeds alert threshold",
            confidence=0.9,
            metadata={"failure_rate": 0.5},
        )
        snapshot = EvolutionTuningSnapshot(
            generated_at=datetime(2024, 1, 4, tzinfo=UTC),
            status=EvolutionTuningStatus.alert,
            summary=summary_block,
            recommendations=(recommendation,),
            metadata={"ingest_success": True},
        )
        app.record_evolution_tuning_snapshot(snapshot)

        summary = app.summary()
        section = summary.get("evolution_tuning")
        assert section is not None
        assert section["snapshot"]["status"] == EvolutionTuningStatus.alert.value
        assert "markdown" in section
        assert "disable_strategy" in section["markdown"]
    finally:
        await app.shutdown()
