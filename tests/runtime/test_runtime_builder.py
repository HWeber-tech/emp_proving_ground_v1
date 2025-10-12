import asyncio
import contextlib
import json
from collections import deque
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, Mapping

import io
import logging

import pytest
import pandas as pd

import src.runtime.runtime_builder as runtime_builder_module

from src.config.risk.risk_config import RiskConfig
from src.governance.system_config import ConnectionProtocol, DataBackboneMode, EmpTier, SystemConfig
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
from src.operations.professional_readiness import (
    ProfessionalReadinessComponent,
    ProfessionalReadinessSnapshot,
    ProfessionalReadinessStatus,
)
from src.operations.incident_response import IncidentResponseStatus
from src.operations.cross_region_failover import CrossRegionFailoverSnapshot
from src.operations.kafka_readiness import (
    KafkaReadinessComponent,
    KafkaReadinessSnapshot,
    KafkaReadinessStatus,
)
from src.data_foundation.ingest.scheduler_telemetry import (
    IngestSchedulerSnapshot,
    IngestSchedulerStatus,
)
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackbonePlan,
)
from src.runtime import (
    RuntimeApplication,
    RuntimeWorkload,
    WorkloadRestartPolicy,
    build_professional_predator_app,
    build_professional_runtime_application,
)
from src.runtime.predator_app import ProfessionalPredatorApp
from src.runtime.task_supervisor import TaskSupervisor
from src.runtime.runtime_builder import (
    _normalise_ingest_plan_metadata,
    _plan_dimensions,
    _process_sensory_status,
    _supervise_background_task,
)


@pytest.mark.asyncio()
async def test_runtime_application_runs_workloads_and_shutdown_callbacks():
    execution_order: deque[str] = deque()

    async def _ingest() -> None:
        execution_order.append("ingest")

    async def _trade() -> None:
        execution_order.append("trade")

    app = RuntimeApplication(
        ingestion=RuntimeWorkload(name="ingest", factory=_ingest, description="test ingest"),
        trading=RuntimeWorkload(name="trade", factory=_trade, description="test trade"),
    )

    called: list[str] = []

    async def _cleanup() -> None:
        called.append("cleanup")

    app.add_shutdown_callback(_cleanup)

    await app.run()

    assert list(execution_order) == ["ingest", "trade"]
    assert called == ["cleanup"]
    assert app.task_snapshots() == ()


@pytest.mark.asyncio()
async def test_runtime_application_restart_policy_recovers_failed_workload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    ingest_ready = asyncio.Event()
    trade_ready = asyncio.Event()
    release_ingest = asyncio.Event()
    release_trade = asyncio.Event()

    async def _ingest() -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("ingest failure")
        ingest_ready.set()
        await release_ingest.wait()

    async def _trade() -> None:
        trade_ready.set()
        await release_trade.wait()

    app = RuntimeApplication(
        ingestion=RuntimeWorkload(
            name="ingest",
            factory=_ingest,
            description="Timescale ingest loop",
            restart_policy=WorkloadRestartPolicy(max_restarts=3, backoff_seconds=0.0),
        ),
        trading=RuntimeWorkload(
            name="trade",
            factory=_trade,
            description="Trading loop",
        ),
    )

    with caplog.at_level(logging.ERROR, logger=app._logger.name):
        run_task = asyncio.create_task(app.run())

        await asyncio.wait_for(trade_ready.wait(), timeout=1.0)

        while attempts < 1:
            await asyncio.sleep(0)

        # Allow the supervisor to log the failure and restart
        await asyncio.sleep(0)
        messages = [record.getMessage() for record in caplog.records]
        assert any("ingest" in message and "failed" in message for message in messages)

        await asyncio.wait_for(ingest_ready.wait(), timeout=1.0)

        snapshots = app.task_snapshots()
        assert any(
            snapshot.get("name") == "ingest-workload" and snapshot.get("state") == "running"
            for snapshot in snapshots
        )

        release_ingest.set()
        release_trade.set()
        await asyncio.wait_for(run_task, timeout=1.0)

    assert attempts == 2
    summary = app.summary()
    assert summary["workload_states"].get("ingest") == "finished"
    assert app.task_snapshots() == ()


@pytest.mark.asyncio()
async def test_runtime_application_shutdown_cancels_active_workloads() -> None:
    ingest_started = asyncio.Event()
    trade_started = asyncio.Event()
    ingest_finished = asyncio.Event()
    trade_finished = asyncio.Event()
    cancellation_flags = {"ingest": False, "trade": False}

    async def _ingest() -> None:
        ingest_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_flags["ingest"] = True
            raise
        finally:
            ingest_finished.set()

    async def _trade() -> None:
        trade_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_flags["trade"] = True
            raise
        finally:
            trade_finished.set()

    app = RuntimeApplication(
        ingestion=RuntimeWorkload(
            name="ingest",
            factory=_ingest,
            description="long-running ingest",
        ),
        trading=RuntimeWorkload(
            name="trade",
            factory=_trade,
            description="long-running trade",
        ),
    )

    run_task = asyncio.create_task(app.run())

    await asyncio.wait_for(ingest_started.wait(), timeout=1.0)
    await asyncio.wait_for(trade_started.wait(), timeout=1.0)

    await app.shutdown()
    await asyncio.wait_for(run_task, timeout=1.0)

    assert ingest_finished.is_set()
    assert trade_finished.is_set()
    assert cancellation_flags["ingest"] is True
    assert cancellation_flags["trade"] is True
    assert app.task_snapshots() == ()
    summary = app.summary()
    assert summary["workload_states"].get("ingest") == "cancelled"
    assert summary["workload_states"].get("trade") == "cancelled"


@pytest.mark.asyncio()
async def test_runtime_application_ingest_failure_does_not_stop_trading(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trade_ready = asyncio.Event()
    release_trade = asyncio.Event()

    async def _ingest() -> None:
        await asyncio.sleep(0)
        raise RuntimeError("ingest panic")

    async def _trade() -> None:
        trade_ready.set()
        await release_trade.wait()

    app = RuntimeApplication(
        ingestion=RuntimeWorkload(
            name="ingest",
            factory=_ingest,
            description="Fail-fast ingest loop",
        ),
        trading=RuntimeWorkload(
            name="trade",
            factory=_trade,
            description="Trading loop",
        ),
    )

    with caplog.at_level(logging.ERROR, logger=app._logger.name):
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trade_ready.wait(), timeout=1.0)

        # Allow the supervisor to process the ingest failure and log it.
        await asyncio.sleep(0)
        messages = [record.getMessage() for record in caplog.records]
        assert any("ingest" in message and "failed" in message for message in messages)

        snapshots = app.task_snapshots()
        assert any(
            snapshot.get("name") == "trade-workload" and snapshot.get("state") == "running"
            for snapshot in snapshots
        )

        release_trade.set()
        await asyncio.wait_for(run_task, timeout=1.0)


@pytest.mark.asyncio()
async def test_runtime_application_ingest_torture_retries_without_crash(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    ingest_ready = asyncio.Event()
    release_ingest = asyncio.Event()
    trade_ready = asyncio.Event()
    release_trade = asyncio.Event()

    async def _ingest() -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError(f"ingest failure #{attempts}")
        ingest_ready.set()
        await release_ingest.wait()

    async def _trade() -> None:
        trade_ready.set()
        await release_trade.wait()

    app = RuntimeApplication(
        ingestion=RuntimeWorkload(
            name="ingest-torture",
            factory=_ingest,
            description="Fail twice before succeeding",
            restart_policy=WorkloadRestartPolicy(max_restarts=5, backoff_seconds=0.0),
        ),
        trading=RuntimeWorkload(
            name="trade-torture",
            factory=_trade,
            description="Trading loop should stay alive",
        ),
    )

    with caplog.at_level(logging.ERROR, logger=app._logger.name):
        run_task = asyncio.create_task(app.run())
        try:
            await asyncio.wait_for(trade_ready.wait(), timeout=1.0)
            await asyncio.wait_for(ingest_ready.wait(), timeout=1.0)

            snapshots = app.task_snapshots()
            assert any(
                snapshot.get("name") == "trade-torture-workload"
                and snapshot.get("state") == "running"
                for snapshot in snapshots
            )
            ingest_snapshot = next(
                snapshot
                for snapshot in snapshots
                if snapshot.get("name") == "ingest-torture-workload"
            )
            assert ingest_snapshot.get("state") == "running"
            assert ingest_snapshot.get("restarts") == 2

            messages = [record.getMessage() for record in caplog.records]
            assert sum("ingest-torture" in message and "failed" in message for message in messages) >= 2
        finally:
            release_ingest.set()
            release_trade.set()
            await asyncio.wait_for(run_task, timeout=1.0)

    assert attempts == 3
    summary = app.summary()
    assert summary["workload_states"].get("ingest-torture") == "finished"
    assert summary["workload_states"].get("trade-torture") == "finished"


@pytest.mark.asyncio()
async def test_supervise_background_task_provisions_fallback_supervisor() -> None:
    class DummyApp:
        pass

    app = DummyApp()
    ran = asyncio.Event()
    release = asyncio.Event()
    started = asyncio.Event()

    async def _job() -> None:
        started.set()
        await release.wait()
        ran.set()

    task = _supervise_background_task(
        app,
        _job(),
        name="dummy-job",
        metadata={"component": "test"},
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)

    supervisor = getattr(app, "_fallback_task_supervisor", None)
    assert isinstance(supervisor, TaskSupervisor)
    snapshots = supervisor.describe()
    assert snapshots
    assert snapshots[0]["metadata"]["component"] == "test"

    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert ran.is_set()
    assert supervisor.active_count == 0


@pytest.mark.asyncio()
async def test_runtime_application_summary_contains_metadata():
    workload = RuntimeWorkload(
        name="example",
        factory=lambda: asyncio.sleep(0),
        description="example workload",
        metadata={"key": "value"},
    )
    app = RuntimeApplication(ingestion=workload)

    summary = app.summary()
    assert summary["ingestion"]["name"] == "example"
    assert summary["ingestion"]["metadata"] == {"key": "value"}
    assert summary["shutdown_callbacks"] == 0
    assert summary["startup_callbacks"] == 0


class _StubEventBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def is_running(self) -> bool:
        return True

    def publish_from_sync(self, event: Any) -> int:
        self.events.append(event)
        return 1


class _StubSensoryApp:
    def __init__(self) -> None:
        self.event_bus = _StubEventBus()
        self.summaries: list[Any] = []
        self.metrics: list[Any] = []

    def record_sensory_summary(self, summary: Any) -> None:
        self.summaries.append(summary)

    def record_sensory_metrics(self, metrics: Any) -> None:
        self.metrics.append(metrics)


def _sample_sensory_status_payload() -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    return {
        "samples": 4,
        "latest": {
            "symbol": "EURUSD",
            "generated_at": generated_at.isoformat(),
            "integrated_signal": {
                "strength": 0.42,
                "confidence": 0.68,
                "direction": 1.0,
                "contributing": ["WHY", "HOW", "ANOMALY"],
            },
            "dimensions": {
                "WHY": {
                    "signal": 0.38,
                    "confidence": 0.70,
                    "metadata": {
                        "state": "bullish",
                        "threshold_assessment": {"state": "nominal"},
                    },
                },
                "HOW": {
                    "signal": 0.22,
                    "confidence": 0.55,
                    "metadata": {
                        "state": "nominal",
                        "threshold_assessment": {"state": "nominal"},
                    },
                },
                "ANOMALY": {
                    "signal": 0.75,
                    "confidence": 0.61,
                    "metadata": {
                        "state": "alert",
                        "threshold_assessment": {"state": "alert"},
                    },
                },
            },
        },
        "sensor_audit": [
            {
                "symbol": "EURUSD",
                "generated_at": generated_at.isoformat(),
                "unified_score": 0.39,
                "confidence": 0.66,
                "dimensions": {
                    "WHY": {"signal": 0.38, "confidence": 0.70},
                    "HOW": {"signal": 0.22, "confidence": 0.55},
                },
            }
        ],
        "drift_summary": {
            "exceeded": [
                {"sensor": "ANOMALY", "z_score": 3.1},
            ]
        },
    }


def test_process_sensory_status_publishes_summary_and_metrics() -> None:
    app = _StubSensoryApp()
    status = _sample_sensory_status_payload()

    audit_entries = _process_sensory_status(app, status)

    assert {event.type for event in app.event_bus.events} == {
        "telemetry.sensory.summary",
        "telemetry.sensory.metrics",
    }
    assert app.summaries and app.summaries[0].symbol == "EURUSD"
    assert app.metrics and app.metrics[0].symbol == "EURUSD"
    assert audit_entries
    assert audit_entries[0]["symbol"] == "EURUSD"


@pytest.mark.asyncio()
async def test_builder_bootstrap_mode(monkeypatch, tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={"BOOTSTRAP_SYMBOLS": "EURUSD"},
    )

    app = await build_professional_predator_app(config=cfg)

    class StubTradingManager:
        def __init__(self) -> None:
            self._risk_config = RiskConfig()

        def get_experiment_events(self) -> list[dict[str, object]]:
            return [
                {
                    "event_id": "exp-1",
                    "status": "executed",
                    "confidence": 0.85,
                    "notional": 10_000.0,
                },
                {
                    "event_id": "exp-2",
                    "status": "rejected",
                    "metadata": {"reason": "low_confidence"},
                },
            ]

        def get_last_roi_snapshot(self) -> None:
            return None

        def get_risk_status(self) -> Mapping[str, object]:
            return {"risk_config": self._risk_config.dict()}

    if getattr(app, "sensory_organ", None) is None:
        app.sensory_organ = SimpleNamespace(trading_manager=StubTradingManager())
    else:
        setattr(app.sensory_organ, "trading_manager", StubTradingManager())
    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=False,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )
        summary = runtime_app.summary()
        assert summary["ingestion"]["name"] == "tier0-ingest"
        assert summary["trading"]["name"] == "professional-trading"
        trading_metadata = summary["trading"].get("metadata", {})
        assert "risk" in trading_metadata
        assert trading_metadata["risk"]["mandatory_stop_loss"] is True
        assert trading_metadata["risk"]["runbook"].endswith("risk_api_contract.md")
    finally:
        await runtime_app.shutdown()
        await app.shutdown()


@pytest.mark.asyncio()
async def test_professional_app_summary_includes_runtime_application(tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={"BOOTSTRAP_SYMBOLS": "EURUSD"},
    )

    app = await build_professional_predator_app(config=cfg)

    runtime_app: RuntimeApplication | None = None

    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=True,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )

        runtime_summary = app.summary().get("runtime_application")
        assert runtime_summary is not None

        ingestion_block = runtime_summary.get("ingestion")
        assert ingestion_block is not None
        assert ingestion_block.get("name") == "skip-ingest"
        assert ingestion_block.get("metadata", {}).get(
            "reason"
        ) == "skip-ingest flag enabled"

        trading_block = runtime_summary.get("trading")
        assert trading_block is not None
        assert trading_block.get("name") == "professional-trading"
        assert trading_block.get("metadata", {}).get("mode") == app.config.run_mode.value

        # ``runtime_app`` should expose the same workload metadata for parity.
        runtime_snapshot = runtime_app.summary()
        assert runtime_snapshot.get("trading", {}).get("name") == "professional-trading"
    finally:
        if runtime_app is not None:
            await runtime_app.shutdown()
        await app.shutdown()


@pytest.mark.asyncio()
async def test_runtime_builder_labels_workload_components(tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={
            "BOOTSTRAP_SYMBOLS": "EURUSD",
            "RUNTIME_HEALTHCHECK_ENABLED": "false",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    runtime_app: RuntimeApplication | None = None

    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=False,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )

        summary = runtime_app.summary()
        ingestion_metadata = summary["ingestion"]["metadata"]
        assert ingestion_metadata["workload_kind"] == "data_backbone"
        assert tuple(ingestion_metadata["supervised_components"]) == (
            "data_backbone",
            "drift_monitor",
        )

        trading_metadata = summary["trading"]["metadata"]
        assert trading_metadata["workload_kind"] == "understanding_loop"
        assert tuple(trading_metadata["supervised_components"]) == (
            "understanding_loop",
            "trade_execution",
        )
    finally:
        if runtime_app is not None:
            await runtime_app.shutdown()
        await app.shutdown()


@pytest.mark.asyncio()
async def test_runtime_builder_ingest_failure_keeps_trading_running(
    caplog: pytest.LogCaptureFixture,
    tmp_path,
):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={
            "BOOTSTRAP_SYMBOLS": "EURUSD",
            "RUNTIME_HEALTHCHECK_ENABLED": "false",
        },
    )

    app = await build_professional_predator_app(config=cfg)

    attempts = {"count": 0}
    first_failure = asyncio.Event()
    successful_retry = asyncio.Event()

    runtime_app: RuntimeApplication | None = None
    run_task: asyncio.Task[None] | None = None

    try:
        await app.start()
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=False,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )

        original_ingestion = runtime_app.ingestion
        assert original_ingestion is not None

        async def _flaky_ingest() -> None:
            attempts["count"] += 1
            if attempts["count"] == 1:
                first_failure.set()
                raise RuntimeError("tier0 ingest failure")
            successful_retry.set()
            await asyncio.sleep(0)

        runtime_app.ingestion = replace(
            original_ingestion,
            factory=_flaky_ingest,
        )

        caplog.set_level(logging.ERROR, logger="src.runtime.predator_app")

        run_task = asyncio.create_task(runtime_app.run())

        await asyncio.wait_for(first_failure.wait(), timeout=2.0)
        await asyncio.sleep(0)

        failure_messages = [
            record.getMessage()
            for record in caplog.records
            if "tier0-ingest-workload failed" in record.getMessage()
        ]
        assert failure_messages, "expected supervised ingest failure logs"

        await asyncio.wait_for(successful_retry.wait(), timeout=5.0)
        await asyncio.sleep(0)

        summary = runtime_app.summary()
        trading_name = runtime_app.trading.name if runtime_app.trading else None
        assert trading_name is not None
        assert summary["workload_states"].get(trading_name) == "running"
        assert attempts["count"] >= 2

        trading_snapshots = runtime_app.task_snapshots()
        assert any(
            snapshot.get("name") == "professional-trading-workload"
            and snapshot.get("state") == "running"
            for snapshot in trading_snapshots
        )
    finally:
        app.request_shutdown()
        if run_task is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(run_task, timeout=5.0)
        if runtime_app is not None:
            await runtime_app.shutdown()
        await app.shutdown()

@pytest.mark.asyncio()
async def test_builder_requires_trading_manager(tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={"BOOTSTRAP_SYMBOLS": "EURUSD"},
    )

    app = await build_professional_predator_app(config=cfg)

    # Remove the trading manager to simulate a misconfigured runtime.
    app.sensory_organ = SimpleNamespace()

    runtime_app: RuntimeApplication | None = None

    try:
        with pytest.raises(RuntimeError) as excinfo:
            build_professional_runtime_application(
                app,
                skip_ingest=True,
                symbols_csv="EURUSD",
                duckdb_path=str(tmp_path / "tier0.duckdb"),
            )

        message = str(excinfo.value)
        assert "Trading manager not attached" in message
        assert "risk_api_contract.md" in message
    finally:
        if runtime_app is not None:
            await runtime_app.shutdown()
        await app.shutdown()


@pytest.mark.asyncio()
async def test_builder_rejects_invalid_trading_risk_config(tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={"BOOTSTRAP_SYMBOLS": "EURUSD"},
    )

    app = await build_professional_predator_app(config=cfg)

    class StubTradingManager:
        def get_risk_status(self) -> Mapping[str, object]:
            return {
                "risk_config": {
                    "max_risk_per_trade_pct": 0.2,
                    "max_total_exposure_pct": 0.1,
                }
            }

    if getattr(app, "sensory_organ", None) is None:
        app.sensory_organ = SimpleNamespace(trading_manager=StubTradingManager())
    else:
        setattr(app.sensory_organ, "trading_manager", StubTradingManager())

    try:
        with pytest.raises(RuntimeError) as excinfo:
            build_professional_runtime_application(
                app,
                skip_ingest=True,
                symbols_csv="EURUSD",
                duckdb_path=str(tmp_path / "tier0.duckdb"),
            )
        message = str(excinfo.value)
        assert "risk configuration is invalid" in message
        assert "risk_api_contract.md" in message
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_builder_rejects_stop_loss_disabled_outside_research_mode(tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={"BOOTSTRAP_SYMBOLS": "EURUSD"},
    )

    app = await build_professional_predator_app(config=cfg)

    class StubTradingManager:
        def __init__(self) -> None:
            self._risk_config = RiskConfig(
                mandatory_stop_loss=False,
                research_mode=False,
            )

    if getattr(app, "sensory_organ", None) is None:
        app.sensory_organ = SimpleNamespace(trading_manager=StubTradingManager())
    else:
        setattr(app.sensory_organ, "trading_manager", StubTradingManager())

    try:
        with pytest.raises(RuntimeError) as excinfo:
            build_professional_runtime_application(
                app,
                skip_ingest=True,
                symbols_csv="EURUSD",
                duckdb_path=str(tmp_path / "tier0.duckdb"),
            )
        message = str(excinfo.value)
        assert "mandatory_stop_loss" in message
        assert "risk_api_contract.md" in message
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_builder_publishes_risk_configuration_event(monkeypatch, tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={
            "BOOTSTRAP_SYMBOLS": "EURUSD",
            "RUNTIME_HEALTHCHECK_ENABLED": False,
        },
    )

    app = await build_professional_predator_app(config=cfg)

    class StubTradingManager:
        def __init__(self) -> None:
            self._risk_config = RiskConfig()

        def get_risk_status(self) -> Mapping[str, object]:
            return {"risk_config": self._risk_config.dict()}

    if getattr(app, "sensory_organ", None) is None:
        app.sensory_organ = SimpleNamespace(trading_manager=StubTradingManager())
    else:
        setattr(app.sensory_organ, "trading_manager", StubTradingManager())

    runtime_app: RuntimeApplication | None = None

    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=True,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )

        captured_events: list[Any] = []

        def _capture_publish_from_sync(event: Any) -> int:
            captured_events.append(event)
            return 1

        async def _capture_publish(event: Any) -> None:
            captured_events.append(event)

        monkeypatch.setattr(
            app.event_bus,
            "publish_from_sync",
            _capture_publish_from_sync,
            raising=False,
        )
        monkeypatch.setattr(
            app.event_bus,
            "publish",
            _capture_publish,
            raising=False,
        )
        monkeypatch.setattr(app.event_bus, "is_running", lambda: True, raising=False)

        callbacks = [
            callback
            for callback in runtime_app.startup_callbacks
            if getattr(callback, "__name__", "") == "enforce_trading_risk_config"
        ]
        assert callbacks, "expected risk enforcement callback to be registered"
        await callbacks[0]()

        risk_events = [
            event
            for event in captured_events
            if getattr(event, "type", None) == "telemetry.risk.configuration"
        ]
        assert risk_events, "risk configuration event was not published"
        payload = risk_events[0].payload
        assert payload["risk"]["mandatory_stop_loss"] is True
        assert payload["runbook"].endswith("risk_api_contract.md")

        summary = app.summary()
        configuration_event = summary.get("risk", {}).get("configuration_event")
        assert configuration_event is not None
        assert configuration_event["risk"]["max_risk_per_trade_pct"] == pytest.approx(0.02)
    finally:
        if runtime_app is not None:
            await runtime_app.shutdown()
        await app.shutdown()


@pytest.mark.asyncio()
async def test_configuration_audit_runs_without_timescale(monkeypatch, tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={"BOOTSTRAP_SYMBOLS": "EURUSD"},
    )

    app = await build_professional_predator_app(config=cfg)

    published: dict[str, object] = {}

    def _capture_publish(event_bus, snapshot):
        published["snapshot"] = snapshot

    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_configuration_audit_snapshot",
        _capture_publish,
    )

    migrator_called = False

    def _mark_migrator(self, *args, **kwargs):
        nonlocal migrator_called
        migrator_called = True

    monkeypatch.setattr(
        "src.runtime.runtime_builder.TimescaleMigrator.ensure_configuration_tables",
        _mark_migrator,
    )

    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=True,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )
        assert isinstance(runtime_app, RuntimeApplication)

        assert "snapshot" in published, "expected configuration audit publication"
        snapshot = published["snapshot"]
        assert snapshot.metadata.get("initial_snapshot") is True

        summary = app.summary()
        assert "configuration_audit" in summary
        audit_block = summary["configuration_audit"]
        assert audit_block["snapshot"]["metadata"]["initial_snapshot"] is True

        assert migrator_called is False
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_builder_registers_runtime_health_server(tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={"BOOTSTRAP_SYMBOLS": "EURUSD"},
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=True,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )
        assert runtime_app.startup_callbacks, "health server startup callback missing"
        assert runtime_app.shutdown_callbacks, "health server shutdown callback missing"
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_builder_health_server_can_be_disabled(tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={
            "BOOTSTRAP_SYMBOLS": "EURUSD",
            "RUNTIME_HEALTHCHECK_ENABLED": "false",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=True,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )
        assert runtime_app.startup_callbacks, "expected risk enforcement callback"
        assert {
            callback.__name__ for callback in runtime_app.startup_callbacks
        } == {"enforce_trading_risk_config"}
    finally:
        await app.shutdown()


@pytest.mark.asyncio()
async def test_builder_institutional_ingest_executes_timescale(monkeypatch, tmp_path):
    db_path = tmp_path / "timescale.sqlite"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        tier=EmpTier.tier_1,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
            "TIMESCALE_SYMBOLS": "EURUSD",
            "SECURITY_TOTAL_USERS": "10",
            "SECURITY_USERS_WITH_MFA": "9",
            "SECURITY_CREDENTIAL_AGE_DAYS": "5",
            "SECURITY_SECRETS_AGE_DAYS": "4",
            "SECURITY_INCIDENT_DRILL_AGE_DAYS": "20",
            "SECURITY_VULNERABILITY_SCAN_AGE_DAYS": "10",
            "SECURITY_INTRUSION_DETECTION_ACTIVE": "true",
            "SECURITY_TLS_VERSIONS": "TLS1.2,TLS1.3",
        },
    )

    executed: dict[str, int] = {"timescale": 0, "tier0": 0}

    async def _fake_timescale(**kwargs):
        executed["timescale"] += 1
        return True, None

    async def _fake_tier0(*args, **kwargs):
        executed["tier0"] += 1

    monkeypatch.setattr("src.runtime.runtime_builder._execute_timescale_ingest", _fake_timescale)
    monkeypatch.setattr("src.runtime.runtime_builder._run_tier0_ingest", _fake_tier0)

    app = await build_professional_predator_app(config=cfg)
    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=False,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )
        await runtime_app.ingestion.factory()  # type: ignore[union-attr]
    finally:
        await app.shutdown()

    assert executed["timescale"] == 1
    assert executed["tier0"] == 0


@pytest.mark.asyncio()
async def test_builder_records_backup_snapshot(monkeypatch, tmp_path):
    db_path = tmp_path / "timescale-backup.sqlite"
    validation_path = tmp_path / "system_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "timestamp": "2025-01-05T00:00:00+00:00",
                "validator": "System Completeness",
                "total_checks": 2,
                "results": {"core": True, "integration": False},
                "summary": {
                    "status": "PARTIAL",
                    "message": "integration check pending",
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        tier=EmpTier.tier_1,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
            "TIMESCALE_SYMBOLS": "EURUSD",
            "SYSTEM_VALIDATION_REPORT_PATH": str(validation_path),
            "INCIDENT_REQUIRED_RUNBOOKS": "redis_outage,kafka_lag",
            "INCIDENT_AVAILABLE_RUNBOOKS": "redis_outage,kafka_lag,fix_restart",
            "INCIDENT_MIN_PRIMARY_RESPONDERS": "2",
            "INCIDENT_PRIMARY_RESPONDERS": "alice,bob",
            "INCIDENT_MIN_SECONDARY_RESPONDERS": "1",
            "INCIDENT_SECONDARY_RESPONDERS": "carol",
            "INCIDENT_TRAINING_INTERVAL_DAYS": "30",
            "INCIDENT_TRAINING_AGE_DAYS": "12",
            "INCIDENT_DRILL_INTERVAL_DAYS": "45",
            "INCIDENT_DRILL_AGE_DAYS": "20",
            "INCIDENT_POSTMORTEM_SLA_HOURS": "24",
            "INCIDENT_POSTMORTEM_BACKLOG_HOURS": "8",
            "INCIDENT_REQUIRE_CHATOPS": "true",
            "INCIDENT_CHATOPS_READY": "true",
            "INCIDENT_MAX_OPEN_INCIDENTS": "3",
            "INCIDENT_SERVICE_NAME": "emp_incidents",
        },
    )

    snapshot = BackupReadinessSnapshot(
        service="timescale",
        generated_at=datetime(2024, 1, 4, tzinfo=UTC),
        status=BackupStatus.ok,
        latest_backup_at=datetime(2024, 1, 4, 0, 0, tzinfo=UTC),
        next_backup_due_at=datetime(2024, 1, 4, 12, 0, tzinfo=UTC),
        retention_days=7,
        issues=tuple(),
    )

    published: dict[str, BackupReadinessSnapshot] = {}
    compliance_published: list[object] = []
    workflow_published: list[object] = []
    security_published: list[object] = []
    cache_published: list[object] = []
    event_bus_published: list[object] = []
    scheduler_published: list[IngestSchedulerSnapshot] = []
    system_validation_published: list[object] = []
    evolution_published: list[object] = []
    strategy_published: list[object] = []
    incident_published: list[object] = []
    tuning_published: list[object] = []
    regulatory_published: list[object] = []
    governance_published: list[object] = []
    backbone_validation_recorded: list[DataBackboneValidationSnapshot] = []
    backbone_validation_published: list[DataBackboneValidationSnapshot] = []
    backbone_recorded: list[DataBackboneReadinessSnapshot] = []
    professional_recorded: list[ProfessionalReadinessSnapshot] = []
    retention_recorded: list[DataRetentionSnapshot] = []

    def _fake_publish(event_bus, snap):
        published["snapshot"] = snap

    def _fake_publish_compliance(event_bus, snapshot):
        compliance_published.append(snapshot)

    def _fake_publish_compliance_workflows(event_bus, snapshot):
        workflow_published.append(snapshot)

    async def _fake_timescale(*args, **kwargs):
        event_bus = kwargs.get("event_bus")
        if event_bus is not None:
            _fake_publish(event_bus, snapshot)
        validation_recorder = kwargs.get("record_backbone_validation_snapshot")
        if callable(validation_recorder):
            validation_snapshot = DataBackboneValidationSnapshot(
                status=BackboneStatus.ok,
                generated_at=datetime(2024, 1, 4, tzinfo=UTC),
                checks=(
                    BackboneComponentSnapshot(
                        name="plan",
                        status=BackboneStatus.ok,
                        summary="configured",
                    ),
                ),
                metadata={},
            )
            validation_recorder(validation_snapshot)
            backbone_validation_recorded.append(validation_snapshot)
            backbone_validation_published.append(validation_snapshot)
        recorder = kwargs.get("record_backbone_snapshot")
        if callable(recorder):
            backbone_snapshot = DataBackboneReadinessSnapshot(
                status=BackboneStatus.ok,
                generated_at=datetime(2024, 1, 4, tzinfo=UTC),
                components=(
                    BackboneComponentSnapshot(
                        name="plan",
                        status=BackboneStatus.ok,
                        summary="configured",
                    ),
                ),
                metadata={},
            )
            recorder(backbone_snapshot)
            backbone_recorded.append(backbone_snapshot)
        professional_recorder = kwargs.get("record_professional_snapshot")
        if callable(professional_recorder):
            professional_snapshot = ProfessionalReadinessSnapshot(
                status=ProfessionalReadinessStatus.ok,
                generated_at=datetime(2024, 1, 4, tzinfo=UTC),
                components=(
                    ProfessionalReadinessComponent(
                        name="data_backbone",
                        status=ProfessionalReadinessStatus.ok,
                        summary="ready",
                    ),
                ),
            )
            professional_recorder(professional_snapshot)
            professional_recorded.append(professional_snapshot)
        retention_recorder = kwargs.get("record_data_retention_snapshot")
        if callable(retention_recorder):
            retention_snapshot = DataRetentionSnapshot(
                status=RetentionStatus.ok,
                generated_at=datetime(2024, 1, 4, tzinfo=UTC),
                components=(
                    RetentionComponentSnapshot(
                        name="daily_bars",
                        status=RetentionStatus.ok,
                        summary="coverage",
                    ),
                ),
            )
            retention_recorder(retention_snapshot)
            retention_recorded.append(retention_snapshot)
        return True, snapshot

    monkeypatch.setattr("src.runtime.runtime_builder._execute_timescale_ingest", _fake_timescale)
    monkeypatch.setattr("src.runtime.runtime_builder._publish_backup_snapshot", _fake_publish)
    monkeypatch.setattr(
        "src.runtime.runtime_builder._publish_data_backbone_validation",
        lambda event_bus, snapshot: backbone_validation_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_compliance_readiness",
        _fake_publish_compliance,
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_compliance_workflows",
        _fake_publish_compliance_workflows,
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_regulatory_telemetry",
        lambda event_bus, snapshot: regulatory_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_governance_report",
        lambda event_bus, report: governance_published.append(report),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.collect_audit_evidence",
        lambda _cfg, strategy_id=None: {
            "metadata": {"configured": True, "dialect": "sqlite"},
            "compliance": {"stats": {"total_records": 5}},
            "kyc": {"stats": {"total_cases": 1}},
        },
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_security_posture",
        lambda event_bus, snapshot: security_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_cache_health",
        lambda event_bus, snapshot: cache_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_event_bus_health",
        lambda event_bus, snapshot: event_bus_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_system_validation_snapshot",
        lambda event_bus, snapshot: system_validation_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_evolution_experiment_snapshot",
        lambda event_bus, snapshot: evolution_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_evolution_tuning_snapshot",
        lambda event_bus, snapshot: tuning_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_strategy_performance_snapshot",
        lambda event_bus, snapshot: strategy_published.append(snapshot),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_incident_response_snapshot",
        lambda event_bus, snapshot: incident_published.append(snapshot),
    )

    async def _fake_publish_scheduler(event_bus, snapshot, **_kwargs):
        scheduler_published.append(snapshot)

    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_scheduler_snapshot",
        _fake_publish_scheduler,
    )

    app = await build_professional_predator_app(config=cfg)

    class StubTradingManager:
        def __init__(self) -> None:
            self._events = [
                {
                    "event_id": "exp-1",
                    "status": "executed",
                    "confidence": 0.9,
                    "notional": 12_000.0,
                },
                {
                    "event_id": "exp-2",
                    "status": "rejected",
                    "metadata": {"reason": "low_confidence"},
                },
            ]
            self._risk_config = RiskConfig()

        def get_execution_stats(self) -> Mapping[str, object]:
            return {
                "orders_submitted": 2,
                "orders_executed": 1,
                "orders_failed": 1,
                "latency_samples": 1,
                "total_latency_ms": 12.5,
                "max_latency_ms": 12.5,
                "last_error": "low_confidence",
            }

        def get_experiment_events(self, limit: int | None = None) -> list[Mapping[str, object]]:
            events = list(self._events)
            if limit is None:
                return events
            if limit <= 0:
                return []
            return events[: int(limit)]

        def get_last_roi_snapshot(self) -> Mapping[str, object]:
            return {"status": "tracking", "roi": 0.05, "net_pnl": 320.0}

        def get_risk_status(self) -> Mapping[str, object]:
            return {"risk_config": self._risk_config.dict()}

    trading_manager = StubTradingManager()
    if getattr(app, "sensory_organ", None) is None:
        app.sensory_organ = SimpleNamespace(trading_manager=trading_manager)
    else:
        setattr(app.sensory_organ, "trading_manager", trading_manager)
    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=False,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )
        await runtime_app.ingestion.factory()  # type: ignore[union-attr]
    finally:
        await app.shutdown()

    assert published["snapshot"] is snapshot
    assert len(backbone_validation_recorded) == 1
    assert len(backbone_validation_published) == 1
    assert app.get_last_backup_snapshot() is snapshot
    assert app.get_last_data_backbone_validation_snapshot() is backbone_validation_recorded[0]
    assert backbone_recorded
    assert app.get_last_data_backbone_snapshot() is backbone_recorded[0]
    assert professional_recorded
    assert app.get_last_professional_readiness_snapshot() is professional_recorded[0]
    assert retention_recorded
    assert app.get_last_data_retention_snapshot() is retention_recorded[0]
    assert compliance_published
    last_compliance = app.get_last_compliance_readiness_snapshot()
    assert last_compliance is compliance_published[0]
    assert workflow_published
    assert app.get_last_compliance_workflow_snapshot() is workflow_published[0]
    assert regulatory_published
    regulatory_snapshot = app.get_last_regulatory_snapshot()
    assert regulatory_snapshot is regulatory_published[0]
    assert regulatory_snapshot.metadata.get("compliance_status") == last_compliance.status.value
    assert governance_published
    governance_report = app.get_last_governance_report()
    assert governance_report is governance_published[0]
    assert governance_report.metadata.get("regulatory_status") == regulatory_snapshot.status.value
    assert governance_report.sections
    assert security_published
    security_snapshot = app.get_last_security_snapshot()
    assert security_snapshot is security_published[0]
    assert security_snapshot.controls
    assert cache_published
    cache_snapshot = app.get_last_cache_snapshot()
    assert cache_snapshot is cache_published[0]
    assert cache_snapshot.metadata
    assert event_bus_published
    event_bus_snapshot = app.get_last_event_bus_snapshot()
    assert event_bus_snapshot is event_bus_published[0]
    assert system_validation_published
    system_validation_snapshot = app.get_last_system_validation_snapshot()
    assert system_validation_snapshot is system_validation_published[0]
    assert system_validation_snapshot.failed_checks == 1
    assert scheduler_published
    scheduler_snapshot = app.get_last_scheduler_snapshot()
    assert scheduler_snapshot is scheduler_published[0]
    assert scheduler_snapshot.status in {
        IngestSchedulerStatus.ok,
        IngestSchedulerStatus.warn,
        IngestSchedulerStatus.fail,
    }
    assert evolution_published
    evolution_snapshot = app.get_last_evolution_experiment_snapshot()
    assert evolution_snapshot is evolution_published[0]
    assert evolution_snapshot.metrics.total_events == 2
    assert evolution_snapshot.rejection_reasons["low_confidence"] == 1
    assert strategy_published
    strategy_snapshot = app.get_last_strategy_performance_snapshot()
    assert strategy_snapshot is strategy_published[0]
    assert strategy_snapshot.totals.total_events == 2
    assert tuning_published
    tuning_snapshot = app.get_last_evolution_tuning_snapshot()
    assert tuning_snapshot is tuning_published[0]
    assert tuning_snapshot.summary.total_recommendations >= 1
    assert incident_published
    incident_snapshot = app.get_last_incident_response_snapshot()
    assert incident_snapshot is incident_published[0]
    assert incident_snapshot.status is IncidentResponseStatus.ok
    assert strategy_snapshot.totals.roi_status == "tracking"
    assert strategy_snapshot.strategies


@pytest.mark.asyncio()
async def test_builder_records_cross_region_snapshot(monkeypatch, tmp_path):
    db_path = tmp_path / "timescale-cross-region.sqlite"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        tier=EmpTier.tier_1,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
            "TIMESCALE_SYMBOLS": "EURUSD",
            "TIMESCALE_CROSS_REGION_ENABLED": "true",
            "TIMESCALE_CROSS_REGION_PRIMARY_REGION": "eu-west",
            "TIMESCALE_CROSS_REGION_REPLICA_REGION": "us-east",
            "TIMESCALE_FAILOVER_DRILL": "false",
        },
    )

    cross_region_published: list[CrossRegionFailoverSnapshot] = []

    def _capture_cross_region(event_bus, snapshot):
        cross_region_published.append(snapshot)

    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_cross_region_snapshot",
        _capture_cross_region,
    )

    sample_daily = pd.DataFrame(
        [
            {
                "date": datetime(2024, 1, 2, tzinfo=UTC),
                "open": 1.0,
                "high": 1.05,
                "low": 0.98,
                "close": 1.02,
                "adj_close": 1.01,
                "volume": 1000,
                "symbol": "EURUSD",
            }
        ]
    )

    def _stub_daily(symbols, days=None):
        return sample_daily

    monkeypatch.setattr(
        "src.runtime.runtime_builder.fetch_daily_bars",
        _stub_daily,
    )

    validation_snapshot = DataBackboneValidationSnapshot(
        status=BackboneStatus.ok,
        generated_at=datetime(2024, 1, 8, tzinfo=UTC),
        checks=tuple(),
    )

    monkeypatch.setattr(
        "src.runtime.runtime_builder.evaluate_data_backbone_validation",
        lambda **kwargs: validation_snapshot,
    )

    app = await build_professional_predator_app(config=cfg)

    class StubTradingManager:
        def __init__(self) -> None:
            self._risk_config = RiskConfig()

        def get_execution_stats(self) -> Mapping[str, object]:
            return {
                "orders_submitted": 1,
                "orders_executed": 1,
                "orders_failed": 0,
                "latency_samples": 1,
                "total_latency_ms": 5.0,
                "max_latency_ms": 5.0,
            }

        def get_experiment_events(self, limit: int | None = None) -> list[Mapping[str, object]]:
            return []

        def get_last_roi_snapshot(self) -> Mapping[str, object]:
            return {"status": "tracking", "roi": 0.0, "net_pnl": 0.0}

        def get_risk_status(self) -> Mapping[str, object]:
            return {"risk_config": self._risk_config.dict()}

    if getattr(app, "sensory_organ", None) is None:
        app.sensory_organ = SimpleNamespace(trading_manager=StubTradingManager())
    else:
        setattr(app.sensory_organ, "trading_manager", StubTradingManager())

    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=False,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )
        await runtime_app.ingestion.factory()  # type: ignore[union-attr]
    finally:
        await app.shutdown()

    assert cross_region_published, "cross-region snapshot not published"
    snapshot = cross_region_published[0]
    assert isinstance(snapshot, CrossRegionFailoverSnapshot)
    recorded = app.get_last_cross_region_snapshot()
    assert recorded is snapshot
    metadata = snapshot.metadata.get("settings", {})
    assert metadata["primary_region"] == "eu-west"
    assert snapshot.status.value in {"warn", "fail"}


@pytest.mark.asyncio()
async def test_builder_records_kafka_readiness_snapshot(monkeypatch, tmp_path):
    db_path = tmp_path / "timescale-kafka.sqlite"
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        tier=EmpTier.tier_1,
        extras={
            "TIMESCALEDB_URL": f"sqlite:///{db_path}",
            "TIMESCALE_SYMBOLS": "EURUSD",
            "KAFKA_BROKERS": "localhost:9092",
            "KAFKA_INGEST_TOPICS": "daily_bars:ingest.daily",
        },
    )

    kafka_published: list[KafkaReadinessSnapshot] = []
    captured_topics: list[tuple[str, ...]] = []

    def _capture_kafka(event_bus, snapshot):
        kafka_published.append(snapshot)

    sample_snapshot = KafkaReadinessSnapshot(
        status=KafkaReadinessStatus.warn,
        generated_at=datetime(2024, 1, 1, tzinfo=UTC),
        brokers="Kafka(bootstrap=localhost:9092, security=PLAINTEXT, acks=all, client_id=emp-institutional-ingest)",
        components=(
            KafkaReadinessComponent(
                name="connection",
                status=KafkaReadinessStatus.ok,
                summary="configured",
            ),
        ),
        metadata={"settings": {"enabled": True}},
    )

    def _fake_evaluate(**kwargs):
        captured_topics.append(tuple(kwargs.get("topics", ())))
        return sample_snapshot

    monkeypatch.setattr(
        "src.runtime.runtime_builder.evaluate_kafka_readiness",
        lambda **kwargs: _fake_evaluate(**kwargs),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.publish_kafka_readiness",
        _capture_kafka,
    )

    sample_daily = pd.DataFrame(
        [
            {
                "date": datetime(2024, 1, 2, tzinfo=UTC),
                "open": 1.0,
                "high": 1.05,
                "low": 0.98,
                "close": 1.02,
                "adj_close": 1.01,
                "volume": 1000,
                "symbol": "EURUSD",
            }
        ]
    )

    monkeypatch.setattr(
        "src.runtime.runtime_builder.fetch_daily_bars",
        lambda symbols, days=None: sample_daily,
    )

    validation_snapshot = DataBackboneValidationSnapshot(
        status=BackboneStatus.ok,
        generated_at=datetime(2024, 1, 8, tzinfo=UTC),
        checks=tuple(),
    )
    monkeypatch.setattr(
        "src.runtime.runtime_builder.evaluate_data_backbone_validation",
        lambda **kwargs: validation_snapshot,
    )

    app = await build_professional_predator_app(config=cfg)

    class StubTradingManager:
        def __init__(self) -> None:
            self._risk_config = RiskConfig()

        def get_execution_stats(self) -> Mapping[str, object]:
            return {
                "orders_submitted": 1,
                "orders_executed": 1,
                "orders_failed": 0,
                "latency_samples": 1,
                "total_latency_ms": 5.0,
                "max_latency_ms": 5.0,
            }

        def get_experiment_events(self, limit: int | None = None) -> list[Mapping[str, object]]:
            return []

        def get_last_roi_snapshot(self) -> Mapping[str, object]:
            return {"status": "tracking", "roi": 0.0, "net_pnl": 0.0}

        def get_risk_status(self) -> Mapping[str, object]:
            return {"risk_config": self._risk_config.dict()}

    if getattr(app, "sensory_organ", None) is None:
        app.sensory_organ = SimpleNamespace(trading_manager=StubTradingManager())
    else:
        setattr(app.sensory_organ, "trading_manager", StubTradingManager())

    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=False,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )
        await runtime_app.ingestion.factory()  # type: ignore[union-attr]
    finally:
        await app.shutdown()

    assert kafka_published, "kafka readiness snapshot not published"
    assert captured_topics and captured_topics[0] == ("ingest.daily",)
    recorded = app.get_last_kafka_readiness_snapshot()
    assert recorded is sample_snapshot


@pytest.mark.asyncio()
async def test_builder_institutional_falls_back_when_plan_disabled(monkeypatch, tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.institutional,
        tier=EmpTier.tier_1,
        extras={"TIMESCALEDB_URL": f"sqlite:///{tmp_path / 'db.sqlite'}"},
    )

    executed = {"tier0": 0}

    async def _fake_tier0(*args, **kwargs):
        executed["tier0"] += 1

    monkeypatch.setattr("src.runtime.runtime_builder._run_tier0_ingest", _fake_tier0)

    app = await build_professional_predator_app(config=cfg)
    try:
        runtime_app = build_professional_runtime_application(
            app,
            skip_ingest=False,
            symbols_csv="EURUSD",
            duckdb_path=str(tmp_path / "tier0.duckdb"),
        )
        await runtime_app.ingestion.factory()  # type: ignore[union-attr]
    finally:
        await app.shutdown()

    assert executed["tier0"] == 1


def test_configure_runtime_logging_skips_without_flag(monkeypatch):
    from src.runtime import runtime_builder

    def _unexpected(**_: object) -> None:
        raise AssertionError("configure_structured_logging should not be invoked")

    monkeypatch.setattr(runtime_builder, "configure_structured_logging", _unexpected)
    runtime_builder._configure_runtime_logging(SystemConfig())


def test_configure_runtime_logging_enables_structured_logging(monkeypatch):
    from src.runtime import runtime_builder

    captured: dict[str, object] = {}
    info_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _fake_configure(**kwargs: object):
        captured.update(kwargs)
        handler = logging.StreamHandler(io.StringIO())
        handler.set_name("emp-structured-logger")
        return handler

    monkeypatch.setattr(runtime_builder, "configure_structured_logging", _fake_configure)
    monkeypatch.setattr(
        runtime_builder.logger, "info", lambda *args, **kwargs: info_calls.append((args, kwargs))
    )

    cfg = SystemConfig().with_updated(
        tier=EmpTier.tier_1,
        extras={
            "RUNTIME_LOG_STRUCTURED": "true",
            "RUNTIME_LOG_LEVEL": "debug",
            "RUNTIME_LOG_CONTEXT": json.dumps({"deployment": "staging"}),
        },
    )

    runtime_builder._configure_runtime_logging(cfg)

    assert captured["component"] == "professional_runtime"
    assert captured["level"] == logging.DEBUG
    static_fields = captured["static_fields"]
    assert static_fields["runtime.tier"] == EmpTier.tier_1.value
    assert static_fields["deployment"] == "staging"
    assert static_fields["runtime.environment"] == cfg.environment.value
    assert info_calls, "expected info log when structured logging is enabled"


def test_configure_runtime_logging_handles_invalid_context(monkeypatch):
    from src.runtime import runtime_builder

    warnings: list[str] = []
    monkeypatch.setattr(
        runtime_builder.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )

    captured: dict[str, object] = {}

    def _fake_configure(**kwargs: object):
        captured.update(kwargs)
        handler = logging.StreamHandler(io.StringIO())
        handler.set_name("emp-structured-logger")
        return handler

    monkeypatch.setattr(runtime_builder, "configure_structured_logging", _fake_configure)
    monkeypatch.setattr(runtime_builder.logger, "info", lambda *args, **kwargs: None)

    cfg = SystemConfig().with_updated(
        extras={"RUNTIME_LOG_STRUCTURED": "true", "RUNTIME_LOG_CONTEXT": "{invalid"}
    )

    runtime_builder._configure_runtime_logging(cfg)

    assert warnings, "expected warning for invalid JSON context"
    static_fields = captured["static_fields"]
    assert static_fields["runtime.environment"] == cfg.environment.value


@pytest.mark.parametrize(
    "plan,expected",
    [
        (
            TimescaleBackbonePlan(
                daily=DailyBarIngestPlan(symbols=("EURUSD",), lookback_days=10),
                intraday=IntradayTradeIngestPlan(symbols=("EURUSD",), lookback_days=1),
                macro=MacroEventIngestPlan(
                    events=({"event_name": "GDP"}, {"event_name": "CPI"})
                ),
            ),
            ["daily_bars", "intraday_trades", "macro_events"],
        ),
        (
            TimescaleBackbonePlan(
                daily=DailyBarIngestPlan(symbols=("AAPL",), lookback_days=5),
                macro=MacroEventIngestPlan(events=({"event_name": "NFP"},)),
            ),
            ["daily_bars", "macro_events"],
        ),
        (TimescaleBackbonePlan(), []),
    ],
)
def test_plan_dimensions_matches_active_slices(
    plan: TimescaleBackbonePlan, expected: list[str]
) -> None:
    assert _plan_dimensions(plan) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"daily_bars": {}, "macro_events": {}}, ["daily_bars", "macro_events"]),
        (("intraday_trades", "daily_bars"), ["intraday_trades", "daily_bars"]),
        ("daily_bars", ["daily_bars"]),
        (None, []),
    ],
)
def test_normalise_ingest_plan_metadata_handles_iterables(
    value: object, expected: list[str]
) -> None:
    assert _normalise_ingest_plan_metadata(value) == expected


@pytest.mark.asyncio()
async def test_builder_configures_governance_cadence(monkeypatch, tmp_path):
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={
            "BOOTSTRAP_SYMBOLS": "EURUSD",
            "RUNTIME_HEALTHCHECK_ENABLED": False,
            "GOVERNANCE_CADENCE_ENABLED": "true",
            "GOVERNANCE_CADENCE_INTERVAL_SECONDS": "3600",
            "GOVERNANCE_CADENCE_POLL_SECONDS": "0.05",
            "GOVERNANCE_CADENCE_FORCE_ON_START": "1",
            "GOVERNANCE_CADENCE_REPORT_PATH": str(tmp_path / "reports" / "governance.json"),
        },
    )

    app = await build_professional_predator_app(config=cfg)

    class _StubRunner:
        def __init__(self) -> None:
            self.compliance_provider = lambda: None
            self.regulatory_provider = lambda: None
            self.run_calls: list[tuple[datetime | None, bool]] = []

        def run(self, *, reference: datetime | None = None, force: bool = False):
            self.run_calls.append((reference, force))
            return object()

    stub_runner = _StubRunner()
    captured_kwargs: list[dict[str, object]] = []

    def _fake_builder(*args, **kwargs):
        captured_kwargs.append(kwargs)
        return stub_runner

    monkeypatch.setattr(
        runtime_builder_module,
        "build_governance_cadence_runner_from_config",
        _fake_builder,
    )

    reports: list[object] = []

    def _record_report(report: object) -> None:
        reports.append(report)

    app.record_governance_report = _record_report  # type: ignore[assignment]

    runtime_app = build_professional_runtime_application(
        app,
        skip_ingest=True,
        symbols_csv="EURUSD",
        duckdb_path=str(tmp_path / "tier0.duckdb"),
    )

    try:
        assert getattr(runtime_app, "_governance_cadence_runner") is stub_runner

        assert captured_kwargs, "Cadence runner builder should be invoked"
        kwargs = captured_kwargs[0]
        metadata = kwargs["metadata"]
        assert metadata["cadence_runner"] == "runtime.builder"
        assert str(kwargs["report_path"]).startswith(str(tmp_path))

        start_cb = next(
            cb for cb in runtime_app.startup_callbacks if cb.__name__ == "_start_governance_cadence"
        )
        await start_cb()

        await asyncio.sleep(0.1)

        assert stub_runner.run_calls, "Cadence loop should trigger runner"
        reference, force_flag = stub_runner.run_calls[0]
        assert isinstance(reference, datetime)
        assert force_flag is True
        assert reports, "Governance report should be recorded"

        await runtime_app.shutdown()
        runs_after_shutdown = len(stub_runner.run_calls)
        await asyncio.sleep(0.1)
        assert len(stub_runner.run_calls) == runs_after_shutdown
        assert not hasattr(runtime_app, "_governance_cadence_runner")
    finally:
        await app.shutdown()
@pytest.mark.asyncio()
async def test_runtime_application_ingest_failure_restarts_and_trading_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = TaskSupervisor(namespace="test-runtime", cancel_timeout=0.1)
    stop_event = asyncio.Event()
    ingest_restart = asyncio.Event()
    trade_started = asyncio.Event()
    attempts = 0

    async def _ingest() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("ingest burst")
        ingest_restart.set()
        await stop_event.wait()

    async def _trade() -> None:
        trade_started.set()
        await stop_event.wait()

    ingestion = RuntimeWorkload(
        name="ingest",
        factory=_ingest,
        description="test ingest",
        restart_policy=WorkloadRestartPolicy(max_restarts=3, backoff_seconds=0.0),
    )
    trading = RuntimeWorkload(
        name="trade",
        factory=_trade,
        description="test trade",
    )
    app = RuntimeApplication(
        ingestion=ingestion,
        trading=trading,
        task_supervisor=supervisor,
    )

    with caplog.at_level(logging.ERROR, logger=supervisor._logger.name):
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trade_started.wait(), timeout=1.0)
        await asyncio.sleep(0)  # allow ingest failure to propagate and restart
        await asyncio.wait_for(ingest_restart.wait(), timeout=1.0)
        assert not run_task.done()
        assert any(
            record.exc_text and "ingest burst" in record.exc_text
            for record in caplog.records
        )
        assert any("restarting" in record.getMessage() for record in caplog.records)

    stop_event.set()
    await asyncio.wait_for(run_task, timeout=1.0)
    assert attempts == 2
    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_runtime_application_ingest_failure_does_not_stop_trading(
    caplog: pytest.LogCaptureFixture,
) -> None:
    supervisor = TaskSupervisor(namespace="test-runtime-failure", cancel_timeout=0.1)
    stop_event = asyncio.Event()
    trade_started = asyncio.Event()

    async def _ingest_fail() -> None:
        raise RuntimeError("ingest catastrophic failure")

    async def _trade() -> None:
        trade_started.set()
        await stop_event.wait()

    ingestion = RuntimeWorkload(
        name="ingest",
        factory=_ingest_fail,
        description="failing ingest",
    )
    trading = RuntimeWorkload(
        name="trade",
        factory=_trade,
        description="resilient trade",
    )

    app = RuntimeApplication(
        ingestion=ingestion,
        trading=trading,
        task_supervisor=supervisor,
    )

    with caplog.at_level(logging.ERROR, logger=supervisor._logger.name):
        run_task = asyncio.create_task(app.run())
        await asyncio.wait_for(trade_started.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert any(
            "ingest-workload" in record.getMessage() and "failed" in record.getMessage()
            for record in caplog.records
        ), "expected ingest failure to be logged"
        assert not run_task.done(), "trading workload should continue running"

    interim_summary = app.summary()
    interim_states = interim_summary.get("workload_states") or {}
    assert interim_states.get("ingest") == "failed"
    assert interim_states.get("trade") == "running"
    supervisor_info = interim_summary.get("task_supervisor") or {}
    assert supervisor_info.get("active_tasks") == 1
    active_names = {task.get("name") for task in supervisor_info.get("tasks", [])}
    assert active_names == {"trade-workload"}

    stop_event.set()
    await asyncio.wait_for(run_task, timeout=1.0)

    final_states = app.summary().get("workload_states") or {}
    assert final_states.get("ingest") == "failed"
    assert final_states.get("trade") == "finished"

    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_runtime_application_summary_surfaces_supervisor_state() -> None:
    supervisor = TaskSupervisor(namespace="test-runtime-summary", cancel_timeout=0.1)
    stop_event = asyncio.Event()
    ingest_started = asyncio.Event()
    trade_started = asyncio.Event()

    async def _ingest() -> None:
        ingest_started.set()
        await stop_event.wait()

    async def _trade() -> None:
        trade_started.set()
        await stop_event.wait()

    ingestion = RuntimeWorkload(
        name="ingest",
        factory=_ingest,
        description="torture ingest",
        restart_policy=WorkloadRestartPolicy(max_restarts=None, backoff_seconds=0.0),
    )
    trading = RuntimeWorkload(
        name="trade",
        factory=_trade,
        description="steady trade",
        restart_policy=WorkloadRestartPolicy(max_restarts=1, backoff_seconds=0.0),
    )

    app = RuntimeApplication(
        ingestion=ingestion,
        trading=trading,
        task_supervisor=supervisor,
    )

    run_task = asyncio.create_task(app.run())
    await asyncio.wait_for(ingest_started.wait(), timeout=1.0)
    await asyncio.wait_for(trade_started.wait(), timeout=1.0)

    summary = app.summary()
    supervisor_info = summary.get("task_supervisor") or {}
    assert supervisor_info.get("namespace") == "test-runtime-summary"
    assert supervisor_info.get("active_tasks") == 2
    task_names = {task.get("name") for task in supervisor_info.get("tasks", [])}
    assert task_names == {"ingest-workload", "trade-workload"}
    ingest_summary = summary.get("ingestion") or {}
    trade_summary = summary.get("trading") or {}
    assert ingest_summary.get("state") == "running"
    assert trade_summary.get("state") == "running"
    assert ingest_summary.get("restart_policy") == {
        "max_restarts": None,
        "backoff_seconds": 0.0,
    }
    assert trade_summary.get("restart_policy") == {
        "max_restarts": 1,
        "backoff_seconds": 0.0,
    }

    stop_event.set()
    await asyncio.wait_for(run_task, timeout=1.0)
    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_runtime_application_create_background_task_tracks_task() -> None:
    supervisor = TaskSupervisor(namespace="test-runtime-background", cancel_timeout=0.1)
    app = RuntimeApplication(task_supervisor=supervisor)

    started = asyncio.Event()
    stop_event = asyncio.Event()

    async def _background() -> None:
        started.set()
        await stop_event.wait()

    task = app.create_background_task(
        _background(),
        name="custom-background-task",
        metadata={"component": "unit-test"},
    )

    await asyncio.wait_for(started.wait(), timeout=1.0)
    snapshots = app.task_snapshots()
    names = {entry.get("name") for entry in snapshots}
    assert "custom-background-task" in names
    summary = app.summary()
    assert summary.get("task_supervisor", {}).get("active_tasks") == 1

    stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)
    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_runtime_application_register_background_task_merges_metadata() -> None:
    supervisor = TaskSupervisor(namespace="test-runtime-register", cancel_timeout=0.1)
    app = RuntimeApplication(task_supervisor=supervisor)

    started = asyncio.Event()
    stop_event = asyncio.Event()

    async def _external() -> None:
        started.set()
        await stop_event.wait()

    task = asyncio.create_task(_external(), name="external-runtime-task")
    try:
        app.register_background_task(
            task,
            metadata={"component": "external", "kind": "probe"},
        )

        await asyncio.wait_for(started.wait(), timeout=1.0)
        snapshots = app.task_snapshots()
        entry = next(item for item in snapshots if item.get("name") == "external-runtime-task")
        metadata = entry.get("metadata") or {}
        assert metadata.get("component") == "external"
        assert metadata.get("kind") == "probe"
    finally:
        stop_event.set()
        await asyncio.wait_for(task, timeout=1.0)
        await supervisor.cancel_all()

async def test_runtime_application_recovers_from_ingest_failure(caplog: pytest.LogCaptureFixture) -> None:
    supervisor = TaskSupervisor(namespace="test-runtime-recovery", cancel_timeout=0.1)
    stop_event = asyncio.Event()
    ingest_first_attempt = asyncio.Event()
    ingest_restarted = asyncio.Event()
    trading_started = asyncio.Event()

    ingest_attempts = 0

    async def _ingest() -> None:
        nonlocal ingest_attempts
        ingest_attempts += 1
        if ingest_attempts == 1:
            ingest_first_attempt.set()
            raise RuntimeError("ingest failure")
        ingest_restarted.set()
        await stop_event.wait()

    async def _trade() -> None:
        trading_started.set()
        await stop_event.wait()

    ingestion = RuntimeWorkload(
        name="ingest",
        factory=_ingest,
        description="resilient ingest",
        restart_policy=WorkloadRestartPolicy(max_restarts=1, backoff_seconds=0.0),
    )
    trading = RuntimeWorkload(
        name="trade",
        factory=_trade,
        description="steady trading",
        restart_policy=WorkloadRestartPolicy(max_restarts=None, backoff_seconds=0.0),
    )

    app = RuntimeApplication(
        ingestion=ingestion,
        trading=trading,
        task_supervisor=supervisor,
    )

    caplog.set_level(
        logging.ERROR,
        logger="src.runtime.task_supervisor.test-runtime-recovery",
    )

    run_task = asyncio.create_task(app.run())

    await asyncio.wait_for(ingest_first_attempt.wait(), timeout=1.0)
    await asyncio.wait_for(ingest_restarted.wait(), timeout=1.0)
    await asyncio.wait_for(trading_started.wait(), timeout=1.0)

    stop_event.set()

    await asyncio.wait_for(run_task, timeout=1.0)

    assert ingest_attempts == 2
    assert any(
        "Background task ingest-workload failed" in message
        for _logger, _level, message in caplog.record_tuples
    )

    summary = app.summary()
    assert summary.get("ingestion", {}).get("state") == "finished"
    assert summary.get("trading", {}).get("state") == "finished"

    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_bootstrap_runtime_uses_app_task_supervisor() -> None:
    cfg = SystemConfig().with_updated(
        connection_protocol=ConnectionProtocol.bootstrap,
        data_backbone_mode=DataBackboneMode.bootstrap,
        extras={
            "BOOTSTRAP_TICK_INTERVAL": "0.01",
            "RUNTIME_HEALTHCHECK_ENABLED": "false",
        },
    )

    app = await build_professional_predator_app(config=cfg)
    try:
        await app.start()
        await asyncio.sleep(0)

        task_details = app.task_supervisor.describe()
        runtime_task = next(
            item for item in task_details if item.get("name") == "bootstrap-runtime-loop"
        )
        metadata = runtime_task.get("metadata")
        assert metadata is not None
        assert metadata.get("component") == "understanding.loop"
        assert tuple(metadata.get("symbols") or ()) == ("EURUSD",)
        assert metadata.get("drift_window")
        assert metadata.get("drift_sensors")

        summary = app.summary()
        assert summary.get("background_tasks", 0) >= 1
        background_details = summary.get("background_task_details") or []
        assert any(item.get("name") == "bootstrap-runtime-loop" for item in background_details)
    finally:
        await app.shutdown()
