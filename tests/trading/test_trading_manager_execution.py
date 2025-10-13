import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import AsyncMock

from decimal import Decimal

import datetime as _datetime
import enum as _enum
import typing as _typing
import sys

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]

if not hasattr(_enum, "StrEnum"):
    class _StrEnum(str, _enum.Enum):
        pass

    _enum.StrEnum = _StrEnum

def _shim_class_getitem(name: str) -> type:
    class _Placeholder:
        @classmethod
        def __class_getitem__(cls, item):
            return item

    _Placeholder.__name__ = name
    return _Placeholder


if not hasattr(_typing, "Unpack"):
    _typing.Unpack = _shim_class_getitem("Unpack")  # type: ignore[attr-defined]

if not hasattr(_typing, "NotRequired"):
    _typing.NotRequired = _shim_class_getitem("NotRequired")  # type: ignore[attr-defined]

import logging

import pytest

from src.config.risk.risk_config import RiskConfig
from src.compliance.workflow import WorkflowTaskStatus
from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyDelta,
    PolicyLedgerFeatureFlags,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.operations.sensory_drift import DriftSeverity, SensoryDimensionDrift, SensoryDriftSnapshot
from src.runtime.task_supervisor import TaskSupervisor
from src.trading.execution.backlog_tracker import EventBacklogTracker
from src.trading.execution.liquidity_prober import LiquidityProber
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.execution.performance_baseline import collect_performance_baseline
from src.trading.execution.performance_monitor import ThroughputMonitor
from src.trading.execution.resource_monitor import ResourceUsageMonitor
from src.trading.execution.release_router import ReleaseAwareExecutionRouter
from src.trading.gating import DriftSentryGate
from src.trading.risk.risk_api import RISK_API_RUNBOOK
from src.trading.trading_manager import TradingManager


class AlwaysActiveRegistry:
    def get_strategy(self, strategy_id: str) -> dict[str, str]:
        return {"status": "active"}


class DummyBus:
    def __init__(self) -> None:
        self.subscriptions: dict[str, list[object]] = {}

    def subscribe(self, topic: str, callback: object) -> None:
        self.subscriptions.setdefault(topic, []).append(callback)

    async def publish(self, *args, **kwargs) -> None:  # pragma: no cover - stubbed
        return None

    def publish_from_sync(self, event: object) -> None:  # pragma: no cover - stubbed
        return None

    def is_running(self) -> bool:  # pragma: no cover - stubbed
        return False


class RecordingBus(DummyBus):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


def _silence_trading_manager_publishers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)


@dataclass
class SimpleIntent:
    symbol: str
    quantity: float
    price: float


@dataclass
class ConfidenceIntent:
    symbol: str
    quantity: float
    price: float
    confidence: float
    strategy_id: str = "alpha"
    created_at: datetime | None = None


class RecordingExecutionEngine:
    def __init__(self) -> None:
        self.calls = 0

    async def process_order(self, intent: Any) -> str:
        self.calls += 1
        return "ok"


class FailingExecutionEngine:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls = 0
        self._error = error or RuntimeError("paper broker rejected order")

    async def process_order(self, _intent: Any) -> str:
        self.calls += 1
        raise self._error


class _ImmediateProbeBroker:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, float | str]] = {}

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> str:
        order_id = f"ORD-{len(self.orders) + 1}"
        self.orders[order_id] = {
            "status": "FILLED",
            "filled_qty": float(quantity),
        }
        return order_id

    def get_order_status(self, order_id: str) -> dict[str, float | str] | None:
        return self.orders.get(order_id)


class _BlockingProbeBroker:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, float | str]] = {}
        self._allow_fill = asyncio.Event()

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> str:
        order_id = f"ORD-{len(self.orders) + 1}"
        self.orders[order_id] = {
            "status": "PENDING",
            "filled_qty": 0.0,
            "target_qty": float(quantity),
        }

        async def _complete() -> None:
            await self._allow_fill.wait()
            order = self.orders[order_id]
            order["status"] = "FILLED"
            order["filled_qty"] = order["target_qty"]

        asyncio.create_task(_complete())
        return order_id

    def get_order_status(self, order_id: str) -> dict[str, float | str] | None:
        return self.orders.get(order_id)

    def release(self) -> None:
        self._allow_fill.set()


class StaticResourceMonitor:
    def __init__(self, snapshot: Mapping[str, object] | None = None) -> None:
        base_snapshot = {
            "timestamp": None,
            "cpu_percent": None,
            "memory_mb": None,
            "memory_percent": None,
        }
        if snapshot:
            base_snapshot.update(snapshot)
        self._snapshot = base_snapshot

    def update(self, **fields: object) -> None:
        self._snapshot.update(fields)

    def snapshot(self) -> Mapping[str, object]:
        return dict(self._snapshot)

    def sample(self) -> Mapping[str, object]:
        return self.snapshot()


def test_trading_manager_requires_risk_config() -> None:
    with pytest.raises(ValueError):
        TradingManager(
            event_bus=DummyBus(),
            strategy_registry=AlwaysActiveRegistry(),
            execution_engine=None,
        )


def test_trading_manager_accepts_custom_monitor_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    bus = DummyBus()
    throughput = ThroughputMonitor(window=32)
    backlog = EventBacklogTracker(threshold_ms=125.0, window=64)
    resource = ResourceUsageMonitor(process=None)

    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        throughput_monitor=throughput,
        backlog_tracker=backlog,
        resource_monitor=resource,
    )

    assert manager._throughput_monitor is throughput
    assert manager._backlog_tracker is backlog
    assert manager._resource_monitor is resource
    assert manager._throughput_monitor.window == 32
    assert manager._backlog_tracker.threshold_ms == pytest.approx(125.0)
    assert manager._backlog_tracker.window == 64

    stats = manager.get_execution_stats()
    assert stats["throughput"]["samples"] == 0
    assert stats["backlog"]["threshold_ms"] == pytest.approx(125.0)


def test_trading_manager_rejects_conflicting_monitor_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    bus = DummyBus()
    with pytest.raises(ValueError):
        TradingManager(
            event_bus=bus,
            strategy_registry=AlwaysActiveRegistry(),
            execution_engine=None,
            risk_config=RiskConfig(
                min_position_size=1,
                mandatory_stop_loss=False,
                research_mode=True,
            ),
            throughput_monitor=ThroughputMonitor(window=8),
            throughput_window=16,
        )

    with pytest.raises(ValueError):
        TradingManager(
            event_bus=bus,
            strategy_registry=AlwaysActiveRegistry(),
            execution_engine=None,
            risk_config=RiskConfig(
                min_position_size=1,
                mandatory_stop_loss=False,
                research_mode=True,
            ),
            backlog_tracker=EventBacklogTracker(threshold_ms=80.0),
            backlog_threshold_ms=100.0,
        )


def test_assess_performance_health_reports_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    now = datetime.now(tz=timezone.utc)
    throughput = ThroughputMonitor(window=16)
    throughput.record(
        started_at=now,
        finished_at=now + timedelta(milliseconds=120),
        ingested_at=now - timedelta(milliseconds=35),
    )
    throughput.record(
        started_at=now + timedelta(seconds=1),
        finished_at=now + timedelta(seconds=1, milliseconds=110),
        ingested_at=now + timedelta(seconds=1, milliseconds=-20),
    )

    backlog = EventBacklogTracker(threshold_ms=200.0)
    backlog.record(lag_ms=120.0, timestamp=now)

    resource_monitor = StaticResourceMonitor(
        {
            "timestamp": now.isoformat(),
            "cpu_percent": 42.0,
            "memory_mb": 512.0,
            "memory_percent": 36.0,
        }
    )

    manager = TradingManager(
        event_bus=DummyBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        throughput_monitor=throughput,
        backlog_tracker=backlog,
        resource_monitor=resource_monitor,
    )

    assessment = manager.assess_performance_health(
        max_processing_ms=200.0,
        max_lag_ms=200.0,
        backlog_threshold_ms=180.0,
        max_cpu_percent=75.0,
        max_memory_mb=1024.0,
        max_memory_percent=70.0,
    )

    assert assessment["healthy"] is True
    assert assessment["throughput"]["healthy"] is True
    assert assessment["backlog"]["healthy"] is True
    resource_result = assessment["resource"]
    assert resource_result["healthy"] is True
    assert resource_result["status"] == "ok"
    assert set(resource_result["evaluated_metrics"]) == {
        "max_cpu_percent",
        "max_memory_mb",
        "max_memory_percent",
    }
    assert resource_result["violations"] == {}
    assert "throttle" not in assessment


def test_generate_performance_health_report_renders_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    now = datetime.now(tz=timezone.utc)
    throughput = ThroughputMonitor(window=16)
    throughput.record(
        started_at=now,
        finished_at=now + timedelta(milliseconds=120),
        ingested_at=now - timedelta(milliseconds=35),
    )
    backlog = EventBacklogTracker(threshold_ms=200.0)
    backlog.record(lag_ms=120.0, timestamp=now)
    resource_monitor = StaticResourceMonitor(
        {
            "timestamp": now.isoformat(),
            "cpu_percent": 42.0,
            "memory_mb": 512.0,
            "memory_percent": 36.0,
        }
    )

    manager = TradingManager(
        event_bus=DummyBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        throughput_monitor=throughput,
        backlog_tracker=backlog,
        resource_monitor=resource_monitor,
    )

    report = manager.generate_performance_health_report(
        max_processing_ms=200.0,
        max_lag_ms=200.0,
        backlog_threshold_ms=180.0,
        max_cpu_percent=75.0,
        max_memory_mb=1024.0,
        max_memory_percent=70.0,
    )

    assert "# Performance health assessment" in report
    assert "Overall status: Healthy" in report
    assert "## Throughput" in report
    assert "## Backlog" in report
    assert "## Resource utilisation" in report


def test_assess_performance_health_flags_violations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    now = datetime.now(tz=timezone.utc)
    throughput = ThroughputMonitor(window=8)
    throughput.record(
        started_at=now,
        finished_at=now + timedelta(milliseconds=220),
        ingested_at=now - timedelta(milliseconds=10),
    )

    backlog = EventBacklogTracker(threshold_ms=90.0)
    backlog.record(lag_ms=240.0, timestamp=now)

    resource_monitor = StaticResourceMonitor(
        {
            "timestamp": now.isoformat(),
            "cpu_percent": 92.0,
            "memory_mb": 4096.0,
            "memory_percent": 88.0,
        }
    )

    manager = TradingManager(
        event_bus=DummyBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        throughput_monitor=throughput,
        backlog_tracker=backlog,
        resource_monitor=resource_monitor,
    )

    assessment = manager.assess_performance_health(
        max_processing_ms=120.0,
        max_lag_ms=100.0,
        backlog_threshold_ms=150.0,
        max_cpu_percent=75.0,
        max_memory_mb=1024.0,
        max_memory_percent=65.0,
    )

    assert assessment["healthy"] is False
    assert assessment["throughput"]["healthy"] is False
    assert assessment["backlog"]["healthy"] is False
    resource_result = assessment["resource"]
    assert resource_result["status"] == "violated"
    assert resource_result["healthy"] is False
    assert {
        "max_cpu_percent",
        "max_memory_mb",
        "max_memory_percent",
    } <= set(resource_result["violations"].keys())
    assert "throttle" not in assessment


@pytest.mark.asyncio()
async def test_assess_performance_health_surfaces_throttle_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        trade_throttle={
            "max_trades": 1,
            "window_seconds": 60.0,
            "scope_fields": ("strategy_id", "symbol"),
        },
    )
    engine = RecordingExecutionEngine()
    manager.execution_engine = engine

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2105,
        confidence=0.9,
        strategy_id="alpha",
    )
    setattr(intent, "event_id", "throttle-meta-1")

    validate_mock: AsyncMock = AsyncMock(side_effect=[intent, intent])
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    await manager.on_trade_intent(intent)
    await manager.on_trade_intent(intent)

    assessment = manager.assess_performance_health(
        max_processing_ms=1_000.0,
        max_lag_ms=1_000.0,
    )

    throttle_summary = assessment.get("throttle")
    assert isinstance(throttle_summary, Mapping)
    assert throttle_summary.get("state") in {
        "open",
        "rate_limited",
        "cooldown",
        "min_interval",
    }
    assert "remaining_trades" in throttle_summary
    assert throttle_summary["remaining_trades"] is not None
    assert "window_utilisation" in throttle_summary
    utilisation = throttle_summary["window_utilisation"]
    assert utilisation is None or 0.0 <= float(utilisation) <= 1.0
    assert "retry_in_seconds" in throttle_summary
    retry_seconds = throttle_summary["retry_in_seconds"]
    assert retry_seconds is None or float(retry_seconds) >= 0.0
    assert "window_reset_in_seconds" in throttle_summary
    window_reset = throttle_summary["window_reset_in_seconds"]
    assert window_reset is None or float(window_reset) >= 0.0
    context = throttle_summary.get("context")
    assert isinstance(context, Mapping)
    assert context.get("symbol") == "EURUSD"
    scope = throttle_summary.get("scope")
    assert isinstance(scope, Mapping)
    assert scope.get("strategy_id") == "alpha"
    assert "scope_key" in throttle_summary
    assert isinstance(throttle_summary["scope_key"], list)

@pytest.mark.asyncio()
async def test_trading_manager_records_execution_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
    )
    engine = ImmediateFillExecutionAdapter(manager.portfolio_monitor)
    manager.execution_engine = engine

    intent = SimpleIntent(symbol="EURUSD", quantity=1.0, price=1.2345)
    await manager.on_trade_intent(intent)

    stats = manager.get_execution_stats()
    assert stats["orders_submitted"] == 1
    assert stats["orders_executed"] == 1
    assert stats.get("avg_latency_ms") is not None
    assert stats.get("pending_orders", 0) == 0
    assert stats.get("fills") == 1
    resource_usage = stats.get("resource_usage")
    assert isinstance(resource_usage, dict)
    assert set(resource_usage.keys()) == {
        "timestamp",
        "cpu_percent",
        "memory_mb",
        "memory_percent",
    }
    backlog = stats.get("backlog")
    assert isinstance(backlog, dict)
    assert "healthy" in backlog

    summary = manager.get_strategy_execution_summary()
    assert summary.get("unknown", {}).get("executed") == 1


@pytest.mark.asyncio()
async def test_trading_manager_records_experiment_events_and_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        min_intent_confidence=0.6,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
    )
    engine = ImmediateFillExecutionAdapter(manager.portfolio_monitor)
    manager.execution_engine = engine

    accepted = ConfidenceIntent(symbol="EURUSD", quantity=1.0, price=1.2010, confidence=0.9)
    rejected = ConfidenceIntent(symbol="EURUSD", quantity=1.0, price=1.1995, confidence=0.1)

    await manager.on_trade_intent(accepted)
    await manager.on_trade_intent(rejected)

    events = manager.get_experiment_events()
    assert events, "expected experiment events to be recorded"
    statuses = {event["status"] for event in events}
    assert "executed" in statuses
    assert "rejected" in statuses

    summary = manager.get_strategy_execution_summary()
    alpha_stats = summary.get("alpha")
    assert alpha_stats is not None
    assert alpha_stats["executed"] >= 1
    assert alpha_stats["rejected"] >= 1


@pytest.mark.asyncio()
async def test_trading_manager_gates_on_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    gate = DriftSentryGate(warn_confidence_floor=0.75)
    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        drift_gate=gate,
    )
    execution_engine = RecordingExecutionEngine()
    manager.execution_engine = execution_engine

    dimension = SensoryDimensionDrift(
        name="WHY",
        current_signal=0.42,
        baseline_signal=0.10,
        delta=0.32,
        current_confidence=0.76,
        baseline_confidence=0.70,
        confidence_delta=0.06,
        severity=DriftSeverity.warn,
        samples=6,
    )
    snapshot = SensoryDriftSnapshot(
        generated_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
        status=DriftSeverity.warn,
        dimensions={"WHY": dimension},
        sample_window=6,
        metadata={"source": "test"},
    )
    manager.update_drift_sentry_snapshot(snapshot)

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2010,
        confidence=0.4,
    )

    validate_mock: AsyncMock = AsyncMock(return_value=intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    await manager.on_trade_intent(intent)

    assert validate_mock.await_count == 1
    assert execution_engine.calls == 1

    events = manager.get_experiment_events()
    assert events, "expected experiment events to be recorded"
    latest = events[0]
    assert latest["status"] == "executed"
    metadata = latest.get("metadata")
    assert isinstance(metadata, dict)
    gate_payload = metadata.get("drift_gate")
    assert isinstance(gate_payload, dict)
    assert gate_payload.get("allowed") is False
    assert gate_payload.get("severity") == DriftSeverity.warn.value
    assert gate_payload.get("force_paper") is True
    forced_events = [event for event in events if event["status"] == "forced_paper"]
    assert forced_events, "expected forced paper gating event"
    forced_meta = forced_events[0].get("metadata")
    assert isinstance(forced_meta, dict)
    assert forced_meta.get("forced_paper") is True
    gate_metadata = forced_events[0].get("metadata")
    assert isinstance(gate_metadata, dict)
    assert gate_metadata.get("forced_paper") is True
    assert "confidence" in str(gate_metadata.get("reason"))
    decision = manager.get_last_drift_gate_decision()
    assert decision is not None
    assert not decision.allowed
    assert decision.severity is DriftSeverity.warn
    assert decision.force_paper is True

    assert bus.events, "expected drift gate telemetry"
    forced_event = next(
        event
        for event in bus.events
        if event.type == "telemetry.trading.drift_gate"
        and event.payload.get("status") == "forced_paper"
    )
    payload = forced_event.payload
    assert payload["status"] == "forced_paper"
    assert payload["decision"]["force_paper"] is True


@pytest.mark.asyncio()
async def test_trading_manager_records_gate_metadata_on_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    gate = DriftSentryGate(warn_confidence_floor=0.6)
    dimension = SensoryDimensionDrift(
        name="WHY",
        current_signal=0.42,
        baseline_signal=0.10,
        delta=0.32,
        current_confidence=0.76,
        baseline_confidence=0.70,
        confidence_delta=0.06,
        severity=DriftSeverity.warn,
        samples=6,
    )
    snapshot = SensoryDriftSnapshot(
        generated_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
        status=DriftSeverity.warn,
        dimensions={"WHY": dimension},
        sample_window=6,
        metadata={"source": "test"},
    )
    gate.update_snapshot(snapshot)

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        drift_gate=gate,
    )
    execution_engine = RecordingExecutionEngine()
    manager.execution_engine = execution_engine

    validated_intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2010,
        confidence=0.95,
        strategy_id="alpha",
    )

    validate_mock: AsyncMock = AsyncMock(return_value=validated_intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    await manager.on_trade_intent(validated_intent)

    events = manager.get_experiment_events()
    assert events, "expected executed event"
    latest = events[0]
    assert latest["status"] == "executed"
    metadata = latest.get("metadata")
    assert isinstance(metadata, dict)
    gate_payload = metadata.get("drift_gate")
    assert isinstance(gate_payload, dict)
    assert gate_payload.get("allowed") is True
    assert gate_payload.get("severity") == DriftSeverity.warn.value
    assert gate_payload.get("force_paper") is True
    thresholds = gate_payload.get("requirements", {})
    assert thresholds.get("confidence_floor") == pytest.approx(0.6)
    assert execution_engine.calls == 1


@pytest.mark.asyncio()
async def test_trading_manager_attaches_gate_metadata_to_fills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    gate = DriftSentryGate(warn_confidence_floor=0.6)
    dimension = SensoryDimensionDrift(
        name="WHEN",
        current_signal=0.30,
        baseline_signal=0.12,
        delta=0.18,
        current_confidence=0.80,
        baseline_confidence=0.70,
        confidence_delta=0.10,
        severity=DriftSeverity.warn,
        samples=8,
    )
    snapshot = SensoryDriftSnapshot(
        generated_at=datetime(2024, 2, 2, tzinfo=timezone.utc),
        status=DriftSeverity.warn,
        dimensions={"WHEN": dimension},
        sample_window=8,
        metadata={"source": "test"},
    )
    gate.update_snapshot(snapshot)

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=75_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        drift_gate=gate,
    )
    engine = ImmediateFillExecutionAdapter(manager.portfolio_monitor)
    manager.execution_engine = engine

    validated_intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=2.0,
        price=1.2050,
        confidence=0.92,
        strategy_id="paper-alpha",
    )

    validate_mock: AsyncMock = AsyncMock(return_value=validated_intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    await manager.on_trade_intent(validated_intent)

    assert engine.fills, "expected paper execution to record a fill"
    fill = engine.fills[-1]
    metadata = fill.get("metadata")
    assert isinstance(metadata, dict)
    payload = metadata.get("drift_gate")
    assert isinstance(payload, dict)
    assert payload.get("allowed") is True
    assert payload.get("severity") == DriftSeverity.warn.value
    assert payload.get("force_paper") is True
    assert validate_mock.await_count == 1


@pytest.mark.asyncio()
async def test_trading_manager_release_thresholds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    gate = DriftSentryGate(warn_confidence_floor=0.6)
    store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PAPER,
        threshold_overrides={"warn_confidence_floor": 0.82},
        evidence_id="diary-alpha",
    )

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        drift_gate=gate,
        release_manager=release_manager,
    )
    execution_engine = RecordingExecutionEngine()
    manager.execution_engine = execution_engine

    dimension = SensoryDimensionDrift(
        name="WHY",
        current_signal=0.42,
        baseline_signal=0.10,
        delta=0.32,
        current_confidence=0.76,
        baseline_confidence=0.70,
        confidence_delta=0.06,
        severity=DriftSeverity.warn,
        samples=6,
    )
    snapshot = SensoryDriftSnapshot(
        generated_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
        status=DriftSeverity.warn,
        dimensions={"WHY": dimension},
        sample_window=6,
        metadata={"source": "test"},
    )
    manager.update_drift_sentry_snapshot(snapshot)

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2010,
        confidence=0.8,
        strategy_id="alpha",
    )

    validate_mock: AsyncMock = AsyncMock(return_value=intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    await manager.on_trade_intent(intent)

    decision = manager.get_last_drift_gate_decision()
    assert decision is not None
    assert not decision.allowed, "confidence below release-managed floor should block"
    assert decision.force_paper is True
    assert "0.870" in (decision.reason or "")
    requirements = dict(decision.requirements)
    assert requirements["release_stage"] == PolicyLedgerStage.PAPER.value
    assert requirements["confidence_floor"] == pytest.approx(0.87, rel=1e-6)
    assert requirements.get("warn_notional_limit") == pytest.approx(35_625.0)
    assert validate_mock.await_count == 1
    assert execution_engine.calls == 1
    events = manager.get_experiment_events()
    assert events, "expected experiment events"
    executed = events[0]
    assert executed["status"] == "executed"
    metadata = executed.get("metadata")
    assert isinstance(metadata, dict)
    gate_payload = metadata.get("drift_gate")
    assert isinstance(gate_payload, dict)
    gate_requirements = gate_payload.get("requirements", {})
    assert gate_payload.get("force_paper") is True
    assert gate_requirements.get("release_stage") == PolicyLedgerStage.PAPER.value
    assert gate_requirements.get("confidence_floor") == pytest.approx(0.87, rel=1e-6)
    assert gate_requirements.get("warn_notional_limit") == pytest.approx(35_625.0)
    forced_events = [event for event in events if event["status"] == "forced_paper"]
    assert forced_events, "expected forced paper gating event"


@pytest.mark.asyncio()
async def test_trading_manager_forces_paper_experiment_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    gate = DriftSentryGate(warn_confidence_floor=0.4)
    store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    release_manager = LedgerReleaseManager(store)

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        drift_gate=gate,
        release_manager=release_manager,
    )
    paper_engine = RecordingExecutionEngine()
    manager.execution_engine = paper_engine
    live_engine = RecordingExecutionEngine()
    router = manager.install_release_execution_router(live_engine=live_engine)

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2010,
        confidence=0.9,
        strategy_id="alpha",
    )

    validate_mock: AsyncMock = AsyncMock(return_value=intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    snapshot = SensoryDriftSnapshot(
        generated_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
        status=DriftSeverity.normal,
        dimensions={},
        sample_window=6,
        metadata={"source": "test"},
    )
    manager.update_drift_sentry_snapshot(snapshot)

    await manager.on_trade_intent(intent)

    decision = manager.get_last_drift_gate_decision()
    assert decision is not None
    assert decision.allowed is True
    assert decision.reason == "release_stage_experiment_requires_paper_or_better"
    assert decision.force_paper is True
    assert decision.requirements.get("release_stage") == PolicyLedgerStage.EXPERIMENT.value

    events = manager.get_experiment_events()
    assert events, "Expected experiment event to be recorded"
    event = events[0]
    assert event["status"] == "executed"
    metadata = event.get("metadata")
    assert isinstance(metadata, dict)
    gate_payload = metadata.get("drift_gate")
    assert isinstance(gate_payload, dict)
    assert gate_payload.get("allowed") is True
    assert gate_payload.get("force_paper") is True
    assert (
        gate_payload.get("reason")
        == "release_stage_experiment_requires_paper_or_better"
    )

    assert validate_mock.await_count == 1
    assert paper_engine.calls == 1
    assert live_engine.calls == 0
    last_route = router.last_route()
    assert last_route is not None
    assert last_route.get("forced_reason") == "release_stage_experiment_requires_paper_or_better"
    assert last_route.get("route") == "paper"


@pytest.mark.asyncio()
async def test_trading_manager_forces_paper_without_audit_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot",
        _noop,
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    gate = DriftSentryGate(warn_confidence_floor=0.2)
    store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    release_manager = LedgerReleaseManager(
        store,
        feature_flags=PolicyLedgerFeatureFlags(require_diary_evidence=False),
    )
    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
    )

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        drift_gate=gate,
        release_manager=release_manager,
    )
    paper_engine = RecordingExecutionEngine()
    manager.execution_engine = paper_engine
    live_engine = RecordingExecutionEngine()
    router = manager.install_release_execution_router(live_engine=live_engine)

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2050,
        confidence=0.95,
        strategy_id="alpha",
    )

    validate_mock: AsyncMock = AsyncMock(return_value=intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    await manager.on_trade_intent(intent)

    decision = manager.get_last_drift_gate_decision()
    assert decision is not None
    assert decision.allowed is True
    assert decision.force_paper is True
    assert decision.reason == "release_stage_experiment_requires_paper_or_better"
    assert validate_mock.await_count == 1
    assert paper_engine.calls == 1
    assert live_engine.calls == 0
    last_route = router.last_route()
    assert last_route is not None
    assert last_route.get("forced_reason") == "release_stage_experiment_requires_paper_or_better"
    assert last_route.get("route") == "paper"
    summary = manager.describe_release_posture("alpha")
    assert summary["stage"] == PolicyLedgerStage.EXPERIMENT.value
    assert summary["declared_stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert summary.get("audit_enforced") is True
    assert "missing_evidence" in summary.get("audit_gaps", [])
    events = manager.get_experiment_events()
    assert events and events[0]["status"] == "executed"
    metadata = events[0].get("metadata")
    assert isinstance(metadata, dict)
    gate_payload = metadata.get("drift_gate")
    assert isinstance(gate_payload, dict)
    assert gate_payload.get("force_paper") is True


@pytest.mark.asyncio()
async def test_trading_manager_auto_installs_release_router(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    gate = DriftSentryGate(warn_confidence_floor=0.6)
    store = PolicyLedgerStore(tmp_path / "policy_auto.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("ops", "risk"),
        evidence_id="diary-alpha",
    )

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        drift_gate=gate,
        release_manager=release_manager,
    )

    paper_engine = RecordingExecutionEngine()
    manager.execution_engine = paper_engine

    router = manager.execution_engine
    assert isinstance(router, ReleaseAwareExecutionRouter)
    assert router.paper_engine is paper_engine

    dimension = SensoryDimensionDrift(
        name="HOW",
        current_signal=0.9,
        baseline_signal=0.6,
        delta=0.3,
        current_confidence=0.82,
        baseline_confidence=0.8,
        confidence_delta=0.02,
        severity=DriftSeverity.warn,
        samples=12,
    )
    snapshot = SensoryDriftSnapshot(
        generated_at=datetime(2024, 4, 5, tzinfo=timezone.utc),
        status=DriftSeverity.warn,
        dimensions={"HOW": dimension},
        sample_window=12,
        metadata={"source": "auto"},
    )
    manager.update_drift_sentry_snapshot(snapshot)

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2210,
        confidence=0.92,
        strategy_id="alpha",
    )

    validate_mock: AsyncMock = AsyncMock(return_value=intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    await manager.on_trade_intent(intent)

    assert paper_engine.calls == 1
    assert validate_mock.await_count == 1

    last_route = manager.get_last_release_route()
    assert last_route is not None
    assert last_route.get("route") == "paper"
    assert last_route.get("forced_reason") == "drift_gate_severity_warn"

    events = manager.get_experiment_events()
    assert events, "expected experiment event recorded"
    metadata = events[0].get("metadata")
    assert isinstance(metadata, dict)
    release_metadata = metadata.get("release_execution")
    assert isinstance(release_metadata, dict)
    assert release_metadata.get("forced") is True
    assert release_metadata.get("forced_reason") == "drift_gate_severity_warn"


def test_describe_release_posture(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="diary-alpha",
    )

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=10_000.0,
        drift_gate=None,
        release_manager=release_manager,
        risk_config=RiskConfig(),
    )

    summary = manager.describe_release_posture("alpha")
    assert summary["stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert summary["managed"] is True
    assert summary["declared_stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert summary.get("audit_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert summary.get("audit_enforced") is False
    assert summary.get("audit_gaps", []) == []
    assert "thresholds" in summary
    thresholds = summary["thresholds"]
    assert thresholds.get("adaptive_source") == DriftSeverity.normal.value
    assert summary.get("approvals") == ["ops", "risk"]
    assert "last_route" not in summary


def test_build_policy_governance_snapshot(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "policy_ledger.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk", "compliance"),
        evidence_id="diary-alpha",
        policy_delta=PolicyDelta(
            regime="balanced",
            risk_config={"max_leverage": 6},
            router_guardrails={"requires_diary": True},
        ),
    )

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=10_000.0,
        drift_gate=None,
        release_manager=release_manager,
        risk_config=RiskConfig(),
    )

    snapshot = manager.build_policy_governance_snapshot()
    assert snapshot is not None
    assert snapshot.metadata.get("policy_count") == 1
    checklist = next(
        workflow for workflow in snapshot.workflows if workflow.name == "Policy Ledger Governance"
    )
    assert checklist.status in {
        WorkflowTaskStatus.completed,
        WorkflowTaskStatus.in_progress,
    }
    assert checklist.tasks
    first_task = checklist.tasks[0]
    assert first_task.metadata["policy_id"] == "alpha"
    assert "diary-alpha" in first_task.summary


@pytest.mark.asyncio()
async def test_install_release_execution_router(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "policy.json")
    release_manager = LedgerReleaseManager(store)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        release_manager=release_manager,
    )

    paper_engine = RecordingExecutionEngine()
    manager.execution_engine = paper_engine
    live_engine = RecordingExecutionEngine()

    router = manager.install_release_execution_router(live_engine=live_engine)
    assert manager.execution_engine is router

    intent = ConfidenceIntent(symbol="EURUSD", quantity=1.0, price=1.1, confidence=0.8)
    await router.process_order(intent)
    assert paper_engine.calls == 1
    assert live_engine.calls == 0

    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("ops", "risk"),
        evidence_id="diary-alpha",
    )

    await router.process_order(intent)
    assert live_engine.calls == 1
    last_route = router.last_route()
    assert last_route is not None
    assert last_route["stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert last_route["route"] == "live"


@pytest.mark.asyncio()
async def test_release_router_forces_paper_on_drift_warn(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "policy_forced.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("ops", "risk"),
        evidence_id="diary-alpha",
    )

    paper_engine = RecordingExecutionEngine()
    live_engine = RecordingExecutionEngine()
    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
        live_engine=live_engine,
    )

    intent = ConfidenceIntent(symbol="EURUSD", quantity=1.0, price=1.2, confidence=0.85)
    intent.metadata = {
        "drift_gate": {
            "allowed": True,
            "severity": DriftSeverity.warn.value,
            "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
            "blocked_dimensions": [],
            "requirements": {},
            "snapshot_metadata": {},
            "force_paper": True,
        }
    }

    await router.process_order(intent)

    assert paper_engine.calls == 1
    assert live_engine.calls == 0
    metadata = getattr(intent, "metadata", {})
    assert metadata.get("release_execution_route") == "paper"
    assert metadata.get("release_execution_forced") == "drift_gate_severity_warn"
    forced_reasons = metadata.get("release_execution_forced_reasons")
    assert forced_reasons == ["drift_gate_severity_warn"]
    audit_meta = metadata.get("release_execution_audit")
    assert isinstance(audit_meta, dict)
    assert audit_meta.get("declared_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert audit_meta.get("audit_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    last_route = router.last_route()
    assert last_route is not None
    assert last_route.get("stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert last_route.get("route") == "paper"
    assert last_route.get("forced_reason") == "drift_gate_severity_warn"
    assert last_route.get("forced_reasons") == ["drift_gate_severity_warn"]
    assert last_route.get("drift_severity") == DriftSeverity.warn.value
    audit_payload = last_route.get("audit")
    assert isinstance(audit_payload, dict)
    assert audit_payload.get("declared_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert audit_payload.get("audit_stage") == PolicyLedgerStage.LIMITED_LIVE.value


@pytest.mark.asyncio()
async def test_trading_manager_forces_paper_execution_under_drift_warn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    gate = DriftSentryGate(warn_confidence_floor=0.6)
    store = PolicyLedgerStore(tmp_path / "policy_forced_manager.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("ops", "risk"),
        evidence_id="diary-alpha",
    )

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        drift_gate=gate,
        release_manager=release_manager,
    )

    paper_engine = RecordingExecutionEngine()
    manager.execution_engine = paper_engine
    live_engine = RecordingExecutionEngine()
    router = manager.install_release_execution_router(live_engine=live_engine)

    dimension = SensoryDimensionDrift(
        name="HOW",
        current_signal=0.9,
        baseline_signal=0.6,
        delta=0.3,
        current_confidence=0.82,
        baseline_confidence=0.8,
        confidence_delta=0.02,
        severity=DriftSeverity.warn,
        samples=12,
    )
    snapshot = SensoryDriftSnapshot(
        generated_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
        status=DriftSeverity.warn,
        dimensions={"HOW": dimension},
        sample_window=12,
        metadata={"source": "test"},
    )
    manager.update_drift_sentry_snapshot(snapshot)

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2345,
        confidence=0.9,
        strategy_id="alpha",
    )

    validate_mock: AsyncMock = AsyncMock(return_value=intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    await manager.on_trade_intent(intent)

    assert paper_engine.calls == 1
    assert live_engine.calls == 0
    last_route = router.last_route()
    assert last_route is not None
    assert last_route.get("forced_reason") == "drift_gate_severity_warn"
    assert last_route.get("route") == "paper"
    decision = manager.get_last_drift_gate_decision()
    assert decision is not None
    assert decision.allowed is True
    assert decision.severity is DriftSeverity.warn
    events = manager.get_experiment_events()
    assert events, "Expected experiment event recorded"
    metadata = events[0].get("metadata")
    assert isinstance(metadata, dict)
    gate_payload = metadata.get("drift_gate")
    assert isinstance(gate_payload, dict)
    assert gate_payload.get("allowed") is True
    assert gate_payload.get("severity") == DriftSeverity.warn.value
    release_metadata = metadata.get("release_execution")
    assert isinstance(release_metadata, dict)
    assert release_metadata.get("stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert release_metadata.get("route") == "paper"
    assert release_metadata.get("forced") is True
    assert release_metadata.get("forced_reason") == "drift_gate_severity_warn"
    assert release_metadata.get("forced_reasons") == ["drift_gate_severity_warn"]
    audit_section = release_metadata.get("audit")
    assert isinstance(audit_section, dict)
    assert audit_section.get("declared_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert audit_section.get("audit_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    posture = manager.describe_release_posture("alpha")
    assert posture.get("stage") == PolicyLedgerStage.LIMITED_LIVE.value
    last_posture_route = posture.get("last_route")
    assert isinstance(last_posture_route, dict)
    assert last_posture_route.get("route") == "paper"
    assert last_posture_route.get("forced_reason") == "drift_gate_severity_warn"
    assert last_posture_route.get("forced_reasons") == ["drift_gate_severity_warn"]
    audit_payload = last_posture_route.get("audit")
    assert isinstance(audit_payload, dict)
    assert audit_payload.get("declared_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert audit_payload.get("audit_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert manager.get_last_release_route() == last_posture_route

    assert bus.events, "expected drift gate telemetry"
    drift_event = next(
        event
        for event in reversed(bus.events)
        if event.type == "telemetry.trading.drift_gate"
    )
    payload = drift_event.payload
    assert payload["status"] == "executed"
    assert payload["decision"]["force_paper"] is True
    assert payload.get("release", {}).get("forced_reason") == "drift_gate_severity_warn"

    release_event = next(
        event
        for event in reversed(bus.events)
        if event.type == "telemetry.trading.release_route"
    )
    release_payload = release_event.payload
    assert release_payload["status"] == "executed"
    assert release_payload["forced"] is True
    assert release_payload["forced_reason"] == "drift_gate_severity_warn"


@pytest.mark.asyncio()
async def test_trading_manager_liquidity_prober_captures_risk_metadata() -> None:
    supervisor = TaskSupervisor(namespace="test-manager-liquidity-meta")
    broker = _ImmediateProbeBroker()
    prober = LiquidityProber(broker)
    manager = TradingManager(
        event_bus=RecordingBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=RecordingExecutionEngine(),
        initial_equity=50_000.0,
        liquidity_prober=prober,
        task_supervisor=supervisor,
        risk_config=RiskConfig(min_position_size=1),
    )

    decision = {"checks": []}
    intent = {"symbol": "EURUSD", "side": "buy", "price": 1.1234}
    await manager.risk_gateway._run_liquidity_probe(
        intent,
        Decimal("1"),
        decision,
        market_price=1.1234,
        portfolio_state={},
    )

    context = manager.risk_gateway.liquidity_prober.describe_risk_context()
    assert context["runbook"] == RISK_API_RUNBOOK
    assert context["risk_api_runbook"] == RISK_API_RUNBOOK
    metadata = context.get("metadata")
    assert metadata is not None
    assert metadata["max_risk_per_trade_pct"] > 0
    await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_trading_manager_liquidity_prober_uses_task_supervisor() -> None:
    supervisor = TaskSupervisor(namespace="test-manager-liquidity-supervisor")
    broker = _BlockingProbeBroker()
    prober = LiquidityProber(
        broker,
        config={"max_concurrent_probes": 2, "timeout_seconds": 0.5},
    )
    manager = TradingManager(
        event_bus=RecordingBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=RecordingExecutionEngine(),
        initial_equity=25_000.0,
        liquidity_prober=prober,
        task_supervisor=supervisor,
        risk_config=RiskConfig(),
    )

    probe_task = asyncio.create_task(
        manager.risk_gateway.liquidity_prober.probe_liquidity(
            "EURUSD", [1.101, 1.102, 1.103], "buy"
        )
    )
    try:
        await asyncio.sleep(0.05)
        assert supervisor.active_count >= 1
        broker.release()
        await probe_task
    finally:
        broker.release()
        await supervisor.cancel_all()
    assert supervisor.active_count == 0


@pytest.mark.asyncio()
async def test_trading_manager_emits_policy_violation_alert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    captured: list[tuple[Any, Any, str]] = []

    async def _capture(event_bus: RecordingBus, alert, *, source: str) -> None:
        captured.append((alert, source, len(event_bus.events)))
        await event_bus.publish({"alert": alert, "source": source})

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _capture)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(
            min_position_size=100,
            mandatory_stop_loss=False,
            research_mode=False,
        ),
    )

    intent = SimpleIntent(symbol="EURUSD", quantity=1.0, price=1.2222)
    await manager.on_trade_intent(intent)

    assert captured, "expected policy violation alert"
    alert, source, _ = captured[-1]
    assert source == "trading_manager"
    assert not alert.snapshot.approved
    assert "policy.min_position_size" in alert.snapshot.violations
    assert bus.events, "expected violation to publish on event bus"


@pytest.mark.asyncio()
async def test_trading_manager_emits_risk_interface_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    captured: list[tuple[Any, str]] = []

    async def _capture(event_bus: RecordingBus, snapshot, *, source: str) -> None:
        captured.append((snapshot, source))
        await event_bus.publish({"snapshot": snapshot.as_dict(), "source": source})

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _capture
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = RecordingBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(),
    )
    engine = ImmediateFillExecutionAdapter(manager.portfolio_monitor)
    manager.execution_engine = engine

    intent = SimpleIntent(symbol="EURUSD", quantity=1.0, price=1.2050)
    await manager.on_trade_intent(intent)

    assert captured, "expected risk interface snapshot"
    snapshot, source = captured[-1]
    assert source == "trading_manager"
    assert snapshot.summary["mandatory_stop_loss"] is True
    assert manager.get_last_risk_interface_snapshot() is not None
    assert manager.get_last_risk_interface_error() is None


@pytest.mark.asyncio()
async def test_trading_manager_emits_risk_interface_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    captured: list[tuple[Any, str]] = []

    async def _capture(event_bus: RecordingBus, alert, *, source: str) -> None:
        captured.append((alert, source))
        await event_bus.publish({"alert": alert.as_dict(), "source": source})

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_error", _capture
    )

    class BrokenTradingManager(TradingManager):
        def __init__(self) -> None:
            super().__init__(
                event_bus=RecordingBus(),
                strategy_registry=AlwaysActiveRegistry(),
                execution_engine=None,
                risk_config=RiskConfig(),
            )
            self._risk_config = None  # type: ignore[assignment]

        def get_risk_status(self) -> dict[str, object]:
            return {"risk_config": {"max_risk_per_trade_pct": -1}}

    bus = RecordingBus()
    manager = BrokenTradingManager()
    manager.event_bus = bus  # override to shared recording bus
    engine = ImmediateFillExecutionAdapter(manager.portfolio_monitor)
    manager.execution_engine = engine

    intent = SimpleIntent(symbol="EURUSD", quantity=1.0, price=1.2000)
    await manager.on_trade_intent(intent)

    assert captured, "expected risk interface error alert"
    alert, source = captured[-1]
    assert source == "trading_manager"
    assert alert.runbook.endswith("risk_api_contract.md")
    assert manager.get_last_risk_interface_snapshot() is None
    stored_alert = manager.get_last_risk_interface_error()
    assert stored_alert is not None
    assert stored_alert.runbook.endswith("risk_api_contract.md")
    assert bus.events, "expected alert to publish on event bus"


@pytest.mark.asyncio()
async def test_trading_manager_routes_limited_live_to_live_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = RecordingBus()
    gate = DriftSentryGate(warn_confidence_floor=0.1)

    store = PolicyLedgerStore(tmp_path / "limited_live.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="dd-alpha-live",
    )

    paper_engine = RecordingExecutionEngine()
    live_engine = RecordingExecutionEngine()

    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=paper_engine,
        pilot_execution_engine=paper_engine,
        live_execution_engine=live_engine,
        initial_equity=75_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        drift_gate=gate,
        release_manager=release_manager,
    )

    validate_mock: AsyncMock = AsyncMock()
    validated_intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2345,
        confidence=0.85,
    )
    validate_mock.return_value = validated_intent
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    snapshot = SensoryDriftSnapshot(
        generated_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        status=DriftSeverity.normal,
        dimensions={},
        sample_window=12,
        metadata={"source": "test"},
    )
    manager.update_drift_sentry_snapshot(snapshot)

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2345,
        confidence=0.85,
    )

    await manager.on_trade_intent(intent)

    assert live_engine.calls == 1
    assert paper_engine.calls == 0

    last_route = manager.get_last_release_route()
    assert last_route is not None
    assert last_route["stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert last_route["route"] == "live"


def test_record_experiment_event_handles_non_mapping_inputs() -> None:
    manager = TradingManager(
        event_bus=DummyBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=10_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
    )

    manager._record_experiment_event(  # type: ignore[attr-defined]
        event_id="exp-1",
        status="executed",
        strategy_id="alpha",
        symbol="EURUSD",
        confidence=0.5,
        metadata="unexpected",  # type: ignore[arg-type]
        decision="not-a-mapping",  # type: ignore[arg-type]
    )

    events = manager.get_experiment_events()
    assert events
    recorded = events[0]
    assert recorded["status"] == "executed"
    assert "metadata" not in recorded
    assert "decision" not in recorded


def test_describe_risk_interface_exposes_canonical_payload() -> None:
    manager = TradingManager(
        event_bus=DummyBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(),
    )

    description = manager.describe_risk_interface()

    assert "config" in description
    assert description["summary"]["mandatory_stop_loss"] is True
    assert description["summary"]["max_risk_per_trade_pct"] == pytest.approx(
        float(manager._risk_config.max_risk_per_trade_pct)  # type: ignore[attr-defined]
    )
    assert "policy_limits" in description["status"]
    reference = description.get("risk_reference")
    assert isinstance(reference, dict)
    assert reference.get("risk_api_runbook", "").endswith("risk_api_contract.md")
    reference_summary = reference.get("risk_config_summary")
    assert isinstance(reference_summary, dict)
    assert reference_summary["mandatory_stop_loss"] is True
    assert description.get("runbook", "").endswith("risk_api_contract.md")


def test_apply_risk_config_refreshes_gateway_and_policy() -> None:
    manager = TradingManager(
        event_bus=DummyBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(),
    )

    updated = manager.apply_risk_config(
        RiskConfig(
            max_risk_per_trade_pct=Decimal("0.03"),
            max_total_exposure_pct=Decimal("0.60"),
            max_drawdown_pct=Decimal("0.12"),
            mandatory_stop_loss=False,
            research_mode=True,
        )
    )

    assert updated.max_risk_per_trade_pct == Decimal("0.03")
    assert manager._risk_policy is not None  # type: ignore[attr-defined]
    assert manager._risk_policy.research_mode is True  # type: ignore[attr-defined]

    description = manager.describe_risk_interface()
    summary = description["summary"]
    assert summary["max_risk_per_trade_pct"] == pytest.approx(0.03)
    assert summary["max_drawdown_pct"] == pytest.approx(0.12)
    assert summary["mandatory_stop_loss"] is False

    limits_payload = manager.risk_gateway.get_risk_limits()
    assert limits_payload["limits"]["max_risk_per_trade_pct"] == pytest.approx(0.03)
    assert limits_payload["risk_config_summary"]["research_mode"] is True

    risk_manager_config = manager._portfolio_risk_manager.config  # type: ignore[attr-defined]
    assert risk_manager_config.max_position_risk == pytest.approx(0.03)

def test_get_risk_status_includes_risk_api_summary() -> None:
    manager = TradingManager(
        event_bus=DummyBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=10_000.0,
        risk_config=RiskConfig(),
    )

    status = manager.get_risk_status()

    summary = status.get("risk_config_summary")
    assert isinstance(summary, dict)
    assert summary["max_risk_per_trade_pct"] == pytest.approx(0.02)
    assert status.get("risk_api_runbook", "").endswith("risk_api_contract.md")
    reference = status.get("risk_reference")
    assert isinstance(reference, dict)
    assert reference.get("risk_api_runbook", "").endswith("risk_api_contract.md")
    reference_summary = reference.get("risk_config_summary")
    assert isinstance(reference_summary, dict)
    assert reference_summary["max_total_exposure_pct"] == pytest.approx(0.5)


@pytest.mark.asyncio()
async def test_trade_throttle_blocks_and_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        trade_throttle={"max_trades": 1, "window_seconds": 120.0},
    )
    engine = RecordingExecutionEngine()
    manager.execution_engine = engine

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2345,
        confidence=0.9,
        strategy_id="alpha",
    )
    validate_mock: AsyncMock = AsyncMock(side_effect=[intent, intent, intent])
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    caplog.set_level(logging.WARNING, logger="src.trading.trading_manager")

    await manager.on_trade_intent(intent)
    await manager.on_trade_intent(intent)

    assert engine.calls == 1
    stats = manager.get_execution_stats()
    assert stats["orders_submitted"] == 1
    assert stats["throttle_blocks"] == 1
    assert stats["throttle_retry_in_seconds"] is not None
    throughput_snapshot = stats.get("throughput")
    assert isinstance(throughput_snapshot, Mapping)
    assert throughput_snapshot.get("throttle_active") is True
    assert throughput_snapshot.get("throttle_state") in {
        "rate_limited",
        "cooldown",
        "min_interval",
    }
    assert throughput_snapshot.get("throttle_retry_in_seconds") == pytest.approx(
        stats["throttle_retry_in_seconds"]
    )
    assert throughput_snapshot.get("throttle_reason")
    assert throughput_snapshot.get("throttle_message")
    assert throughput_snapshot.get("throttle_remaining_trades") is not None
    assert throughput_snapshot.get("throttle_max_trades") is not None
    throttle_stats = stats.get("trade_throttle")
    assert isinstance(throttle_stats, dict)
    assert throttle_stats.get("active") is True
    scope_snapshots = stats.get("trade_throttle_scopes")
    assert isinstance(scope_snapshots, list)
    assert scope_snapshots, "expected per-scope snapshots to be recorded"
    assert any(
        snapshot.get("state") in {"rate_limited", "cooldown", "min_interval"}
        for snapshot in scope_snapshots
    )
    method_snapshots = manager.get_trade_throttle_scope_snapshots()
    assert method_snapshots
    assert len(method_snapshots) == len(scope_snapshots)
    events = manager.get_experiment_events()
    statuses = [event["status"] for event in events]
    assert "throttled" in statuses
    throttle_events = [event for event in events if event["status"] == "throttled"]
    assert throttle_events
    throttle_metadata = throttle_events[0].get("metadata")
    assert isinstance(throttle_metadata, Mapping)
    assert throttle_metadata.get("reason")
    assert "message" in throttle_metadata
    assert throttle_metadata.get("retry_in_seconds") == pytest.approx(
        stats["throttle_retry_in_seconds"]
    )
    throttle_snapshot = throttle_metadata.get("throttle")
    assert isinstance(throttle_snapshot, Mapping)
    assert throttle_snapshot.get("state") in {"rate_limited", "cooldown", "min_interval"}
    assert "message" in throttle_snapshot
    throttle_context = throttle_snapshot.get("metadata", {}).get("context", {})
    assert throttle_context.get("symbol") == "EURUSD"
    assert throttle_context.get("strategy_id") == "alpha"

    snapshot = manager.get_trade_throttle_snapshot()
    assert snapshot is not None
    assert snapshot.get("state") in {"rate_limited", "cooldown", "min_interval"}

    manager.configure_trade_throttle(None)
    assert manager.get_trade_throttle_snapshot() is None
    assert manager.get_trade_throttle_scope_snapshots() == tuple()
    stats_after_disable = manager.get_execution_stats()
    assert "trade_throttle_scopes" not in stats_after_disable
    throughput_after_disable = stats_after_disable.get("throughput")
    assert isinstance(throughput_after_disable, Mapping)
    assert throughput_after_disable.get("throttle_active") is False
    assert throughput_after_disable.get("throttle_state") is None
    retry_after_disable = throughput_after_disable.get("throttle_retry_in_seconds")
    if retry_after_disable is not None:
        assert retry_after_disable == pytest.approx(0.0)

    await manager.on_trade_intent(intent)
    assert engine.calls == 2

    throttled_logs = [record for record in caplog.records if "Throttled trade intent" in record.getMessage()]
    assert throttled_logs, "expected throttle warning log to be emitted"
    assert any("too many trades" in record.getMessage() for record in throttled_logs)


@pytest.mark.asyncio()
async def test_backlog_breach_enforces_throttle_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        trade_throttle={"max_trades": 5, "window_seconds": 120.0},
        backlog_cooldown_seconds=30.0,
    )
    engine = RecordingExecutionEngine()
    manager.execution_engine = engine

    def _intent_with_lag() -> ConfidenceIntent:
        created = datetime.now(tz=timezone.utc) - timedelta(seconds=1.5)
        return ConfidenceIntent(
            symbol="EURUSD",
            quantity=1.0,
            price=1.2345,
            confidence=0.9,
            strategy_id="alpha",
            created_at=created,
        )

    first_intent = _intent_with_lag()
    validate_mock: AsyncMock = AsyncMock(side_effect=[first_intent, _intent_with_lag()])
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    await manager.on_trade_intent(first_intent)
    assert engine.calls == 1

    second_intent = _intent_with_lag()
    await manager.on_trade_intent(second_intent)

    assert engine.calls == 1, "Second intent should be throttled by backlog cooldown"
    stats = manager.get_execution_stats()
    assert stats.get("throttle_blocks", 0) >= 1
    throughput_snapshot = stats.get("throughput")
    assert isinstance(throughput_snapshot, Mapping)
    assert throughput_snapshot.get("throttle_reason") == "backlog_cooldown"
    assert throughput_snapshot.get("throttle_state") == "cooldown"
    throttle_snapshot = stats.get("trade_throttle")
    assert isinstance(throttle_snapshot, Mapping)
    assert throttle_snapshot.get("state") == "cooldown"
    metadata = throttle_snapshot.get("metadata", {})
    assert metadata.get("cooldown_reason") == "backlog_cooldown"
    cooldown_context = metadata.get("cooldown_context", {})
    assert cooldown_context.get("lag_ms") is not None

    events = manager.get_experiment_events()
    assert any(event["status"] == "throttled" for event in events)
    throttled_metadata = next(
        event["metadata"]
        for event in events
        if event["status"] == "throttled"
    )
    assert throttled_metadata.get("reason") == "backlog_cooldown"
    assert throttled_metadata.get("message")


@pytest.mark.asyncio()
async def test_trade_throttle_rolls_back_on_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        trade_throttle={"max_trades": 1, "window_seconds": 60.0},
    )

    failing_engine = FailingExecutionEngine()
    manager.execution_engine = failing_engine

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2345,
        confidence=0.9,
        strategy_id="alpha",
    )
    manager.risk_gateway.validate_trade_intent = AsyncMock(return_value=intent)  # type: ignore[assignment]

    outcome = await manager.on_trade_intent(intent)
    assert outcome.status == "failed"
    assert outcome.executed is False
    assert failing_engine.calls == 1

    stats_after_failure = manager.get_execution_stats()
    throttle_stats = stats_after_failure.get("trade_throttle")
    assert isinstance(throttle_stats, Mapping)
    assert throttle_stats.get("state") == "open"
    throttle_meta = throttle_stats.get("metadata", {})
    assert throttle_meta.get("recent_trades") == 0
    assert throttle_meta.get("remaining_trades") == 1
    assert stats_after_failure.get("throttle_retry_at") is None
    assert stats_after_failure.get("throttle_retry_in_seconds") in (None, 0.0)

    recording_engine = RecordingExecutionEngine()
    manager.execution_engine = recording_engine
    manager.risk_gateway.validate_trade_intent = AsyncMock(return_value=intent)  # type: ignore[assignment]

    success_outcome = await manager.on_trade_intent(intent)
    assert success_outcome.executed is True
    assert recording_engine.calls == 1


@pytest.mark.asyncio()
async def test_trade_throttle_handles_high_frequency_burst(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        trade_throttle={"max_trades": 1, "window_seconds": 60.0},
    )
    engine = RecordingExecutionEngine()
    manager.execution_engine = engine

    class _BurstIntent:
        def __init__(self, idx: int) -> None:
            self.symbol = "EURUSD"
            self.quantity = 1.0
            self.price = 1.2345
            self.confidence = 0.9
            self.strategy_id = "alpha"
            self.event_id = f"burst-{idx}"

    intents = [_BurstIntent(idx) for idx in range(5)]

    validate_mock: AsyncMock = AsyncMock(side_effect=intents)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    caplog.set_level(logging.WARNING, logger="src.trading.trading_manager")

    await asyncio.gather(*(manager.on_trade_intent(intent) for intent in intents))

    assert engine.calls == 1
    stats = manager.get_execution_stats()
    assert stats["orders_submitted"] == 1
    assert stats["throttle_blocks"] == len(intents) - 1
    assert stats["throttle_retry_in_seconds"] is not None
    assert stats["throttle_retry_in_seconds"] >= 0.0
    assert stats["throttle_retry_in_seconds"] is not None
    assert stats["throttle_retry_in_seconds"] >= 0.0

    throttle_events = [
        event
        for event in manager.get_experiment_events()
        if event["status"] == "throttled"
    ]
    assert len(throttle_events) == len(intents) - 1
    for event in throttle_events:
        metadata = event.get("metadata", {})
        assert metadata.get("message", "").startswith("Throttled: too many trades")
        assert metadata.get("retry_in_seconds") is not None
        assert metadata.get("retry_in_seconds") >= 0.0

    throttled_logs = [
        record.getMessage()
        for record in caplog.records
        if "Throttled trade intent" in record.getMessage()
    ]
    assert len(throttled_logs) == len(intents) - 1
    assert all("too many trades" in message for message in throttled_logs)


@pytest.mark.asyncio()
async def test_trade_throttle_applies_multiplier_to_quantity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        trade_throttle={
            "max_trades": 5,
            "window_seconds": 120.0,
            "multiplier": 0.5,
        },
    )
    engine = RecordingExecutionEngine()
    manager.execution_engine = engine

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=2.0,
        price=1.2345,
        confidence=0.85,
        strategy_id="alpha",
    )
    setattr(intent, "event_id", "multiplier-1")

    validate_mock: AsyncMock = AsyncMock(return_value=intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    outcome = await manager.on_trade_intent(intent)

    assert engine.calls == 1
    assert intent.quantity == pytest.approx(1.0)

    stats = manager.get_execution_stats()
    assert stats["throttle_scalings"] == 1
    assert stats["last_throttle_multiplier"] == pytest.approx(0.5)

    throttle_snapshot = manager.get_trade_throttle_snapshot()
    assert throttle_snapshot is not None
    assert throttle_snapshot.get("multiplier") == pytest.approx(0.5)

    events = manager.get_experiment_events()
    scaling_events = [event for event in events if event["status"] == "throttle_scaled"]
    assert scaling_events, "expected throttle scaling experiment event"
    scaling_summary = scaling_events[0].get("metadata", {}).get("summary", {})
    assert scaling_summary.get("throttle_multiplier") == pytest.approx(0.5)
    assert scaling_summary.get("quantity_after_throttle") == pytest.approx(1.0)

    assert outcome.executed is True
    assert outcome.metadata.get("throttle_multiplier") == pytest.approx(0.5)
    assert outcome.metadata.get("quantity_after_throttle") == pytest.approx(1.0)


@pytest.mark.asyncio()
async def test_high_frequency_replay_throughput_remains_healthy(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        trade_throttle={"max_trades": 1, "window_seconds": 45.0},
    )
    engine = RecordingExecutionEngine()
    manager.execution_engine = engine

    base_time = datetime.now(tz=timezone.utc)
    intents: list[ConfidenceIntent] = []
    for idx in range(6):
        intent = ConfidenceIntent(
            symbol="EURUSD",
            quantity=1.0,
            price=1.23 + idx * 0.0001,
            confidence=0.95,
            strategy_id="alpha",
        )
        setattr(intent, "event_id", f"hft-{idx}")
        setattr(intent, "ingested_at", base_time - timedelta(milliseconds=idx * 2))
        intents.append(intent)

    validate_mock: AsyncMock = AsyncMock(side_effect=intents)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    caplog.set_level(logging.WARNING, logger="src.trading.trading_manager")

    await asyncio.gather(*(manager.on_trade_intent(intent) for intent in intents))

    assert engine.calls == 1

    stats = manager.get_execution_stats()
    assert stats["orders_submitted"] == 1
    assert stats["throttle_blocks"] == len(intents) - 1

    throughput = stats.get("throughput")
    assert isinstance(throughput, Mapping)
    assert throughput.get("samples") == len(intents)
    assert throughput.get("throughput_per_min") is not None
    assert throughput.get("max_processing_ms") is not None
    assert throughput.get("max_lag_ms") is not None
    assert float(throughput["max_lag_ms"]) <= 500.0

    health = manager.assess_throughput_health(
        max_processing_ms=100.0, max_lag_ms=500.0
    )
    assert health["healthy"] is True
    assert health["samples"] == len(intents)
    assert health["processing_within_limit"] is True
    assert health["lag_within_limit"] is True

    throttled_logs = [
        record.getMessage()
        for record in caplog.records
        if "Throttled trade intent" in record.getMessage()
    ]
    assert len(throttled_logs) == len(intents) - 1
    assert all("too many trades" in message for message in throttled_logs)


@pytest.mark.asyncio()
async def test_backlog_breach_records_event(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        backlog_threshold_ms=10.0,
    )
    engine = RecordingExecutionEngine()
    manager.execution_engine = engine

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2345,
        confidence=0.9,
        strategy_id="alpha",
    )
    setattr(intent, "ingested_at", datetime.now(tz=timezone.utc) - timedelta(seconds=1.5))

    validate_mock: AsyncMock = AsyncMock(return_value=intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    caplog.set_level(logging.WARNING, logger="src.trading.trading_manager")

    await manager.on_trade_intent(intent)

    stats = manager.get_execution_stats()
    assert stats["orders_submitted"] == 1
    backlog_snapshot = stats.get("backlog")
    assert isinstance(backlog_snapshot, Mapping)
    assert backlog_snapshot.get("breaches", 0) >= 1
    assert stats["backlog_breaches"] == 1
    assert stats["last_backlog_breach"] is not None

    events = manager.get_experiment_events()
    backlog_events = [event for event in events if event.get("status") == "backlog_breach"]
    assert backlog_events, "expected backlog breach event to be recorded"
    metadata = backlog_events[0].get("metadata", {})
    assert metadata.get("lag_ms") is not None
    assert metadata.get("threshold_ms") == 10.0

    backlog_logs = [
        record.getMessage()
        for record in caplog.records
        if "Backlog threshold exceeded" in record.getMessage()
    ]
    assert backlog_logs, "expected backlog breach warning to be logged"


@pytest.mark.asyncio()
async def test_trading_manager_records_throughput_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=RecordingExecutionEngine(),
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        trade_throttle={"max_trades": 10, "window_seconds": 60.0},
    )

    intents = []
    for idx in range(3):
        intent = ConfidenceIntent(
            symbol="EURUSD",
            quantity=1.0,
            price=1.2 + idx * 0.001,
            confidence=0.8,
            strategy_id="alpha",
        )
        setattr(intent, "ingested_at", datetime.now(tz=timezone.utc))
        intents.append(intent)

    for intent in intents:
        await manager.on_trade_intent(intent)

    stats = manager.get_execution_stats()
    throughput = stats.get("throughput")
    assert isinstance(throughput, Mapping)
    assert throughput.get("samples") == len(intents)
    assert throughput.get("avg_processing_ms") is not None
    assert throughput.get("throughput_per_min") is not None


@pytest.mark.asyncio()
async def test_generate_execution_report_renders_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=RecordingExecutionEngine(),
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        trade_throttle={"max_trades": 2, "window_seconds": 60.0},
    )

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2,
        confidence=0.9,
    )
    setattr(intent, "event_id", "report-test")
    setattr(intent, "ingested_at", datetime.now(tz=timezone.utc))

    await manager.on_trade_intent(intent)

    report = manager.generate_execution_report()

    assert report.startswith("# Execution performance summary")
    assert "Throughput window" in report


@pytest.mark.asyncio()
async def test_trading_manager_releases_reservation_on_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)

    bus = DummyBus()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=50_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
    )

    engine = FailingExecutionEngine(error=RuntimeError("paper broker offline"))
    manager.execution_engine = engine

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2345,
        confidence=0.9,
        strategy_id="alpha",
    )
    setattr(intent, "event_id", "fail-001")

    validate_mock: AsyncMock = AsyncMock(return_value=intent)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    release_calls: list[tuple[str, float]] = []
    original_release = manager.portfolio_monitor.release_position

    def _tracking_release(symbol: str, quantity: float) -> None:
        release_calls.append((symbol, quantity))
        original_release(symbol, quantity)

    manager.portfolio_monitor.release_position = _tracking_release  # type: ignore[assignment]

    await manager.on_trade_intent(intent)

    assert engine.calls == 1
    assert release_calls, "expected reserved position to be released after failure"
    released_symbol, released_qty = release_calls[-1]
    assert released_symbol == "EURUSD"
    assert released_qty == pytest.approx(intent.quantity)
    assert manager.portfolio_monitor.get_position("EURUSD") is None

    stats = manager.get_execution_stats()
    assert stats.get("orders_failed") == 1


@pytest.mark.asyncio()
async def test_collect_performance_baseline_reports_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    manager = TradingManager(
        event_bus=DummyBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        throughput_monitor=ThroughputMonitor(window=8),
        trade_throttle={"max_trades": 1, "window_seconds": 60.0},
    )
    engine = RecordingExecutionEngine()
    manager.execution_engine = engine

    intent = ConfidenceIntent(
        symbol="EURUSD",
        quantity=1.0,
        price=1.2150,
        confidence=0.9,
        strategy_id="alpha",
    )

    validate_mock: AsyncMock = AsyncMock(side_effect=[intent, intent])
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

    await manager.on_trade_intent(intent)
    await manager.on_trade_intent(intent)

    baseline = collect_performance_baseline(
        manager,
        max_processing_ms=1_000.0,
        max_lag_ms=1_000.0,
    )

    execution_stats = baseline["execution"]
    assert isinstance(execution_stats, Mapping)
    assert execution_stats.get("orders_submitted") == 1
    assert execution_stats.get("throttle_retry_in_seconds") is not None

    throughput_summary = baseline["throughput"]
    assert isinstance(throughput_summary, Mapping)
    assert throughput_summary.get("samples", 0) >= 1

    performance_snapshot = baseline["performance"]
    assert isinstance(performance_snapshot, Mapping)
    assert performance_snapshot.get("throughput", {}).get("samples") == throughput_summary.get(
        "samples"
    )

    throttle_summary = baseline.get("throttle")
    assert isinstance(throttle_summary, Mapping)
    assert throttle_summary.get("state") in {"rate_limited", "cooldown", "min_interval"}

    reports = baseline.get("reports")
    assert isinstance(reports, Mapping)
    throttle_scopes = baseline.get("throttle_scopes")
    assert isinstance(throttle_scopes, list)
    assert throttle_scopes, "expected baseline to include per-scope snapshots"
    assert "execution" in reports and "performance" in reports


def test_describe_risk_interface_returns_runbook_on_error() -> None:
    class BrokenTradingManager(TradingManager):
        def __init__(self) -> None:
            super().__init__(
                event_bus=DummyBus(),
                strategy_registry=AlwaysActiveRegistry(),
                execution_engine=None,
                risk_config=RiskConfig(),
            )
            # Invalidate the canonical risk configuration to trigger fallback path.
            self._risk_config = None  # type: ignore[assignment]

        def get_risk_status(self) -> dict[str, object]:
            return {"risk_config": {"max_risk_per_trade_pct": -1}}

    manager = BrokenTradingManager()

    description = manager.describe_risk_interface()

    assert description["runbook"].endswith("risk_api_contract.md")
    assert "BrokenTradingManager" in description["details"].get("manager", "")
    assert "error" in description
