from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

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

import pytest

from src.config.risk.risk_config import RiskConfig
from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.operations.sensory_drift import DriftSeverity, SensoryDimensionDrift, SensoryDriftSnapshot
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.gating import DriftSentryGate
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


class RecordingExecutionEngine:
    def __init__(self) -> None:
        self.calls = 0

    async def process_order(self, intent: Any) -> str:
        self.calls += 1
        return "ok"


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
    manager = TradingManager(
        event_bus=DummyBus(),
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

    validate_mock: AsyncMock = AsyncMock(return_value=None)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

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

    await manager.on_trade_intent(intent)

    assert validate_mock.await_count == 0
    assert execution_engine.calls == 0

    events = manager.get_experiment_events()
    assert events, "expected gating event"
    latest = events[0]
    assert latest["status"] == "gated"
    metadata = latest.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("drift_severity") == DriftSeverity.warn.value
    assert "confidence" in str(metadata.get("reason"))
    gate_payload = metadata.get("drift_gate")
    assert isinstance(gate_payload, dict)
    assert gate_payload.get("allowed") is False
    assert gate_payload.get("severity") == DriftSeverity.warn.value
    decision = manager.get_last_drift_gate_decision()
    assert decision is not None
    assert not decision.allowed
    assert decision.severity is DriftSeverity.warn


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

    manager = TradingManager(
        event_bus=DummyBus(),
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
    thresholds = gate_payload.get("requirements", {})
    assert thresholds.get("confidence_floor") == pytest.approx(0.6)
    assert execution_engine.calls == 1


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

    manager = TradingManager(
        event_bus=DummyBus(),
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

    validate_mock: AsyncMock = AsyncMock(return_value=None)
    manager.risk_gateway.validate_trade_intent = validate_mock  # type: ignore[assignment]

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

    await manager.on_trade_intent(intent)

    decision = manager.get_last_drift_gate_decision()
    assert decision is not None
    assert not decision.allowed, "confidence below release-managed floor should block"
    assert "0.870" in (decision.reason or "")
    requirements = dict(decision.requirements)
    assert requirements["release_stage"] == PolicyLedgerStage.PAPER.value
    assert requirements["confidence_floor"] == pytest.approx(0.87, rel=1e-6)
    assert requirements.get("warn_notional_limit") == pytest.approx(35_625.0)
    assert validate_mock.await_count == 0
    assert execution_engine.calls == 0
    events = manager.get_experiment_events()
    assert events and events[0]["status"] == "gated"
    metadata = events[0].get("metadata")
    assert isinstance(metadata, dict)
    gate_payload = metadata.get("drift_gate")
    assert isinstance(gate_payload, dict)
    gate_requirements = gate_payload.get("requirements", {})
    assert gate_requirements.get("release_stage") == PolicyLedgerStage.PAPER.value
    assert gate_requirements.get("confidence_floor") == pytest.approx(0.87, rel=1e-6)
    assert gate_requirements.get("warn_notional_limit") == pytest.approx(35_625.0)


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

    manager = TradingManager(
        event_bus=DummyBus(),
        strategy_registry=AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=10_000.0,
        drift_gate=None,
        release_manager=release_manager,
    )

    summary = manager.describe_release_posture("alpha")
    assert summary["stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert summary["managed"] is True
    assert "thresholds" in summary
    thresholds = summary["thresholds"]
    assert thresholds.get("adaptive_source") == DriftSeverity.normal.value
    assert summary.get("approvals") == ["ops", "risk"]
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
