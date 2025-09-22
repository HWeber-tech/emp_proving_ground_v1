from dataclasses import dataclass

import pytest

from src.config.risk.risk_config import RiskConfig
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
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


@pytest.mark.asyncio()
async def test_trading_manager_records_execution_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)

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
