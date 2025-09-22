from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from src.config.risk.risk_config import RiskConfig as TradingRiskConfig
from src.core.event_bus import AsyncEventBus, Event
from src.operations.roi import RoiTelemetrySnapshot
from src.risk.telemetry import RiskTelemetrySnapshot
from src.trading.trading_manager import TradingManager
from src.trading.risk.risk_policy import RiskPolicy


@dataclass
class _RecordedEvents:
    topics: tuple[str, ...]
    events: dict[str, list[Event]]
    futures: dict[str, asyncio.Future[None]]

    @classmethod
    def create(cls, topics: tuple[str, ...]) -> "_RecordedEvents":
        loop = asyncio.get_running_loop()
        futures: dict[str, asyncio.Future[None]] = {topic: loop.create_future() for topic in topics}
        return cls(topics=topics, events=defaultdict(list), futures=futures)

    async def handler(self, event: Event) -> None:
        self.events[event.type].append(event)
        future = self.futures.get(event.type)
        if future is not None and not future.done():
            future.set_result(None)

    async def wait_for_all(self, timeout: float = 1.0) -> None:
        await asyncio.wait_for(
            asyncio.gather(*(future for future in self.futures.values())),
            timeout=timeout,
        )

    def snapshot_counts(self) -> dict[str, int]:
        return {topic: len(self.events.get(topic, [])) for topic in self.topics}


class _AlwaysActiveRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, dict[str, str]] = {}

    def ensure(self, strategy_id: str, status: str = "active") -> None:
        self._strategies[strategy_id] = {"id": strategy_id, "status": status}

    def get_strategy(self, strategy_id: str) -> dict[str, str] | None:
        return self._strategies.get(strategy_id)


class _RecordingExecutionEngine:
    def __init__(self, *, should_fail: Callable[[Any], bool] | None = None) -> None:
        self._orders: list[dict[str, Any]] = []
        self._should_fail = should_fail or (lambda _intent: False)

    @property
    def orders(self) -> list[dict[str, Any]]:
        return list(self._orders)

    async def process_order(self, intent: Any) -> dict[str, Any]:
        if self._should_fail(intent):
            raise RuntimeError("execution failure")
        payload = {
            "order_id": getattr(intent, "event_id", "unknown"),
            "symbol": getattr(intent, "symbol", "UNKNOWN"),
            "quantity": float(getattr(intent, "quantity", 0)),
            "status": "FILLED",
        }
        self._orders.append(payload)
        return payload

    def iter_orders(self):
        return iter(self._orders)


@pytest.mark.asyncio()
async def test_trading_manager_event_bus_flow_and_fallback() -> None:
    topics = (
        "telemetry.risk.posture",
        "telemetry.risk.policy",
        "telemetry.operational.roi",
    )

    event_bus = AsyncEventBus()
    await event_bus.start()

    recorder = _RecordedEvents.create(topics)
    for topic in topics:
        event_bus.subscribe(topic, recorder.handler)

    registry = _AlwaysActiveRegistry()
    registry.ensure("strat-1", status="active")

    execution_engine = _RecordingExecutionEngine()

    risk_config = TradingRiskConfig(
        max_risk_per_trade_pct=Decimal("0.2"),
        max_total_exposure_pct=Decimal("1.0"),
        max_drawdown_pct=Decimal("0.5"),
        max_leverage=Decimal("20"),
        min_position_size=1,
        max_position_size=10_000,
        mandatory_stop_loss=False,
        research_mode=True,
    )

    manager = TradingManager(
        event_bus=event_bus,
        strategy_registry=registry,
        execution_engine=execution_engine,
        initial_equity=100_000.0,
        risk_config=risk_config,
        risk_policy=RiskPolicy.from_config(risk_config),
        min_intent_confidence=0.05,
    )

    successful_intent = SimpleNamespace(
        event_id="intent-1",
        strategy_id="strat-1",
        symbol="EURUSD",
        quantity=Decimal("1000"),
        price=Decimal("1.12"),
        confidence=0.9,
        stop_loss_pct=0.01,
    )

    await manager.on_trade_intent(successful_intent)
    await recorder.wait_for_all()

    assert len(execution_engine.orders) == 1

    risk_events = recorder.events["telemetry.risk.posture"]
    assert risk_events, "risk telemetry not published"
    first_risk_payload = risk_events[0].payload
    assert isinstance(first_risk_payload, dict)
    assert first_risk_payload.get("status") in {"ok", "warn", "alert"}

    policy_events = recorder.events["telemetry.risk.policy"]
    assert policy_events, "policy telemetry not published"
    policy_payload = policy_events[0].payload
    assert policy_payload.get("approved") is True
    assert policy_payload.get("symbol") == "EURUSD"

    roi_events = recorder.events["telemetry.operational.roi"]
    assert roi_events, "ROI telemetry not published"
    roi_payload = roi_events[0].payload
    assert roi_payload.get("executed_trades", 0) >= 1
    assert roi_payload.get("total_notional", 0.0) > 0.0

    await event_bus.stop()

    baseline_counts = recorder.snapshot_counts()

    rejected_intent = SimpleNamespace(
        event_id="intent-2",
        strategy_id="strat-1",
        symbol="EURUSD",
        quantity=Decimal("1000"),
        price=Decimal("1.10"),
        confidence=0.0,
        stop_loss_pct=0.0,
    )

    await manager.on_trade_intent(rejected_intent)
    await asyncio.sleep(0.05)

    assert len(execution_engine.orders) == 1
    for topic, before in baseline_counts.items():
        assert len(recorder.events.get(topic, [])) == before

    risk_snapshot = manager.get_last_risk_snapshot()
    assert isinstance(risk_snapshot, RiskTelemetrySnapshot)

    policy_snapshot = manager.get_last_policy_snapshot()
    assert policy_snapshot is not None
    assert policy_snapshot.symbol == "EURUSD"
    assert policy_snapshot.approved in {True, False}

    roi_snapshot = manager.get_last_roi_snapshot()
    assert isinstance(roi_snapshot, RoiTelemetrySnapshot)
    assert roi_snapshot.executed_trades >= 1
