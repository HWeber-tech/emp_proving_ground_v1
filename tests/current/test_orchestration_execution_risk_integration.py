from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.core.event_bus import Event, EventBus
from src.orchestration.compose import compose_validation_adapters
from src.risk.risk_manager_impl import RiskManagerImpl
from src.trading.execution.execution_engine import ExecutionEngine
from tests.util.orchestration_stubs import (
    InMemoryStateStore,
    install_phase3_orchestrator,
)


@pytest.mark.asyncio
async def test_orchestration_risk_execution_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    adapters = compose_validation_adapters()
    orchestrator_module = install_phase3_orchestrator(monkeypatch)

    event_bus = EventBus()
    await event_bus.start()

    state_store = InMemoryStateStore()
    orchestrator = orchestrator_module.Phase3Orchestrator(
        state_store=state_store,
        event_bus=event_bus,
        adaptation_service=adapters["adaptation_service"],
    )
    risk_manager = RiskManagerImpl(initial_balance=75_000)
    execution = ExecutionEngine()

    recorded_events: list[tuple[str, dict[str, Any]]] = []

    async def _collector(event: Event) -> None:
        recorded_events.append((event.type, event.payload or {}))

    handle = event_bus.subscribe("trade.execution", _collector)

    position_size = 0.0

    try:
        assert await orchestrator.initialize() is True

        analysis = await orchestrator.run_full_analysis()
        predictive = analysis["systems"]["predictive"]
        signal = {
            "symbol": "EURUSD",
            "confidence": predictive["average_confidence"],
            "stop_loss_pct": 0.015,
        }

        assert await risk_manager.validate_position(
            {"symbol": "EURUSD", "size": 10_000, "entry_price": 1.2345}
        )

        position_size = await risk_manager.calculate_position_size(signal)
        assert position_size > 0

        risk_manager.add_position("EURUSD", position_size, 1.2345)

        order_id = await execution.send_order(
            "EURUSD",
            "BUY",
            position_size,
            price=1.2345,
        )
        execution.record_fill(order_id, position_size * 0.4, 1.2347)
        execution.record_fill(order_id, position_size * 0.6, 1.2351)

        risk_manager.update_position_value("EURUSD", 1.2351)

        reconciliation = execution.reconcile()
        filled_snapshot = next(
            entry for entry in reconciliation["filled_orders"] if entry["order_id"] == order_id
        )

        await event_bus.publish(
            Event(
                type="trade.execution",
                payload={
                    "order": filled_snapshot,
                    "positions": reconciliation["positions"],
                    "risk": risk_manager.get_position_risk("EURUSD"),
                },
            )
        )
        await asyncio.sleep(0.05)
    finally:
        event_bus.unsubscribe(handle)
        await orchestrator.stop()
        await event_bus.stop()

    assert recorded_events, "expected execution telemetry to be emitted"
    event_type, payload = recorded_events[-1]
    assert event_type == "trade.execution"
    assert payload["order"]["status"] == "FILLED"
    assert payload["risk"]["symbol"] == "EURUSD"
    assert payload["positions"]["EURUSD"]["quantity"] == pytest.approx(position_size)
