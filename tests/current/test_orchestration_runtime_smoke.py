"""Smoke test wiring the orchestration adapters, event bus, and phase 3 orchestrator."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.core.event_bus import Event, EventBus
from src.orchestration.compose import compose_validation_adapters
from src.runtime.task_supervisor import TaskSupervisor
from tests.util.orchestration_stubs import (
    InMemoryStateStore,
    install_phase3_orchestrator,
)


@pytest.mark.asyncio
async def test_phase3_orchestrator_pipeline_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    adapters = compose_validation_adapters()
    orchestrator_module = install_phase3_orchestrator(monkeypatch)

    event_bus = EventBus()
    await event_bus.start()

    received: list[dict[str, Any]] = []

    async def _capture(event: Event) -> None:
        received.append(event.payload or {})

    handle = event_bus.subscribe("orchestration.smoke", _capture)
    await event_bus.publish(Event(type="orchestration.smoke", payload={"ok": True}))
    await asyncio.sleep(0.05)
    event_bus.unsubscribe(handle)
    assert received == [{"ok": True}]

    state_store = InMemoryStateStore()
    orchestrator = orchestrator_module.Phase3Orchestrator(
        state_store=state_store,
        event_bus=event_bus,
        adaptation_service=adapters["adaptation_service"],
    )

    try:
        init_ok = await orchestrator.initialize()
        assert init_ok is True

        analysis = await orchestrator.run_full_analysis()
        assert set(analysis["systems"]) == {
            "sentient",
            "predictive",
            "adversarial",
            "specialized",
            "understanding",
        }
        assert "competitive" not in analysis["systems"]

        predictive = analysis["systems"]["predictive"]
        assert predictive["scenarios_generated"] == 2
        assert predictive["average_confidence"] == pytest.approx(0.65, rel=1e-3)

        adversarial = analysis["systems"]["adversarial"]
        assert adversarial["strategies_improved"] == 2
        assert adversarial["gan_training_complete"] is True

        assert "overall_metrics" in analysis
        overall = analysis["overall_metrics"]
        assert overall["systems_count"] == 5
        assert overall["success_ratio"] >= 0.0
        assert overall["presence"]["adversarial"] is True
        assert overall["presence"]["understanding"] is True
    finally:
        await orchestrator.stop()
        await event_bus.stop()


@pytest.mark.asyncio
async def test_phase3_orchestrator_supervised_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    adapters = compose_validation_adapters()
    orchestrator_module = install_phase3_orchestrator(monkeypatch)

    event_bus = EventBus()
    await event_bus.start()
    supervisor = TaskSupervisor(namespace="phase3-supervisor-test")

    state_store = InMemoryStateStore()
    orchestrator = orchestrator_module.Phase3Orchestrator(
        state_store=state_store,
        event_bus=event_bus,
        adaptation_service=adapters["adaptation_service"],
        task_supervisor=supervisor,
    )

    try:
        assert await orchestrator.initialize() is True
        assert await orchestrator.start() is True
        await asyncio.sleep(0.05)
        assert supervisor.active_count >= 2
        await orchestrator.stop()
        await asyncio.sleep(0)
        assert supervisor.active_count == 0
    finally:
        await supervisor.cancel_all()
        await event_bus.stop()
