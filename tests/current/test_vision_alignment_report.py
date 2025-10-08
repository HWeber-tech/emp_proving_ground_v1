from __future__ import annotations

import pytest

from src.core.event_bus import EventBus
from src.governance.vision_alignment import VisionAlignmentReport
from src.runtime.bootstrap_runtime import BootstrapRuntime


class _StubChampion:
    def __init__(self) -> None:
        self.genome_id = "bootstrap-champion"
        self.fitness = 1.08

    def as_payload(self) -> dict[str, object]:
        return {"genome_id": self.genome_id, "fitness": self.fitness}


class _StubOrchestrator:
    def __init__(self) -> None:
        self.telemetry = {
            "total_generations": 5,
            "champion": {"genome_id": "bootstrap-champion"},
            "adaptive_runs": {"enabled": True, "reason": "override_enabled"},
        }
        self.population_statistics = {"population_size": 12, "best_fitness": 1.08}
        self.champion = _StubChampion()

    def build_readiness_snapshot(self) -> dict[str, object]:
        return {
            "status": "ready",
            "adaptive_runs_enabled": True,
            "seed_source": "catalogue",
            "seed_templates": ("baseline",),
            "champion_id": "bootstrap-champion",
            "lineage_generation": 3,
            "issues": [],
        }


@pytest.mark.asyncio()
async def test_vision_alignment_report_tracks_layer_progress() -> None:
    orchestrator = _StubOrchestrator()
    runtime = BootstrapRuntime(event_bus=EventBus(), evolution_orchestrator=orchestrator)

    for _ in range(12):
        await runtime.trading_stack.evaluate_tick(runtime.symbols[0])

    reporter = VisionAlignmentReport(
        fabric=runtime.fabric,
        pipeline=runtime.pipeline,
        trading_manager=runtime.trading_manager,
        control_center=runtime.control_center,
        evolution_orchestrator=orchestrator,
    )

    payload = reporter.build()

    assert payload["summary"]["status"] in {"ready", "progressing", "gap"}
    assert payload["summary"]["readiness"] >= 0.0
    assert len(payload["layers"]) == 5
    assert payload["layers"][0]["layer"].startswith("Layer 1")
    assert any(layer["status"] == "ready" for layer in payload["layers"])
    assert payload["strengths"]
    assert payload["gaps"]

    evolution_layer = next(
        layer for layer in payload["layers"] if layer["layer"].startswith("Layer 4")
    )
    evidence = evolution_layer.get("evidence", {})
    readiness = evidence.get("readiness") if isinstance(evidence, dict) else None
    if isinstance(readiness, dict):
        assert readiness.get("status") == "ready"
