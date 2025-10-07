from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerFeatureFlags,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.trading.execution.release_router import ReleaseAwareExecutionRouter


class StubEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[Any] = []

    async def process_order(self, intent: Any) -> str:
        self.calls.append(intent)
        return f"{self.name}-ok"


@pytest.mark.asyncio()
async def test_release_router_routes_live_stage(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    release_manager = LedgerReleaseManager(store)

    paper_engine = StubEngine("paper")
    live_engine = StubEngine("live")

    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
        live_engine=live_engine,
    )

    intent: dict[str, Any] = {"strategy_id": "alpha", "metadata": {}}
    first = await router.process_order(intent)
    assert first == "paper-ok"
    assert len(paper_engine.calls) == 1
    assert not live_engine.calls
    assert intent["metadata"]["release_stage"] == PolicyLedgerStage.EXPERIMENT.value
    assert intent["metadata"]["release_execution_route"] == "paper"

    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="diary-alpha",
    )

    second = await router.process_order(intent)
    assert second == "live-ok"
    assert len(live_engine.calls) == 1
    last_route = router.last_route()
    assert last_route is not None
    assert last_route["stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert last_route["route"] == "live"
    assert "audit" in last_route
    audit_payload = last_route["audit"]
    assert audit_payload.get("declared_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert audit_payload.get("audit_stage") == PolicyLedgerStage.LIMITED_LIVE.value


@pytest.mark.asyncio()
async def test_release_router_falls_back_to_paper(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    release_manager = LedgerReleaseManager(store)

    paper_engine = StubEngine("paper")
    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
    )

    release_manager.promote(
        policy_id="beta",
        tactic_id="beta",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk",),
        evidence_id="diary-beta",
    )

    intent: dict[str, Any] = {"strategy_id": "beta"}
    result = await router.process_order(intent)
    assert result == "paper-ok"
    assert len(paper_engine.calls) == 1
    last_route = router.last_route()
    assert last_route is not None
    assert last_route["route"] == "paper"
    assert last_route.get("forced_reason") is None
    audit_payload = last_route.get("audit")
    assert audit_payload is not None
    assert audit_payload.get("enforced") is True
    assert audit_payload.get("declared_stage") == PolicyLedgerStage.LIMITED_LIVE.value
    assert audit_payload.get("audit_stage") == PolicyLedgerStage.PILOT.value
    assert "additional_approval_needed" in audit_payload.get("gaps", [])

    metadata = intent.get("metadata")
    assert isinstance(metadata, dict)
    audit_meta = metadata.get("release_execution_audit")
    assert isinstance(audit_meta, dict)
    assert audit_meta.get("enforced") is True
    assert "additional_approval_needed" in audit_meta.get("gaps", [])


@pytest.mark.asyncio()
async def test_release_router_enforces_missing_evidence(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger_missing_evidence.json")
    release_manager = LedgerReleaseManager(
        store,
        feature_flags=PolicyLedgerFeatureFlags(require_diary_evidence=False),
    )

    paper_engine = StubEngine("paper")
    live_engine = StubEngine("live")
    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
        live_engine=live_engine,
    )

    release_manager.promote(
        policy_id="gamma",
        tactic_id="gamma",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id=None,
    )

    intent: dict[str, Any] = {"strategy_id": "gamma"}
    result = await router.process_order(intent)

    assert result == "paper-ok"
    assert len(paper_engine.calls) == 1
    assert not live_engine.calls

    metadata = intent.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("release_stage") == PolicyLedgerStage.EXPERIMENT.value
    assert metadata.get("release_execution_route") == "paper"
    assert metadata.get("release_execution_route_overridden") is True
    assert metadata.get("release_execution_forced") == "release_audit_gap_missing_evidence"
    forced_reasons = metadata.get("release_execution_forced_reasons")
    assert forced_reasons == ["release_audit_gap_missing_evidence"]
    audit_meta = metadata.get("release_execution_audit")
    assert isinstance(audit_meta, dict)
    assert audit_meta.get("enforced") is True
    assert "missing_evidence" in audit_meta.get("gaps", [])

    last_route = router.last_route()
    assert last_route is not None
    assert last_route.get("route") == "paper"
    assert last_route.get("forced_reason") == "release_audit_gap_missing_evidence"
    assert last_route.get("forced_reasons") == ["release_audit_gap_missing_evidence"]
    assert last_route.get("audit_forced") is True
    audit_payload = last_route.get("audit")
    assert isinstance(audit_payload, dict)
    assert audit_payload.get("enforced") is True
    assert "missing_evidence" in audit_payload.get("gaps", [])
