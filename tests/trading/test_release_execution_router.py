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
from tests.util import promotion_checklist_metadata


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
    metadata = intent["metadata"]
    assert metadata["release_stage"] == PolicyLedgerStage.EXPERIMENT.value
    assert metadata["release_execution_route"] == "paper"
    assert (
        metadata["release_execution_forced"]
        == "release_stage_experiment_requires_paper_or_better"
    )
    assert metadata["release_execution_forced_reasons"] == [
        "release_stage_experiment_requires_paper_or_better"
    ]
    assert metadata["release_execution_route_overridden"] is True

    initial_route = router.last_route()
    assert initial_route is not None
    assert initial_route["forced_reason"] == "release_stage_experiment_requires_paper_or_better"
    assert initial_route["forced_route"] == "paper"

    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="diary-alpha",
        metadata={
            "promotion_checklist": {
                "oos_regime_grid": True,
                "leakage_checks": True,
                "risk_audit": True,
            }
        },
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
    metadata = intent["metadata"]
    assert "release_execution_forced" not in metadata
    assert "release_execution_forced_reasons" not in metadata
    assert "release_execution_route_overridden" not in metadata


@pytest.mark.asyncio()
async def test_release_router_requires_ledger_record_for_limited_live(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger_default.json")
    release_manager = LedgerReleaseManager(
        store,
        default_stage=PolicyLedgerStage.LIMITED_LIVE,
    )

    paper_engine = StubEngine("paper")
    live_engine = StubEngine("live")

    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
        live_engine=live_engine,
        default_stage=PolicyLedgerStage.LIMITED_LIVE,
    )

    intent: dict[str, Any] = {"strategy_id": "alpha", "metadata": {}}
    result = await router.process_order(intent)

    assert result == "paper-ok"
    metadata = intent["metadata"]
    assert metadata["release_stage"] == PolicyLedgerStage.PILOT.value
    assert metadata["release_execution_route"] == "paper"
    assert metadata["release_execution_route_overridden"] is True
    assert (
        metadata["release_execution_forced"]
        == "release_stage_pilot_requires_policy_ledger_record"
    )
    assert metadata["release_execution_forced_reasons"] == [
        "release_stage_pilot_requires_policy_ledger_record"
    ]

    last_route = router.last_route()
    assert last_route is not None
    assert last_route["stage"] == PolicyLedgerStage.PILOT.value
    assert last_route["route"] == "paper"
    assert last_route["forced_route"] == "paper"
    assert (
        last_route["forced_reason"]
        == "release_stage_pilot_requires_policy_ledger_record"
    )


@pytest.mark.asyncio()
async def test_release_router_requires_ledger_when_policy_missing(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger_missing_policy.json")
    release_manager = LedgerReleaseManager(store)

    paper_engine = StubEngine("paper")
    live_engine = StubEngine("live")

    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
        live_engine=live_engine,
        default_stage=PolicyLedgerStage.LIMITED_LIVE,
    )

    intent: dict[str, Any] = {"metadata": {}}
    result = await router.process_order(intent)

    assert result == "paper-ok"
    metadata = intent["metadata"]
    assert metadata["release_stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert metadata["release_execution_route"] == "paper"
    assert metadata["release_execution_route_overridden"] is True
    assert (
        metadata["release_execution_forced"]
        == "release_stage_limited_live_requires_policy_ledger_record"
    )
    assert metadata["release_execution_forced_reasons"] == [
        "release_stage_limited_live_requires_policy_ledger_record"
    ]

    last_route = router.last_route()
    assert last_route is not None
    assert last_route["forced_reason"] == "release_stage_limited_live_requires_policy_ledger_record"
    assert last_route["forced_route"] == "paper"


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
        metadata={
            "promotion_checklist": {
                "oos_regime_grid": True,
                "leakage_checks": True,
                "risk_audit": True,
            }
        },
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
        metadata={
            "promotion_checklist": {
                "oos_regime_grid": True,
                "leakage_checks": True,
                "risk_audit": True,
            }
        },
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
    assert forced_reasons == [
        "release_audit_gap_missing_evidence",
        "release_stage_experiment_requires_paper_or_better",
    ]
    audit_meta = metadata.get("release_execution_audit")
    assert isinstance(audit_meta, dict)
    assert audit_meta.get("enforced") is True
    assert "missing_evidence" in audit_meta.get("gaps", [])

    last_route = router.last_route()
    assert last_route is not None
    assert last_route.get("route") == "paper"
    assert last_route.get("forced_reason") == "release_audit_gap_missing_evidence"
    assert last_route.get("forced_reasons") == [
        "release_audit_gap_missing_evidence",
        "release_stage_experiment_requires_paper_or_better",
    ]
    assert last_route.get("audit_forced") is True
    audit_payload = last_route.get("audit")
    assert isinstance(audit_payload, dict)
    assert audit_payload.get("enforced") is True
    assert "missing_evidence" in audit_payload.get("gaps", [])


@pytest.mark.asyncio()
async def test_release_router_stage_gate_paper_forces_paper(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger_paper.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="shadow", 
        tactic_id="shadow", 
        stage=PolicyLedgerStage.PAPER, 
        approvals=(), 
        evidence_id="dd-shadow",
    )

    paper_engine = StubEngine("paper")
    live_engine = StubEngine("live")

    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
        live_engine=live_engine,
    )

    intent: dict[str, Any] = {"strategy_id": "shadow"}
    result = await router.process_order(intent)

    assert result == "paper-ok"
    assert paper_engine.calls == [intent]
    metadata = intent.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("release_stage") == PolicyLedgerStage.PAPER.value
    assert metadata.get("release_execution_route") == "paper"
    assert (
        metadata.get("release_execution_forced")
        == "release_stage_paper_requires_paper_execution"
    )
    assert metadata.get("release_execution_forced_reasons") == [
        "release_stage_paper_requires_paper_execution"
    ]
    assert metadata.get("release_execution_route_overridden") is True
    last_route = router.last_route()
    assert last_route is not None
    assert last_route.get("forced_reason") == "release_stage_paper_requires_paper_execution"
    assert last_route.get("forced_route") == "paper"


@pytest.mark.asyncio()
async def test_release_router_counterfactual_guardrail_forces_paper(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger_guardrail.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha_guard",
        tactic_id="alpha_guard",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="dd-alpha-guard",
        metadata={
            "promotion_checklist": {
                "oos_regime_grid": True,
                "leakage_checks": True,
                "risk_audit": True,
            }
        },
    )

    paper_engine = StubEngine("paper")
    live_engine = StubEngine("live")
    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
        live_engine=live_engine,
    )

    intent: dict[str, Any] = {
        "strategy_id": "alpha_guard",
        "metadata": {
            "guardrails": {
                "counterfactual_guardrail": {
                    "breached": True,
                    "reason": "counterfactual_guardrail_delta_exceeded",
                    "severity": "aggro",
                    "action": "force_paper",
                }
            }
        },
    }

    result = await router.process_order(intent)

    assert result == "paper-ok"
    assert len(paper_engine.calls) == 1
    assert not live_engine.calls

    metadata = intent.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("release_execution_route") == "paper"
    assert metadata.get("release_execution_route_overridden") is True
    assert metadata.get("release_execution_forced") == "counterfactual_guardrail_delta_exceeded"
    assert metadata.get("release_execution_forced_reasons") == [
        "counterfactual_guardrail_delta_exceeded"
    ]

    last_route = router.last_route()
    assert last_route is not None
    assert last_route.get("forced_route") == "paper"
    assert last_route.get("forced_reason") == "counterfactual_guardrail_delta_exceeded"


@pytest.mark.asyncio()
async def test_release_router_counterfactual_guardrail_passive_does_not_force(
    tmp_path: Path,
) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger_guardrail_passive.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha_guard",
        tactic_id="alpha_guard",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="dd-alpha-guard",
    )

    paper_engine = StubEngine("paper")
    live_engine = StubEngine("live")
    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
        live_engine=live_engine,
    )

    intent: dict[str, Any] = {
        "strategy_id": "alpha_guard",
        "metadata": {
            "guardrails": {
                "counterfactual_guardrail": {
                    "breached": True,
                    "reason": "counterfactual_guardrail_delta_exceeded",
                    "severity": "passive",
                    "delta_direction": "passive",
                }
            }
        },
    }

    result = await router.process_order(intent)

    assert result == "live-ok"
    assert len(live_engine.calls) == 1
    assert not paper_engine.calls

    metadata = intent.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("release_execution_route") == "live"
    assert metadata.get("release_execution_route_overridden") is None
    assert metadata.get("release_execution_forced") is None
    assert metadata.get("release_execution_forced_reasons") in (None, [])

    last_route = router.last_route()
    assert last_route is not None
    assert last_route.get("route") == "live"
    assert "forced_route" not in last_route


@pytest.mark.asyncio()
async def test_release_router_configure_engines_for_stage_routing(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger_configure.json")
    release_manager = LedgerReleaseManager(store)

    paper_engine = StubEngine("paper")
    pilot_engine = StubEngine("pilot")
    live_engine = StubEngine("live")

    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
    )

    router.configure_engines(pilot_engine=pilot_engine, live_engine=live_engine)

    intent: dict[str, Any] = {"strategy_id": "alpha"}

    await router.process_order(intent)
    assert len(paper_engine.calls) == 1
    assert not pilot_engine.calls
    assert not live_engine.calls

    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk",),
        evidence_id="dd-alpha-pilot",
    )

    await router.process_order(intent)
    assert len(pilot_engine.calls) == 1
    assert len(paper_engine.calls) == 1
    assert not live_engine.calls

    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="dd-alpha-live",
        metadata={
            "promotion_checklist": {
                "oos_regime_grid": True,
                "leakage_checks": True,
                "risk_audit": True,
            }
        },
    )

    await router.process_order(intent)
    assert len(live_engine.calls) == 1
    last_route = router.last_route()
    assert last_route is not None
    assert last_route["stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert last_route["route"] == "live"
