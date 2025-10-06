from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage, PolicyLedgerStore
from src.governance.policy_rebuilder import rebuild_policy_artifacts


def test_rebuild_policy_artifacts_applies_deltas(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    manager = LedgerReleaseManager(store)

    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id="diary-1",
        policy_delta={
            "risk_config": {"max_leverage": 7},
            "router_guardrails": {"requires_diary": True},
        },
    )

    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk", "ops"),
        evidence_id="diary-2",
        policy_delta={
            "risk_config": {"max_total_exposure_pct": 0.4},
            "router_guardrails": {"max_latency_ms": 250},
        },
    )

    artifacts = rebuild_policy_artifacts(store)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.stage is PolicyLedgerStage.PILOT
    assert isinstance(artifact.risk_config.max_leverage, Decimal)
    assert artifact.risk_config.max_leverage == Decimal("7")
    assert artifact.risk_config.max_total_exposure_pct == Decimal("0.4")
    assert artifact.router_guardrails["requires_diary"] is True
    assert artifact.router_guardrails["max_latency_ms"] == 250
    assert artifact.thresholds["stage"] == PolicyLedgerStage.PILOT.value
    assert artifact.policy_delta is not None
    assert artifact.history

    payload = artifact.as_dict()
    assert payload["risk_config"]["max_leverage"] == 7.0
    assert payload["router_guardrails"]["max_latency_ms"] == 250
