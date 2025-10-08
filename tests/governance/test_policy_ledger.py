from __future__ import annotations

from datetime import datetime, timedelta, timezone
try:  # Python < 3.11 compatibility
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    UTC = timezone.utc  # type: ignore[assignment]
from pathlib import Path
from typing import Callable

import pytest
from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyDelta,
    PolicyLedgerFeatureFlags,
    PolicyLedgerStage,
    PolicyLedgerStore,
    _parse_bool,
    build_policy_governance_workflow,
)


def _incrementing_now_factory(start: datetime) -> Callable[[], datetime]:
    counter = {"offset": 0}

    def _now() -> datetime:
        current = start + timedelta(seconds=counter["offset"])
        counter["offset"] += 1
        return current

    return _now


def test_policy_ledger_promotion_progression(tmp_path: Path) -> None:
    path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(path, now=_incrementing_now_factory(datetime(2024, 1, 1, tzinfo=UTC)))
    manager = LedgerReleaseManager(store)

    first = manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id="diary-001",
        threshold_overrides={"warn_confidence_floor": 0.8},
        policy_delta=PolicyDelta(
            regime="balanced",
            risk_config={"max_leverage": "8"},
            router_guardrails={"requires_diary": True},
        ),
    )

    assert first.stage is PolicyLedgerStage.PAPER
    assert first.approvals == ("compliance", "risk")
    assert first.evidence_id == "diary-001"
    assert first.threshold_overrides["warn_confidence_floor"] == 0.8
    assert first.history
    history_entry = first.history[0]
    assert history_entry["prior_stage"] is None
    assert history_entry["policy_delta"]["risk_config"]["max_leverage"] == "8"
    assert path.exists()

    second = manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk", "ops"),
        evidence_id="diary-002",
        threshold_overrides={"warn_notional_limit": 60_000.0},
        policy_delta={
            "risk_config": {"max_total_exposure_pct": 0.45},
            "router_guardrails": {"max_latency_ms": 250},
        },
    )
    assert second.stage is PolicyLedgerStage.PILOT
    assert second.approvals == ("ops", "risk")
    assert second.threshold_overrides["warn_notional_limit"] == 60_000.0
    assert second.history
    last_transition = second.history[-1]
    assert last_transition["prior_stage"] == "paper"
    assert last_transition["next_stage"] == "pilot"

    persisted = PolicyLedgerStore(path).get("alpha.policy")
    assert persisted is not None
    assert persisted.stage is PolicyLedgerStage.PILOT
    assert persisted.policy_delta is not None
    assert persisted.policy_delta.risk_config["max_total_exposure_pct"] == 0.45


def test_policy_ledger_rejects_stage_regression(tmp_path: Path) -> None:
    path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(path)
    manager = LedgerReleaseManager(
        store,
        feature_flags=PolicyLedgerFeatureFlags(require_diary_evidence=False),
    )

    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PILOT,
    )

    with pytest.raises(ValueError):
        manager.promote(
            policy_id="alpha.policy",
            tactic_id="tactic.alpha",
            stage=PolicyLedgerStage.PAPER,
        )


def test_release_manager_threshold_resolution(tmp_path: Path) -> None:
    path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(path)
    manager = LedgerReleaseManager(
        store,
        feature_flags=PolicyLedgerFeatureFlags(require_diary_evidence=False),
    )

    thresholds = manager.resolve_thresholds("missing")
    assert thresholds["stage"] == PolicyLedgerStage.EXPERIMENT.value
    assert thresholds["warn_confidence_floor"] == pytest.approx(0.85)

    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="diary-alpha",
        threshold_overrides={"warn_confidence_floor": 0.58},
    )

    overrides = manager.resolve_thresholds("alpha.policy")
    assert overrides["stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert overrides["warn_confidence_floor"] == pytest.approx(0.58)
    assert overrides["warn_notional_limit"] == pytest.approx(100_000.0)


def test_policy_ledger_requires_diary_evidence(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    manager = LedgerReleaseManager(store)

    with pytest.raises(ValueError):
        manager.promote(
            policy_id="alpha.policy",
            tactic_id="tactic.alpha",
            stage=PolicyLedgerStage.PAPER,
        )


def test_policy_ledger_rejects_blank_evidence(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    manager = LedgerReleaseManager(store)

    with pytest.raises(ValueError):
        manager.promote(
            policy_id="alpha.policy",
            tactic_id="tactic.alpha",
            stage=PolicyLedgerStage.PAPER,
            evidence_id="   ",
            approvals=("risk",),
        )


def test_policy_ledger_rejects_unsigned_policy_delta(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")

    with pytest.raises(ValueError):
        store.upsert(
            policy_id="alpha.policy",
            tactic_id="tactic.alpha",
            stage=PolicyLedgerStage.PAPER,
            approvals=(),
            policy_delta={"risk_config": {"max_leverage": 9}},
        )


def test_policy_governance_workflow_builds_tasks(tmp_path: Path) -> None:
    path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(path)
    manager = LedgerReleaseManager(store)
    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk", "compliance"),
        evidence_id="diary-1",
        policy_delta={"risk_config": {"max_leverage": 8}},
    )

    snapshot = build_policy_governance_workflow(store)
    assert snapshot.workflows
    checklist = snapshot.workflows[0]
    assert checklist.name == "Policy Ledger Governance"
    assert checklist.tasks
    task = checklist.tasks[0]
    assert task.status.value in {"completed", "in_progress"}
    assert task.metadata["policy_id"] == "alpha.policy"


def test_policy_ledger_trims_evidence_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(path)
    manager = LedgerReleaseManager(
        store,
        feature_flags=PolicyLedgerFeatureFlags(require_diary_evidence=False),
    )

    record = manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PAPER,
        evidence_id="  diary-123  ",
        approvals=("risk",),
    )

    assert record.evidence_id == "diary-123"

    reloaded = PolicyLedgerStore(path).get("alpha.policy")
    assert reloaded is not None
    assert reloaded.evidence_id == "diary-123"


def test_release_manager_enforces_audit_coverage(tmp_path: Path) -> None:
    path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(path)
    manager = LedgerReleaseManager(
        store,
        feature_flags=PolicyLedgerFeatureFlags(require_diary_evidence=False),
    )

    manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
    )

    enforced_stage = manager.resolve_stage("alpha")
    assert enforced_stage is PolicyLedgerStage.EXPERIMENT
    summary = manager.describe("alpha")
    assert summary["stage"] == PolicyLedgerStage.EXPERIMENT.value
    assert summary["declared_stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert summary["audit_enforced"] is True
    assert "missing_evidence" in summary.get("audit_gaps", [])

    manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        evidence_id="diary-alpha",
        approvals=("risk",),
    )

    enforced_stage = manager.resolve_stage("alpha")
    assert enforced_stage is PolicyLedgerStage.PILOT
    summary = manager.describe("alpha")
    assert summary["stage"] == PolicyLedgerStage.PILOT.value
    assert summary["audit_stage"] == PolicyLedgerStage.PILOT.value
    assert "additional_approval_needed" in summary.get("audit_gaps", [])

    manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        evidence_id="diary-alpha",
        approvals=("risk", "ops"),
    )

    enforced_stage = manager.resolve_stage("alpha")
    assert enforced_stage is PolicyLedgerStage.LIMITED_LIVE
    summary = manager.describe("alpha")
    assert summary["stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert summary.get("audit_gaps", []) == []
    assert summary["audit_enforced"] is False


def test_policy_ledger_feature_flags_env_parsing() -> None:
    enabled = PolicyLedgerFeatureFlags.from_env({"POLICY_LEDGER_REQUIRE_DIARY": "ON"})
    disabled = PolicyLedgerFeatureFlags.from_env({"POLICY_LEDGER_REQUIRE_DIARY": "off"})
    null_coerced = PolicyLedgerFeatureFlags.from_env({"POLICY_LEDGER_REQUIRE_DIARY": None})

    assert enabled.require_diary_evidence is True
    assert disabled.require_diary_evidence is False
    assert null_coerced.require_diary_evidence is True


def test_parse_bool_handles_none() -> None:
    assert _parse_bool(None) is False
