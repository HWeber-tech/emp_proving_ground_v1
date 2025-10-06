from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable

import pytest

from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerStage,
    PolicyLedgerStore,
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
    )

    assert first.stage is PolicyLedgerStage.PAPER
    assert first.approvals == ("compliance", "risk")
    assert first.evidence_id == "diary-001"
    assert first.threshold_overrides["warn_confidence_floor"] == 0.8
    assert first.history == ()
    assert path.exists()

    second = manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk", "ops"),
        threshold_overrides={"warn_notional_limit": 60_000.0},
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


def test_policy_ledger_rejects_stage_regression(tmp_path: Path) -> None:
    path = tmp_path / "policy_ledger.json"
    store = PolicyLedgerStore(path)
    manager = LedgerReleaseManager(store)

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
    manager = LedgerReleaseManager(store)

    thresholds = manager.resolve_thresholds("missing")
    assert thresholds["stage"] == PolicyLedgerStage.EXPERIMENT.value
    assert thresholds["warn_confidence_floor"] == pytest.approx(0.85)

    manager.promote(
        policy_id="alpha.policy",
        tactic_id="tactic.alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        threshold_overrides={"warn_confidence_floor": 0.58},
    )

    overrides = manager.resolve_thresholds("alpha.policy")
    assert overrides["stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert overrides["warn_confidence_floor"] == pytest.approx(0.58)
    assert overrides["warn_notional_limit"] == pytest.approx(100_000.0)
