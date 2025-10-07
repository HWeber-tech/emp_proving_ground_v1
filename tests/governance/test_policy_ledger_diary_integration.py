from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.thinking.adaptation.policy_router import PolicyDecision, RegimeState
from src.understanding.decision_diary import DecisionDiaryStore
from src.understanding.probe_registry import ProbeRegistry

try:
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - python 3.10 fallback
    UTC = timezone.utc  # type: ignore[assignment]


@pytest.fixture()
def fixed_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    deterministic = uuid.UUID("abcdef12-3456-7890-abcd-ef1234567890")
    monkeypatch.setattr(
        "src.understanding.decision_diary.uuid.uuid4",
        lambda: deterministic,
    )


def _now() -> datetime:
    return datetime(2024, 2, 1, 9, 0, tzinfo=UTC)


def test_policy_ledger_requires_diary_evidence(tmp_path, fixed_uuid) -> None:
    ledger_store = PolicyLedgerStore(tmp_path / "policy_ledger.json", now=_now)
    diary_store = DecisionDiaryStore(tmp_path / "decision_diary.json", now=_now, probe_registry=ProbeRegistry())

    decision = PolicyDecision(
        tactic_id="alpha.shadow",
        parameters={"size": 2},
        selected_weight=1.1,
        guardrails={"requires_diary": True},
        rationale="Shadow cadence",
        experiments_applied=(),
        reflection_summary={},
    )
    regime = RegimeState(regime="balanced", confidence=0.75, features={}, timestamp=_now())

    entry = diary_store.record(
        policy_id="alpha.policy",
        decision=decision,
        regime_state=regime,
        outcomes={"reviewer_confidence": 0.81},
    )

    manager = LedgerReleaseManager(ledger_store, evidence_resolver=diary_store.exists)

    with pytest.raises(ValueError, match="DecisionDiary evidence_id is required"):
        manager.promote(
            policy_id="alpha.policy",
            tactic_id="alpha.shadow",
            stage=PolicyLedgerStage.PAPER,
            approvals=("risk",),
        )

    record = manager.promote(
        policy_id="alpha.policy",
        tactic_id="alpha.shadow",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id=entry.entry_id,
    )

    assert record.evidence_id == entry.entry_id
    assert ledger_store.get("alpha.policy").stage is PolicyLedgerStage.PAPER
