from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.governance.policy_ledger import PolicyDelta, PolicyLedgerStage
from src.governance.policy_traceability import build_traceability_metadata
from src.understanding.decision_diary import DecisionDiaryStore


def _record_diary_entry(store: DecisionDiaryStore, entry_id: str) -> None:
    store.record(
        policy_id="alpha",
        decision={
            "tactic_id": "alpha",
            "parameters": {},
            "guardrails": {},
            "rationale": "traceability",
            "selected_weight": 1.0,
            "experiments_applied": (),
            "reflection_summary": {},
            "weight_breakdown": {},
        },
        regime_state={
            "regime": "balanced",
            "confidence": 0.9,
            "features": {},
        },
        outcomes={"paper_pnl": 1.5},
        metadata={"release_stage": "paper"},
        entry_id=entry_id,
        recorded_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
    )


def test_build_traceability_metadata_includes_diary_slice(tmp_path: Path) -> None:
    diary_path = tmp_path / "diary.json"
    diary_store = DecisionDiaryStore(diary_path, publish_on_record=False)
    _record_diary_entry(diary_store, "dd-alpha")

    traceability = build_traceability_metadata(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PILOT,
        evidence_id="dd-alpha",
        diary_store=diary_store,
        diary_path=diary_path,
        threshold_overrides={"warn_confidence_floor": 0.7},
        policy_delta=PolicyDelta(notes=("increase leverage",)),
        metadata={"owner": "ops"},
    )

    assert traceability is not None
    assert traceability["stage"] == PolicyLedgerStage.PILOT.value
    assert traceability["evidence_id"] == "dd-alpha"
    assert traceability["code_hash"]
    config_hash = traceability["config_hash"]
    assert isinstance(config_hash, str) and len(config_hash) == 64
    diary_slice = traceability["diary_slice"]
    assert diary_slice["status"] == "ok"
    assert diary_slice["entry"]["entry_id"] == "dd-alpha"
    assert diary_slice["diary_path"] == diary_path.as_posix()


def test_build_traceability_metadata_without_diary_store() -> None:
    traceability = build_traceability_metadata(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PAPER,
        evidence_id="dd-missing",
        diary_store=None,
        diary_path=None,
        threshold_overrides=None,
        policy_delta=None,
        metadata=None,
    )

    assert traceability is not None
    diary_slice = traceability["diary_slice"]
    assert diary_slice["status"] == "store_unavailable"
    assert diary_slice["entry_id"] == "dd-missing"
