from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from src.governance.policy_graduation import PolicyGraduationEvaluator
from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.thinking.adaptation.policy_router import RegimeState
from src.understanding.decision_diary import DecisionDiaryStore
from tools.governance import alpha_trade_graduation as graduation_cli
from tests.util import promotion_checklist_metadata


_UTC = timezone.utc


def _record_entry(
    store: DecisionDiaryStore,
    *,
    policy_id: str,
    stage: PolicyLedgerStage,
    severity: str = "normal",
    force_paper: bool = False,
    recorded_at: datetime,
) -> None:
    decision_payload = {
        "tactic_id": policy_id,
        "parameters": {},
        "selected_weight": 1.0,
        "guardrails": {},
        "rationale": "auto",
        "experiments_applied": (),
        "reflection_summary": {},
    }
    regime = RegimeState(
        regime="balanced",
        confidence=0.8,
        features={},
        timestamp=recorded_at,
    )
    metadata = {
        "release_stage": stage.value,
        "drift_decision": {
            "severity": severity,
            "force_paper": force_paper,
        },
        "release_execution": {
            "stage": stage.value,
            "route": "live" if stage is PolicyLedgerStage.LIMITED_LIVE else "paper",
            "forced": force_paper,
        },
    }
    store.record(
        policy_id=policy_id,
        decision=decision_payload,
        regime_state=regime,
        outcomes={"paper_pnl": 0.0},
        metadata=metadata,
        recorded_at=recorded_at,
    )


def _build_release_manager(path: Path) -> LedgerReleaseManager:
    store = PolicyLedgerStore(path)
    return LedgerReleaseManager(store)


def test_assessment_recommends_paper(tmp_path: Path) -> None:
    diary = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)
    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.EXPERIMENT,
        evidence_id="dd-alpha-initial",
    )
    release_manager = LedgerReleaseManager(store)

    base = datetime(2024, 1, 1, tzinfo=_UTC)
    for index in range(25):
        severity = "normal" if index < 23 else "warn"
        _record_entry(
            diary,
            policy_id="alpha",
            stage=PolicyLedgerStage.EXPERIMENT,
            severity=severity,
            force_paper=False,
            recorded_at=base + timedelta(minutes=index),
        )

    evaluator = PolicyGraduationEvaluator(release_manager, diary)
    assessment = evaluator.assess("alpha")

    assert assessment.recommended_stage is PolicyLedgerStage.PAPER
    assert assessment.stage_blockers[PolicyLedgerStage.PAPER] == ()
    assert assessment.stage_blockers[PolicyLedgerStage.PILOT], "expected pilot blockers when still paper"


def test_assessment_recommends_pilot_when_paper_metrics_green(tmp_path: Path) -> None:
    diary = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    store.upsert(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.EXPERIMENT,
        evidence_id="dd-alpha-exp",
    )
    release_manager = LedgerReleaseManager(store)

    base = datetime(2024, 1, 1, tzinfo=_UTC)
    for index in range(25):
        _record_entry(
            diary,
            policy_id="alpha",
            stage=PolicyLedgerStage.EXPERIMENT,
            recorded_at=base + timedelta(minutes=index),
        )

    # Advance ledger to paper stage and add fresh entries.
    store.upsert(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="dd-alpha-paper",
    )
    paper_start = base + timedelta(hours=5)
    warn_indices = {5, 17, 29}
    for index in range(45):
        severity = "warn" if index in warn_indices else "normal"
        _record_entry(
            diary,
            policy_id="alpha",
            stage=PolicyLedgerStage.PAPER,
            severity=severity,
            recorded_at=paper_start + timedelta(minutes=index),
        )

    evaluator = PolicyGraduationEvaluator(release_manager, diary)
    assessment = evaluator.assess("alpha")

    assert assessment.recommended_stage is PolicyLedgerStage.PILOT
    assert assessment.stage_blockers[PolicyLedgerStage.PAPER] == ()
    assert assessment.stage_blockers[PolicyLedgerStage.PILOT] == ()
    assert assessment.stage_blockers[PolicyLedgerStage.LIMITED_LIVE], "limited live should not be ready yet"


def test_assessment_requires_additional_approvals_for_limited_live(tmp_path: Path) -> None:
    diary = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)
    store = PolicyLedgerStore(tmp_path / "ledger.json")

    store.upsert(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.EXPERIMENT,
        evidence_id="dd-alpha-exp",
    )
    release_manager = LedgerReleaseManager(store)

    base = datetime(2024, 1, 1, tzinfo=_UTC)
    for index in range(20):
        _record_entry(
            diary,
            policy_id="alpha",
            stage=PolicyLedgerStage.EXPERIMENT,
            recorded_at=base + timedelta(minutes=index),
        )

    store.upsert(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="dd-alpha-paper",
    )
    paper_start = base + timedelta(hours=4)
    for index in range(40):
        _record_entry(
            diary,
            policy_id="alpha",
            stage=PolicyLedgerStage.PAPER,
            recorded_at=paper_start + timedelta(minutes=index),
        )

    store.upsert(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk",),
        evidence_id="dd-alpha-pilot",
    )
    pilot_start = base + timedelta(hours=8)
    for index in range(60):
        _record_entry(
            diary,
            policy_id="alpha",
            stage=PolicyLedgerStage.PILOT,
            recorded_at=pilot_start + timedelta(minutes=index),
        )

    evaluator = PolicyGraduationEvaluator(release_manager, diary)
    assessment = evaluator.assess("alpha")

    assert assessment.recommended_stage is PolicyLedgerStage.PILOT
    blockers = assessment.stage_blockers[PolicyLedgerStage.LIMITED_LIVE]
    assert any("approvals_required" in blocker for blocker in blockers)


def test_cli_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    diary_path = tmp_path / "cli_diary.json"
    diary = DecisionDiaryStore(diary_path, publish_on_record=False)
    ledger_path = tmp_path / "cli_ledger.json"
    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.EXPERIMENT,
        evidence_id="dd-alpha-exp",
    )

    base = datetime(2024, 2, 1, tzinfo=_UTC)
    for index in range(20):
        _record_entry(
            diary,
            policy_id="alpha",
            stage=PolicyLedgerStage.EXPERIMENT,
            recorded_at=base + timedelta(minutes=index),
        )

    store.upsert(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="dd-alpha-paper",
    )
    paper_start = base + timedelta(hours=4)
    for index in range(45):
        _record_entry(
            diary,
            policy_id="alpha",
            stage=PolicyLedgerStage.PAPER,
            recorded_at=paper_start + timedelta(minutes=index),
        )

    exit_code = graduation_cli.main(
        [
            "--ledger",
            str(ledger_path),
            "--diary",
            str(diary_path),
            "--policy-id",
            "alpha",
            "--json",
            "--indent",
            "0",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload and payload[0]["policy_id"] == "alpha"
    assert payload[0]["recommended_stage"] == PolicyLedgerStage.PILOT.value
    assert payload[0]["applied_stage"] is None


def test_cli_apply_promotes_stage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    diary_path = tmp_path / "apply_diary.json"
    diary = DecisionDiaryStore(diary_path, publish_on_record=False)
    ledger_path = tmp_path / "apply_ledger.json"
    store = PolicyLedgerStore(ledger_path)
    log_path = tmp_path / "promotions.jsonl"

    store.upsert(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.EXPERIMENT,
        evidence_id="dd-alpha-exp",
    )

    base = datetime(2024, 3, 1, 9, 0, tzinfo=_UTC)
    for index in range(25):
        _record_entry(
            diary,
            policy_id="alpha",
            stage=PolicyLedgerStage.EXPERIMENT,
            recorded_at=base + timedelta(minutes=index),
        )

    store.upsert(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="dd-alpha-paper",
    )
    paper_start = base + timedelta(hours=4)
    warn_indices = {5, 17, 29}
    for index in range(45):
        severity = "warn" if index in warn_indices else "normal"
        _record_entry(
            diary,
            policy_id="alpha",
            stage=PolicyLedgerStage.PAPER,
            severity=severity,
            recorded_at=paper_start + timedelta(minutes=index),
        )

    exit_code = graduation_cli.main(
        [
            "--ledger",
            str(ledger_path),
            "--diary",
            str(diary_path),
            "--policy-id",
            "alpha",
            "--apply",
            "--log-file",
            str(log_path),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "promotion_applied: pilot" in captured.out
    assert "Applied promotions:" in captured.out
    assert "alpha: pilot" in captured.out

    refreshed_store = PolicyLedgerStore(ledger_path)
    record = refreshed_store.get("alpha")
    assert record is not None
    assert record.stage is PolicyLedgerStage.PILOT
    log_entries = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    assert log_entries
    assert log_entries[0]["policy_id"] == "alpha"
    assert log_entries[0]["stage"] == PolicyLedgerStage.PILOT.value


def test_limited_live_blocked_without_paper_green_duration(tmp_path: Path) -> None:
    diary = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)
    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path)
    policy_id = "alpha"

    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.EXPERIMENT,
        evidence_id="dd-alpha-exp",
    )

    base = datetime(2024, 4, 1, 9, 0, tzinfo=_UTC)
    for index in range(20):
        _record_entry(
            diary,
            policy_id=policy_id,
            stage=PolicyLedgerStage.EXPERIMENT,
            recorded_at=base + timedelta(minutes=index * 5),
        )

    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="dd-alpha-paper",
        metadata=promotion_checklist_metadata(),
    )

    paper_start = base + timedelta(hours=4)
    for index in range(40):
        _record_entry(
            diary,
            policy_id=policy_id,
            stage=PolicyLedgerStage.PAPER,
            recorded_at=paper_start + timedelta(hours=6 * index),
        )

    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk", "compliance"),
        evidence_id="dd-alpha-pilot",
        metadata=promotion_checklist_metadata(),
    )

    pilot_start = paper_start + timedelta(hours=6 * 39 + 1)
    for index in range(60):
        _record_entry(
            diary,
            policy_id=policy_id,
            stage=PolicyLedgerStage.PILOT,
            recorded_at=pilot_start + timedelta(minutes=30 * index),
        )

    release_manager = LedgerReleaseManager(store)
    evaluator = PolicyGraduationEvaluator(release_manager, diary)
    assessment = evaluator.assess(policy_id)

    assert assessment.recommended_stage is PolicyLedgerStage.PILOT
    blockers = assessment.stage_blockers[PolicyLedgerStage.LIMITED_LIVE]
    assert any(blocker.startswith("paper_green_gate_duration_below") for blocker in blockers)


def test_limited_live_recommended_after_paper_green_duration(tmp_path: Path) -> None:
    diary = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)
    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path)
    policy_id = "alpha"

    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.EXPERIMENT,
        evidence_id="dd-alpha-exp",
    )

    base = datetime(2024, 5, 1, 9, 0, tzinfo=_UTC)
    for index in range(20):
        _record_entry(
            diary,
            policy_id=policy_id,
            stage=PolicyLedgerStage.EXPERIMENT,
            recorded_at=base + timedelta(minutes=4 * index),
        )

    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="dd-alpha-paper",
        metadata=promotion_checklist_metadata(),
    )

    paper_start = base + timedelta(hours=3)
    for index in range(40):
        _record_entry(
            diary,
            policy_id=policy_id,
            stage=PolicyLedgerStage.PAPER,
            recorded_at=paper_start + timedelta(hours=12 * index),
        )

    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk", "compliance"),
        evidence_id="dd-alpha-pilot",
        metadata=promotion_checklist_metadata(),
    )

    pilot_start = paper_start + timedelta(hours=12 * 39 + 2)
    for index in range(60):
        _record_entry(
            diary,
            policy_id=policy_id,
            stage=PolicyLedgerStage.PILOT,
            recorded_at=pilot_start + timedelta(minutes=20 * index),
        )

    release_manager = LedgerReleaseManager(store)
    evaluator = PolicyGraduationEvaluator(release_manager, diary)
    assessment = evaluator.assess(policy_id)

    assert assessment.recommended_stage is PolicyLedgerStage.LIMITED_LIVE
    assert assessment.stage_blockers[PolicyLedgerStage.LIMITED_LIVE] == ()


def test_limited_live_blocked_when_recent_paper_gate_regresses(tmp_path: Path) -> None:
    diary = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)
    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path)
    policy_id = "alpha"

    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.EXPERIMENT,
        evidence_id="dd-alpha-exp",
    )

    base = datetime(2024, 6, 1, 9, 0, tzinfo=_UTC)
    for index in range(20):
        _record_entry(
            diary,
            policy_id=policy_id,
            stage=PolicyLedgerStage.EXPERIMENT,
            recorded_at=base + timedelta(minutes=6 * index),
        )

    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="dd-alpha-paper",
        metadata=promotion_checklist_metadata(),
    )

    paper_start = base + timedelta(hours=2)
    for index in range(41):
        severity = "normal" if index < 40 else "warn"
        _record_entry(
            diary,
            policy_id=policy_id,
            stage=PolicyLedgerStage.PAPER,
            severity=severity,
            recorded_at=paper_start + timedelta(hours=9 * index),
        )

    store.upsert(
        policy_id=policy_id,
        tactic_id=policy_id,
        stage=PolicyLedgerStage.PILOT,
        approvals=("risk", "compliance"),
        evidence_id="dd-alpha-pilot",
        metadata=promotion_checklist_metadata(),
    )

    pilot_start = paper_start + timedelta(hours=9 * 40 + 1)
    for index in range(60):
        _record_entry(
            diary,
            policy_id=policy_id,
            stage=PolicyLedgerStage.PILOT,
            recorded_at=pilot_start + timedelta(minutes=20 * index),
        )

    release_manager = LedgerReleaseManager(store)
    evaluator = PolicyGraduationEvaluator(release_manager, diary)
    assessment = evaluator.assess(policy_id)

    assert assessment.recommended_stage is PolicyLedgerStage.PILOT
    blockers = assessment.stage_blockers[PolicyLedgerStage.LIMITED_LIVE]
    assert any(blocker.startswith("paper_green_gate_duration_below") for blocker in blockers)
