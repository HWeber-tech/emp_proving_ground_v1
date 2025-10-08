from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from src.understanding.decision_diary import DecisionDiaryStore
from tools.governance.promote_policy import main


def test_promote_policy_cli_with_delta(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger_path = tmp_path / "ledger.json"
    log_path = tmp_path / "promotions.jsonl"

    result = main(
        [
            "--ledger",
            str(ledger_path),
            "--log-file",
            str(log_path),
            "--policy-id",
            "alpha",
            "--stage",
            "pilot",
            "--approval",
            "risk",
            "--approval",
            "compliance",
            "--evidence-id",
            "dd-alpha",
            "--threshold",
            "warn_confidence_floor=0.68",
            "--threshold",
            "block_severity=alert",
            "--metadata",
            "owner=loop",
            "--delta-regime",
            "balanced",
            "--delta-note",
            "Increase leverage",
            "--delta-risk-config",
            "max_leverage=6",
            "--delta-guardrail",
            "requires_diary=true",
        ]
    )
    assert result == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["policy_id"] == "alpha"
    assert payload["stage"] == PolicyLedgerStage.PILOT.value
    posture = payload["release_posture"]
    assert posture["stage"] == PolicyLedgerStage.PILOT.value
    thresholds = posture["thresholds"]
    assert pytest.approx(thresholds["warn_confidence_floor"], abs=1e-6) == 0.68
    assert thresholds["block_severity"] == "alert"
    assert payload["log_path"] == str(log_path)

    store = PolicyLedgerStore(ledger_path)
    record = store.get("alpha")
    assert record is not None
    assert record.stage is PolicyLedgerStage.PILOT
    assert record.approvals == ("compliance", "risk")
    assert record.evidence_id == "dd-alpha"
    assert record.threshold_overrides["warn_confidence_floor"] == 0.68
    assert record.policy_delta is not None
    assert record.policy_delta.router_guardrails["requires_diary"] is True

    log_lines = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    assert log_lines
    assert log_lines[0]["policy_id"] == "alpha"
    assert log_lines[0]["stage"] == PolicyLedgerStage.PILOT.value


def test_promote_policy_cli_requires_diary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger_path = tmp_path / "ledger.json"
    diary_path = tmp_path / "diary.json"
    DecisionDiaryStore(diary_path, publish_on_record=False)

    result = main(
        [
            "--ledger",
            str(ledger_path),
            "--log-file",
            str(tmp_path / "promotions.jsonl"),
            "--policy-id",
            "alpha",
            "--stage",
            "paper",
            "--diary",
            str(diary_path),
            "--evidence-id",
            "dd-missing",
        ]
    )
    assert result == 1
    captured = capsys.readouterr()
    assert "failed to promote policy" in captured.err

    store = PolicyLedgerStore(ledger_path)
    assert store.get("alpha") is None


def test_promote_policy_cli_allows_missing_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger_path = tmp_path / "ledger.json"
    log_path = tmp_path / "promotions.jsonl"

    result = main(
        [
            "--ledger",
            str(ledger_path),
            "--log-file",
            str(log_path),
            "--policy-id",
            "alpha",
            "--stage",
            "experiment",
            "--allow-missing-evidence",
        ]
    )
    assert result == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["stage"] == PolicyLedgerStage.EXPERIMENT.value

    store = PolicyLedgerStore(ledger_path)
    record = store.get("alpha")
    assert record is not None
    assert record.stage is PolicyLedgerStage.EXPERIMENT
    assert record.evidence_id is None
    assert log_path.read_text().strip()


def test_promote_policy_cli_rejects_limited_live_with_single_approval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger_path = tmp_path / "ledger.json"
    log_path = tmp_path / "promotions.jsonl"

    result = main(
        [
            "--ledger",
            str(ledger_path),
            "--log-file",
            str(log_path),
            "--policy-id",
            "alpha",
            "--stage",
            "limited_live",
            "--approval",
            "risk",
            "--evidence-id",
            "dd-alpha",
        ]
    )
    assert result == 2
    captured = capsys.readouterr()
    assert "limited_live require at least 2 approval" in captured.err
    assert not log_path.exists()


def test_promote_policy_cli_writes_markdown_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger_path = tmp_path / "ledger.json"
    log_path = tmp_path / "promotions.jsonl"
    summary_path = tmp_path / "promotion.md"

    result = main(
        [
            "--ledger",
            str(ledger_path),
            "--log-file",
            str(log_path),
            "--summary-path",
            str(summary_path),
            "--policy-id",
            "beta",
            "--stage",
            "pilot",
            "--approval",
            "risk",
            "--approval",
            "compliance",
            "--evidence-id",
            "dd-beta",
        ]
    )
    assert result == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["summary_path"] == str(summary_path)
    assert summary_path.exists()
    summary_text = summary_path.read_text()
    assert "Policy Promotion Summary" in summary_text
    assert "beta" in summary_text
