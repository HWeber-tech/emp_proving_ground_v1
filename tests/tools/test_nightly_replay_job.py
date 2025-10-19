from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.operations import nightly_replay_job


def test_nightly_replay_job_generates_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "mirror"
    monkeypatch.setenv("ALPHATRADE_ARTIFACT_ROOT", str(archive_root))
    run_root = tmp_path / "artifacts" / "nightly_replay"
    exit_code = nightly_replay_job.main(
        [
            "--run-root",
            str(run_root),
            "--timestamp",
            "2025-01-02T03:04:05Z",
            "--log-level",
            "WARNING",
        ]
    )

    assert exit_code == 0

    run_dir = run_root / "20250102T030405Z"
    assert run_dir.exists()

    dataset_path = run_dir / "recorded_snapshots.jsonl"
    assert dataset_path.exists()
    assert dataset_path.read_text(encoding="utf-8").strip()

    evaluation_report = run_dir / "replay_evaluation.json"
    drift_report = run_dir / "sensor_drift_summary.json"
    diary_path = run_dir / "decision_diary.json"
    ledger_path = run_dir / "policy_ledger.json"

    summary_payload = json.loads(evaluation_report.read_text(encoding="utf-8"))
    assert summary_payload["run_id"] == "nightly-replay-20250102T030405Z"
    assert len(summary_payload["results"]) == 2
    for result in summary_payload["results"]:
        assert "tactic_id" in result
        assert "metrics" in result
        assert result["snapshot_count"] >= 2

    drift_payload = json.loads(drift_report.read_text(encoding="utf-8"))
    assert drift_payload["total_observations"] >= 2
    assert "results" in drift_payload

    diary_payload = json.loads(diary_path.read_text(encoding="utf-8"))
    assert len(diary_payload.get("entries", [])) == 2

    ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert "records" in ledger_payload

    archive_dir = archive_root / "diaries" / "2025" / "01" / "02" / "nightly-replay-20250102T030405Z"
    assert (archive_dir / "decision_diary.json").exists()

    drift_dir = archive_root / "drift_reports" / "2025" / "01" / "02" / "nightly-replay-20250102T030405Z"
    assert (drift_dir / "sensor_drift_summary.json").exists()

    ledger_dir = archive_root / "ledger_exports" / "2025" / "01" / "02" / "nightly-replay-20250102T030405Z"
    assert (ledger_dir / "policy_ledger.json").exists()
