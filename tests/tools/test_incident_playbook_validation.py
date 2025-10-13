from __future__ import annotations

import json
from pathlib import Path

from tools.operations import incident_playbook_validation


def test_incident_playbook_validation_creates_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "artifacts" / "incident"
    exit_code = incident_playbook_validation.main(
        [
            "--run-root",
            str(run_root),
            "--timestamp",
            "20250102T030405Z",
            "--log-level",
            "WARNING",
        ]
    )

    assert exit_code == 0

    run_dir = run_root / "20250102T030405Z"
    assert run_dir.exists()

    summary_path = run_dir / "incident_playbook_summary.json"
    kill_path = run_dir / "kill_switch.json"
    replay_path = run_dir / "nightly_replay.json"
    rollback_path = run_dir / "trade_rollback.json"

    for artifact in (summary_path, kill_path, replay_path, rollback_path):
        assert artifact.exists(), f"missing artifact: {artifact}"

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary_payload["results"]

    assert results["kill_switch"]["status"] == "passed"
    assert "kill_switch_path" in results["kill_switch"]

    replay_result = results["nightly_replay"]
    assert replay_result["status"] == "passed"
    assert replay_result["tactic_count"] == 2
    assert replay_result["diary_entry_count"] == 2

    rollback_result = results["trade_rollback"]
    assert rollback_result["status"] == "passed"
    assert rollback_result["retry_allowed"] is True
