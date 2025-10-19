from __future__ import annotations

import json
from pathlib import Path

from tools.operations import live_pilot_drill


def test_live_pilot_drill_creates_artifacts(tmp_path: Path) -> None:
    run_root = tmp_path / "artifacts" / "live_pilot"
    exit_code = live_pilot_drill.main(
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

    tiny_path = run_dir / "tiny_capital.json"
    kill_path = run_dir / "kill_switch.json"
    rollback_path = run_dir / "trade_rollback.json"
    reconciliation_path = run_dir / "reconciliation.json"
    summary_path = run_dir / "live_pilot_drill_summary.json"

    for artifact in (tiny_path, kill_path, rollback_path, reconciliation_path, summary_path):
        assert artifact.exists(), f"missing artifact: {artifact}"

    tiny_payload = json.loads(tiny_path.read_text(encoding="utf-8"))
    assert tiny_payload["status"] == "passed"
    assert tiny_payload["confirm_live"] is True
    assert tiny_payload["resolved_initial_capital"] < 50_000

    kill_payload = json.loads(kill_path.read_text(encoding="utf-8"))
    assert kill_payload["status"] == "passed"

    rollback_payload = json.loads(rollback_path.read_text(encoding="utf-8"))
    assert rollback_payload["status"] == "passed"
    assert rollback_payload["retry_allowed"] is True

    reconciliation_payload = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    assert reconciliation_payload["status"] == "passed"
    assert reconciliation_payload["differences"] == []

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary_payload["results"]
    assert all(entry["status"] == "passed" for entry in results.values())
