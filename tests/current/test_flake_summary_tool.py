from __future__ import annotations

import json
from pathlib import Path

from tools.telemetry.summarize_flakes import format_summary, summarize_flake_log


def test_summarize_flake_log_parses_failures(tmp_path: Path) -> None:
    payload = {
        "meta": {"session_start": 1, "session_end": 2, "exit_status": 1},
        "events": [
            {"nodeid": "tests/test_example.py::test_ok", "outcome": "passed"},
            {"nodeid": "tests/test_example.py::test_fail", "outcome": "failed"},
        ],
        "history": [
            {"run_id": 10, "conclusion": "failure"},
            {"run_id": 11, "conclusion": "success"},
        ],
    }
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps(payload))

    summary = summarize_flake_log(log_path)

    assert summary["failure_count"] == 1
    assert summary["failing_tests"] == ["tests/test_example.py::test_fail"]
    assert summary["history"] == [
        {"run_id": 10, "conclusion": "failure"},
        {"run_id": 11, "conclusion": "success"},
    ]


def test_format_summary_includes_key_metrics(tmp_path: Path) -> None:
    payload = {
        "meta": {"session_start": 5, "session_end": 9, "exit_status": 0},
        "events": [],
        "history": [],
    }
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps(payload))

    summary = summarize_flake_log(log_path)
    formatted = format_summary(summary)

    assert "Pytest flake telemetry" in formatted
    assert "Failure count: 0" in formatted
    assert "Failing tests: none" in formatted
