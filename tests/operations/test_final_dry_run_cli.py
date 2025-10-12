from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from emp.__main__ import cli as emp_cli


_HEARTBEAT_SCRIPT = textwrap.dedent(
    """
    import datetime
    import json
    import signal
    import time

    running = True

    def _stop(_signum, _frame):  # pragma: no cover - signal path
        global running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    start = time.time()
    count = 0
    while running and time.time() - start < 2.0:
        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": "info",
            "event": "heartbeat",
            "counter": count,
        }
        print(json.dumps(payload), flush=True)
        count += 1
        time.sleep(0.1)
    """
)


_INSTANT_EXIT_SCRIPT = "import sys; sys.exit(0)"


@pytest.mark.slow
def test_final_dry_run_cli_success(tmp_path: Path) -> None:
    runner = CliRunner()
    log_dir = tmp_path / "logs"
    progress_path = tmp_path / "progress.json"
    json_report = tmp_path / "summary.json"
    markdown_report = tmp_path / "summary.md"
    review_output = tmp_path / "review.md"

    result = runner.invoke(
        emp_cli,
        [
            "final-dry-run",
            "--log-dir",
            str(log_dir),
            "--duration-hours",
            "0.0005",
            "--required-duration-hours",
            "0.0003",
            "--minimum-uptime-ratio",
            "0.8",
            "--no-diary-required",
            "--no-performance-required",
            "--progress-path",
            str(progress_path),
            "--json-report",
            str(json_report),
            "--markdown-report",
            str(markdown_report),
            "--review-output",
            str(review_output),
            "--metadata",
            "run_id=test-cli",
            "--review-run-label",
            "CLI Dry Run",
            "--attendee",
            "Ops",
            "--note",
            "Check latency charts",
            "--review-skip-sign-off",
            "--",
            sys.executable,
            "-c",
            _HEARTBEAT_SCRIPT,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Final dry run status: PASS" in result.output
    assert log_dir.exists()
    assert any(log_dir.glob("final_dry_run_*.jsonl"))
    assert progress_path.exists()

    payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert "summary" in payload

    review_text = review_output.read_text(encoding="utf-8")
    assert "Final Dry Run Review" in review_text


def test_final_dry_run_cli_failure_on_short_run(tmp_path: Path) -> None:
    runner = CliRunner()
    log_dir = tmp_path / "logs"

    result = runner.invoke(
        emp_cli,
        [
            "final-dry-run",
            "--log-dir",
            str(log_dir),
            "--duration-hours",
            "0.0005",
            "--required-duration-hours",
            "0.0005",
            "--no-diary-required",
            "--no-performance-required",
            "--",
            sys.executable,
            "-c",
            _INSTANT_EXIT_SCRIPT,
        ],
    )

    assert result.exit_code == 2
    assert "FAIL" in result.output
