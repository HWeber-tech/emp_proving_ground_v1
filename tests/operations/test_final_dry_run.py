from __future__ import annotations

import json
import sys
from datetime import timedelta

import pytest

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run import FinalDryRunConfig, run_final_dry_run
from src.operations.final_dry_run_workflow import run_final_dry_run_workflow


_HEARTBEAT_SCRIPT = r"""
import datetime
import json
import signal
import sys
import time

_running = True

def _stop(_signum, _frame):  # pragma: no cover - signal path
    global _running
    _running = False

signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)

start = time.time()
count = 0
while _running and time.time() - start < 2.0:
    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "level": "info",
        "event": "heartbeat",
        "counter": count,
    }
    print(json.dumps(payload), flush=True)
    count += 1
    time.sleep(0.1)

sys.exit(0)
"""


@pytest.mark.slow
def test_final_dry_run_success(tmp_path):
    config = FinalDryRunConfig(
        command=[sys.executable, "-c", _HEARTBEAT_SCRIPT],
        duration=timedelta(seconds=0.8),
        required_duration=timedelta(seconds=0.6),
        log_directory=tmp_path,
        minimum_uptime_ratio=0.8,
        require_diary_evidence=False,
        require_performance_evidence=False,
    )

    result = run_final_dry_run(config)

    assert result.status is DryRunStatus.pass_
    assert result.summary.log_summary is not None
    assert result.summary.log_summary.status is DryRunStatus.pass_
    assert result.log_path.exists()

    first_line = result.log_path.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(first_line)
    assert payload["stream"] == "stdout"
    assert payload["payload"]["structured"]["event"] == "heartbeat"


def test_final_dry_run_detects_early_exit(tmp_path):
    config = FinalDryRunConfig(
        command=[sys.executable, "-c", "import sys; sys.exit(0)"],
        duration=timedelta(seconds=0.5),
        required_duration=timedelta(seconds=0.5),
        log_directory=tmp_path,
        minimum_uptime_ratio=0.0,
        require_diary_evidence=False,
        require_performance_evidence=False,
    )

    result = run_final_dry_run(config)

    assert result.status is DryRunStatus.fail
    assert any(
        incident.message.startswith("Dry run completed before") for incident in result.incidents
    )
    assert result.summary.status is DryRunStatus.fail


def test_final_dry_run_workflow_builds_packet_and_review(tmp_path):
    packet_dir = tmp_path / "packet"
    archive_path = tmp_path / "packet.tar.gz"

    config = FinalDryRunConfig(
        command=[sys.executable, "-c", _HEARTBEAT_SCRIPT],
        duration=timedelta(seconds=0.8),
        required_duration=timedelta(seconds=0.6),
        log_directory=tmp_path,
        minimum_uptime_ratio=0.8,
        require_diary_evidence=False,
        require_performance_evidence=False,
    )

    workflow = run_final_dry_run_workflow(
        config,
        evidence_dir=packet_dir,
        evidence_archive=archive_path,
        review_run_label="Test Run",
        review_attendees=("Alice", "Bob"),
        review_notes=("Inspect metrics",),
    )

    assert workflow.run_result.status is DryRunStatus.pass_

    packet = workflow.evidence_packet
    assert packet is not None
    assert packet.summary_json.exists()
    assert packet.summary_markdown.exists()
    assert packet.archive_path == archive_path
    assert archive_path.exists()

    review = workflow.review
    assert review is not None
    assert review.status is DryRunStatus.pass_
    assert review.run_label == "Test Run"
    assert set(review.attendees) == {"Alice", "Bob"}
    assert review.evidence_packet == packet
