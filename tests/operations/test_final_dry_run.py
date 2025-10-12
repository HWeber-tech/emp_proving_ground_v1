from __future__ import annotations

import json
import sys
import asyncio
from datetime import timedelta

import pytest

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run import (
    FinalDryRunConfig,
    perform_final_dry_run,
    run_final_dry_run,
)
from src.operations.final_dry_run_workflow import run_final_dry_run_workflow
from src.runtime.task_supervisor import TaskSupervisor


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


_ERROR_ONCE_SCRIPT = r"""
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


def _emit(level, event, message):
    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "level": level,
        "event": event,
        "message": message,
    }
    print(json.dumps(payload), flush=True)


_emit("info", "startup", "Runtime initialised")
_emit("error", "unexpected", "Boom! encountered fatal condition")

start = time.time()
while _running and time.time() - start < 4.0:
    time.sleep(0.1)

sys.exit(0)
"""


async def _wait_for_file(path, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if path.exists():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for {path}")


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

    assert result.progress_path is not None
    progress = json.loads(result.progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == DryRunStatus.pass_.value
    assert progress["phase"] == "complete"
    assert progress["total_lines"] >= 1


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
    assert result.progress_path is not None
    progress = json.loads(result.progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == DryRunStatus.fail.value
    assert progress["phase"] == "complete"


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
    raw_names = {path.name for path in packet.raw_artifacts}
    assert any(name.endswith("_progress.json") for name in raw_names)

    review = workflow.review
    assert review is not None
    assert review.status is DryRunStatus.pass_
    assert review.run_label == "Test Run"
    assert set(review.attendees) == {"Alice", "Bob"}
    assert review.evidence_packet == packet


@pytest.mark.asyncio()
async def test_perform_final_dry_run_supervises_background_tasks(tmp_path):
    config = FinalDryRunConfig(
        command=[sys.executable, "-c", _HEARTBEAT_SCRIPT],
        duration=timedelta(seconds=0.6),
        required_duration=timedelta(seconds=0.5),
        log_directory=tmp_path,
        minimum_uptime_ratio=0.5,
        require_diary_evidence=False,
        require_performance_evidence=False,
    )

    supervisor = TaskSupervisor(namespace="test-final-dry-run")

    run_task = asyncio.create_task(
        perform_final_dry_run(config, task_supervisor=supervisor)
    )

    # Allow the harness to start its background tasks under supervision.
    await asyncio.sleep(0.1)
    snapshots = supervisor.describe()
    task_names = {snapshot.get("name") for snapshot in snapshots}
    expected_names = {
        "dry-run-stdout",
        "dry-run-stderr",
        "dry-run-duration-timeout",
        "dry-run-process-wait",
        "dry-run-progress-reporter",
    }
    assert expected_names.issubset(task_names)

    result = await run_task

    assert result.status is DryRunStatus.pass_
    assert supervisor.active_count == 0

    await supervisor.cancel_all()


def test_final_dry_run_records_log_incidents(tmp_path):
    config = FinalDryRunConfig(
        command=[sys.executable, "-c", _ERROR_ONCE_SCRIPT],
        duration=timedelta(seconds=0.6),
        required_duration=timedelta(seconds=0.3),
        log_directory=tmp_path,
        minimum_uptime_ratio=0.5,
        require_diary_evidence=False,
        require_performance_evidence=False,
    )

    result = run_final_dry_run(config)

    assert result.status is DryRunStatus.fail
    assert any(
        incident.severity is DryRunStatus.fail and "ERROR log" in incident.message
        for incident in result.incidents
    )


def test_final_dry_run_monitor_can_be_disabled(tmp_path):
    config = FinalDryRunConfig(
        command=[sys.executable, "-c", _ERROR_ONCE_SCRIPT],
        duration=timedelta(seconds=0.6),
        required_duration=timedelta(seconds=0.3),
        log_directory=tmp_path,
        minimum_uptime_ratio=0.5,
        require_diary_evidence=False,
        require_performance_evidence=False,
        monitor_log_levels=False,
    )

    result = run_final_dry_run(config)

    assert result.status is DryRunStatus.fail
    assert not result.incidents


@pytest.mark.asyncio()
async def test_final_dry_run_progress_updates_on_incident(tmp_path):
    progress_path = tmp_path / "progress.json"
    config = FinalDryRunConfig(
        command=[sys.executable, "-c", _ERROR_ONCE_SCRIPT],
        duration=timedelta(seconds=1.4),
        required_duration=timedelta(seconds=1.0),
        log_directory=tmp_path,
        minimum_uptime_ratio=0.0,
        require_diary_evidence=False,
        require_performance_evidence=False,
        progress_path=progress_path,
        progress_interval=timedelta(seconds=30),
    )

    run_task = asyncio.create_task(perform_final_dry_run(config))

    await _wait_for_file(progress_path)
    incidents: list[dict[str, object]] = []
    for _ in range(40):
        progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
        incidents = progress_payload.get("incidents", [])
        if incidents:
            break
        await asyncio.sleep(0.05)
    else:  # pragma: no cover - defensive guard
        pytest.fail("expected incidents to be written to progress snapshot")

    assert progress_payload["status"] == DryRunStatus.fail.value
    assert any(incident.get("severity") == DryRunStatus.fail.value for incident in incidents)
    assert progress_payload["phase"] in {"running", "complete"}

    result = await run_task

    assert result.status is DryRunStatus.fail
    final_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert final_payload["phase"] == "complete"
