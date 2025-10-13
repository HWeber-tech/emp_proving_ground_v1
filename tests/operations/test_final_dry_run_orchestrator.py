from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from tools.operations import final_dry_run_orchestrator as orchestrator


_RUNTIME_SCRIPT = textwrap.dedent(
    """
    import datetime
    import json
    import os
    import signal
    import sys
    import time

    running = True

    def _stop(_signum, _frame):  # pragma: no cover - signal path
        global running
        running = False

    for _name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, _name):
            signal.signal(getattr(signal, _name), _stop)

    diary_path = os.environ.get("FINAL_DRY_RUN_DIARY_PATH")
    performance_path = os.environ.get("FINAL_DRY_RUN_PERFORMANCE_PATH")

    entries = []
    start = time.time()
    counter = 0
    while running and time.time() - start < 1.2:
        now = datetime.datetime.now(datetime.timezone.utc)
        record = {
            "timestamp": now.isoformat(),
            "level": "info",
            "event": "heartbeat",
            "counter": counter,
        }
        print(json.dumps(record), flush=True)

        if diary_path:
            entry = {
                "entry_id": f"unit-{counter:05d}",
                "recorded_at": now.isoformat(),
                "policy_id": "unit-test-policy",
                "decision": {
                    "tactic_id": "unit-test",
                    "rationale": "Final dry run orchestrator test",
                    "parameters": {"counter": counter},
                },
                "regime_state": {
                    "regime": "unit",
                    "confidence": 0.9,
                    "features": {"counter": counter},
                },
                "outcomes": {"status": "pending"},
                "metadata": {"source": "unit-test"},
            }
            entries.append(entry)
            diary_payload = {
                "generated_at": now.isoformat(),
                "entries": list(entries),
                "probe_registry": {
                    "generated_at": now.isoformat(),
                    "probes": {},
                },
            }
            with open(diary_path, "w", encoding="utf-8") as diary_file:
                json.dump(diary_payload, diary_file)

        if performance_path:
            performance_payload = {
                "generated_at": now.isoformat(),
                "period_start": now.isoformat(),
                "trades": counter,
                "roi": 0.01 * counter,
                "win_rate": 0.6,
                "sharpe_ratio": 1.5,
                "window_duration_seconds": 120.0,
                "metadata": {"note": "unit-test"},
            }
            with open(performance_path, "w", encoding="utf-8") as performance_file:
                json.dump(performance_payload, performance_file)

        counter += 1
        time.sleep(0.1)

    shutdown_record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "level": "info",
        "event": "shutdown",
    }
    print(json.dumps(shutdown_record), flush=True)
    sys.exit(0)
    """
)


@pytest.mark.slow
def test_orchestrator_creates_evidence_bundle(tmp_path, capsys):
    output_root = tmp_path / "runs"
    args = [
        "--output-root",
        str(output_root),
        "--run-label",
        "UAT rehearsal",
        "--duration-hours",
        "0.0006",
        "--required-duration-hours",
        "0.0002",
        "--minimum-uptime-ratio",
        "0.1",
        "--progress-interval-minutes",
        "0.0005",
        "--evidence-initial-grace-minutes",
        "0.0",
        "--objective",
        "governance=pass:Controls enforced",
        "--attendee",
        "Ops Lead",
        "--note",
        "Verify throttles",
        "--",
        sys.executable,
        "-c",
        _RUNTIME_SCRIPT,
    ]

    exit_code = orchestrator.main(args)
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Final dry run status: PASS" in captured.out

    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"
    review_json = run_dir / "review.json"
    review_md = run_dir / "review.md"
    wrap_up_json = run_dir / "wrap_up.json"
    wrap_up_md = run_dir / "wrap_up.md"
    log_dir = run_dir / "logs"
    timeline_dir = run_dir / "progress_timeline"
    packet_dir = run_dir / "packet"
    packet_archive = run_dir / "packet.tar.gz"

    assert summary_json.exists()
    assert summary_md.exists()
    assert review_json.exists()
    assert review_md.exists()
    assert wrap_up_json.exists()
    assert wrap_up_md.exists()
    assert log_dir.exists()
    assert any(log_dir.glob("final_dry_run_*.jsonl"))
    assert timeline_dir.exists()
    assert list(timeline_dir.glob("snapshot-*.json"))

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["status"] == "pass"
    assert payload["review"]["status"] == "pass"
    assert payload["summary"]["metadata"]["run_label"] == "UAT rehearsal"
    assert payload["wrap_up"]["status"] == "pass"
    assert payload["progress_timeline_dir"] == timeline_dir.as_posix()

    review_payload = json.loads(review_json.read_text(encoding="utf-8"))
    assert review_payload["status"] == "pass"
    assert review_payload["objectives"][0]["name"] == "governance"
    assert review_payload["attendees"] == ["Ops Lead"]

    wrap_payload = json.loads(wrap_up_json.read_text(encoding="utf-8"))
    assert wrap_payload["status"] == "pass"
    assert wrap_payload["backlog_items"] == []
    assert wrap_payload["incidents"] == []

    diary_path = run_dir / "decision_diary.jsonl"
    performance_path = run_dir / "performance_metrics.json"
    assert diary_path.exists()
    assert performance_path.exists()

    if packet_dir.exists():
        summary_manifest = packet_dir / "manifest.json"
        assert summary_manifest.exists()
        assert packet_archive.exists()


def test_orchestrator_skip_wrap_up(tmp_path, capsys):
    output_root = tmp_path / "runs"
    args = [
        "--output-root",
        str(output_root),
        "--duration-hours",
        "0.0005",
        "--required-duration-hours",
        "0.0002",
        "--minimum-uptime-ratio",
        "0.1",
        "--progress-interval-minutes",
        "0.0004",
        "--evidence-initial-grace-minutes",
        "0.0",
        "--no-wrap-up",
        "--no-progress-timeline",
        "--",
        sys.executable,
        "-c",
        _RUNTIME_SCRIPT,
    ]

    exit_code = orchestrator.main(args)
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Wrap-up generation skipped." in captured.out

    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    summary_json = run_dir / "summary.json"
    wrap_up_json = run_dir / "wrap_up.json"
    wrap_up_md = run_dir / "wrap_up.md"
    timeline_dir = run_dir / "progress_timeline"

    assert summary_json.exists()
    assert not wrap_up_json.exists()
    assert not wrap_up_md.exists()
    assert not timeline_dir.exists()

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "wrap_up" not in payload
    assert payload.get("progress_timeline_dir") is None
