from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run_wrap_up import build_wrap_up_report
from tools.operations import final_dry_run_wrap_up as wrap_cli


def _iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat().replace("+00:00", "Z")


def test_build_wrap_up_report_aggregates_backlog_and_duration():
    generated = datetime(2025, 10, 13, 6, 0, tzinfo=UTC)
    start = datetime(2025, 10, 10, 6, 0, tzinfo=UTC)
    end = datetime(2025, 10, 10, 7, 0, tzinfo=UTC)

    bundle = {
        "generated_at": _iso(generated),
        "status": "warn",
        "summary": {
            "generated_at": _iso(generated - timedelta(minutes=1)),
            "status": "warn",
            "metadata": {"run_label": "Phase II"},
            "logs": {
                "started_at": _iso(start),
                "ended_at": _iso(end),
                "duration_seconds": 3600.0,
                "level_counts": {"info": 120, "error": 2},
                "errors": [
                    {
                        "severity": "fail",
                        "occurred_at": _iso(start + timedelta(minutes=20)),
                        "summary": "Unhandled exception",
                        "event": "runtime_error",
                    }
                ],
            },
            "diary": {
                "issues": [
                    {
                        "severity": "warn",
                        "reason": "Missing execution summary",
                        "entry_id": "dry-001",
                    }
                ]
            },
            "performance": {
                "status": "warn",
                "roi": -0.02,
            },
        },
        "sign_off": {
            "status": "warn",
            "findings": [
                {
                    "severity": "warn",
                    "message": "Sharpe ratio below policy threshold",
                }
            ],
        },
        "review": {
            "status": "warn",
            "notes": ["Throttle tweaks required"],
            "action_items": [
                {
                    "severity": "warn",
                    "category": "operations",
                    "description": "Tune throttle windows for FX pairs",
                }
            ],
            "objectives": [
                {
                    "name": "duration",
                    "status": "fail",
                    "note": "Observed window < 72h",
                }
            ],
        },
        "incidents": [
            {
                "severity": "warn",
                "occurred_at": _iso(start + timedelta(minutes=45)),
                "message": "Live log gap detected",
                "metadata": {"gap_minutes": 12},
            }
        ],
    }

    report = build_wrap_up_report(
        bundle,
        required_duration=timedelta(hours=72),
        duration_tolerance=timedelta(minutes=5),
    )

    assert report.status is DryRunStatus.fail
    assert report.duration_met is False
    assert report.summary_status is DryRunStatus.warn
    assert len(report.backlog_items) == 9
    assert any(item.category == "duration" for item in report.backlog_items)
    assert any(item.category == "logs" and "Unhandled" in item.description for item in report.backlog_items)
    assert any(item.category == "operations" for item in report.backlog_items)
    assert report.notes == ("Throttle tweaks required",)
    assert report.incidents and report.incidents[0].message == "Live log gap detected"
    assert "Log levels" in report.highlights


def test_build_wrap_up_report_escalates_warn_when_requested():
    now = datetime(2025, 10, 15, 12, 0, tzinfo=UTC)
    bundle = {
        "generated_at": _iso(now),
        "status": "warn",
        "summary": {
            "generated_at": _iso(now),
            "status": "warn",
            "logs": {
                "duration_seconds": 7200,
                "level_counts": {"info": 100},
            },
        },
        "review": {
            "status": "warn",
            "action_items": [
                {
                    "severity": "warn",
                    "description": "Tidy configs",
                }
            ],
        },
    }

    report_warn = build_wrap_up_report(bundle, required_duration=None)
    assert report_warn.status is DryRunStatus.warn

    report_fail = build_wrap_up_report(
        bundle,
        required_duration=None,
        treat_warn_as_failure=True,
    )
    assert report_fail.status is DryRunStatus.fail


def test_final_dry_run_wrap_up_cli(tmp_path: Path):
    now = datetime(2025, 10, 20, 9, 0, tzinfo=UTC)
    bundle = {
        "generated_at": _iso(now),
        "status": "pass",
        "summary": {
            "generated_at": _iso(now),
            "status": "pass",
            "logs": {
                "duration_seconds": 3600.0,
                "level_counts": {"info": 80},
            },
        },
        "review": {
            "status": "pass",
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(bundle), encoding="utf-8")

    json_path = tmp_path / "wrap.json"
    markdown_path = tmp_path / "wrap.md"

    exit_code = wrap_cli.main(
        [
            str(summary_path),
            "--required-duration-hours",
            "0",
            "--output-json",
            str(json_path),
            "--output-markdown",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == DryRunStatus.pass_.value
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "Final Dry Run Wrap-up" in markdown_text
