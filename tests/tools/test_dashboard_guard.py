from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import json

import pytest

from tools.telemetry.dashboard_guard import (
    DashboardGuardStatus,
    evaluate_dashboard_health,
    main,
)


def _base_summary(now: datetime) -> dict[str, object]:
    return {
        "generated_at": now.astimezone(UTC).isoformat(),
        "overall_status": "ok",
        "panel_counts": {"ok": 2, "warn": 0, "fail": 0},
        "failing_panels": (),
        "warning_panels": (),
        "healthy_panels": ("PnL & ROI", "System health"),
    }


def test_evaluate_dashboard_health_reports_ok() -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    summary = _base_summary(now)

    report = evaluate_dashboard_health(
        summary,
        max_age=timedelta(minutes=30),
        required_panels=("PnL & ROI", "System health"),
        current_time=now + timedelta(minutes=10),
    )

    assert report.status is DashboardGuardStatus.ok
    assert report.missing_panels == ()
    assert report.issues == ()
    assert report.age_seconds == pytest.approx(600.0)
    assert report.overall_status is DashboardGuardStatus.ok


def test_evaluate_dashboard_health_detects_failing_panels() -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    summary = _base_summary(now)
    summary["failing_panels"] = ("Risk & exposure",)
    summary["panel_counts"] = {"ok": 1, "warn": 0, "fail": 1}

    report = evaluate_dashboard_health(summary, current_time=now + timedelta(minutes=5))

    assert report.status is DashboardGuardStatus.fail
    assert any("Failing panels" in issue for issue in report.issues)
    assert report.overall_status is DashboardGuardStatus.ok


def test_evaluate_dashboard_health_flags_stale_snapshot() -> None:
    now = datetime(2025, 1, 1, 14, 0, tzinfo=UTC)
    summary = _base_summary(now - timedelta(hours=2))

    report = evaluate_dashboard_health(
        summary,
        max_age=timedelta(minutes=30),
        current_time=now,
    )

    assert report.status is DashboardGuardStatus.fail
    assert any("stale" in issue.lower() for issue in report.issues)


def test_evaluate_dashboard_health_requires_panels() -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    summary = _base_summary(now)

    report = evaluate_dashboard_health(
        summary,
        required_panels=("Operational readiness",),
        current_time=now + timedelta(minutes=5),
    )

    assert report.status is DashboardGuardStatus.fail
    assert report.missing_panels == ("Operational readiness",)
    assert report.overall_status is DashboardGuardStatus.ok


def test_evaluate_dashboard_health_escalates_overall_status_warn() -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    summary = _base_summary(now)
    summary["overall_status"] = "warn"
    report = evaluate_dashboard_health(summary, current_time=now + timedelta(minutes=1))

    assert report.status is DashboardGuardStatus.warn
    assert report.overall_status is DashboardGuardStatus.warn
    assert any("Overall dashboard status" in issue for issue in report.issues)


def test_evaluate_dashboard_health_escalates_overall_status_fail() -> None:
    now = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    summary = _base_summary(now)
    summary["overall_status"] = "FAIL"
    report = evaluate_dashboard_health(summary, current_time=now + timedelta(minutes=1))

    assert report.status is DashboardGuardStatus.fail
    assert report.overall_status is DashboardGuardStatus.fail
    assert any(issue.endswith("FAIL") for issue in report.issues)


def test_cli_outputs_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    now = datetime.now(tz=UTC)
    payload = _base_summary(now)
    payload_path = tmp_path / "dashboard.json"
    payload_path.write_text(json.dumps(payload))

    exit_code = main(
        [
            str(payload_path),
            "--format",
            "json",
            "--max-age-minutes",
            "30",
            "--require-panel",
            "PnL & ROI",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "ok"
    assert output["overall_status"] == "ok"
    assert output["missing_panels"] == []


def test_cli_exit_code_for_failing_dashboard(tmp_path: Path) -> None:
    now = datetime.now(tz=UTC)
    payload = _base_summary(now)
    payload["failing_panels"] = ("Risk & exposure",)
    payload_path = tmp_path / "dashboard.json"
    payload_path.write_text(json.dumps(payload))

    exit_code = main(
        [
            str(payload_path),
            "--max-age-minutes",
            "30",
        ]
    )

    assert exit_code == 2


def test_cli_exit_code_for_warning_dashboard(tmp_path: Path) -> None:
    now = datetime.now(tz=UTC)
    payload = _base_summary(now)
    payload["warning_panels"] = ("Latency",)
    payload_path = tmp_path / "dashboard.json"
    payload_path.write_text(json.dumps(payload))

    exit_code = main([
        str(payload_path),
        "--max-age-minutes",
        "30",
    ])

    assert exit_code == 1
