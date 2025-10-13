from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.operations.dry_run_audit import DryRunStatus
from src.operations.final_dry_run_backlog import collect_backlog_items
from tools.operations import final_dry_run_backlog


def _iso(ts: str) -> str:
    return datetime.fromisoformat(ts).astimezone(UTC).isoformat()


def test_collect_backlog_items_merges_sources() -> None:
    summary = {
        "logs": {
            "warnings": [
                {
                    "severity": "warn",
                    "summary": "Latency spike detected",
                    "occurred_at": _iso("2025-01-01T00:00:00+00:00"),
                    "metadata": {"stream": "runtime"},
                }
            ]
        },
        "diary": {
            "issues": [
                {
                    "severity": "warn",
                    "policy_id": "risk-policy",
                    "entry_id": "entry-1",
                    "reason": "Missing stop loss",
                    "recorded_at": _iso("2025-01-01T01:00:00+00:00"),
                    "metadata": {"symbol": "EURUSD"},
                }
            ]
        },
        "performance": {
            "status": "warn",
            "generated_at": _iso("2025-01-01T02:00:00+00:00"),
            "total_trades": 8,
            "roi": -0.01,
            "metadata": {"window_hours": 72},
        },
    }
    sign_off = {
        "evaluated_at": _iso("2025-01-01T03:00:00+00:00"),
        "findings": [
            {
                "severity": "fail",
                "message": "Run duration below 72h",
                "metadata": {"actual_hours": 64},
            }
        ],
    }
    review = {
        "action_items": [
            {
                "severity": "warn",
                "category": "ops",
                "description": "Investigate latency spike",
                "context": {"ticket": "OPS-1"},
            }
        ],
        "objectives": [
            {
                "name": "governance",
                "status": "fail",
                "note": "Missing risk sign-off",
                "evidence": {"doc": "governance-review.md"},
            }
        ],
    }

    items = collect_backlog_items(summary, sign_off=sign_off, review=review)

    severities = {item.severity for item in items}
    categories = {item.category for item in items}

    assert DryRunStatus.fail in severities
    assert DryRunStatus.warn in severities
    assert {"log_warning", "diary", "performance", "sign_off", "ops", "objective"}.issubset(categories)


def test_collect_backlog_items_deduplicates_and_includes_pass() -> None:
    summary = {
        "logs": {
            "warnings": [
                {
                    "severity": "warn",
                    "summary": "Duplicate warning",
                    "occurred_at": _iso("2025-01-02T00:00:00+00:00"),
                }
            ]
        }
    }
    review = {
        "action_items": [
            {
                "severity": "warn",
                "category": "log_warning",
                "description": "Duplicate warning",
            }
        ]
    }
    sign_off = {
        "evaluated_at": _iso("2025-01-02T01:00:00+00:00"),
        "findings": [
            {
                "severity": "pass",
                "message": "Duration satisfied",
            }
        ],
    }

    without_pass = collect_backlog_items(summary, sign_off=sign_off, review=review)
    with_pass = collect_backlog_items(
        summary,
        sign_off=sign_off,
        review=review,
        include_pass=True,
    )

    assert len(without_pass) == 1
    assert len(with_pass) == 2
    assert any(item.severity is DryRunStatus.pass_ for item in with_pass)


def test_final_dry_run_backlog_cli_markdown(tmp_path: Path, capsys) -> None:
    summary = {
        "logs": {
            "warnings": [
                {
                    "severity": "warn",
                    "summary": "Latency spike",
                    "occurred_at": _iso("2025-01-03T00:00:00+00:00"),
                }
            ]
        }
    }
    bundle = {
        "summary": summary,
        "review": {
            "action_items": [
                {
                    "severity": "warn",
                    "category": "ops",
                    "description": "Investigate latency",
                }
            ]
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(bundle), encoding="utf-8")

    code = final_dry_run_backlog.main(["--summary", str(summary_path)])
    output = capsys.readouterr().out

    assert code == 0
    assert "Final Dry Run Backlog" in output
    assert "Latency spike" in output

    code_warn = final_dry_run_backlog.main(
        ["--summary", str(summary_path), "--fail-on-warn"]
    )
    assert code_warn == 1
