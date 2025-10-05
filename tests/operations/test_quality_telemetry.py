from datetime import UTC, datetime

import pytest

from src.operations.quality_telemetry import (
    QualityStatus,
    build_quality_telemetry_snapshot,
)


def test_quality_snapshot_ok_with_recent_coverage() -> None:
    now = datetime(2024, 6, 5, 12, 0, tzinfo=UTC)
    metrics = {
        "coverage_trend": [
            {
                "label": "2024-06-05T11:00:00+00:00",
                "coverage_percent": 84.5,
            }
        ],
        "coverage_domain_trend": [
            {
                "generated_at": "2024-06-05T11:00:00+00:00",
                "lagging_domains": [],
            }
        ],
        "remediation_trend": [],
    }

    snapshot = build_quality_telemetry_snapshot(metrics, generated_at=now)

    assert snapshot.status is QualityStatus.ok
    assert snapshot.coverage_percent == pytest.approx(84.5, rel=1e-6)
    assert snapshot.staleness_hours == pytest.approx(1.0, rel=1e-6)
    assert any("Coverage 84.50%" in note for note in snapshot.notes)
    assert snapshot.metadata["trend_counts"]["coverage_trend"] == 1


def test_quality_snapshot_flags_low_coverage_and_remediation() -> None:
    now = datetime(2024, 6, 5, 12, 0, tzinfo=UTC)
    metrics = {
        "coverage_trend": [
            {
                "label": "2024-06-01T12:00:00+00:00",
                "coverage_percent": 70.0,
            }
        ],
        "coverage_domain_trend": [
            {
                "generated_at": "2024-06-01T12:00:00+00:00",
                "lagging_domains": ["ingest"],
            }
        ],
        "remediation_trend": [
            {
                "label": "2024-06-04T09:00:00+00:00",
                "note": "Ingest coverage remediation backlog",
                "statuses": {"ingest": "fail"},
            }
        ],
    }

    snapshot = build_quality_telemetry_snapshot(
        metrics,
        generated_at=now,
        coverage_target=80.0,
        max_staleness_hours=24.0,
    )

    assert snapshot.status is QualityStatus.fail
    assert snapshot.coverage_percent == pytest.approx(70.0, rel=1e-6)
    assert snapshot.staleness_hours == pytest.approx(96.0, rel=1e-6)
    assert any("Lagging domains" in note for note in snapshot.notes)
    assert snapshot.remediation_items == ("Ingest coverage remediation backlog",)
    assert snapshot.metadata["remediation_entry"]["statuses"]["ingest"] == "fail"
