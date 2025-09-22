from datetime import datetime, timezone

from src.data_foundation.ingest.health import (
    IngestHealthCheck,
    IngestHealthReport,
    IngestHealthStatus,
)
from src.data_foundation.ingest.metrics import (
    IngestDimensionMetrics,
    IngestMetricsSnapshot,
)
from src.operations.slo import SLOStatus, evaluate_ingest_slos


def test_evaluate_ingest_slos_maps_dimension_status() -> None:
    generated = datetime(2024, 1, 3, tzinfo=timezone.utc)
    health_check = IngestHealthCheck(
        dimension="daily_bars",
        status=IngestHealthStatus.warn,
        message="Freshness 900s exceeds SLA 600s",
        rows_written=10,
        freshness_seconds=900.0,
        expected_symbols=("EURUSD", "GBPUSD"),
        observed_symbols=("EURUSD",),
        missing_symbols=("GBPUSD",),
        ingest_duration_seconds=42.0,
        metadata={
            "freshness_sla_seconds": 600.0,
            "min_rows_required": 1,
        },
    )
    report = IngestHealthReport(
        status=IngestHealthStatus.warn,
        generated_at=generated,
        checks=(health_check,),
        metadata={
            "planned_dimensions": ["daily_bars"],
            "observed_dimensions": ["daily_bars"],
        },
    )
    metrics = IngestMetricsSnapshot(
        generated_at=generated,
        dimensions=(
            IngestDimensionMetrics(
                dimension="daily_bars",
                rows=10,
                symbols=("EURUSD",),
                ingest_duration_seconds=42.0,
                freshness_seconds=900.0,
                source="yahoo",
            ),
        ),
    )

    snapshot = evaluate_ingest_slos(
        metrics,
        report,
        alert_routes={"timescale_ingest.daily_bars": "pagerduty:test"},
        metadata={"recovery_attempts": 1},
    )

    assert snapshot.status is SLOStatus.at_risk
    assert snapshot.slos[0].name == "timescale_ingest"
    assert snapshot.slos[0].observed["total_rows"] == 10
    per_dimension = snapshot.slos[1]
    assert per_dimension.alert_route == "pagerduty:test"
    assert per_dimension.metadata["missing_symbols"] == ["GBPUSD"]
    assert snapshot.metadata["recovery_attempts"] == 1


def test_evaluate_ingest_slos_handles_missing_metrics() -> None:
    generated = datetime(2024, 1, 3, tzinfo=timezone.utc)
    report = IngestHealthReport(
        status=IngestHealthStatus.ok,
        generated_at=generated,
        checks=(),
        metadata={"planned_dimensions": [], "observed_dimensions": []},
    )

    snapshot = evaluate_ingest_slos(None, report)

    assert snapshot.status is SLOStatus.met
    assert snapshot.slos[0].message == "No ingest checks executed"
    assert "total_rows" not in snapshot.slos[0].observed

    markdown = snapshot.to_markdown()
    assert "Operational SLOs" in markdown
    assert "timescale_ingest" in markdown
