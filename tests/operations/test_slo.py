from datetime import datetime, timedelta, timezone

import pytest

from src.data_foundation.ingest.health import (
    IngestHealthCheck,
    IngestHealthReport,
    IngestHealthStatus,
)
from src.data_foundation.ingest.metrics import (
    IngestDimensionMetrics,
    IngestMetricsSnapshot,
)
import src.operations.slo as slo_module
from src.operations.slo import (
    DriftAlertFreshnessProbe,
    LoopLatencyProbe,
    ReplayDeterminismProbe,
    SLOStatus,
    evaluate_ingest_slos,
    evaluate_understanding_loop_slos,
)


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


def test_evaluate_understanding_loop_slos_records_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)

    metric_calls: list[tuple[str, tuple[object, ...]]] = []

    def _record(name: str):
        def _inner(*args: object, **kwargs: object) -> None:
            metric_calls.append((name, args))

        return _inner

    monkeypatch.setattr(
        slo_module.operational_metrics,
        "set_understanding_loop_latency",
        _record("loop_latency"),
    )
    monkeypatch.setattr(
        slo_module.operational_metrics,
        "set_understanding_loop_latency_status",
        _record("loop_status"),
    )
    monkeypatch.setattr(
        slo_module.operational_metrics,
        "set_drift_alert_freshness",
        _record("drift_freshness"),
    )
    monkeypatch.setattr(
        slo_module.operational_metrics,
        "set_drift_alert_status",
        _record("drift_status"),
    )
    monkeypatch.setattr(
        slo_module.operational_metrics,
        "set_replay_determinism_drift",
        _record("replay_drift"),
    )
    monkeypatch.setattr(
        slo_module.operational_metrics,
        "set_replay_determinism_status",
        _record("replay_status"),
    )
    monkeypatch.setattr(
        slo_module.operational_metrics,
        "set_replay_determinism_mismatches",
        _record("replay_mismatches"),
    )

    loop_probe = LoopLatencyProbe(
        loop="professional",
        target_p95_seconds=1.5,
        p95_seconds=1.8,
        max_seconds=2.4,
        breach_p95_seconds=3.0,
        sample_count=120,
        window_seconds=60.0,
        runbook="docs/operations/runbooks/understanding_loop_latency.md",
        metadata={"note": "integration"},
    )
    drift_probe = DriftAlertFreshnessProbe(
        alert="page_hinkley",
        warn_after_seconds=200.0,
        fail_after_seconds=600.0,
        last_alert_at=now - timedelta(seconds=250),
        alerts_sent=5,
        runbook="docs/operations/runbooks/drift_sentry_response.md",
        metadata={"route": "pagerduty"},
    )
    replay_probe = ReplayDeterminismProbe(
        probe="fast_weights",
        warn_threshold=0.05,
        fail_threshold=0.1,
        drift_score=0.03,
        checksum_match=True,
        mismatched_fields=(),
    )

    snapshot = evaluate_understanding_loop_slos(
        loop_latency_probes=(loop_probe,),
        drift_alert_probes=(drift_probe,),
        replay_probes=(replay_probe,),
        generated_at=now,
        now=now,
        metadata={"environment": "ci"},
    )

    assert snapshot.service == "understanding_loop"
    assert snapshot.status is SLOStatus.at_risk
    assert snapshot.metadata["environment"] == "ci"

    summary = snapshot.slos[0]
    assert summary.name == "understanding_loop"
    assert summary.status is SLOStatus.at_risk
    assert summary.observed["loop_latency_probes"] == 1

    loop_record = next(
        record for record in snapshot.slos if record.name.startswith("understanding_loop.latency.")
    )
    assert loop_record.status is SLOStatus.at_risk
    assert loop_record.observed["p95_seconds"] == pytest.approx(1.8)

    drift_record = next(
        record for record in snapshot.slos if record.name.startswith("understanding_loop.drift_alert.")
    )
    assert drift_record.status is SLOStatus.at_risk
    assert drift_record.observed["freshness_seconds"] == pytest.approx(250.0)

    replay_record = next(
        record for record in snapshot.slos if record.name.startswith("understanding_loop.replay.")
    )
    assert replay_record.status is SLOStatus.met
    assert replay_record.observed["drift_score"] == pytest.approx(0.03)

    assert ("loop_latency", ("professional", "p95", 1.8)) in metric_calls
    assert ("loop_status", ("professional", 1)) in metric_calls
    assert any(
        name == "drift_freshness"
        and args[0] == "page_hinkley"
        and args[1] == pytest.approx(250.0)
        for name, args in metric_calls
    )
    assert any(name == "replay_status" for name, _ in metric_calls)


def test_evaluate_understanding_loop_slos_without_probes() -> None:
    snapshot = evaluate_understanding_loop_slos()

    assert snapshot.service == "understanding_loop"
    assert snapshot.status is SLOStatus.met
    assert snapshot.slos[0].message == "No understanding-loop probes evaluated"
