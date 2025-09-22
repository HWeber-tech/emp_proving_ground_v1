from datetime import UTC, datetime

import pytest

from src.data_foundation.batch.spark_export import (
    SparkExportFormat,
    SparkExportJobResult,
    SparkExportSnapshot,
    SparkExportStatus,
)
from src.operations.spark_stress import (
    SparkStressStatus,
    execute_spark_stress_drill,
)


def _make_snapshot(
    status: SparkExportStatus = SparkExportStatus.ok,
    *,
    issues: tuple[str, ...] = (),
) -> SparkExportSnapshot:
    job = SparkExportJobResult(
        dimension="daily_bars",
        status=status,
        rows=10,
        paths=("exports/daily.csv",),
        issues=issues,
        metadata={"symbols": ["EURUSD"]},
    )
    return SparkExportSnapshot(
        generated_at=datetime.now(tz=UTC),
        status=status,
        format=SparkExportFormat.csv,
        root_path="exports",
        jobs=(job,),
    )


def test_execute_spark_stress_drill_records_cycles(monkeypatch) -> None:
    snapshots = iter([_make_snapshot(), _make_snapshot()])

    def runner() -> SparkExportSnapshot:
        return next(snapshots)

    perf_values = iter([0.0, 0.05, 0.5, 0.65])
    monkeypatch.setattr("src.operations.spark_stress.perf_counter", lambda: next(perf_values))

    snapshot = execute_spark_stress_drill(
        label="integration",
        cycles=2,
        runner=runner,
    )

    assert snapshot.status is SparkStressStatus.ok
    assert len(snapshot.cycles) == 2
    assert pytest.approx(snapshot.metadata["average_duration_seconds"], rel=1e-3) == 0.1
    assert snapshot.metadata["cycles"] == 2


def test_execute_spark_stress_drill_applies_duration_thresholds(monkeypatch) -> None:
    def runner() -> SparkExportSnapshot:
        return _make_snapshot(status=SparkExportStatus.warn, issues=("no_rows",))

    perf_values = iter([0.0, 0.4, 0.5, 1.7])
    monkeypatch.setattr("src.operations.spark_stress.perf_counter", lambda: next(perf_values))

    snapshot = execute_spark_stress_drill(
        label="thresholds",
        cycles=2,
        runner=runner,
        warn_after_seconds=0.3,
        fail_after_seconds=1.0,
    )

    assert snapshot.status is SparkStressStatus.fail
    assert snapshot.cycles[0].status is SparkStressStatus.warn
    assert "duration_exceeded_warn_threshold" in snapshot.cycles[0].issues
    assert snapshot.cycles[1].status is SparkStressStatus.fail
    assert "duration_exceeded_fail_threshold" in snapshot.cycles[1].issues
    assert any(issue.startswith("job:") for issue in snapshot.cycles[0].issues)


def test_execute_spark_stress_drill_requires_positive_cycles() -> None:
    with pytest.raises(ValueError):
        execute_spark_stress_drill(
            label="invalid",
            cycles=0,
            runner=lambda: _make_snapshot(),
        )
