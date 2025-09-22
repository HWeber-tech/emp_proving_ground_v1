"""Stress testing utilities for Spark export workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from time import perf_counter
from typing import Callable, Mapping

from src.data_foundation.batch.spark_export import (
    SparkExportSnapshot,
    SparkExportStatus,
)


class SparkStressStatus(StrEnum):
    """Severity levels for Spark stress drill outcomes."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: dict[SparkStressStatus, int] = {
    SparkStressStatus.ok: 0,
    SparkStressStatus.warn: 1,
    SparkStressStatus.fail: 2,
}


_EXPORT_TO_STRESS: dict[SparkExportStatus, SparkStressStatus] = {
    SparkExportStatus.ok: SparkStressStatus.ok,
    SparkExportStatus.warn: SparkStressStatus.warn,
    SparkExportStatus.fail: SparkStressStatus.fail,
}


def _escalate_status(current: SparkStressStatus, candidate: SparkStressStatus) -> SparkStressStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class SparkStressCycleResult:
    """Result captured for a single stress-test export cycle."""

    cycle: int
    status: SparkStressStatus
    export_status: SparkExportStatus
    duration_seconds: float
    issues: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "cycle": self.cycle,
            "status": self.status.value,
            "export_status": self.export_status.value,
            "duration_seconds": self.duration_seconds,
        }
        if self.issues:
            payload["issues"] = list(self.issues)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class SparkStressSnapshot:
    """Aggregated snapshot describing a Spark stress drill."""

    label: str
    status: SparkStressStatus
    generated_at: datetime
    cycles: tuple[SparkStressCycleResult, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "label": self.label,
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "cycles": [cycle.as_dict() for cycle in self.cycles],
            "metadata": dict(self.metadata),
        }
        return payload

    def to_markdown(self) -> str:
        if not self.cycles:
            return "| Cycle | Status | Export | Duration (s) | Issues |\n| --- | --- | --- | --- | --- |\n"

        rows = [
            f"**Spark stress drill – {self.label}**",
            f"- Generated: {self.generated_at.isoformat()}",
            f"- Status: {self.status.value}",
            "",
            "| Cycle | Status | Export | Duration (s) | Issues |",
            "| --- | --- | --- | --- | --- |",
        ]
        for cycle in self.cycles:
            issues = ", ".join(cycle.issues) if cycle.issues else ""
            rows.append(
                f"| {cycle.cycle} | {cycle.status.value.upper()} | "
                f"{cycle.export_status.value.upper()} | {cycle.duration_seconds:.2f} | {issues} |"
            )
        return "\n".join(rows)


def format_spark_stress_markdown(snapshot: SparkStressSnapshot) -> str:
    """Compatibility wrapper aligning with other telemetry helpers."""

    return snapshot.to_markdown()


def execute_spark_stress_drill(
    *,
    label: str,
    cycles: int,
    runner: Callable[[], SparkExportSnapshot],
    warn_after_seconds: float | None = None,
    fail_after_seconds: float | None = None,
    metadata: Mapping[str, object] | None = None,
    now: datetime | None = None,
) -> SparkStressSnapshot:
    """Execute a Spark export multiple times to gauge resilience."""

    if cycles <= 0:
        raise ValueError("cycles must be positive")

    generated_at = now or datetime.now(tz=UTC)
    cycles_results: list[SparkStressCycleResult] = []
    durations: list[float] = []
    overall_status = SparkStressStatus.ok

    for index in range(1, cycles + 1):
        started = perf_counter()
        snapshot = runner()
        duration = perf_counter() - started
        durations.append(duration)

        issues: list[str] = []
        cycle_status = _EXPORT_TO_STRESS.get(snapshot.status, SparkStressStatus.fail)

        if fail_after_seconds is not None and duration >= fail_after_seconds:
            cycle_status = SparkStressStatus.fail
            issues.append("duration_exceeded_fail_threshold")
        elif warn_after_seconds is not None and duration >= warn_after_seconds:
            cycle_status = _escalate_status(cycle_status, SparkStressStatus.warn)
            issues.append("duration_exceeded_warn_threshold")

        job_issues = [issue for job in snapshot.jobs for issue in job.issues]
        if job_issues:
            issues.extend(f"job:{issue}" for issue in job_issues)

        cycle_metadata: dict[str, object] = {
            "job_count": len(snapshot.jobs),
            "rows": sum(job.rows for job in snapshot.jobs),
            "duration_seconds": duration,
            "export_root": snapshot.root_path,
            "export_format": snapshot.format.value,
            "jobs": [
                {
                    "dimension": job.dimension,
                    "status": job.status.value,
                    "rows": job.rows,
                    "issues": list(job.issues),
                }
                for job in snapshot.jobs
            ],
        }
        if snapshot.metadata:
            cycle_metadata["snapshot_metadata"] = dict(snapshot.metadata)

        cycle_result = SparkStressCycleResult(
            cycle=index,
            status=cycle_status,
            export_status=snapshot.status,
            duration_seconds=duration,
            issues=tuple(issues),
            metadata=cycle_metadata,
        )
        cycles_results.append(cycle_result)
        overall_status = _escalate_status(overall_status, cycle_result.status)

    metadata_payload: dict[str, object] = dict(metadata) if metadata else {}
    metadata_payload.update(
        {
            "cycles": cycles,
            "average_duration_seconds": sum(durations) / len(durations),
            "max_duration_seconds": max(durations),
            "min_duration_seconds": min(durations),
        }
    )
    if warn_after_seconds is not None:
        metadata_payload["warn_after_seconds"] = warn_after_seconds
    if fail_after_seconds is not None:
        metadata_payload["fail_after_seconds"] = fail_after_seconds

    return SparkStressSnapshot(
        label=label,
        status=overall_status,
        generated_at=generated_at,
        cycles=tuple(cycles_results),
        metadata=metadata_payload,
    )


__all__ = [
    "SparkStressCycleResult",
    "SparkStressSnapshot",
    "SparkStressStatus",
    "execute_spark_stress_drill",
    "format_spark_stress_markdown",
]
