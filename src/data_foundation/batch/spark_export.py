"""Utilities for exporting Timescale data into Spark-friendly datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

from src.data_foundation.persist.timescale_reader import TimescaleQueryResult, TimescaleReader


class SparkExportFormat(StrEnum):
    """Supported serialization formats for Spark export datasets."""

    csv = "csv"
    jsonl = "jsonl"

    @property
    def extension(self) -> str:
        return "csv" if self is SparkExportFormat.csv else "jsonl"


class SparkExportStatus(StrEnum):
    """Severity levels for Spark export execution."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[SparkExportStatus, int] = {
    SparkExportStatus.ok: 0,
    SparkExportStatus.warn: 1,
    SparkExportStatus.fail: 2,
}


def _escalate(current: SparkExportStatus, candidate: SparkExportStatus) -> SparkExportStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class SparkExportJob:
    """Description of a dataset to export for Spark consumption."""

    dimension: str
    symbols: tuple[str, ...] = ()
    start: datetime | None = None
    end: datetime | None = None
    limit: int | None = None
    partition_by: tuple[str, ...] = ()
    filename: str | None = None


@dataclass(frozen=True)
class SparkExportPlan:
    """Concrete export plan compiled from configuration."""

    root_path: Path
    format: SparkExportFormat
    jobs: tuple[SparkExportJob, ...]
    partition_columns: tuple[str, ...] = ()
    include_metadata: bool = True
    publish_telemetry: bool = True


@dataclass(frozen=True)
class SparkExportJobResult:
    """Outcome of an individual Spark export job."""

    dimension: str
    status: SparkExportStatus
    rows: int
    paths: tuple[str, ...]
    issues: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "dimension": self.dimension,
            "status": self.status.value,
            "rows": self.rows,
            "paths": list(self.paths),
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }
        return payload


@dataclass(frozen=True)
class SparkExportSnapshot:
    """Aggregated view of Spark export execution."""

    generated_at: datetime
    status: SparkExportStatus
    format: SparkExportFormat
    root_path: str
    jobs: tuple[SparkExportJobResult, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "format": self.format.value,
            "root_path": self.root_path,
            "jobs": [job.as_dict() for job in self.jobs],
            "metadata": dict(self.metadata),
        }
        return payload

    def to_markdown(self) -> str:
        lines = [
            "**Spark export snapshot**",
            f"- Generated: {self.generated_at.isoformat()}",
            f"- Status: {self.status.value}",
            f"- Format: {self.format.value}",
            f"- Root: {self.root_path}",
        ]
        for job in self.jobs:
            lines.append("")
            lines.append(f"**{job.dimension}** – status={job.status.value} rows={job.rows}")
            if job.paths:
                lines.append(f"- Files: {', '.join(job.paths)}")
            if job.issues:
                lines.append("- Issues:")
                for issue in job.issues:
                    lines.append(f"  - {issue}")
        return "\n".join(lines)


def format_spark_export_markdown(snapshot: SparkExportSnapshot) -> str:
    """Compatibility wrapper matching other telemetry helpers."""

    return snapshot.to_markdown()


def _normalise_partition_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    text = str(value)
    return text.replace("/", "_")


def _write_frame(frame: pd.DataFrame, path: Path, fmt: SparkExportFormat) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt is SparkExportFormat.csv:
        frame.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    else:
        frame.to_json(
            path,
            orient="records",
            lines=True,
            date_format="iso",
        )


def _partition_frame(
    frame: pd.DataFrame,
    *,
    partition_by: Sequence[str],
    base_dir: Path,
    fmt: SparkExportFormat,
    filename: str,
) -> list[Path]:
    if not partition_by:
        target = base_dir / filename
        _write_frame(frame, target, fmt)
        return [target]

    partitions: list[Path] = []
    grouped = frame.groupby(list(partition_by), dropna=False, sort=True)
    for index, (keys, subset) in enumerate(grouped):
        if not isinstance(keys, tuple):
            keys = (keys,)
        partition_dir = base_dir
        for column, value in zip(partition_by, keys):
            partition_dir /= f"{column}={_normalise_partition_value(value)}"
        target = partition_dir / f"{index:05d}_{filename}"
        _write_frame(subset, target, fmt)
        partitions.append(target)
    if not partitions:
        target = base_dir / filename
        _write_frame(frame, target, fmt)
        partitions.append(target)
    return partitions


def _query_dimension(reader: TimescaleReader, job: SparkExportJob) -> TimescaleQueryResult:
    if job.dimension == "daily_bars":
        return reader.fetch_daily_bars(
            symbols=list(job.symbols) or None,
            start=job.start,
            end=job.end,
            limit=job.limit,
        )
    if job.dimension == "intraday_trades":
        return reader.fetch_intraday_trades(
            symbols=list(job.symbols) or None,
            start=job.start,
            end=job.end,
            limit=job.limit,
        )
    if job.dimension == "macro_events":
        return reader.fetch_macro_events(
            calendars=list(job.symbols) or None,
            start=job.start,
            end=job.end,
            limit=job.limit,
        )
    raise ValueError(f"Unsupported Spark export dimension: {job.dimension}")


def _create_job_metadata(result: TimescaleQueryResult) -> dict[str, object]:
    metadata: dict[str, object] = {
        "symbols": list(result.symbols),
        "rowcount": result.rowcount,
    }
    if result.start_ts is not None:
        metadata["start_ts"] = result.start_ts.isoformat()
    if result.end_ts is not None:
        metadata["end_ts"] = result.end_ts.isoformat()
    if result.max_ingested_at is not None:
        metadata["max_ingested_at"] = result.max_ingested_at.isoformat()
    return metadata


def execute_spark_export_plan(
    reader: TimescaleReader,
    plan: SparkExportPlan,
    *,
    now: datetime | None = None,
    metadata_writer: Callable[[Path, Mapping[str, object]], None] | None = None,
) -> SparkExportSnapshot:
    """Execute the Spark export plan and return an aggregated snapshot."""

    generated_at = now or datetime.now(tz=UTC)
    plan.root_path.mkdir(parents=True, exist_ok=True)
    job_results: list[SparkExportJobResult] = []
    status = SparkExportStatus.ok

    jobs_manifest: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "generated_at": generated_at.isoformat(),
        "format": plan.format.value,
        "jobs": jobs_manifest,
    }

    for job in plan.jobs:
        base_dir = plan.root_path / job.dimension
        filename = job.filename or f"{job.dimension}.{plan.format.extension}"
        try:
            query_result = _query_dimension(reader, job)
        except Exception as exc:  # pragma: no cover - error path exercised in tests
            job_status = SparkExportStatus.fail
            issue = f"query_failed: {exc}"[:500]
            job_results.append(
                SparkExportJobResult(
                    dimension=job.dimension,
                    status=job_status,
                    rows=0,
                    paths=tuple(),
                    issues=(issue,),
                )
            )
            status = _escalate(status, job_status)
            continue

        frame = query_result.frame
        if frame.empty:
            job_status = SparkExportStatus.warn
            job_metadata = _create_job_metadata(query_result)
            job_metadata["partitions"] = []
            job_results.append(
                SparkExportJobResult(
                    dimension=job.dimension,
                    status=job_status,
                    rows=0,
                    paths=tuple(),
                    issues=("no_rows_returned",),
                    metadata=job_metadata,
                )
            )
            status = _escalate(status, job_status)
            continue

        try:
            paths = _partition_frame(
                frame,
                partition_by=job.partition_by or plan.partition_columns,
                base_dir=base_dir,
                fmt=plan.format,
                filename=filename,
            )
        except Exception as exc:  # pragma: no cover - filesystem errors
            job_status = SparkExportStatus.fail
            job_results.append(
                SparkExportJobResult(
                    dimension=job.dimension,
                    status=job_status,
                    rows=query_result.rowcount,
                    paths=tuple(),
                    issues=(f"write_failed: {exc}",),
                )
            )
            status = _escalate(status, job_status)
            continue

        job_metadata = _create_job_metadata(query_result)
        partitions: list[str] = [str(path) for path in paths]
        job_metadata["partitions"] = partitions
        manifest_entry = {
            "dimension": job.dimension,
            "paths": list(partitions),
            "rowcount": query_result.rowcount,
        }
        jobs_manifest.append(manifest_entry)

        if plan.include_metadata:
            metadata_path = base_dir / "_metadata.json"
            metadata_payload = {
                "dimension": job.dimension,
                "generated_at": generated_at.isoformat(),
                "metadata": job_metadata,
            }
            try:
                _write_metadata(metadata_path, metadata_payload, metadata_writer)
            except Exception:  # pragma: no cover - diagnostics only
                pass

        job_results.append(
            SparkExportJobResult(
                dimension=job.dimension,
                status=SparkExportStatus.ok,
                rows=query_result.rowcount,
                paths=tuple(str(path) for path in paths),
                metadata=job_metadata,
            )
        )

    if plan.include_metadata:
        manifest_path = plan.root_path / "manifest.json"
        manifest["status"] = status.value
        manifest["job_count"] = len(plan.jobs)
        try:
            _write_metadata(manifest_path, manifest, metadata_writer)
        except Exception:  # pragma: no cover - diagnostics only
            pass

    snapshot = SparkExportSnapshot(
        generated_at=generated_at,
        status=status,
        format=plan.format,
        root_path=str(plan.root_path),
        jobs=tuple(job_results),
        metadata={
            "job_count": len(plan.jobs),
            "publish_telemetry": plan.publish_telemetry,
        },
    )
    return snapshot


def _write_metadata(
    path: Path,
    payload: Mapping[str, object],
    metadata_writer: Callable[[Path, Mapping[str, object]], None] | None,
) -> None:
    if metadata_writer is not None:
        metadata_writer(path, payload)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = [
    "SparkExportFormat",
    "SparkExportJob",
    "SparkExportJobResult",
    "SparkExportPlan",
    "SparkExportSnapshot",
    "SparkExportStatus",
    "execute_spark_export_plan",
    "format_spark_export_markdown",
]
