"""Data backbone readiness evaluation and telemetry helpers.

This module translates the roadmap's institutional data backbone requirements
into an actionable readiness snapshot.  It inspects recent ingest health,
quality, failover, recovery, and backup outcomes alongside runtime context
about Redis and Kafka wiring so operators inherit a single source of truth for
Timescale-backed readiness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Mapping, Sequence

from src.data_foundation.batch.spark_export import (
    SparkExportSnapshot,
    SparkExportStatus,
)
from src.data_foundation.ingest.configuration import InstitutionalIngestConfig
from src.data_foundation.ingest.failover import IngestFailoverDecision
from src.data_foundation.ingest.health import IngestHealthReport, IngestHealthStatus
from src.data_foundation.ingest.metrics import IngestMetricsSnapshot
from src.data_foundation.ingest.quality import IngestQualityReport, IngestQualityStatus
from src.data_foundation.ingest.recovery import IngestRecoveryRecommendation
from src.data_foundation.ingest.scheduler import IngestSchedulerState
from src.data_foundation.ingest.scheduler_telemetry import (
    IngestSchedulerSnapshot,
    IngestSchedulerStatus,
)
from src.operations.backup import BackupReadinessSnapshot, BackupStatus
from src.operations.spark_stress import SparkStressSnapshot, SparkStressStatus


class BackboneStatus(Enum):
    """Severity levels used by the backbone readiness snapshot."""

    ok = "ok"
    warn = "warn"
    fail = "fail"

    @classmethod
    def from_health(cls, status: IngestHealthStatus) -> "BackboneStatus":
        if status is IngestHealthStatus.ok:
            return cls.ok
        if status is IngestHealthStatus.warn:
            return cls.warn
        return cls.fail

    @classmethod
    def from_quality(cls, status: IngestQualityStatus) -> "BackboneStatus":
        if status is IngestQualityStatus.ok:
            return cls.ok
        if status is IngestQualityStatus.warn:
            return cls.warn
        return cls.fail

    @classmethod
    def from_backup(cls, status: BackupStatus) -> "BackboneStatus":
        if status is BackupStatus.ok:
            return cls.ok
        if status is BackupStatus.warn:
            return cls.warn
        return cls.fail


_STATUS_ORDER: dict[BackboneStatus, int] = {
    BackboneStatus.ok: 0,
    BackboneStatus.warn: 1,
    BackboneStatus.fail: 2,
}


def _combine_status(current: BackboneStatus, candidate: BackboneStatus) -> BackboneStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class BackboneComponentSnapshot:
    """Point-in-time readiness view for an individual backbone component."""

    name: str
    status: BackboneStatus
    summary: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class BackboneRuntimeContext:
    """Ambient runtime information used when grading readiness."""

    redis_expected: bool = False
    redis_configured: bool = False
    redis_namespace: str | None = None
    redis_backing: str | None = None
    kafka_expected: bool = False
    kafka_configured: bool = False
    kafka_topics: tuple[str, ...] = ()
    kafka_publishers: tuple[str, ...] = ()
    scheduler_enabled: bool = False
    scheduler_state: IngestSchedulerState | None = None

    @property
    def scheduler_running(self) -> bool:
        return bool(self.scheduler_state and self.scheduler_state.running)


@dataclass(frozen=True)
class DataBackboneReadinessSnapshot:
    """Aggregated readiness snapshot for the institutional data backbone."""

    status: BackboneStatus
    generated_at: datetime
    components: tuple[BackboneComponentSnapshot, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "components": [component.as_dict() for component in self.components],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        if not self.components:
            return "| Component | Status | Summary |\n| --- | --- | --- |\n"

        rows = ["| Component | Status | Summary |", "| --- | --- | --- |"]
        for component in self.components:
            rows.append(
                f"| {component.name} | {component.status.value.upper()} | {component.summary} |"
            )
        return "\n".join(rows)


def _planned_dimensions(plan_metadata: Mapping[str, object] | None) -> tuple[str, ...]:
    if not plan_metadata:
        return tuple()
    dimensions: list[str] = []
    for key in ("daily_bars", "intraday_trades", "macro_events"):
        if key in plan_metadata:
            dimensions.append(key)
    return tuple(dimensions)


def _summarise_metrics(metrics: IngestMetricsSnapshot | None) -> Mapping[str, object]:
    if metrics is None or not metrics.dimensions:
        return {}
    return {
        "total_rows": metrics.total_rows(),
        "active_dimensions": list(metrics.active_dimensions()),
    }


@dataclass(frozen=True)
class DataBackboneValidationSnapshot:
    """Static validation snapshot for institutional data backbone toggles."""

    status: BackboneStatus
    generated_at: datetime
    checks: tuple[BackboneComponentSnapshot, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "checks": [check.as_dict() for check in self.checks],
            "metadata": dict(self.metadata),
        }

    def to_markdown(self) -> str:
        if not self.checks:
            return "| Check | Status | Summary |\n| --- | --- | --- |\n"

        rows = ["| Check | Status | Summary |", "| --- | --- | --- |"]
        for check in self.checks:
            rows.append(f"| {check.name} | {check.status.value.upper()} | {check.summary} |")
        return "\n".join(rows)


def _scheduler_status_to_backbone(status: IngestSchedulerStatus) -> BackboneStatus:
    if status is IngestSchedulerStatus.ok:
        return BackboneStatus.ok
    if status is IngestSchedulerStatus.warn:
        return BackboneStatus.warn
    return BackboneStatus.fail


def _spark_status_to_backbone(status: SparkExportStatus) -> BackboneStatus:
    if status is SparkExportStatus.ok:
        return BackboneStatus.ok
    if status is SparkExportStatus.warn:
        return BackboneStatus.warn
    return BackboneStatus.fail


def _spark_stress_status_to_backbone(status: SparkStressStatus) -> BackboneStatus:
    if status is SparkStressStatus.ok:
        return BackboneStatus.ok
    if status is SparkStressStatus.warn:
        return BackboneStatus.warn
    return BackboneStatus.fail


def evaluate_data_backbone_validation(
    *,
    ingest_config: InstitutionalIngestConfig,
    context: BackboneRuntimeContext | None = None,
    scheduler_snapshot: IngestSchedulerSnapshot | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DataBackboneValidationSnapshot:
    """Validate institutional backbone prerequisites and toggles."""

    generated_at = datetime.now(tz=UTC)
    checks: list[BackboneComponentSnapshot] = []
    overall = BackboneStatus.ok
    context = context or BackboneRuntimeContext()

    plan = ingest_config.plan
    planned = [
        key
        for key, value in (
            ("daily", getattr(plan, "daily", None)),
            ("intraday", getattr(plan, "intraday", None)),
            ("macro", getattr(plan, "macro", None)),
        )
        if value
    ]
    if ingest_config.should_run:
        summary = "Ingest plan configured" if planned else "Plan enabled without dimensions"
        status = BackboneStatus.ok if planned else BackboneStatus.warn
    else:
        summary = ingest_config.reason or "Institutional ingest disabled"
        status = BackboneStatus.fail
    checks.append(
        BackboneComponentSnapshot(
            name="plan",
            status=status,
            summary=summary,
            metadata={"dimensions": planned},
        )
    )
    overall = _combine_status(overall, status)

    timescale_metadata = {
        "url": ingest_config.timescale_settings.url,
        "configured": ingest_config.timescale_settings.configured,
    }
    if ingest_config.timescale_settings.configured:
        ts_status = BackboneStatus.ok
        ts_summary = "Timescale connection configured"
    else:
        ts_status = BackboneStatus.fail if ingest_config.should_run else BackboneStatus.warn
        ts_summary = "Using fallback Timescale simulation"
    checks.append(
        BackboneComponentSnapshot(
            name="timescale_connection",
            status=ts_status,
            summary=ts_summary,
            metadata=timescale_metadata,
        )
    )
    overall = _combine_status(overall, ts_status)

    redis_metadata = {
        "expected": context.redis_expected,
        "configured": context.redis_configured,
        "namespace": context.redis_namespace,
        "backing": context.redis_backing,
    }
    if context.redis_expected:
        if context.redis_configured:
            redis_status = BackboneStatus.ok
            redis_summary = "Redis namespace active"
        else:
            redis_status = BackboneStatus.fail
            redis_summary = "Redis expected but not configured"
    else:
        redis_status = BackboneStatus.warn
        redis_summary = "Redis optional for current run"
    checks.append(
        BackboneComponentSnapshot(
            name="redis", status=redis_status, summary=redis_summary, metadata=redis_metadata
        )
    )
    overall = _combine_status(overall, redis_status)

    kafka_metadata = {
        "expected": context.kafka_expected,
        "configured": context.kafka_configured,
        "topics": list(context.kafka_topics),
        "publishers": list(context.kafka_publishers),
        "config_flags": {
            "extras_configured": bool(ingest_config.metadata.get("kafka_configured")),
        },
    }
    if context.kafka_expected:
        configured = context.kafka_configured and bool(context.kafka_topics)
        if configured:
            kafka_status = BackboneStatus.ok
            kafka_summary = "Kafka publishers ready"
        else:
            kafka_status = BackboneStatus.fail
            kafka_summary = "Kafka expected but brokers/topics missing"
    else:
        kafka_status = BackboneStatus.warn
        kafka_summary = "Kafka optional for current run"
    checks.append(
        BackboneComponentSnapshot(
            name="kafka", status=kafka_status, summary=kafka_summary, metadata=kafka_metadata
        )
    )
    overall = _combine_status(overall, kafka_status)

    if context.scheduler_enabled or scheduler_snapshot is not None:
        if scheduler_snapshot is None:
            scheduler_status = BackboneStatus.fail
            scheduler_summary = "Scheduler telemetry unavailable"
            scheduler_metadata: dict[str, object] = {"enabled": context.scheduler_enabled}
        else:
            scheduler_status = _scheduler_status_to_backbone(scheduler_snapshot.status)
            scheduler_summary = "Scheduler telemetry captured"
            scheduler_metadata = dict(scheduler_snapshot.as_dict())
        checks.append(
            BackboneComponentSnapshot(
                name="scheduler",
                status=scheduler_status,
                summary=scheduler_summary,
                metadata=scheduler_metadata,
            )
        )
        overall = _combine_status(overall, scheduler_status)

    snapshot_metadata: dict[str, object] = {}
    if isinstance(metadata, Mapping):
        snapshot_metadata.update(dict(metadata))
    snapshot_metadata.setdefault("plan_dimensions", planned)
    snapshot_metadata.setdefault("kafka_topics", list(context.kafka_topics))
    snapshot_metadata.setdefault("redis_namespace", context.redis_namespace)

    return DataBackboneValidationSnapshot(
        status=overall,
        generated_at=generated_at,
        checks=tuple(checks),
        metadata=snapshot_metadata,
    )


def evaluate_data_backbone_readiness(
    *,
    ingest_config: InstitutionalIngestConfig,
    health_report: IngestHealthReport | None = None,
    quality_report: IngestQualityReport | None = None,
    metrics_snapshot: IngestMetricsSnapshot | None = None,
    failover_decision: IngestFailoverDecision | None = None,
    recovery_recommendation: IngestRecoveryRecommendation | None = None,
    backup_snapshot: BackupReadinessSnapshot | None = None,
    context: BackboneRuntimeContext | None = None,
    metadata: Mapping[str, object] | None = None,
    spark_snapshot: SparkExportSnapshot | None = None,
    spark_stress_snapshot: SparkStressSnapshot | None = None,
) -> DataBackboneReadinessSnapshot:
    """Fuse ingest outcomes and runtime wiring into a readiness snapshot."""

    generated_at = datetime.now(tz=UTC)
    components: list[BackboneComponentSnapshot] = []
    overall = BackboneStatus.ok
    context = context or BackboneRuntimeContext()

    plan_metadata_obj = ingest_config.metadata.get("plan") if ingest_config.metadata else None
    plan_metadata = plan_metadata_obj if isinstance(plan_metadata_obj, Mapping) else None
    planned_dimensions = _planned_dimensions(plan_metadata)
    if ingest_config.should_run:
        summary = (
            "Timescale ingest plan configured"
            if planned_dimensions
            else "Timescale ingest plan empty"
        )
        plan_status = BackboneStatus.ok if planned_dimensions else BackboneStatus.warn
    else:
        summary = ingest_config.reason or "Institutional ingest disabled"
        plan_status = BackboneStatus.warn
    plan_component = BackboneComponentSnapshot(
        name="plan",
        status=plan_status,
        summary=summary,
        metadata={"dimensions": list(planned_dimensions)},
    )
    components.append(plan_component)
    overall = _combine_status(overall, plan_status)

    if health_report is not None:
        health_component = BackboneComponentSnapshot(
            name="ingest_health",
            status=BackboneStatus.from_health(health_report.status),
            summary=f"{health_report.status.value} with {len(health_report.checks)} checks",
            metadata={
                "generated_at": health_report.generated_at.isoformat(),
            },
        )
        components.append(health_component)
        overall = _combine_status(overall, health_component.status)

    if quality_report is not None:
        quality_component = BackboneComponentSnapshot(
            name="ingest_quality",
            status=BackboneStatus.from_quality(quality_report.status),
            summary=f"score={quality_report.score:.2f}",
            metadata={"generated_at": quality_report.generated_at.isoformat()},
        )
        components.append(quality_component)
        overall = _combine_status(overall, quality_component.status)

    if metrics_snapshot is not None and metrics_snapshot.dimensions:
        metrics_component = BackboneComponentSnapshot(
            name="ingest_metrics",
            status=BackboneStatus.ok,
            summary="metrics collected",
            metadata=dict(_summarise_metrics(metrics_snapshot)),
        )
        components.append(metrics_component)

    if spark_snapshot is not None:
        spark_status = _spark_status_to_backbone(spark_snapshot.status)
        job_summaries = [
            {
                "dimension": job.dimension,
                "status": job.status.value,
                "rows": job.rows,
                "paths": list(job.paths),
                "issues": list(job.issues),
            }
            for job in spark_snapshot.jobs
        ]
        spark_metadata: dict[str, object] = {
            "generated_at": spark_snapshot.generated_at.isoformat(),
            "format": spark_snapshot.format.value,
            "root_path": spark_snapshot.root_path,
            "job_count": len(spark_snapshot.jobs),
            "jobs": job_summaries,
        }
        if spark_snapshot.metadata:
            spark_metadata["metadata"] = dict(spark_snapshot.metadata)
        job_label = "job" if len(spark_snapshot.jobs) == 1 else "jobs"
        spark_component = BackboneComponentSnapshot(
            name="spark_exports",
            status=spark_status,
            summary=f"spark {spark_snapshot.status.value} ({len(spark_snapshot.jobs)} {job_label})",
            metadata=spark_metadata,
        )
        components.append(spark_component)
        overall = _combine_status(overall, spark_component.status)

    if spark_stress_snapshot is not None:
        stress_status = _spark_stress_status_to_backbone(spark_stress_snapshot.status)
        cycles = [cycle.as_dict() for cycle in spark_stress_snapshot.cycles]
        stress_metadata: dict[str, object] = {
            "generated_at": spark_stress_snapshot.generated_at.isoformat(),
            "cycle_count": len(cycles),
            "cycles": cycles,
        }
        if spark_stress_snapshot.metadata:
            stress_metadata["metadata"] = dict(spark_stress_snapshot.metadata)
        stress_component = BackboneComponentSnapshot(
            name="spark_stress",
            status=stress_status,
            summary=(f"stress {spark_stress_snapshot.status.value} ({len(cycles)} cycles)"),
            metadata=stress_metadata,
        )
        components.append(stress_component)
        overall = _combine_status(overall, stress_component.status)

    if failover_decision is not None:
        if failover_decision.should_failover:
            summary = failover_decision.reason or "Timescale failover triggered"
            status = BackboneStatus.fail
        elif failover_decision.optional_triggers:
            summary = "optional slices degraded"
            status = BackboneStatus.warn
        else:
            summary = "Timescale healthy"
            status = BackboneStatus.ok
        failover_component = BackboneComponentSnapshot(
            name="failover",
            status=status,
            summary=summary,
            metadata={
                "triggered": list(failover_decision.triggered_dimensions),
                "optional_triggers": list(failover_decision.optional_triggers),
            },
        )
        components.append(failover_component)
        overall = _combine_status(overall, failover_component.status)

    if recovery_recommendation is not None and not recovery_recommendation.is_empty():
        recovery_component = BackboneComponentSnapshot(
            name="recovery",
            status=BackboneStatus.warn,
            summary="automated recovery attempted",
            metadata=recovery_recommendation.summary(),
        )
        components.append(recovery_component)
        overall = _combine_status(overall, recovery_component.status)

    if backup_snapshot is not None:
        backup_component = BackboneComponentSnapshot(
            name="backups",
            status=BackboneStatus.from_backup(backup_snapshot.status),
            summary=f"backup status={backup_snapshot.status.value}",
            metadata={"service": backup_snapshot.service},
        )
        components.append(backup_component)
        overall = _combine_status(overall, backup_component.status)

    if context.redis_expected:
        if context.redis_configured:
            summary = f"Redis active ({context.redis_backing or 'redis'})"
            status = BackboneStatus.ok
        elif context.redis_backing:
            summary = f"Redis fallback in use ({context.redis_backing})"
            status = BackboneStatus.warn
        else:
            summary = "Redis unavailable"
            status = BackboneStatus.warn
        redis_component = BackboneComponentSnapshot(
            name="redis_cache",
            status=status,
            summary=summary,
            metadata={
                "namespace": context.redis_namespace,
                "configured": context.redis_configured,
            },
        )
        components.append(redis_component)
        overall = _combine_status(overall, redis_component.status)

    if context.kafka_expected:
        if context.kafka_configured and context.kafka_publishers:
            summary = "Kafka publishers active"
            status = BackboneStatus.ok
        elif context.kafka_configured:
            summary = "Kafka configured without publishers"
            status = BackboneStatus.warn
        else:
            summary = "Kafka topics pending"
            status = BackboneStatus.warn
        kafka_component = BackboneComponentSnapshot(
            name="kafka_streaming",
            status=status,
            summary=summary,
            metadata={
                "topics": list(context.kafka_topics),
                "publishers": list(context.kafka_publishers),
            },
        )
        components.append(kafka_component)
        overall = _combine_status(overall, kafka_component.status)

    if context.scheduler_enabled or context.scheduler_state is not None:
        running = context.scheduler_running
        scheduler_status = BackboneStatus.ok if running else BackboneStatus.warn
        summary = "scheduler running" if running else "scheduler enabled"
        scheduler_metadata: dict[str, object] = {
            "enabled": context.scheduler_enabled,
            "running": running,
        }
        if context.scheduler_state is not None:
            scheduler_metadata.update(context.scheduler_state.as_dict())
        scheduler_component = BackboneComponentSnapshot(
            name="scheduler",
            status=scheduler_status,
            summary=summary,
            metadata=scheduler_metadata,
        )
        components.append(scheduler_component)
        overall = _combine_status(overall, scheduler_component.status)

    combined_metadata: dict[str, object] = {
        "ingest": dict(ingest_config.metadata),
    }
    if metadata:
        combined_metadata.update({"telemetry": dict(metadata)})
    if context.kafka_topics:
        combined_metadata["kafka_topics"] = list(context.kafka_topics)
    if context.redis_namespace:
        combined_metadata["redis_namespace"] = context.redis_namespace
    if context.scheduler_state is not None or context.scheduler_enabled:
        scheduler_metadata = (
            context.scheduler_state.as_dict()
            if context.scheduler_state is not None
            else {"running": False}
        )
        scheduler_metadata["enabled"] = context.scheduler_enabled
        combined_metadata["scheduler"] = scheduler_metadata

    return DataBackboneReadinessSnapshot(
        status=overall,
        generated_at=generated_at,
        components=tuple(components),
        metadata=combined_metadata,
    )


__all__ = [
    "BackboneComponentSnapshot",
    "BackboneRuntimeContext",
    "BackboneStatus",
    "DataBackboneValidationSnapshot",
    "DataBackboneReadinessSnapshot",
    "evaluate_data_backbone_validation",
    "evaluate_data_backbone_readiness",
]
