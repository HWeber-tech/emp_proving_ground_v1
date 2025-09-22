"""Telemetry helpers for the Timescale ingest scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping

from src.core.event_bus import Event, EventBus

from .scheduler import IngestSchedule, IngestSchedulerState


class IngestSchedulerStatus(StrEnum):
    """Severity levels for scheduler telemetry."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[IngestSchedulerStatus, int] = {
    IngestSchedulerStatus.ok: 0,
    IngestSchedulerStatus.warn: 1,
    IngestSchedulerStatus.fail: 2,
}


def _escalate(
    current: IngestSchedulerStatus, candidate: IngestSchedulerStatus
) -> IngestSchedulerStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class IngestSchedulerSnapshot:
    """Aggregated telemetry describing the ingest scheduler state."""

    status: IngestSchedulerStatus
    generated_at: datetime
    enabled: bool
    running: bool
    consecutive_failures: int
    interval_seconds: float
    jitter_seconds: float
    max_failures: int
    last_started_at: datetime | None = None
    last_completed_at: datetime | None = None
    last_success_at: datetime | None = None
    next_run_at: datetime | None = None
    issues: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Serialise the snapshot into primitives for logging or JSON."""

        def _iso(moment: datetime | None) -> str | None:
            if moment is None:
                return None
            ts = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
            return ts.astimezone(UTC).isoformat()

        payload: dict[str, object] = {
            "status": self.status.value,
            "generated_at": _iso(self.generated_at),
            "enabled": self.enabled,
            "running": self.running,
            "consecutive_failures": self.consecutive_failures,
            "interval_seconds": self.interval_seconds,
            "jitter_seconds": self.jitter_seconds,
            "max_failures": self.max_failures,
            "issues": list(self.issues),
        }
        if self.last_started_at is not None:
            payload["last_started_at"] = _iso(self.last_started_at)
        if self.last_completed_at is not None:
            payload["last_completed_at"] = _iso(self.last_completed_at)
        if self.last_success_at is not None:
            payload["last_success_at"] = _iso(self.last_success_at)
        if self.next_run_at is not None:
            payload["next_run_at"] = _iso(self.next_run_at)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_markdown(self) -> str:
        """Render a concise Markdown summary for dashboards or logs."""

        lines = [
            f"**Status:** {self.status.value.upper()}",
            f"- Enabled: {'yes' if self.enabled else 'no'}",
            f"- Running: {'yes' if self.running else 'no'}",
            f"- Interval: {self.interval_seconds:.1f}s (±{self.jitter_seconds:.1f}s jitter)",
            f"- Max failures: {self.max_failures}",
            f"- Consecutive failures: {self.consecutive_failures}",
        ]
        if self.last_success_at is not None:
            lines.append(f"- Last success: {self.last_success_at.isoformat()}")
        elif self.last_completed_at is not None:
            lines.append(f"- Last completed: {self.last_completed_at.isoformat()}")
        elif self.last_started_at is not None:
            lines.append(f"- Last started: {self.last_started_at.isoformat()}")
        else:
            lines.append("- Last run: never")
        if self.next_run_at is not None:
            lines.append(f"- Next run at: {self.next_run_at.isoformat()}")
        if self.issues:
            lines.append("")
            lines.append("**Issues:**")
            for issue in self.issues:
                lines.append(f"- {issue}")
        return "\n".join(lines)


def _resolve_interval(
    schedule: IngestSchedule | None, state: IngestSchedulerState | None
) -> tuple[float, float, int]:
    if schedule is not None:
        return (
            float(schedule.interval_seconds),
            float(schedule.jitter_seconds),
            int(schedule.max_failures),
        )
    if state is not None:
        return (
            float(state.interval_seconds),
            float(state.jitter_seconds),
            int(state.max_failures),
        )
    return 0.0, 0.0, 0


def _latest_activity(state: IngestSchedulerState | None) -> datetime | None:
    if state is None:
        return None
    for attr in ("last_success_at", "last_completed_at", "last_started_at"):
        moment = getattr(state, attr)
        if moment is not None:
            return moment
    return None


def build_scheduler_snapshot(
    *,
    enabled: bool,
    schedule: IngestSchedule | None,
    state: IngestSchedulerState | None,
    now: datetime | None = None,
    stale_after_seconds: float | None = None,
) -> IngestSchedulerSnapshot:
    """Translate scheduler state into a reusable telemetry snapshot."""

    moment = now or datetime.now(tz=UTC)
    interval, jitter, max_failures = _resolve_interval(schedule, state)
    running = bool(state and state.running)
    consecutive_failures = state.consecutive_failures if state else 0
    next_run_at = state.next_run_at if state else None
    last_started = state.last_started_at if state else None
    last_completed = state.last_completed_at if state else None
    last_success = state.last_success_at if state else None

    status = IngestSchedulerStatus.ok
    issues: list[str] = []

    if not enabled:
        status = IngestSchedulerStatus.warn
        issues.append("Scheduler disabled in configuration")

    if state is None:
        if enabled:
            status = IngestSchedulerStatus.fail
            issues.append("Scheduler telemetry unavailable")
        return IngestSchedulerSnapshot(
            status=status,
            generated_at=moment,
            enabled=enabled,
            running=False,
            consecutive_failures=0,
            interval_seconds=interval,
            jitter_seconds=jitter,
            max_failures=max_failures,
            last_started_at=None,
            last_completed_at=None,
            last_success_at=None,
            next_run_at=None,
            issues=tuple(issues),
            metadata={"stale_after_seconds": stale_after_seconds},
        )

    if enabled and not running:
        status = _escalate(status, IngestSchedulerStatus.fail)
        issues.append("Scheduler loop not running")

    if consecutive_failures > 0:
        status = _escalate(status, IngestSchedulerStatus.warn)
        issues.append(f"Consecutive failures recorded ({consecutive_failures})")
        if max_failures and consecutive_failures >= max_failures:
            status = _escalate(status, IngestSchedulerStatus.fail)
            issues.append("Failure threshold reached – scheduler will stop without intervention")

    latest_activity = _latest_activity(state)
    effective_interval = interval or (state.interval_seconds if state else 0.0)
    if stale_after_seconds is not None:
        stale_threshold = max(stale_after_seconds, 0.0)
    elif effective_interval > 0:
        stale_threshold = max(effective_interval * 3.0, 60.0)
    else:
        stale_threshold = 300.0

    if latest_activity is None:
        status = _escalate(status, IngestSchedulerStatus.warn)
        issues.append("Scheduler has not completed a run yet")
    else:
        age = (moment - latest_activity).total_seconds()
        if age > stale_threshold:
            overdue = int(age - stale_threshold)
            status = _escalate(
                status,
                IngestSchedulerStatus.warn if running else IngestSchedulerStatus.fail,
            )
            issues.append(f"Last success {int(age)}s ago exceeds threshold {int(stale_threshold)}s")
            if overdue > 0:
                issues.append(f"Scheduler overdue by ~{overdue}s")

    if next_run_at is not None and next_run_at < moment:
        late_by = int((moment - next_run_at).total_seconds())
        if late_by > 0:
            status = _escalate(
                status,
                IngestSchedulerStatus.warn if running else IngestSchedulerStatus.fail,
            )
            issues.append(f"Next run overdue by {late_by}s")

    metadata: dict[str, object] = {
        "stale_after_seconds": stale_threshold,
    }

    return IngestSchedulerSnapshot(
        status=status,
        generated_at=moment,
        enabled=enabled,
        running=running,
        consecutive_failures=consecutive_failures,
        interval_seconds=interval,
        jitter_seconds=jitter,
        max_failures=max_failures,
        last_started_at=last_started,
        last_completed_at=last_completed,
        last_success_at=last_success,
        next_run_at=next_run_at,
        issues=tuple(issues),
        metadata=metadata,
    )


def format_scheduler_markdown(snapshot: IngestSchedulerSnapshot) -> str:
    """Convenience wrapper mirroring other telemetry helpers."""

    return snapshot.to_markdown()


async def publish_scheduler_snapshot(
    event_bus: EventBus,
    snapshot: IngestSchedulerSnapshot,
    *,
    source: str = "timescale_ingest_scheduler",
) -> None:
    """Publish scheduler telemetry on ``telemetry.ingest.scheduler``."""

    payload = snapshot.as_dict()
    payload["markdown"] = format_scheduler_markdown(snapshot)
    event = Event(
        type="telemetry.ingest.scheduler",
        payload=payload,
        source=source,
    )
    await event_bus.publish(event)


__all__ = [
    "IngestSchedulerSnapshot",
    "IngestSchedulerStatus",
    "build_scheduler_snapshot",
    "format_scheduler_markdown",
    "publish_scheduler_snapshot",
]
