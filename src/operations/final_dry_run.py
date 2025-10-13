from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import math
import os
import signal
from asyncio.subprocess import PIPE, Process
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence, TextIO

from src.operations.dry_run_audit import (
    DryRunSignOffReport,
    DryRunStatus,
    DryRunSummary,
    assess_sign_off_readiness,
    evaluate_dry_run,
    humanise_timedelta,
)
from src.runtime.task_supervisor import TaskSupervisor
from src.trading.execution.resource_monitor import ResourceUsageMonitor

__all__ = [
    "FinalDryRunConfig",
    "FinalDryRunResult",
    "HarnessIncident",
    "perform_final_dry_run",
    "run_final_dry_run",
]


_MIN_DURATION_TOLERANCE = timedelta(seconds=0.1)
_LOG_FAIL_LEVELS = ("error", "exception", "critical", "fatal")
_LOG_WARN_LEVELS = ("warning", "warn")


@dataclass(slots=True, frozen=True)
class FinalDryRunConfig:
    """Configuration for executing a final dry run harness."""

    command: Sequence[str]
    duration: timedelta
    log_directory: Path
    progress_path: Path | None = None
    progress_interval: timedelta | None = timedelta(minutes=5)
    diary_path: Path | None = None
    performance_path: Path | None = None
    minimum_uptime_ratio: float = 0.98
    required_duration: timedelta | None = None
    shutdown_grace: timedelta = timedelta(seconds=60)
    require_diary_evidence: bool = True
    require_performance_evidence: bool = True
    allow_warnings: bool = False
    minimum_sharpe_ratio: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, str] | None = None
    monitor_log_levels: bool = True
    log_gap_warn: timedelta | None = None
    log_gap_fail: timedelta | None = None
    live_gap_alert: timedelta | None = None
    live_gap_severity: DryRunStatus = DryRunStatus.warn
    diary_stale_warn: timedelta | None = None
    diary_stale_fail: timedelta | None = None
    performance_stale_warn: timedelta | None = None
    performance_stale_fail: timedelta | None = None
    evidence_check_interval: timedelta | None = None
    evidence_initial_grace: timedelta = timedelta(minutes=15)
    compress_logs: bool = False
    log_rotate_interval: timedelta | None = None
    resource_sample_interval: timedelta | None = timedelta(minutes=1)
    resource_max_cpu_percent: float | None = None
    resource_max_memory_mb: float | None = None
    resource_max_memory_percent: float | None = None
    resource_violation_severity: DryRunStatus = DryRunStatus.fail

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("command cannot be empty")
        normalised_command = tuple(str(token) for token in self.command)
        object.__setattr__(self, "command", normalised_command)

        if self.duration <= timedelta(0):
            raise ValueError("duration must be positive")

        required_duration = self.required_duration or self.duration
        if required_duration <= timedelta(0):
            raise ValueError("required_duration must be positive")
        object.__setattr__(self, "required_duration", required_duration)

        if not (0.0 <= self.minimum_uptime_ratio <= 1.0):
            raise ValueError("minimum_uptime_ratio must be between 0.0 and 1.0")

        if self.shutdown_grace < timedelta(0):
            raise ValueError("shutdown_grace must be non-negative")

        object.__setattr__(self, "log_directory", Path(self.log_directory))

        diary_path = Path(self.diary_path) if self.diary_path is not None else None
        performance_path = (
            Path(self.performance_path) if self.performance_path is not None else None
        )
        object.__setattr__(self, "diary_path", diary_path)
        object.__setattr__(self, "performance_path", performance_path)

        progress_path = (
            Path(self.progress_path) if self.progress_path is not None else None
        )
        object.__setattr__(self, "progress_path", progress_path)

        if self.progress_interval is not None and self.progress_interval <= timedelta(0):
            raise ValueError("progress_interval must be positive when provided")

        if self.minimum_sharpe_ratio is not None and self.minimum_sharpe_ratio < 0:
            raise ValueError("minimum_sharpe_ratio must be non-negative when provided")

        normalised_metadata = {
            str(key): value for key, value in (self.metadata or {}).items()
        }
        object.__setattr__(self, "metadata", normalised_metadata)

        if self.environment is not None:
            normalised_env = {
                str(key): str(value)
                for key, value in self.environment.items()
            }
            object.__setattr__(self, "environment", normalised_env)

        object.__setattr__(self, "monitor_log_levels", bool(self.monitor_log_levels))

        log_gap_warn = self.log_gap_warn
        if log_gap_warn is not None and log_gap_warn <= timedelta(0):
            raise ValueError("log_gap_warn must be positive when provided")
        log_gap_fail = self.log_gap_fail
        if log_gap_fail is not None and log_gap_fail <= timedelta(0):
            raise ValueError("log_gap_fail must be positive when provided")
        if (
            log_gap_warn is not None
            and log_gap_fail is not None
            and log_gap_fail < log_gap_warn
        ):
            raise ValueError("log_gap_fail must be greater than or equal to log_gap_warn")
        object.__setattr__(self, "log_gap_warn", log_gap_warn)
        object.__setattr__(self, "log_gap_fail", log_gap_fail)

        live_gap_alert = self.live_gap_alert
        if live_gap_alert is not None and live_gap_alert <= timedelta(0):
            raise ValueError("live_gap_alert must be positive when provided")
        object.__setattr__(self, "live_gap_alert", live_gap_alert)

        try:
            live_gap_severity = DryRunStatus(self.live_gap_severity)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError("live_gap_severity must map to a DryRunStatus value") from exc
        object.__setattr__(self, "live_gap_severity", live_gap_severity)

        def _validate_evidence_threshold(
            label: str,
            warn: timedelta | None,
            fail: timedelta | None,
        ) -> tuple[timedelta | None, timedelta | None]:
            if warn is not None and warn <= timedelta(0):
                raise ValueError(f"{label}_warn must be positive when provided")
            if fail is not None and fail <= timedelta(0):
                raise ValueError(f"{label}_fail must be positive when provided")
            if warn is not None and fail is not None and fail < warn:
                raise ValueError(
                    f"{label}_fail must be greater than or equal to {label}_warn"
                )
            return warn, fail

        diary_warn, diary_fail = _validate_evidence_threshold(
            "diary_stale",
            self.diary_stale_warn,
            self.diary_stale_fail,
        )
        performance_warn, performance_fail = _validate_evidence_threshold(
            "performance_stale",
            self.performance_stale_warn,
            self.performance_stale_fail,
        )
        object.__setattr__(self, "diary_stale_warn", diary_warn)
        object.__setattr__(self, "diary_stale_fail", diary_fail)
        object.__setattr__(self, "performance_stale_warn", performance_warn)
        object.__setattr__(self, "performance_stale_fail", performance_fail)

        check_interval = self.evidence_check_interval
        if check_interval is not None and check_interval <= timedelta(0):
            raise ValueError(
                "evidence_check_interval must be positive when provided"
            )
        object.__setattr__(self, "evidence_check_interval", check_interval)

        if self.evidence_initial_grace < timedelta(0):
            raise ValueError("evidence_initial_grace must be non-negative")

        object.__setattr__(self, "compress_logs", bool(self.compress_logs))

        rotate_interval = self.log_rotate_interval
        if rotate_interval is not None:
            if rotate_interval <= timedelta(0):
                raise ValueError(
                    "log_rotate_interval must be positive when provided"
                )
            object.__setattr__(self, "log_rotate_interval", rotate_interval)

        sample_interval = self.resource_sample_interval
        if sample_interval is not None:
            if sample_interval <= timedelta(0):
                raise ValueError(
                    "resource_sample_interval must be positive when provided"
                )
            object.__setattr__(self, "resource_sample_interval", sample_interval)

        def _validate_non_negative(name: str, value: float | None) -> float | None:
            if value is None:
                return None
            if value < 0:
                raise ValueError(f"{name} must be non-negative when provided")
            return value

        cpu_limit = _validate_non_negative(
            "resource_max_cpu_percent",
            self.resource_max_cpu_percent,
        )
        memory_mb_limit = _validate_non_negative(
            "resource_max_memory_mb",
            self.resource_max_memory_mb,
        )
        memory_percent_limit = _validate_non_negative(
            "resource_max_memory_percent",
            self.resource_max_memory_percent,
        )
        object.__setattr__(self, "resource_max_cpu_percent", cpu_limit)
        object.__setattr__(self, "resource_max_memory_mb", memory_mb_limit)
        object.__setattr__(self, "resource_max_memory_percent", memory_percent_limit)

        try:
            severity = DryRunStatus(self.resource_violation_severity)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                "resource_violation_severity must map to a DryRunStatus value"
            ) from exc
        if severity not in {DryRunStatus.warn, DryRunStatus.fail}:
            raise ValueError(
                "resource_violation_severity must be either 'warn' or 'fail'"
            )
        object.__setattr__(self, "resource_violation_severity", severity)


@dataclass(slots=True, frozen=True)
class HarnessIncident:
    """Incident captured by the dry run harness during orchestration."""

    severity: DryRunStatus
    occurred_at: datetime
    message: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "severity": self.severity.value,
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat(),
            "message": self.message,
        }
        if self.metadata:
            payload["metadata"] = {str(k): v for k, v in self.metadata.items()}
        return payload


@dataclass(slots=True, frozen=True)
class FinalDryRunResult:
    """Result bundle produced after executing a final dry run."""

    config: FinalDryRunConfig
    started_at: datetime
    ended_at: datetime
    exit_code: int | None
    summary: DryRunSummary
    sign_off: DryRunSignOffReport | None
    log_paths: tuple[Path, ...]
    raw_log_paths: tuple[Path, ...]
    progress_path: Path | None
    incidents: tuple[HarnessIncident, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> timedelta:
        return self.ended_at - self.started_at

    @property
    def status(self) -> DryRunStatus:
        statuses: list[DryRunStatus] = [self.summary.status]
        if self.sign_off is not None:
            statuses.append(self.sign_off.status)
        statuses.extend(incident.severity for incident in self.incidents)
        if any(status is DryRunStatus.fail for status in statuses):
            return DryRunStatus.fail
        if any(status is DryRunStatus.warn for status in statuses):
            return DryRunStatus.warn
        return DryRunStatus.pass_

    @property
    def log_path(self) -> Path:
        return self.log_paths[0]

    @property
    def raw_log_path(self) -> Path:
        return self.raw_log_paths[0]


class _LogStats:
    """Track aggregated statistics for runtime log streams."""

    def __init__(self) -> None:
        self._stream_counts: Counter[str] = Counter()
        self._level_counts: Counter[str] = Counter()
        self._total_lines = 0
        self._first_line_at: datetime | None = None
        self._last_line_at: datetime | None = None
        self._lock = asyncio.Lock()

    async def record(self, stream: str, level: Any, observed_at: datetime) -> None:
        stream_key = str(stream or "unknown")
        level_key = str(level or "unknown").lower()
        async with self._lock:
            self._stream_counts[stream_key] += 1
            self._level_counts[level_key] += 1
            self._total_lines += 1
            if self._first_line_at is None or observed_at < self._first_line_at:
                self._first_line_at = observed_at
            if self._last_line_at is None or observed_at > self._last_line_at:
                self._last_line_at = observed_at

    async def snapshot(self) -> Mapping[str, Any]:
        async with self._lock:
            return {
                "lines": Counter(self._stream_counts),
                "levels": Counter(self._level_counts),
                "total_lines": self._total_lines,
                "first_line_at": (
                    self._first_line_at.astimezone(UTC).isoformat()
                    if self._first_line_at is not None
                    else None
                ),
                "last_line_at": (
                    self._last_line_at.astimezone(UTC).isoformat()
                    if self._last_line_at is not None
                    else None
                ),
            }


class _LogSink:
    """Write structured/raw logs with optional rotation support."""

    def __init__(
        self,
        *,
        directory: Path,
        slug: str,
        compress: bool,
        rotate_interval: timedelta | None,
    ) -> None:
        self._directory = Path(directory)
        self._slug = slug
        self._compress = compress
        self._rotate_interval = rotate_interval
        self._part_index = 0
        self._structured_paths: list[Path] = []
        self._raw_paths: list[Path] = []
        self._structured_handle: TextIO | None = None
        self._raw_handle: TextIO | None = None
        self._structured_suffix = "jsonl.gz" if compress else "jsonl"
        self._raw_suffix = "log.gz" if compress else "log"
        self._part_started_at = datetime.now(tz=UTC)
        self._open_new_handles(initial=True)

    def write(
        self,
        *,
        structured_line: str,
        raw_line: str,
        observed_at: datetime,
    ) -> None:
        self._rotate_if_needed(observed_at)
        assert self._structured_handle is not None
        assert self._raw_handle is not None
        self._structured_handle.write(structured_line + "\n")
        self._raw_handle.write(raw_line + "\n")
        self._structured_handle.flush()
        self._raw_handle.flush()

    def close(self) -> None:
        if self._structured_handle is not None:
            self._structured_handle.flush()
            self._structured_handle.close()
            self._structured_handle = None
        if self._raw_handle is not None:
            self._raw_handle.flush()
            self._raw_handle.close()
            self._raw_handle = None

    @property
    def structured_paths(self) -> tuple[Path, ...]:
        return tuple(self._structured_paths)

    @property
    def raw_paths(self) -> tuple[Path, ...]:
        return tuple(self._raw_paths)

    def _rotate_if_needed(self, observed_at: datetime) -> None:
        if self._rotate_interval is None:
            return
        interval = self._rotate_interval
        assert interval is not None
        while observed_at - self._part_started_at >= interval:
            self._open_new_handles(initial=False, start_at=observed_at)

    def _open_new_handles(
        self,
        *,
        initial: bool,
        start_at: datetime | None = None,
    ) -> None:
        if not initial:
            self.close()
            self._part_index += 1
            if start_at is not None:
                self._part_started_at = start_at
            else:
                self._part_started_at = datetime.now(tz=UTC)
        else:
            self._part_started_at = datetime.now(tz=UTC)

        structured_path = self._build_path(self._structured_suffix, self._part_index)
        raw_path = self._build_path(self._raw_suffix, self._part_index)
        self._structured_paths.append(structured_path)
        self._raw_paths.append(raw_path)

        if self._compress:
            self._structured_handle = gzip.open(structured_path, "wt", encoding="utf-8")
            self._raw_handle = gzip.open(raw_path, "wt", encoding="utf-8")
        else:
            self._structured_handle = structured_path.open("w", encoding="utf-8")
            self._raw_handle = raw_path.open("w", encoding="utf-8")

    def _build_path(self, suffix: str, part_index: int) -> Path:
        if part_index == 0:
            name = f"final_dry_run_{self._slug}.{suffix}"
        else:
            name = f"final_dry_run_{self._slug}_p{part_index:03d}.{suffix}"
        return self._directory / name


async def _append_incident(
    *,
    incident: HarnessIncident,
    key: tuple[str, ...],
    incidents: list[HarnessIncident],
    incident_keys: set[tuple[str, ...]],
    lock: asyncio.Lock,
    progress_reporter: "_ProgressReporter | None",
) -> bool:
    """Record a harness incident if it has not been observed before."""

    async with lock:
        if key in incident_keys:
            return False
        incident_keys.add(key)
        incidents.append(incident)
        snapshot = tuple(incidents)

    if progress_reporter is not None:
        await progress_reporter.record_incident(
            incident=incident,
            incidents=snapshot,
        )

    return True


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isnan(result):
            return None
        return result
    if isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            return None
        if math.isnan(result):
            return None
        return result
    return None


@dataclass(slots=True)
class _ResourceMonitorMetrics:
    """Track resource usage extrema observed during the dry run."""

    samples: int = 0
    peak_cpu_percent: float | None = None
    peak_cpu_timestamp: str | None = None
    peak_memory_mb: float | None = None
    peak_memory_mb_timestamp: str | None = None
    peak_memory_percent: float | None = None
    peak_memory_percent_timestamp: str | None = None
    last_sample: Mapping[str, Any] | None = None

    def record(self, sample: Mapping[str, Any]) -> None:
        timestamp_value = str(sample.get("timestamp") or "").strip() or None
        cpu_percent = _coerce_float(sample.get("cpu_percent"))
        memory_mb = _coerce_float(sample.get("memory_mb"))
        memory_percent = _coerce_float(sample.get("memory_percent"))

        self.samples += 1
        self.last_sample = {
            "timestamp": timestamp_value,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "memory_percent": memory_percent,
        }

        if cpu_percent is not None:
            if self.peak_cpu_percent is None or cpu_percent > self.peak_cpu_percent:
                self.peak_cpu_percent = cpu_percent
                self.peak_cpu_timestamp = timestamp_value

        if memory_mb is not None:
            if self.peak_memory_mb is None or memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = memory_mb
                self.peak_memory_mb_timestamp = timestamp_value

        if memory_percent is not None:
            if (
                self.peak_memory_percent is None
                or memory_percent > self.peak_memory_percent
            ):
                self.peak_memory_percent = memory_percent
                self.peak_memory_percent_timestamp = timestamp_value

    def as_metadata(
        self,
        *,
        enabled: bool,
        reason: str | None,
        interval_seconds: float | None,
        severity: DryRunStatus | None,
        thresholds: Mapping[str, float | None],
    ) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "enabled": enabled,
        }
        if reason:
            payload["reason"] = reason
        if interval_seconds is not None:
            payload["interval_seconds"] = interval_seconds
        if severity is not None:
            payload["violation_severity"] = severity.value

        threshold_payload = {
            key: value for key, value in thresholds.items() if value is not None
        }
        if threshold_payload:
            payload["thresholds"] = threshold_payload

        payload["samples"] = self.samples
        if self.samples <= 0:
            return payload

        if self.peak_cpu_percent is not None:
            payload["peak_cpu_percent"] = self.peak_cpu_percent
            if self.peak_cpu_timestamp:
                payload["peak_cpu_percent_timestamp"] = self.peak_cpu_timestamp
        if self.peak_memory_mb is not None:
            payload["peak_memory_mb"] = self.peak_memory_mb
            if self.peak_memory_mb_timestamp:
                payload["peak_memory_mb_timestamp"] = self.peak_memory_mb_timestamp
        if self.peak_memory_percent is not None:
            payload["peak_memory_percent"] = self.peak_memory_percent
            if self.peak_memory_percent_timestamp:
                payload["peak_memory_percent_timestamp"] = (
                    self.peak_memory_percent_timestamp
                )
        if self.last_sample is not None:
            payload["last_sample"] = dict(self.last_sample)
        return payload


def _create_resource_monitor(process_pid: int) -> ResourceUsageMonitor | None:
    try:
        import psutil  # type: ignore  # noqa: WPS433 (module import inside function)
    except Exception:  # pragma: no cover - optional dependency missing
        return None

    try:
        ps_process = psutil.Process(process_pid)
    except Exception:  # pragma: no cover - process lookup failure
        return None

    try:
        return ResourceUsageMonitor(process=ps_process)
    except Exception:  # pragma: no cover - monitor initialisation failure
        return None


async def _run_resource_monitor(
    *,
    monitor: ResourceUsageMonitor,
    process: Process,
    interval: timedelta,
    metrics: _ResourceMonitorMetrics,
    max_cpu_percent: float | None,
    max_memory_mb: float | None,
    max_memory_percent: float | None,
    severity: DryRunStatus,
    incidents: list[HarnessIncident],
    incident_keys: set[tuple[str, ...]],
    incident_lock: asyncio.Lock,
    progress_reporter: "_ProgressReporter | None",
) -> None:
    interval_seconds = max(interval.total_seconds(), 0.05)
    # Prime psutil's cpu_percent estimation without recording the initial zero
    with contextlib.suppress(Exception):
        await asyncio.to_thread(monitor.sample)

    cpu_reported = max_cpu_percent is None
    memory_mb_reported = max_memory_mb is None
    memory_percent_reported = max_memory_percent is None

    try:
        while True:
            sample: Mapping[str, Any] | None
            try:
                sample = await asyncio.to_thread(monitor.sample)
            except Exception:  # pragma: no cover - defensive guard
                break

            if sample is None or not sample:
                if process.returncode is not None:
                    break
                await asyncio.sleep(interval_seconds)
                continue

            metrics.record(sample)

            last_sample = metrics.last_sample or {}
            cpu_percent = last_sample.get("cpu_percent")
            memory_mb = last_sample.get("memory_mb")
            memory_percent = last_sample.get("memory_percent")
            observed_at = last_sample.get("timestamp")

            if (
                not cpu_reported
                and cpu_percent is not None
                and max_cpu_percent is not None
                and cpu_percent > max_cpu_percent
            ):
                incident = HarnessIncident(
                    severity=severity,
                    occurred_at=datetime.now(tz=UTC),
                    message=(
                        "Resource usage exceeded CPU threshold "
                        f"({cpu_percent:.2f}% > {max_cpu_percent:.2f}%)."
                    ),
                    metadata={
                        "observed_cpu_percent": cpu_percent,
                        "threshold_cpu_percent": max_cpu_percent,
                        "observed_at": observed_at,
                    },
                )
                recorded = await _append_incident(
                    incident=incident,
                    key=("resource_monitor", "cpu", severity.value),
                    incidents=incidents,
                    incident_keys=incident_keys,
                    lock=incident_lock,
                    progress_reporter=progress_reporter,
                )
                if recorded:
                    cpu_reported = True

            if (
                not memory_mb_reported
                and memory_mb is not None
                and max_memory_mb is not None
                and memory_mb > max_memory_mb
            ):
                incident = HarnessIncident(
                    severity=severity,
                    occurred_at=datetime.now(tz=UTC),
                    message=(
                        "Resource usage exceeded memory threshold "
                        f"({memory_mb:.2f} MiB > {max_memory_mb:.2f} MiB)."
                    ),
                    metadata={
                        "observed_memory_mb": memory_mb,
                        "threshold_memory_mb": max_memory_mb,
                        "observed_at": observed_at,
                    },
                )
                recorded = await _append_incident(
                    incident=incident,
                    key=("resource_monitor", "memory_mb", severity.value),
                    incidents=incidents,
                    incident_keys=incident_keys,
                    lock=incident_lock,
                    progress_reporter=progress_reporter,
                )
                if recorded:
                    memory_mb_reported = True

            if (
                not memory_percent_reported
                and memory_percent is not None
                and max_memory_percent is not None
                and memory_percent > max_memory_percent
            ):
                incident = HarnessIncident(
                    severity=severity,
                    occurred_at=datetime.now(tz=UTC),
                    message=(
                        "Resource usage exceeded memory percent threshold "
                        f"({memory_percent:.2f}% > {max_memory_percent:.2f}%)."
                    ),
                    metadata={
                        "observed_memory_percent": memory_percent,
                        "threshold_memory_percent": max_memory_percent,
                        "observed_at": observed_at,
                    },
                )
                recorded = await _append_incident(
                    incident=incident,
                    key=("resource_monitor", "memory_percent", severity.value),
                    incidents=incidents,
                    incident_keys=incident_keys,
                    lock=incident_lock,
                    progress_reporter=progress_reporter,
                )
                if recorded:
                    memory_percent_reported = True

            if process.returncode is not None:
                break

            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        raise
_PROGRESS_SEVERITY_ORDER: Mapping[DryRunStatus, int] = {
    DryRunStatus.fail: 3,
    DryRunStatus.warn: 2,
    DryRunStatus.pass_: 1,
}


class _ProgressReporter:
    """Persist periodic progress snapshots for the dry run harness."""

    def __init__(
        self,
        *,
        path: Path,
        config: FinalDryRunConfig,
        stats: _LogStats,
        started_at: datetime,
        interval: timedelta,
    ) -> None:
        self._path = Path(path)
        self._config = config
        self._stats = stats
        self._started_at = started_at
        self._interval_seconds = max(interval.total_seconds(), 0.01)
        self._status = "pending"
        self._phase = "startup"
        self._highest_severity: DryRunStatus | None = None
        self._lock = asyncio.Lock()

    async def run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._interval_seconds)
                await self.write(status=self._status, phase="running")
        except asyncio.CancelledError:
            raise

    async def record_incident(
        self,
        *,
        incident: HarnessIncident,
        incidents: Sequence[HarnessIncident],
    ) -> None:
        """Capture a new incident in the progress snapshot immediately."""

        severity = incident.severity
        current_rank = (
            _PROGRESS_SEVERITY_ORDER.get(self._highest_severity, 0)
            if self._highest_severity is not None
            else 0
        )
        new_rank = _PROGRESS_SEVERITY_ORDER.get(severity, 0)
        if new_rank >= current_rank:
            self._highest_severity = severity

        status_value = (
            self._highest_severity.value
            if self._highest_severity is not None
            else self._status
        )
        await self.write(
            status=status_value,
            phase="running",
            incidents=incidents,
        )

    async def write(
        self,
        *,
        status: str | None = None,
        phase: str | None = None,
        now: datetime | None = None,
        exit_code: int | None = None,
        incidents: Sequence[HarnessIncident] | None = None,
        summary: DryRunSummary | None = None,
        sign_off: DryRunSignOffReport | None = None,
    ) -> None:
        if status is not None:
            self._status = status
            severity = _status_to_severity(status)
            if severity is not None:
                current_rank = (
                    _PROGRESS_SEVERITY_ORDER.get(self._highest_severity, 0)
                    if self._highest_severity is not None
                    else 0
                )
                new_rank = _PROGRESS_SEVERITY_ORDER.get(severity, 0)
                if new_rank >= current_rank:
                    self._highest_severity = severity
        if phase is not None:
            self._phase = phase
        now_value = now or datetime.now(tz=UTC)

        stats_snapshot = await self._stats.snapshot()
        line_counts = {
            str(key): value for key, value in stats_snapshot.get("lines", {}).items()
        }
        level_counts = {
            str(key): value for key, value in stats_snapshot.get("levels", {}).items()
        }

        payload: dict[str, Any] = {
            "status": self._status,
            "phase": self._phase,
            "now": now_value.astimezone(UTC).isoformat(),
            "started_at": self._started_at.astimezone(UTC).isoformat(),
            "elapsed_seconds": (now_value - self._started_at).total_seconds(),
            "target_duration_seconds": self._config.duration.total_seconds(),
            "required_duration_seconds": self._config.required_duration.total_seconds()
            if self._config.required_duration is not None
            else None,
            "total_lines": stats_snapshot.get("total_lines", 0),
            "line_counts": line_counts,
            "level_counts": level_counts,
            "first_line_at": stats_snapshot.get("first_line_at"),
            "last_line_at": stats_snapshot.get("last_line_at"),
            "command": list(self._config.command),
            "minimum_uptime_ratio": self._config.minimum_uptime_ratio,
            "require_diary_evidence": self._config.require_diary_evidence,
            "require_performance_evidence": self._config.require_performance_evidence,
        }

        if exit_code is not None:
            payload["exit_code"] = exit_code
        if incidents:
            payload["incidents"] = [incident.as_dict() for incident in incidents]
        if summary is not None:
            payload["summary"] = summary.as_dict()
        if sign_off is not None:
            payload["sign_off"] = sign_off.as_dict()
        if self._config.metadata:
            payload["config_metadata"] = {
                str(key): value for key, value in self._config.metadata.items()
            }

        data = json.dumps(payload, indent=2, sort_keys=True)
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._path.write_text, data, encoding="utf-8")


class _LogGapMonitor:
    """Emit incidents when runtime logs fall silent beyond a threshold."""

    def __init__(
        self,
        *,
        threshold: timedelta,
        severity: DryRunStatus,
        incidents: list[HarnessIncident],
        incident_keys: set[tuple[str, ...]],
        incident_lock: asyncio.Lock,
        progress_reporter: _ProgressReporter | None,
    ) -> None:
        self._threshold = threshold
        self._severity = severity
        self._incidents = incidents
        self._incident_keys = incident_keys
        self._incident_lock = incident_lock
        self._progress_reporter = progress_reporter
        self._last_log_at: datetime | None = None
        self._alert_active = False
        self._lock = asyncio.Lock()

    async def note(self, observed_at: datetime) -> None:
        async with self._lock:
            self._last_log_at = observed_at
            self._alert_active = False

    async def run(self) -> None:
        interval_seconds = max(
            min(self._threshold.total_seconds() / 2.0, 60.0),
            0.5,
        )
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                async with self._lock:
                    last_log_at = self._last_log_at
                    alert_active = self._alert_active
                if last_log_at is None or alert_active:
                    continue
                gap = datetime.now(tz=UTC) - last_log_at
                if gap < self._threshold:
                    continue

                message = (
                    "No runtime logs observed for "
                    f"{humanise_timedelta(gap)} (threshold "
                    f"{humanise_timedelta(self._threshold)})."
                )
                metadata = {
                    "threshold_seconds": self._threshold.total_seconds(),
                    "gap_seconds": gap.total_seconds(),
                    "last_log_at": last_log_at.astimezone(UTC).isoformat(),
                }
                incident = HarnessIncident(
                    severity=self._severity,
                    occurred_at=datetime.now(tz=UTC),
                    message=message,
                    metadata=metadata,
                )

                recorded = await _append_incident(
                    incident=incident,
                    key=(
                        "log_gap_monitor",
                        self._severity.value,
                        last_log_at.astimezone(UTC).isoformat(),
                    ),
                    incidents=self._incidents,
                    incident_keys=self._incident_keys,
                    lock=self._incident_lock,
                    progress_reporter=self._progress_reporter,
                )
                if recorded:
                    async with self._lock:
                        self._alert_active = True
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            raise


class _EvidenceMonitor:
    """Watch an evidence artefact for freshness and emit harness incidents."""

    def __init__(
        self,
        *,
        name: str,
        path: Path,
        warn_after: timedelta | None,
        fail_after: timedelta | None,
        check_interval: timedelta,
        initial_grace: timedelta,
        started_at: datetime,
        incidents: list[HarnessIncident],
        incident_keys: set[tuple[str, ...]],
        incident_lock: asyncio.Lock,
        progress_reporter: _ProgressReporter | None,
    ) -> None:
        self._name = name
        self._path = Path(path)
        self._warn_after = warn_after
        self._fail_after = fail_after
        self._check_seconds = max(check_interval.total_seconds(), 0.05)
        self._initial_grace = max(initial_grace, timedelta(0))
        self._started_at = started_at
        self._incidents = incidents
        self._incident_keys = incident_keys
        self._incident_lock = incident_lock
        self._progress_reporter = progress_reporter
        self._active_severity: DryRunStatus | None = None

    async def run(self) -> None:
        try:
            while True:
                await self._check()
                await asyncio.sleep(self._check_seconds)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            raise

    async def _check(self) -> None:
        now = datetime.now(tz=UTC)
        if now - self._started_at < self._initial_grace:
            return

        evaluation = self._evaluate(now)
        if evaluation is None:
            self._active_severity = None
            return

        severity, state, age, threshold_seconds, last_observed = evaluation
        current_rank = (
            _PROGRESS_SEVERITY_ORDER.get(self._active_severity, 0)
            if self._active_severity is not None
            else 0
        )
        new_rank = _PROGRESS_SEVERITY_ORDER.get(severity, 0)
        if new_rank <= current_rank:
            return

        age_delta = timedelta(seconds=age)
        threshold_delta = timedelta(seconds=threshold_seconds)
        if state == "missing":
            state_msg = "missing"
        else:
            state_msg = "stale"
        message = (
            f"{self._name} {state_msg} for {humanise_timedelta(age_delta)}; "
            f"exceeds {severity.value.upper()} threshold "
            f"{humanise_timedelta(threshold_delta)}"
        )
        metadata: dict[str, Any] = {
            "path": self._path.as_posix(),
            "state": state,
            "age_seconds": age,
            "threshold_seconds": threshold_seconds,
        }
        if last_observed is not None:
            metadata["last_observed_at"] = last_observed.astimezone(UTC).isoformat()

        incident = HarnessIncident(
            severity=severity,
            occurred_at=now,
            message=message,
            metadata=metadata,
        )

        recorded = await _append_incident(
            incident=incident,
            key=("evidence", self._name, severity.value),
            incidents=self._incidents,
            incident_keys=self._incident_keys,
            lock=self._incident_lock,
            progress_reporter=self._progress_reporter,
        )
        if recorded:
            self._active_severity = severity

    def _evaluate(
        self, now: datetime
    ) -> tuple[DryRunStatus, str, float, float, datetime | None] | None:
        warn_seconds = (
            self._warn_after.total_seconds() if self._warn_after is not None else None
        )
        fail_seconds = (
            self._fail_after.total_seconds() if self._fail_after is not None else None
        )

        try:
            stat = self._path.stat()
        except FileNotFoundError:
            stat = None
        except OSError:
            stat = None

        exists = stat is not None and stat.st_size > 0 if stat is not None else False
        if exists:
            last_observed = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            age_seconds = max((now - last_observed).total_seconds(), 0.0)
        else:
            last_observed = None
            age_seconds = max((now - self._started_at).total_seconds(), 0.0)

        severity: DryRunStatus | None = None
        threshold_seconds: float | None = None

        if fail_seconds is not None and age_seconds >= fail_seconds:
            severity = DryRunStatus.fail
            threshold_seconds = fail_seconds
        elif warn_seconds is not None and age_seconds >= warn_seconds:
            severity = DryRunStatus.warn
            threshold_seconds = warn_seconds

        if severity is None or threshold_seconds is None:
            return None

        state = "missing" if not exists else "stale"
        return severity, state, age_seconds, threshold_seconds, last_observed

    @property
    def name(self) -> str:
        return self._name

    @property
    def check_seconds(self) -> float:
        return self._check_seconds


def _default_evidence_interval(
    explicit: timedelta | None,
    warn: timedelta | None,
    fail: timedelta | None,
) -> timedelta:
    if explicit is not None:
        return explicit
    candidates = [value for value in (warn, fail) if value is not None]
    if not candidates:
        return timedelta(minutes=1)
    minimum = min(candidates)
    seconds = minimum.total_seconds()
    if seconds <= 0:
        return timedelta(seconds=1)
    return timedelta(seconds=max(seconds / 4.0, 0.1))


async def perform_final_dry_run(
    config: FinalDryRunConfig,
    *,
    task_supervisor: TaskSupervisor | None = None,
) -> FinalDryRunResult:
    """Execute the final dry run harness according to ``config``."""

    config.log_directory.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(tz=UTC)
    slug = started_at.strftime("%Y%m%dT%H%M%SZ")
    log_suffix = "jsonl.gz" if config.compress_logs else "jsonl"
    raw_suffix = "log.gz" if config.compress_logs else "log"
    log_path = config.log_directory / f"final_dry_run_{slug}.{log_suffix}"
    raw_log_path = config.log_directory / f"final_dry_run_{slug}.{raw_suffix}"

    incidents: list[HarnessIncident] = []
    incident_lock = asyncio.Lock()
    incident_keys: set[tuple[str, ...]] = set()

    log_lock = asyncio.Lock()
    sink = _LogSink(
        directory=config.log_directory,
        slug=slug,
        compress=config.compress_logs,
        rotate_interval=config.log_rotate_interval,
    )
    log_path = sink.structured_paths[0]
    raw_log_path = sink.raw_paths[0]

    stats = _LogStats()

    resource_metrics = _ResourceMonitorMetrics()
    resource_monitor: ResourceUsageMonitor | None = None
    resource_task: asyncio.Task[None] | None = None
    resource_enabled = False
    resource_reason: str | None = None
    resource_interval_seconds: float | None = None

    env = None
    if config.environment is not None:
        env = os.environ.copy()
        env.update(config.environment)

    try:
        process = await asyncio.create_subprocess_exec(
            *config.command,
            stdout=PIPE,
            stderr=PIPE,
            env=env,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        sink.close()
        raise RuntimeError(f"Failed to launch dry run command: {exc}") from exc

    resource_thresholds: Mapping[str, float | None] = {
        "cpu_percent": config.resource_max_cpu_percent,
        "memory_mb": config.resource_max_memory_mb,
        "memory_percent": config.resource_max_memory_percent,
    }

    if config.resource_sample_interval is None:
        resource_reason = "disabled_by_config"
    else:
        resource_interval_seconds = config.resource_sample_interval.total_seconds()
        resource_monitor = _create_resource_monitor(process.pid)
        if resource_monitor is None:
            resource_reason = "monitor_unavailable"
        else:
            resource_reason = None

    progress_reporter: _ProgressReporter | None = None
    progress_task: asyncio.Task[None] | None = None
    progress_path_value: Path | None = None
    if config.progress_interval is not None:
        progress_path_value = (
            config.progress_path
            or config.log_directory / f"final_dry_run_{slug}_progress.json"
        )
        interval_value = config.progress_interval
        assert interval_value is not None  # safeguard for type narrowing
        progress_reporter = _ProgressReporter(
            path=progress_path_value,
            config=config,
            stats=stats,
            started_at=started_at,
            interval=interval_value,
        )
        await progress_reporter.write(
            status="starting",
            now=started_at,
            phase="startup",
        )
    else:
        progress_path_value = config.progress_path

    gap_monitor: _LogGapMonitor | None = None
    if config.live_gap_alert is not None:
        gap_monitor = _LogGapMonitor(
            threshold=config.live_gap_alert,
            severity=config.live_gap_severity,
            incidents=incidents,
            incident_keys=incident_keys,
            incident_lock=incident_lock,
            progress_reporter=progress_reporter,
        )

    evidence_tasks: list[asyncio.Task[Any]] = []
    evidence_monitors: list[_EvidenceMonitor] = []
    if config.diary_path is not None and (
        config.diary_stale_warn is not None or config.diary_stale_fail is not None
    ):
        evidence_monitors.append(
            _EvidenceMonitor(
                name="Decision diary",
                path=config.diary_path,
                warn_after=config.diary_stale_warn,
                fail_after=config.diary_stale_fail,
                check_interval=_default_evidence_interval(
                    config.evidence_check_interval,
                    config.diary_stale_warn,
                    config.diary_stale_fail,
                ),
                initial_grace=config.evidence_initial_grace,
                started_at=started_at,
                incidents=incidents,
                incident_keys=incident_keys,
                incident_lock=incident_lock,
                progress_reporter=progress_reporter,
            )
        )

    if config.performance_path is not None and (
        config.performance_stale_warn is not None
        or config.performance_stale_fail is not None
    ):
        evidence_monitors.append(
            _EvidenceMonitor(
                name="Performance telemetry",
                path=config.performance_path,
                warn_after=config.performance_stale_warn,
                fail_after=config.performance_stale_fail,
                check_interval=_default_evidence_interval(
                    config.evidence_check_interval,
                    config.performance_stale_warn,
                    config.performance_stale_fail,
                ),
                initial_grace=config.evidence_initial_grace,
                started_at=started_at,
                incidents=incidents,
                incident_keys=incident_keys,
                incident_lock=incident_lock,
                progress_reporter=progress_reporter,
            )
        )

    async def _pump_stream(
        stream: asyncio.StreamReader | None,
        stream_name: str,
        progress: _ProgressReporter | None,
        monitor: _LogGapMonitor | None,
    ) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            observed_at = datetime.now(tz=UTC)
            record = _normalise_log_line(text, stream_name, observed_at)
            await _maybe_record_log_incident(
                record,
                stream_name,
                observed_at,
                config,
                incidents,
                incident_keys,
                incident_lock,
                progress,
            )
            json_line = json.dumps(record, separators=(",", ":"))
            async with log_lock:
                sink.write(
                    structured_line=json_line,
                    raw_line=text,
                    observed_at=observed_at,
                )
            await stats.record(stream_name, record["level"], observed_at)
            if monitor is not None:
                await monitor.note(observed_at)

    owns_supervisor = False
    supervisor = task_supervisor
    if supervisor is None:
        supervisor = TaskSupervisor(namespace="operations.final_dry_run")
        owns_supervisor = True

    if resource_monitor is not None and config.resource_sample_interval is not None:
        resource_enabled = True
        resource_task = supervisor.create(
            _run_resource_monitor(
                monitor=resource_monitor,
                process=process,
                interval=config.resource_sample_interval,
                metrics=resource_metrics,
                max_cpu_percent=config.resource_max_cpu_percent,
                max_memory_mb=config.resource_max_memory_mb,
                max_memory_percent=config.resource_max_memory_percent,
                severity=config.resource_violation_severity,
                incidents=incidents,
                incident_keys=incident_keys,
                incident_lock=incident_lock,
                progress_reporter=progress_reporter,
            ),
            name="dry-run-resource-monitor",
            metadata={
                "component": "operations.final_dry_run.resource_monitor",
                "interval_seconds": config.resource_sample_interval.total_seconds(),
                "severity": config.resource_violation_severity.value,
                "threshold_cpu_percent": config.resource_max_cpu_percent,
                "threshold_memory_mb": config.resource_max_memory_mb,
                "threshold_memory_percent": config.resource_max_memory_percent,
            },
        )

    pump_tasks = [
        supervisor.create(
            _pump_stream(process.stdout, "stdout", progress_reporter, gap_monitor),
            name="dry-run-stdout",
            metadata={
                "component": "operations.final_dry_run.pump",
                "stream": "stdout",
            },
        ),
        supervisor.create(
            _pump_stream(process.stderr, "stderr", progress_reporter, gap_monitor),
            name="dry-run-stderr",
            metadata={
                "component": "operations.final_dry_run.pump",
                "stream": "stderr",
            },
        ),
    ]

    for monitor in evidence_monitors:
        evidence_tasks.append(
            supervisor.create(
                monitor.run(),
                name=f"dry-run-evidence-{monitor.name.lower().replace(' ', '-')}",
                metadata={
                    "component": "operations.final_dry_run.evidence_monitor",
                    "evidence": monitor.name,
                    "check_interval_seconds": monitor.check_seconds,
                },
            )
        )

    gap_monitor_task: asyncio.Task[None] | None = None
    if gap_monitor is not None:
        gap_monitor_task = supervisor.create(
            gap_monitor.run(),
            name="dry-run-gap-monitor",
            metadata={
                "component": "operations.final_dry_run.gap_monitor",
                "threshold_seconds": config.live_gap_alert.total_seconds(),
                "severity": config.live_gap_severity.value,
            },
        )

    if progress_reporter is not None and config.progress_interval is not None:
        progress_task = supervisor.create(
            progress_reporter.run(),
            name="dry-run-progress-reporter",
            metadata={
                "component": "operations.final_dry_run.progress",
                "interval_seconds": config.progress_interval.total_seconds(),
            },
        )
        await progress_reporter.write(status="running", phase="running")

    duration_seconds = config.duration.total_seconds()
    timeout_task = supervisor.create(
        asyncio.sleep(duration_seconds),
        name="dry-run-duration-timeout",
        metadata={
            "component": "operations.final_dry_run.timeout",
            "duration_seconds": duration_seconds,
        },
    )
    wait_task = supervisor.create(
        process.wait(),
        name="dry-run-process-wait",
        metadata={"component": "operations.final_dry_run.process"},
    )

    timed_out = False
    exit_code: int | None = None

    done, _ = await asyncio.wait(
        {timeout_task, wait_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    if wait_task in done:
        exit_code = wait_task.result()
    else:
        timed_out = True
        exit_code = await _terminate_process(
            process,
            config.shutdown_grace.total_seconds(),
        )

    if not wait_task.done():
        exit_code = await wait_task

    timeout_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await timeout_task

    await asyncio.gather(*pump_tasks, return_exceptions=True)

    for task in evidence_tasks:
        task.cancel()
    for task in evidence_tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task

    if progress_task is not None:
        progress_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await progress_task

    if gap_monitor_task is not None:
        gap_monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gap_monitor_task

    if resource_task is not None:
        resource_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await resource_task

    if owns_supervisor:
        await supervisor.cancel_all()

    ended_at = datetime.now(tz=UTC)
    actual_duration = ended_at - started_at

    sink.close()
    structured_paths = sink.structured_paths
    raw_paths = sink.raw_paths

    if exit_code not in (0, None):
        incidents.append(
            HarnessIncident(
                severity=DryRunStatus.fail,
                occurred_at=ended_at,
                message="Dry run process exited with non-zero status",
                metadata={"exit_code": exit_code},
            )
        )

    required_duration = config.required_duration or config.duration
    if actual_duration + _MIN_DURATION_TOLERANCE < required_duration:
        incidents.append(
            HarnessIncident(
                severity=DryRunStatus.fail,
                occurred_at=ended_at,
                message="Dry run completed before the required duration",
                metadata={
                    "required_seconds": required_duration.total_seconds(),
                    "actual_seconds": actual_duration.total_seconds(),
                },
            )
        )

    stats_snapshot = await stats.snapshot()

    resource_metadata = resource_metrics.as_metadata(
        enabled=resource_enabled,
        reason=resource_reason,
        interval_seconds=resource_interval_seconds,
        severity=(
            config.resource_violation_severity if resource_enabled else None
        ),
        thresholds=resource_thresholds,
    )

    metadata: dict[str, Any] = {
        "command": list(config.command),
        "started_at": started_at.astimezone(UTC).isoformat(),
        "ended_at": ended_at.astimezone(UTC).isoformat(),
        "exit_code": exit_code,
        "target_duration_seconds": config.duration.total_seconds(),
        "actual_duration_seconds": actual_duration.total_seconds(),
        "timed_out": timed_out,
        "log_path": str(log_path),
        "raw_log_path": str(raw_log_path),
        "harness_incidents": [incident.as_dict() for incident in incidents],
        "log_line_counts": dict(stats_snapshot["lines"]),
        "log_level_counts": dict(stats_snapshot["levels"]),
        "log_total_lines": stats_snapshot.get("total_lines", 0),
    }
    metadata["structured_log_paths"] = [path.as_posix() for path in structured_paths]
    metadata["raw_log_paths"] = [path.as_posix() for path in raw_paths]
    if progress_path_value is not None:
        metadata["progress_path"] = str(progress_path_value)
    metadata["resource_monitor"] = resource_metadata
    metadata.update(config.metadata)

    summary = evaluate_dry_run(
        log_paths=list(structured_paths),
        diary_path=config.diary_path,
        performance_path=config.performance_path,
        metadata=metadata,
        log_gap_warn=config.log_gap_warn,
        log_gap_fail=config.log_gap_fail,
        minimum_run_duration=required_duration,
        minimum_uptime_ratio=config.minimum_uptime_ratio,
    )

    sign_off = assess_sign_off_readiness(
        summary,
        minimum_duration=required_duration,
        minimum_uptime_ratio=config.minimum_uptime_ratio,
        require_diary=config.require_diary_evidence,
        require_performance=config.require_performance_evidence,
        allow_warnings=config.allow_warnings,
        minimum_sharpe_ratio=config.minimum_sharpe_ratio,
    )

    result = FinalDryRunResult(
        config=config,
        started_at=started_at,
        ended_at=ended_at,
        exit_code=exit_code,
        summary=summary,
        sign_off=sign_off,
        log_paths=structured_paths,
        raw_log_paths=raw_paths,
        progress_path=progress_path_value,
        incidents=tuple(incidents),
    )

    if progress_reporter is not None:
        await progress_reporter.write(
            status=result.status.value,
            now=ended_at,
            phase="complete",
            exit_code=exit_code,
            incidents=result.incidents,
            summary=result.summary,
            sign_off=result.sign_off,
        )

    return result
def run_final_dry_run(config: FinalDryRunConfig) -> FinalDryRunResult:
    """Synchronous helper wrapping :func:`perform_final_dry_run`."""

    return asyncio.run(perform_final_dry_run(config))


async def _terminate_process(process: Process, grace_seconds: float) -> int | None:
    if process.returncode is not None:
        return process.returncode

    signalled = False
    if hasattr(process, "send_signal"):
        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                process.send_signal(sig)
            except (ProcessLookupError, ValueError):
                return process.returncode
            else:
                signalled = True
                break

    if not signalled:
        try:
            process.terminate()
        except ProcessLookupError:
            return process.returncode
    wait_timeout = max(grace_seconds, 0.0)
    if wait_timeout:
        try:
            return await asyncio.wait_for(process.wait(), timeout=wait_timeout)
        except asyncio.TimeoutError:
            pass

    try:
        process.kill()
    except ProcessLookupError:
        return process.returncode
    await process.wait()
    return process.returncode


def _normalise_log_line(text: str, stream: str, observed_at: datetime) -> Mapping[str, Any]:
    level = "error" if stream == "stderr" else "info"
    event: str | None = None
    message = text
    timestamp_value: Any = observed_at.astimezone(UTC).isoformat()
    payload: dict[str, Any] = {
        "stream": stream,
        "ingested_at": observed_at.astimezone(UTC).isoformat(),
    }

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, Mapping):
        payload["structured"] = parsed
        level_raw = parsed.get("level") or parsed.get("severity")
        if isinstance(level_raw, str):
            level = level_raw.lower()
        elif isinstance(level_raw, (int, float)):
            level = str(level_raw)
        event_raw = parsed.get("event")
        if event_raw is not None:
            event = str(event_raw)
        message_raw = (
            parsed.get("message")
            or parsed.get("msg")
            or parsed.get("event")
        )
        if message_raw is not None:
            message = str(message_raw)
        timestamp_candidate = (
            parsed.get("timestamp")
            or parsed.get("ts")
            or parsed.get("@timestamp")
            or parsed.get("time")
        )
        if isinstance(timestamp_candidate, (str, int, float)):
            timestamp_value = timestamp_candidate
    else:
        payload["raw"] = text

    record: dict[str, Any] = {
        "timestamp": timestamp_value,
        "level": level,
        "event": event,
        "message": message,
        "stream": stream,
        "payload": payload,
    }
    return record


def _status_to_severity(value: str) -> DryRunStatus | None:
    try:
        return DryRunStatus(value)
    except ValueError:
        return None


async def _maybe_record_log_incident(
    record: Mapping[str, Any],
    stream: str,
    observed_at: datetime,
    config: FinalDryRunConfig,
    incidents: list[HarnessIncident],
    incident_keys: set[tuple[str, str, str, str]],
    lock: asyncio.Lock,
    progress_reporter: _ProgressReporter | None,
) -> None:
    if not config.monitor_log_levels:
        return

    level = str(record.get("level") or "").lower()

    if _looks_like_stack_trace(record):
        stack_excerpt = _derive_stack_trace_excerpt(record)
        incident = HarnessIncident(
            severity=DryRunStatus.fail,
            occurred_at=observed_at,
            message=_format_stack_trace_incident_message(stream, stack_excerpt),
            metadata={
                "level": level or None,
                "stream": stream,
                "detected_via": "stack_trace",
            },
        )
        await _append_incident(
            incident=incident,
            key=("stack_trace", stream, stack_excerpt),
            incidents=incidents,
            incident_keys=incident_keys,
            lock=lock,
            progress_reporter=progress_reporter,
        )
        return

    severity: DryRunStatus | None = None
    if level in _LOG_FAIL_LEVELS or (not level and stream == "stderr"):
        severity = DryRunStatus.fail
    elif level in _LOG_WARN_LEVELS:
        severity = DryRunStatus.warn
    elif stream == "stderr":
        severity = DryRunStatus.fail

    if severity is None:
        return

    message = _derive_incident_message(record)
    event = record.get("event")
    incident = HarnessIncident(
        severity=severity,
        occurred_at=observed_at,
        message=_format_incident_message(level, stream, message, event),
        metadata={
            "level": level or None,
            "stream": stream,
            "event": event,
        },
    )

    await _append_incident(
        incident=incident,
        key=("log_level", severity.value, stream, level, message),
        incidents=incidents,
        incident_keys=incident_keys,
        lock=lock,
        progress_reporter=progress_reporter,
    )


def _derive_incident_message(record: Mapping[str, Any]) -> str:
    message = record.get("message")
    if isinstance(message, str) and message.strip():
        return message
    payload = record.get("payload")
    if isinstance(payload, Mapping):
        structured = payload.get("structured")
        if isinstance(structured, Mapping):
            structured_message = structured.get("message")
            if isinstance(structured_message, str) and structured_message.strip():
                return structured_message
            structured_event = structured.get("event")
            if isinstance(structured_event, str) and structured_event.strip():
                return structured_event
        raw = payload.get("raw")
        if isinstance(raw, str) and raw.strip():
            return raw
    return "runtime emitted log entry"


def _format_incident_message(
    level: str,
    stream: str,
    message: str,
    event: Any,
) -> str:
    level_token = level.upper() if level else stream.upper()
    suffix = f" — event: {event}" if event else ""
    return f"Runtime emitted {level_token} log on {stream}: {message}{suffix}"


def _looks_like_stack_trace(record: Mapping[str, Any]) -> bool:
    for candidate in _stack_trace_candidates(record):
        text = candidate.lower()
        if "traceback (most recent call last)" in text:
            return True
    return False


def _derive_stack_trace_excerpt(record: Mapping[str, Any]) -> str:
    for candidate in _stack_trace_candidates(record):
        lower = candidate.lower()
        marker_index = lower.find("traceback (most recent call last)")
        if marker_index >= 0:
            snippet = candidate[marker_index:].splitlines()[0].strip()
            if snippet:
                return snippet
    return _derive_incident_message(record)


def _format_stack_trace_incident_message(stream: str, excerpt: str) -> str:
    return f"Runtime emitted stack trace on {stream}: {excerpt}"


def _stack_trace_candidates(record: Mapping[str, Any]) -> tuple[str, ...]:
    candidates: list[str] = []

    message = record.get("message")
    if isinstance(message, str) and message.strip():
        candidates.append(message)

    payload = record.get("payload")
    if isinstance(payload, Mapping):
        raw = payload.get("raw")
        if isinstance(raw, str) and raw.strip():
            candidates.append(raw)
        structured = payload.get("structured")
        if isinstance(structured, Mapping):
            for key in (
                "message",
                "event",
                "exc_info",
                "exc_text",
                "stack",
                "stacktrace",
                "traceback",
            ):
                value = structured.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value)

    return tuple(candidates)
