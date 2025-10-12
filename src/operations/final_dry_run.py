from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from asyncio.subprocess import PIPE, Process
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from src.operations.dry_run_audit import (
    DryRunSignOffReport,
    DryRunStatus,
    DryRunSummary,
    assess_sign_off_readiness,
    evaluate_dry_run,
)
from src.runtime.task_supervisor import TaskSupervisor

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
    log_path: Path
    raw_log_path: Path
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


async def perform_final_dry_run(
    config: FinalDryRunConfig,
    *,
    task_supervisor: TaskSupervisor | None = None,
) -> FinalDryRunResult:
    """Execute the final dry run harness according to ``config``."""

    config.log_directory.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(tz=UTC)
    slug = started_at.strftime("%Y%m%dT%H%M%SZ")
    log_path = config.log_directory / f"final_dry_run_{slug}.jsonl"
    raw_log_path = config.log_directory / f"final_dry_run_{slug}.log"

    incidents: list[HarnessIncident] = []
    incident_lock = asyncio.Lock()
    incident_keys: set[tuple[str, str, str, str]] = set()

    log_lock = asyncio.Lock()
    log_file = log_path.open("w", encoding="utf-8")
    raw_file = raw_log_path.open("w", encoding="utf-8")

    stats = _LogStats()

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
        log_file.close()
        raw_file.close()
        raise RuntimeError(f"Failed to launch dry run command: {exc}") from exc

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

    async def _pump_stream(
        stream: asyncio.StreamReader | None,
        stream_name: str,
        progress: _ProgressReporter | None,
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
                log_file.write(json_line + "\n")
                raw_file.write(text + "\n")
                log_file.flush()
                raw_file.flush()
            await stats.record(stream_name, record["level"], observed_at)

    owns_supervisor = False
    supervisor = task_supervisor
    if supervisor is None:
        supervisor = TaskSupervisor(namespace="operations.final_dry_run")
        owns_supervisor = True

    pump_tasks = [
        supervisor.create(
            _pump_stream(process.stdout, "stdout", progress_reporter),
            name="dry-run-stdout",
            metadata={
                "component": "operations.final_dry_run.pump",
                "stream": "stdout",
            },
        ),
        supervisor.create(
            _pump_stream(process.stderr, "stderr", progress_reporter),
            name="dry-run-stderr",
            metadata={
                "component": "operations.final_dry_run.pump",
                "stream": "stderr",
            },
        ),
    ]

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

    if progress_task is not None:
        progress_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await progress_task

    if owns_supervisor:
        await supervisor.cancel_all()

    ended_at = datetime.now(tz=UTC)
    actual_duration = ended_at - started_at

    log_file.flush()
    raw_file.flush()
    log_file.close()
    raw_file.close()

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
    if progress_path_value is not None:
        metadata["progress_path"] = str(progress_path_value)
    metadata.update(config.metadata)

    summary = evaluate_dry_run(
        log_paths=[log_path],
        diary_path=config.diary_path,
        performance_path=config.performance_path,
        metadata=metadata,
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
        log_path=log_path,
        raw_log_path=raw_log_path,
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
    key = (severity.value, stream, level, message)

    async with lock:
        if key in incident_keys:
            return
        incident_keys.add(key)
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
        incidents.append(incident)
        snapshot = tuple(incidents)

    if progress_reporter is not None:
        await progress_reporter.record_incident(
            incident=incident,
            incidents=snapshot,
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
