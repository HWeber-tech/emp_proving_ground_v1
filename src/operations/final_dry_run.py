from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from asyncio.subprocess import PIPE, Process
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


@dataclass(slots=True, frozen=True)
class FinalDryRunConfig:
    """Configuration for executing a final dry run harness."""

    command: Sequence[str]
    duration: timedelta
    log_directory: Path
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
    incidents: tuple[HarnessIncident, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> timedelta:
        return self.ended_at - self.started_at

    @property
    def status(self) -> DryRunStatus:
        statuses: list[DryRunStatus] = [self.summary.status]
        if self.sign_off is not None:
            statuses.append(self.sign_off.status)
        if any(status is DryRunStatus.fail for status in statuses):
            return DryRunStatus.fail
        if any(status is DryRunStatus.warn for status in statuses):
            return DryRunStatus.warn
        return DryRunStatus.pass_


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

    log_lock = asyncio.Lock()
    log_file = log_path.open("w", encoding="utf-8")
    raw_file = raw_log_path.open("w", encoding="utf-8")

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

    async def _pump_stream(stream: asyncio.StreamReader | None, stream_name: str) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            observed_at = datetime.now(tz=UTC)
            record = _normalise_log_line(text, stream_name, observed_at)
            json_line = json.dumps(record, separators=(",", ":"))
            async with log_lock:
                log_file.write(json_line + "\n")
                raw_file.write(text + "\n")
                log_file.flush()
                raw_file.flush()

    owns_supervisor = False
    supervisor = task_supervisor
    if supervisor is None:
        supervisor = TaskSupervisor(namespace="operations.final_dry_run")
        owns_supervisor = True

    pump_tasks = [
        supervisor.create(
            _pump_stream(process.stdout, "stdout"),
            name="dry-run-stdout",
            metadata={
                "component": "operations.final_dry_run.pump",
                "stream": "stdout",
            },
        ),
        supervisor.create(
            _pump_stream(process.stderr, "stderr"),
            name="dry-run-stderr",
            metadata={
                "component": "operations.final_dry_run.pump",
                "stream": "stderr",
            },
        ),
    ]

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
        exit_code = await _terminate_process(process, config.shutdown_grace.total_seconds())

    if not wait_task.done():
        exit_code = await wait_task

    timeout_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await timeout_task

    await asyncio.gather(*pump_tasks, return_exceptions=True)

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
    }
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

    return FinalDryRunResult(
        config=config,
        started_at=started_at,
        ended_at=ended_at,
        exit_code=exit_code,
        summary=summary,
        sign_off=sign_off,
        log_path=log_path,
        raw_log_path=raw_log_path,
        incidents=tuple(incidents),
    )


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
