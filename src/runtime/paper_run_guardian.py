"""24/7 paper trading guardian utilities.

This module provides orchestration helpers that keep the paper trading runtime
under continuous observation.  It extends the existing paper trading simulation
support with long-horizon monitoring so the roadmap's "24/7 paper run" acceptance
criterion becomes operationally testable.

The guardian tracks latency percentiles, invariant breaches, and process memory
usage while exposing structured summaries that can be persisted for governance
sign-off.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import sys
import tracemalloc
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - platform dependent import
    import resource  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Windows fallback
    resource = None  # type: ignore[assignment]

from src.governance.system_config import SystemConfig
from src.runtime.paper_simulation import (
    PaperTradingSimulationProgress,
    PaperTradingSimulationReport,
    run_paper_trading_simulation,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryTracker",
    "PaperRunConfig",
    "PaperRunMonitor",
    "PaperRunSample",
    "PaperRunStatus",
    "PaperRunSummary",
    "persist_summary",
    "run_guarded_paper_session",
]


DEFAULT_MINIMUM_RUNTIME_SECONDS = 7 * 24 * 60 * 60


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate):
        return None
    return candidate


class PaperRunStatus(Enum):
    """Lifecycle states for the paper run guardian."""

    PREPARING = "preparing"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"
    COMPLETED = "completed"


@dataclass(slots=True)
class PaperRunConfig:
    """Run configuration for the paper guardian."""

    duration_seconds: float | None = None
    progress_interval: float = 60.0
    latency_p99_threshold: float | None = None
    memory_growth_threshold_mb: float | None = None
    allow_invariant_errors: bool = False
    report_path: Path | None = None
    sample_history: int = 2048
    min_orders: int = 0
    minimum_runtime_seconds: float = DEFAULT_MINIMUM_RUNTIME_SECONDS

    def __post_init__(self) -> None:
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            self.duration_seconds = None
        self.progress_interval = max(1.0, float(self.progress_interval or 60.0))
        self.sample_history = max(32, int(self.sample_history or 2048))
        self.min_orders = max(0, int(self.min_orders or 0))
        minimum_runtime = float(self.minimum_runtime_seconds or 0.0)
        self.minimum_runtime_seconds = max(0.0, minimum_runtime)


@dataclass(slots=True)
class PaperRunSample:
    """Atom of recorded progress during a paper run."""

    timestamp: datetime
    runtime_seconds: float
    orders_observed: int
    errors_observed: int
    decisions_observed: int
    memory_mb: float | None
    latency_metrics: Mapping[str, float | None]
    failover: Mapping[str, Any] | None = None
    last_error: Mapping[str, Any] | None = None


@dataclass(slots=True)
class PaperRunSummary:
    """Structured summary of a guarded paper trading session."""

    status: PaperRunStatus
    started_at: datetime | None
    finished_at: datetime | None
    runtime_seconds: float | None
    alerts: Sequence[str]
    stop_reasons: Sequence[str]
    invariant_breaches: Sequence[Mapping[str, Any]]
    metrics: Mapping[str, Any]
    samples: Sequence[PaperRunSample]
    report: Mapping[str, Any] | None

    def to_dict(self) -> Mapping[str, Any]:
        def _serialise_sample(sample: PaperRunSample) -> Mapping[str, Any]:
            payload: MutableMapping[str, Any] = {
                "timestamp": sample.timestamp.isoformat(),
                "runtime_seconds": sample.runtime_seconds,
                "orders_observed": sample.orders_observed,
                "errors_observed": sample.errors_observed,
                "decisions_observed": sample.decisions_observed,
                "memory_mb": sample.memory_mb,
                "latency_metrics": dict(sample.latency_metrics),
            }
            if sample.failover:
                payload["failover"] = dict(sample.failover)
            if sample.last_error:
                payload["last_error"] = dict(sample.last_error)
            return payload

        summary_payload: MutableMapping[str, Any] = {
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "runtime_seconds": self.runtime_seconds,
            "alerts": list(self.alerts),
            "stop_reasons": list(self.stop_reasons),
            "invariant_breaches": [dict(breach) for breach in self.invariant_breaches],
            "metrics": dict(self.metrics),
            "samples": [_serialise_sample(sample) for sample in self.samples],
        }
        if self.report is not None:
            summary_payload["report"] = dict(self.report)
        return summary_payload


class MemoryTracker:
    """Track process memory usage across a long-running session."""

    def __init__(
        self,
        sampler: Callable[[], float | None] | None = None,
    ) -> None:
        self._sampler = sampler or _default_memory_sampler
        self.baseline_mb: float | None = None
        self.max_mb: float | None = None
        self.readings: list[tuple[datetime, float | None]] = []

    def sample(self, timestamp: datetime | None = None) -> float | None:
        timestamp = timestamp or _utc_now()
        value = self._sampler()
        if value is None:
            logger.debug("Memory sampler returned None; skipping reading")
            self.readings.append((timestamp, None))
            return None
        if self.baseline_mb is None:
            self.baseline_mb = value
        if self.max_mb is None or value > self.max_mb:
            self.max_mb = value
        self.readings.append((timestamp, value))
        return value

    def growth_mb(self) -> float | None:
        if self.baseline_mb is None or self.max_mb is None:
            return None
        return max(0.0, self.max_mb - self.baseline_mb)


def _default_memory_sampler() -> float | None:
    try:  # pragma: no cover - platform specific branch
        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore[arg-type]
            if sys.platform == "darwin":
                return usage / (1024.0 * 1024.0)
            return usage / 1024.0
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("resource.getrusage failed; falling back to tracemalloc", exc_info=True)
    try:
        if not tracemalloc.is_tracing():  # pragma: no cover - tracing global state
            tracemalloc.start()
        current, _peak = tracemalloc.get_traced_memory()
        return current / (1024.0 * 1024.0)
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("tracemalloc sampling failed", exc_info=True)
    return None


class PaperRunMonitor:
    """Capture progress updates and enforce guardian policies."""

    def __init__(
        self,
        run_config: PaperRunConfig,
        *,
        logger: logging.Logger | None = None,
        memory_tracker: MemoryTracker | None = None,
    ) -> None:
        self.config = run_config
        self._logger = logger or logging.getLogger(f"{__name__}.monitor")
        self.memory_tracker = memory_tracker or MemoryTracker()
        self.samples: deque[PaperRunSample] = deque(maxlen=self.config.sample_history)
        self.alerts: list[str] = []
        self.stop_reasons: list[str] = []
        self.invariant_breaches: list[Mapping[str, Any]] = []
        self.error_events: list[Mapping[str, Any]] = []
        self.status: PaperRunStatus = PaperRunStatus.PREPARING
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self._should_stop = False
        self._latency_p99_max: float | None = None
        self._latency_avg_max: float | None = None
        self._last_progress: PaperTradingSimulationProgress | None = None
        self._seen_error_signatures: set[tuple[Any, ...]] = set()

    def record_progress(self, snapshot: PaperTradingSimulationProgress) -> None:
        """Record telemetry emitted by the simulation loop."""

        self._last_progress = snapshot
        if self.started_at is None:
            self.started_at = snapshot.timestamp
            self.status = PaperRunStatus.RUNNING

        memory_mb = self.memory_tracker.sample(snapshot.timestamp)
        metrics = snapshot.paper_metrics or {}
        latency_metrics = {
            "avg_latency_s": _coerce_float(metrics.get("avg_latency_s")),
            "last_latency_s": _coerce_float(metrics.get("last_latency_s")),
            "p50_latency_s": _coerce_float(metrics.get("p50_latency_s")),
            "p90_latency_s": _coerce_float(metrics.get("p90_latency_s")),
            "p99_latency_s": _coerce_float(metrics.get("p99_latency_s")),
            "latency_samples": int(metrics.get("latency_samples", 0) or 0),
        }

        p99_latency = latency_metrics.get("p99_latency_s")
        if p99_latency is not None:
            if self._latency_p99_max is None or p99_latency > self._latency_p99_max:
                self._latency_p99_max = p99_latency
            threshold = self.config.latency_p99_threshold
            if threshold is not None and p99_latency > threshold:
                self._register_alert(
                    f"Latency p99 exceeded threshold: {p99_latency:.4f}s > {threshold:.4f}s"
                )
                self._update_status(PaperRunStatus.DEGRADED)

        avg_latency = latency_metrics.get("avg_latency_s")
        if avg_latency is not None:
            if self._latency_avg_max is None or avg_latency > self._latency_avg_max:
                self._latency_avg_max = avg_latency

        failover_snapshot = snapshot.failover
        if isinstance(failover_snapshot, Mapping) and failover_snapshot.get("active"):
            self._register_alert("Paper broker failover active")
            self._update_status(PaperRunStatus.DEGRADED)

        last_error = snapshot.last_error
        if isinstance(last_error, Mapping) and last_error:
            self._handle_error(last_error)

        sample = PaperRunSample(
            timestamp=snapshot.timestamp,
            runtime_seconds=snapshot.runtime_seconds,
            orders_observed=snapshot.orders_observed,
            errors_observed=snapshot.errors_observed,
            decisions_observed=snapshot.decisions_observed,
            memory_mb=memory_mb,
            latency_metrics=latency_metrics,
            failover=dict(failover_snapshot) if isinstance(failover_snapshot, Mapping) else None,
            last_error=dict(last_error) if isinstance(last_error, Mapping) else None,
        )
        self.samples.append(sample)

    def request_stop(self, reason: str) -> None:
        """Signal that the guardian should halt the run."""

        if reason not in self.stop_reasons:
            self.stop_reasons.append(reason)
        self._should_stop = True
        if self.status not in (PaperRunStatus.FAILED, PaperRunStatus.DEGRADED):
            self.status = PaperRunStatus.STOPPED

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    def finalise(self, report: PaperTradingSimulationReport) -> PaperRunSummary:
        """Produce a summary once the simulation concludes."""

        self.finished_at = _utc_now()
        memory_growth = self.memory_tracker.growth_mb()
        observed_runtime = float(report.runtime_seconds or 0.0)
        if observed_runtime <= 0 and self.started_at and self.finished_at:
            observed_runtime = max(
                0.0, (self.finished_at - self.started_at).total_seconds()
            )
        minimum_runtime = self.config.minimum_runtime_seconds
        meets_minimum_runtime = True
        runtime_shortfall: float | None = None
        if (
            memory_growth is not None
            and self.config.memory_growth_threshold_mb is not None
            and memory_growth > self.config.memory_growth_threshold_mb
        ):
            self._register_alert(
                (
                    "Memory growth exceeded threshold: "
                    f"{memory_growth:.2f}MB > {self.config.memory_growth_threshold_mb:.2f}MB"
                )
            )
            self._update_status(PaperRunStatus.DEGRADED)

        if minimum_runtime > 0.0 and observed_runtime < minimum_runtime:
            meets_minimum_runtime = False
            runtime_shortfall = max(0.0, minimum_runtime - observed_runtime)
            self._register_alert(
                (
                    "Paper run ended before minimum duration: "
                    f"{observed_runtime:.2f}s < {minimum_runtime:.2f}s"
                )
            )
            self._update_status(PaperRunStatus.DEGRADED)

        if self.status not in (PaperRunStatus.FAILED, PaperRunStatus.DEGRADED, PaperRunStatus.STOPPED):
            self.status = PaperRunStatus.COMPLETED

        metrics: MutableMapping[str, Any] = {
            "runtime_seconds": observed_runtime,
            "decisions": report.decisions,
            "diary_entries": report.diary_entries,
            "orders": len(report.orders),
            "errors": len(report.errors),
            "max_memory_mb": self.memory_tracker.max_mb,
            "memory_growth_mb": memory_growth,
            "latency_p99_s": self._latency_p99_max,
            "latency_avg_peak_s": self._latency_avg_max,
        }
        if minimum_runtime > 0.0:
            metrics["minimum_runtime_seconds"] = minimum_runtime
            metrics["meets_minimum_runtime"] = meets_minimum_runtime
            if runtime_shortfall is not None:
                metrics["runtime_shortfall_seconds"] = runtime_shortfall
        else:
            metrics["meets_minimum_runtime"] = True
        if report.paper_metrics is not None:
            metrics["paper_metrics"] = dict(report.paper_metrics)
        if report.execution_stats is not None:
            metrics["execution_stats"] = dict(report.execution_stats)
        if report.incident_response is not None:
            metrics["incident_response"] = dict(report.incident_response)

        summary = PaperRunSummary(
            status=self.status,
            started_at=self.started_at,
            finished_at=self.finished_at,
            runtime_seconds=observed_runtime,
            alerts=tuple(self.alerts),
            stop_reasons=tuple(self.stop_reasons),
            invariant_breaches=tuple(self.invariant_breaches),
            metrics=metrics,
            samples=tuple(self.samples),
            report=report.to_dict(),
        )
        return summary

    def _handle_error(self, payload: Mapping[str, Any]) -> None:
        signature = (
            payload.get("stage"),
            payload.get("message"),
            payload.get("exception"),
        )
        if signature in self._seen_error_signatures:
            return
        self._seen_error_signatures.add(signature)

        error_snapshot = dict(payload)
        self.error_events.append(error_snapshot)

        if self.config.allow_invariant_errors:
            return
        if _looks_like_invariant(error_snapshot):
            self.invariant_breaches.append(error_snapshot)
            self._register_alert("Risk invariant breach detected")
            self._update_status(PaperRunStatus.FAILED)
            self.request_stop("risk-invariant-breach")

    def _register_alert(self, message: str) -> None:
        if message not in self.alerts:
            self.alerts.append(message)
            self._logger.warning(message)

    def _update_status(self, state: PaperRunStatus) -> None:
        if self.status is PaperRunStatus.FAILED:
            return
        if state is PaperRunStatus.FAILED:
            self.status = PaperRunStatus.FAILED
        elif state is PaperRunStatus.DEGRADED and self.status not in (
            PaperRunStatus.FAILED,
            PaperRunStatus.STOPPED,
        ):
            self.status = PaperRunStatus.DEGRADED


def _looks_like_invariant(payload: Mapping[str, Any]) -> bool:
    stage = str(payload.get("stage", "")).lower()
    message = str(payload.get("message", "")).lower()
    if "invariant" in stage or "invariant" in message:
        return True
    if "risk" in stage and any(keyword in message for keyword in ("breach", "violation")):
        return True
    risk_error = payload.get("risk_error")
    if isinstance(risk_error, Mapping):
        inner_message = str(risk_error.get("message", "")).lower()
        if "invariant" in inner_message or "violation" in inner_message:
            return True
    return False


async def run_guarded_paper_session(
    system_config: SystemConfig,
    run_config: PaperRunConfig,
    *,
    logger: logging.Logger | None = None,
    memory_tracker: MemoryTracker | None = None,
) -> PaperRunSummary:
    """Execute a long-running paper session under guardian supervision."""

    monitor = PaperRunMonitor(
        run_config,
        logger=logger,
        memory_tracker=memory_tracker,
    )
    stop_event = asyncio.Event()

    async def _progress(snapshot: PaperTradingSimulationProgress) -> None:
        monitor.record_progress(snapshot)
        if monitor.should_stop:
            stop_event.set()

    timer_task: asyncio.Task[None] | None = None
    if run_config.duration_seconds is not None:
        async def _timer() -> None:
            await asyncio.sleep(run_config.duration_seconds)
            monitor.request_stop("duration-elapsed")
            stop_event.set()

        timer_task = asyncio.create_task(_timer())

    try:
        report = await run_paper_trading_simulation(
            system_config,
            min_orders=run_config.min_orders,
            max_runtime=None,
            stop_when_complete=False,
            stop_event=stop_event,
            progress_callback=_progress,
            progress_interval=run_config.progress_interval,
        )
    finally:
        if timer_task is not None:
            timer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await timer_task

    summary = monitor.finalise(report)

    destination = run_config.report_path
    if destination is not None:
        persist_summary(summary, destination)

    return summary


def persist_summary(summary: PaperRunSummary, destination: str | Path) -> None:
    """Persist the guardian summary to ``destination`` as JSON."""

    path = Path(destination)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - diagnostics only
        logger.debug("Failed to create directories for %s", path, exc_info=True)
    payload = summary.to_dict()
    json_payload = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(json_payload, encoding="utf-8")
