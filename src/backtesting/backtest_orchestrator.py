"""Parallel orchestration utilities for running strategy backtests."""

from __future__ import annotations

import asyncio
import logging
import numbers
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import fmean
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from src.runtime.task_supervisor import TaskSupervisor

__all__ = [
    "BacktestBatchResult",
    "BacktestBatchSummary",
    "BacktestOrchestrator",
    "BacktestRequest",
    "BacktestResult",
    "BacktestRunner",
]

logger = logging.getLogger(__name__)

ProgressCallback = Callable[["BacktestResult"], Awaitable[None] | None]


@dataclass(slots=True)
class BacktestRequest:
    """Description of a single backtest execution request."""

    request_id: str
    strategy_id: str
    dataset: str
    parameters: Mapping[str, Any] | None = None
    start: datetime | None = None
    end: datetime | None = None
    metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable payload for tracing and logging."""

        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "strategy_id": self.strategy_id,
            "dataset": self.dataset,
        }
        if self.parameters:
            payload["parameters"] = dict(self.parameters)
        if self.start:
            payload["start"] = self.start.isoformat()
        if self.end:
            payload["end"] = self.end.isoformat()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class BacktestResult:
    """Outcome of an executed backtest request."""

    request_id: str
    strategy_id: str
    status: str
    metrics: Mapping[str, float] | None = None
    artifacts: Mapping[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    def succeeded(self) -> bool:
        """Return ``True`` when the backtest completed without errors."""

        return self.status.lower() == "completed" and self.error is None

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation suitable for telemetry surfaces."""

        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "strategy_id": self.strategy_id,
            "status": self.status,
            "metrics": dict(self.metrics or {}),
        }
        if self.artifacts:
            payload["artifacts"] = dict(self.artifacts)
        if self.error is not None:
            payload["error"] = self.error
        if self.started_at:
            payload["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            payload["completed_at"] = self.completed_at.isoformat()
        if self.duration_seconds is not None:
            payload["duration_seconds"] = self.duration_seconds
        return payload


@dataclass(slots=True)
class BacktestBatchSummary:
    """Aggregated statistics for a batch of backtests."""

    total: int
    succeeded: int
    failed: int
    cancelled: int
    success_rate: float
    wall_time_seconds: float
    average_duration_seconds: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "success_rate": self.success_rate,
            "wall_time_seconds": self.wall_time_seconds,
            "average_duration_seconds": self.average_duration_seconds,
        }


@dataclass(slots=True)
class BacktestBatchResult:
    """Result container produced by :class:`BacktestOrchestrator`."""

    requests: tuple[BacktestRequest, ...]
    results: tuple[BacktestResult, ...]
    summary: BacktestBatchSummary
    started_at: datetime
    finished_at: datetime

    def result_for(self, request_id: str) -> BacktestResult | None:
        """Lookup a result by request identifier."""

        for result in self.results:
            if result.request_id == request_id:
                return result
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "summary": self.summary.as_dict(),
            "results": [result.as_dict() for result in self.results],
        }


@runtime_checkable
class BacktestRunner(Protocol):
    """Protocol implemented by backtest engines that can execute requests."""

    async def run_backtest(self, request: BacktestRequest) -> BacktestResult | Mapping[str, Any]:
        """Execute a backtest request and return the resulting metrics."""


class BacktestOrchestrator:
    """Coordinate the parallel execution of strategy backtests."""

    def __init__(
        self,
        runner: BacktestRunner,
        *,
        max_concurrency: int = 4,
        task_supervisor: TaskSupervisor | None = None,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than zero")
        if not isinstance(max_concurrency, int):
            raise TypeError("max_concurrency must be an integer")

        self._runner = runner
        self._max_concurrency = max_concurrency
        if task_supervisor is None:
            task_supervisor = TaskSupervisor(namespace="backtest-orchestrator")
            self._owns_supervisor = True
        else:
            self._owns_supervisor = False
        self._task_supervisor = task_supervisor
        self._active_tasks: MutableMapping[str, asyncio.Task[BacktestResult]] = {}

    async def run_backtests(
        self,
        requests: Iterable[BacktestRequest],
        *,
        progress_callback: ProgressCallback | None = None,
        cancel_event: asyncio.Event | Any | None = None,
    ) -> BacktestBatchResult:
        """Execute the supplied backtest requests with bounded concurrency."""

        request_sequence = tuple(requests)
        self._validate_requests(request_sequence)

        if not request_sequence:
            now = datetime.now(tz=UTC)
            summary = BacktestBatchSummary(
                total=0,
                succeeded=0,
                failed=0,
                cancelled=0,
                success_rate=0.0,
                wall_time_seconds=0.0,
                average_duration_seconds=0.0,
            )
            return BacktestBatchResult(
                requests=(),
                results=(),
                summary=summary,
                started_at=now,
                finished_at=now,
            )

        started_at = datetime.now(tz=UTC)
        semaphore = asyncio.Semaphore(self._max_concurrency)
        results_map: dict[str, BacktestResult] = {}
        lock = asyncio.Lock()

        async def _record_result(result: BacktestResult) -> None:
            async with lock:
                results_map[result.request_id] = result
            if progress_callback is not None:
                await self._invoke_progress(progress_callback, result)

        async def _execute(request: BacktestRequest) -> BacktestResult:
            async with semaphore:
                if self._is_cancelled(cancel_event):
                    result = self._build_cancelled_result(request)
                    await _record_result(result)
                    return result

                run_started = datetime.now(tz=UTC)
                try:
                    raw_result = await self._runner.run_backtest(request)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive logging path
                    logger.exception(
                        "Backtest %s failed with unhandled exception", request.request_id
                    )
                    run_completed = datetime.now(tz=UTC)
                    result = self._build_failure_result(request, run_started, run_completed, exc)
                else:
                    run_completed = datetime.now(tz=UTC)
                    try:
                        result = self._normalize_result(
                            request, raw_result, run_started, run_completed
                        )
                    except Exception as exc:  # pragma: no cover - misbehaving runner
                        logger.exception(
                            "Backtest %s returned an invalid payload", request.request_id
                        )
                        result = self._build_failure_result(
                            request, run_started, run_completed, exc
                        )

                if self._is_cancelled(cancel_event):
                    result = self._build_cancelled_result(request, result=result)
                await _record_result(result)
                return result

        tasks = [
            self._spawn_backtest_task(
                request,
                _execute(request),
            )
            for request in request_sequence
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            await self._drain_tasks(tasks)

        finished_at = datetime.now(tz=UTC)
        ordered_results = tuple(
            results_map.get(request.request_id)
            or self._build_cancelled_result(request)
            for request in request_sequence
        )
        summary = self._build_summary(ordered_results, started_at, finished_at)
        return BacktestBatchResult(
            requests=request_sequence,
            results=ordered_results,
            summary=summary,
            started_at=started_at,
            finished_at=finished_at,
        )

    async def _drain_tasks(self, tasks: Sequence[asyncio.Task[BacktestResult]]) -> None:
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for request_id in list(self._active_tasks):
            task = self._active_tasks.get(request_id)
            if task is None or task.done():
                self._active_tasks.pop(request_id, None)

    def _spawn_backtest_task(
        self,
        request: BacktestRequest,
        coro: Awaitable[BacktestResult],
    ) -> asyncio.Task[BacktestResult]:
        task_name = f"backtest-{request.request_id}"
        metadata = {
            "component": "backtesting.orchestrator",
            "strategy_id": request.strategy_id,
            "dataset": request.dataset,
        }
        task = self._task_supervisor.create(
            coro,
            name=task_name,
            metadata=metadata,
        )
        self._active_tasks[request.request_id] = task
        return task

    def _validate_requests(self, requests: Sequence[BacktestRequest]) -> None:
        seen: set[str] = set()
        for request in requests:
            if not request.request_id:
                raise ValueError("Each BacktestRequest must define request_id")
            if request.request_id in seen:
                raise ValueError(f"Duplicate request_id detected: {request.request_id}")
            seen.add(request.request_id)
            if not request.strategy_id:
                raise ValueError("BacktestRequest.strategy_id must be provided")
            if not request.dataset:
                raise ValueError("BacktestRequest.dataset must be provided")

    def _normalize_result(
        self,
        request: BacktestRequest,
        raw_result: BacktestResult | Mapping[str, Any],
        started_at: datetime,
        completed_at: datetime,
    ) -> BacktestResult:
        if isinstance(raw_result, BacktestResult):
            result = raw_result
        elif isinstance(raw_result, Mapping):
            metrics = self._extract_metrics(raw_result.get("metrics"))
            result = BacktestResult(
                request_id=str(raw_result.get("request_id", request.request_id)),
                strategy_id=str(raw_result.get("strategy_id", request.strategy_id)),
                status=str(raw_result.get("status", "completed")),
                metrics=metrics,
                artifacts=self._extract_artifacts(raw_result.get("artifacts")),
                error=self._coerce_optional_str(raw_result.get("error")),
                started_at=self._coerce_datetime(raw_result.get("started_at")),
                completed_at=self._coerce_datetime(raw_result.get("completed_at")),
                duration_seconds=self._coerce_optional_float(
                    raw_result.get("duration_seconds")
                ),
            )
        else:
            raise TypeError(
                "Backtest runner must return BacktestResult or mapping-compatible payload"
            )

        if not result.request_id:
            result.request_id = request.request_id
        if result.strategy_id != request.strategy_id:
            result.strategy_id = request.strategy_id
        if result.started_at is None:
            result.started_at = started_at
        if result.completed_at is None:
            result.completed_at = completed_at
        if result.duration_seconds is None and result.started_at and result.completed_at:
            result.duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()
        if result.metrics is None:
            result.metrics = {}
        else:
            result.metrics = self._extract_metrics(result.metrics)
        return result

    def _build_failure_result(
        self,
        request: BacktestRequest,
        started_at: datetime,
        completed_at: datetime,
        exc: Exception,
    ) -> BacktestResult:
        return BacktestResult(
            request_id=request.request_id,
            strategy_id=request.strategy_id,
            status="failed",
            metrics={},
            artifacts=None,
            error=str(exc),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
        )

    def _build_cancelled_result(
        self,
        request: BacktestRequest,
        *,
        result: BacktestResult | None = None,
    ) -> BacktestResult:
        if result is not None:
            return BacktestResult(
                request_id=result.request_id,
                strategy_id=result.strategy_id,
                status="cancelled",
                metrics=dict(result.metrics or {}),
                artifacts=result.artifacts,
                error=result.error or "cancelled",
                started_at=result.started_at,
                completed_at=result.completed_at,
                duration_seconds=result.duration_seconds,
            )
        now = datetime.now(tz=UTC)
        return BacktestResult(
            request_id=request.request_id,
            strategy_id=request.strategy_id,
            status="cancelled",
            metrics={},
            artifacts=None,
            error="cancelled",
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
        )

    def _extract_metrics(self, metrics: Any) -> Mapping[str, float]:
        if metrics is None:
            return {}
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping of string -> numeric values")
        normalised: dict[str, float] = {}
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise TypeError("metrics keys must be strings")
            if isinstance(value, numbers.Real):
                normalised[key] = float(value)
            else:
                raise TypeError("metrics values must be numeric")
        return normalised

    def _extract_artifacts(self, artifacts: Any) -> Mapping[str, Any] | None:
        if artifacts is None:
            return None
        if not isinstance(artifacts, Mapping):
            raise TypeError("artifacts must be a mapping when provided")
        return dict(artifacts)

    def _coerce_optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def _coerce_datetime(self, value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError as exc:  # pragma: no cover - defensive branch
                raise ValueError("Invalid datetime string") from exc
        raise TypeError("datetime values must be datetime or ISO-8601 string")

    def _coerce_optional_float(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, numbers.Real):
            return float(value)
        raise TypeError("duration_seconds must be numeric when provided")

    async def _invoke_progress(
        self, callback: ProgressCallback, result: BacktestResult
    ) -> None:
        try:
            maybe_coro = callback(result)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Progress callback failed for request %s", result.request_id
            )

    def _is_cancelled(self, cancel_event: asyncio.Event | Any | None) -> bool:
        if cancel_event is None:
            return False
        checker = getattr(cancel_event, "is_set", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:  # pragma: no cover - defensive
                logger.debug("cancel_event.is_set() raised an exception", exc_info=True)
                return False
        return False

    def _build_summary(
        self,
        results: Sequence[BacktestResult],
        started_at: datetime,
        finished_at: datetime,
    ) -> BacktestBatchSummary:
        total = len(results)
        succeeded = sum(1 for result in results if result.status == "completed" and result.error is None)
        cancelled = sum(1 for result in results if result.status == "cancelled")
        failed = total - succeeded - cancelled
        success_rate = float(succeeded / total) if total else 0.0
        wall_time_seconds = (finished_at - started_at).total_seconds()
        durations = [
            result.duration_seconds
            for result in results
            if result.duration_seconds is not None
        ]
        average_duration = float(fmean(durations)) if durations else 0.0
        return BacktestBatchSummary(
            total=total,
            succeeded=succeeded,
            failed=failed,
            cancelled=cancelled,
            success_rate=success_rate,
            wall_time_seconds=wall_time_seconds,
            average_duration_seconds=average_duration,
        )
