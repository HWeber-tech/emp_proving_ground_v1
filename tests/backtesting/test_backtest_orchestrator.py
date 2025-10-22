from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from src.backtesting import (
    BacktestBatchResult,
    BacktestOrchestrator,
    BacktestRequest,
    BacktestResult,
)


class _ConcurrencyTrackingRunner:
    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay
        self.max_active = 0
        self._active = 0
        self._lock = asyncio.Lock()

    async def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        async with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            async with self._lock:
                self._active -= 1
        return BacktestResult(
            request_id=request.request_id,
            strategy_id=request.strategy_id,
            status="completed",
            metrics={"return": 0.01},
            started_at=datetime.now(tz=UTC),
            completed_at=datetime.now(tz=UTC),
            duration_seconds=self.delay,
        )


class _FailingRunner:
    def __init__(self, failing_request: str) -> None:
        self.failing_request = failing_request

    async def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        if request.request_id == self.failing_request:
            raise RuntimeError("boom")
        return BacktestResult(
            request_id=request.request_id,
            strategy_id=request.strategy_id,
            status="completed",
            metrics={"return": 0.02},
        )


@pytest.mark.asyncio
async def test_orchestrator_limits_concurrency() -> None:
    runner = _ConcurrencyTrackingRunner(delay=0.02)
    orchestrator = BacktestOrchestrator(runner, max_concurrency=2)
    requests = [
        BacktestRequest(
            request_id=f"req-{idx}",
            strategy_id="s1",
            dataset="historical",
        )
        for idx in range(5)
    ]

    batch = await orchestrator.run_backtests(requests)

    assert isinstance(batch, BacktestBatchResult)
    assert len(batch.results) == len(requests)
    assert batch.summary.succeeded == len(requests)
    assert runner.max_active <= 2


@pytest.mark.asyncio
async def test_orchestrator_handles_runner_exceptions() -> None:
    failing_id = "req-2"
    runner = _FailingRunner(failing_request=failing_id)
    orchestrator = BacktestOrchestrator(runner, max_concurrency=3)
    requests = [
        BacktestRequest(request_id=f"req-{idx}", strategy_id="s1", dataset="hist")
        for idx in range(4)
    ]

    batch = await orchestrator.run_backtests(requests)

    failed = batch.result_for(failing_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error is not None
    assert batch.summary.failed == 1
    assert batch.summary.succeeded == len(requests) - 1


@pytest.mark.asyncio
async def test_orchestrator_emits_progress_updates() -> None:
    runner = _ConcurrencyTrackingRunner(delay=0.0)
    orchestrator = BacktestOrchestrator(runner, max_concurrency=4)
    requests = [
        BacktestRequest(request_id=f"req-{idx}", strategy_id="s1", dataset="hist")
        for idx in range(3)
    ]
    seen: list[str] = []

    async def progress(result: BacktestResult) -> None:
        seen.append(result.request_id)

    await orchestrator.run_backtests(requests, progress_callback=progress)

    assert set(seen) == {request.request_id for request in requests}


@pytest.mark.asyncio
async def test_orchestrator_honours_cancellation_event() -> None:
    runner = _ConcurrencyTrackingRunner(delay=0.05)
    orchestrator = BacktestOrchestrator(runner, max_concurrency=3)
    requests = [
        BacktestRequest(request_id=f"req-{idx}", strategy_id="s1", dataset="hist")
        for idx in range(3)
    ]
    cancel_event = asyncio.Event()
    cancel_event.set()

    batch = await orchestrator.run_backtests(requests, cancel_event=cancel_event)

    assert all(result.status == "cancelled" for result in batch.results)
    assert batch.summary.cancelled == len(requests)
    assert batch.summary.failed == 0
    assert batch.summary.succeeded == 0
