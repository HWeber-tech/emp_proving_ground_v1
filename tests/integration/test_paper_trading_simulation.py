from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

import pytest
from aiohttp import web

from src.core.event_bus import EventBus
from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from src.governance.system_config import ConnectionProtocol, SystemConfig
from src.runtime.predator_app import _build_bootstrap_runtime
from src.runtime.paper_simulation import run_paper_trading_simulation


async def _start_paper_server(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> tuple[str, Callable[[], Awaitable[None]], list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []

    async def wrapped(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        captured.append(payload)
        return await handler(request)

    app = web.Application()
    app.router.add_post("/orders", wrapped)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    assert site._server and site._server.sockets  # type: ignore[attr-defined]
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[attr-defined]
    base_url = f"http://127.0.0.1:{port}"

    async def shutdown() -> None:
        await runner.cleanup()

    return base_url, shutdown, captured


@pytest.mark.asyncio()
async def test_bootstrap_runtime_paper_trading_simulation_records_diary(tmp_path) -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.json_response({"order_id": "sim-123"})

    base_url, shutdown, captured = await _start_paper_server(handler)

    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id="bootstrap-strategy",
        tactic_id="bootstrap-strategy",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "qa"),
        evidence_id="bootstrap-evidence",
    )

    diary_path = tmp_path / "diary.json"

    extras = {
        "PAPER_TRADING_API_URL": base_url,
        "PAPER_TRADING_ORDER_ENDPOINT": "/orders",
        "PAPER_TRADING_ORDER_ID_FIELD": "order_id",
        "PAPER_TRADING_DEFAULT_STAGE": PolicyLedgerStage.LIMITED_LIVE.value,
        "PAPER_TRADING_ORDER_TIMEOUT": "5",
        "POLICY_LEDGER_PATH": str(ledger_path),
        "DECISION_DIARY_PATH": str(diary_path),
        "BOOTSTRAP_TICK_INTERVAL": "0.0",
        "BOOTSTRAP_MAX_TICKS": "3",
        "BOOTSTRAP_BUY_THRESHOLD": "0.0",
        "BOOTSTRAP_SELL_THRESHOLD": "1.0",
        "BOOTSTRAP_MIN_CONFIDENCE": "0.0",
        "BOOTSTRAP_MIN_LIQ_CONF": "0.0",
        "BOOTSTRAP_ORDER_SIZE": "1",
    }

    config = SystemConfig(connection_protocol=ConnectionProtocol.paper, extras=extras)
    bus = EventBus()
    runtime, cleanups = _build_bootstrap_runtime(config, bus)

    try:
        await runtime.start()

        for _ in range(50):
            if captured:
                break
            await asyncio.sleep(0.05)

        await runtime.stop()

        assert captured, "Paper trading API did not receive any orders"
        order_payload = captured[0]
        assert order_payload["symbol"] == "EURUSD"
        assert order_payload["side"] == "buy"

        diary_data = json.loads(diary_path.read_text())
        entries = diary_data.get("entries", [])
        assert entries, "Decision diary did not record any entries"
        latest_entry = entries[-1]
        release_outcome = latest_entry.get("outcomes", {}).get("release", {})
        assert release_outcome.get("stage") == PolicyLedgerStage.LIMITED_LIVE.value
        execution_outcome = latest_entry.get("outcomes", {}).get("execution", {})
        assert "last_order" in execution_outcome
        assert execution_outcome["last_order"]["symbol"] == "EURUSD"
        assert "placed_at" in execution_outcome["last_order"]
        broker_snapshot = execution_outcome["last_order"].get("broker_submission", {})
        assert broker_snapshot.get("response", {}).get("order_id") == "sim-123"
        assert broker_snapshot.get("request", {}).get("payload", {}).get("symbol") == "EURUSD"
        assert execution_outcome.get("last_error") in (None, {})
        metrics_snapshot = execution_outcome.get("paper_metrics")
        assert isinstance(metrics_snapshot, dict)
        assert metrics_snapshot.get("total_orders", 0) >= 1
        assert metrics_snapshot.get("successful_orders", 0) >= 1
        assert metrics_snapshot.get("failed_orders", 0) == 0
        assert metrics_snapshot.get("success_ratio") == pytest.approx(1.0)
        assert metrics_snapshot.get("failure_ratio") == pytest.approx(0.0)
        state = runtime.trading_manager.portfolio_monitor.get_state()
        assert isinstance(state, dict)
        assert "equity" in state and "total_pnl" in state
        initial_equity = state["equity"] - state["total_pnl"]
        assert initial_equity >= 0.0
        summary = runtime.trading_manager.get_strategy_execution_summary()
        assert summary.get("bootstrap-strategy", {}).get("executed", 0) >= 1
    finally:
        await runtime.stop()
        for cleanup in cleanups:
            result = cleanup()
            if asyncio.iscoroutine(result):
                await result
        await shutdown()


@pytest.mark.asyncio()
async def test_paper_trading_simulation_respects_audit_enforcement(tmp_path) -> None:
    call_counter = {"count": 0}

    async def handler(_request: web.Request) -> web.Response:
        call_counter["count"] += 1
        return web.json_response({"order_id": "unexpected"})

    base_url, shutdown, captured = await _start_paper_server(handler)

    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id="bootstrap-strategy",
        tactic_id="bootstrap-strategy",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk",),
        evidence_id="paper-sim-evidence",
    )

    diary_path = tmp_path / "diary.json"

    extras = {
        "PAPER_TRADING_API_URL": base_url,
        "PAPER_TRADING_ORDER_ENDPOINT": "/orders",
        "PAPER_TRADING_ORDER_ID_FIELD": "order_id",
        "PAPER_TRADING_DEFAULT_STAGE": PolicyLedgerStage.LIMITED_LIVE.value,
        "PAPER_TRADING_ORDER_TIMEOUT": "5",
        "POLICY_LEDGER_PATH": str(ledger_path),
        "DECISION_DIARY_PATH": str(diary_path),
        "BOOTSTRAP_TICK_INTERVAL": "0.0",
        "BOOTSTRAP_MAX_TICKS": "6",
        "BOOTSTRAP_BUY_THRESHOLD": "0.0",
        "BOOTSTRAP_SELL_THRESHOLD": "1.0",
        "BOOTSTRAP_MIN_CONFIDENCE": "0.0",
        "BOOTSTRAP_MIN_LIQ_CONF": "0.0",
        "BOOTSTRAP_ORDER_SIZE": "1",
    }

    config = SystemConfig(connection_protocol=ConnectionProtocol.paper, extras=extras)
    bus = EventBus()
    runtime, cleanups = _build_bootstrap_runtime(config, bus)

    try:
        await runtime.start()

        for _ in range(120):
            if not runtime.running:
                break
            await asyncio.sleep(0.05)

        await runtime.stop()

        assert call_counter["count"] == 0
        assert captured == []

        assert diary_path.exists(), "Decision diary was not written"
        diary_data = json.loads(diary_path.read_text())
        entries = diary_data.get("entries", [])
        assert entries, "Decision diary did not capture any entries"
        latest_entry = entries[-1]
        release_outcome = latest_entry.get("outcomes", {}).get("release", {})
        assert release_outcome.get("stage") == PolicyLedgerStage.PILOT.value

        execution_summary = runtime.trading_manager.describe_release_execution()
        assert execution_summary is not None
        last_route = execution_summary.get("last_route") if execution_summary else None
        assert last_route is not None
        assert last_route.get("stage") == PolicyLedgerStage.PILOT.value
        assert last_route.get("route") in {"pilot", "paper"}
        execution_outcome = latest_entry.get("outcomes", {}).get("execution", {})
        metrics_snapshot = execution_outcome.get("paper_metrics")
        assert isinstance(metrics_snapshot, dict)
        assert metrics_snapshot.get("total_orders", 0) == 0
        assert metrics_snapshot.get("success_ratio") == 0.0
        assert metrics_snapshot.get("failure_ratio") == 0.0

        strategy_summary = runtime.trading_manager.get_strategy_execution_summary()
        summary_entry = strategy_summary.get("bootstrap-strategy", {})
        assert summary_entry.get("executed", 0) >= 1
    finally:
        await runtime.stop()
        for cleanup in cleanups:
            result = cleanup()
            if asyncio.iscoroutine(result):
                await result
        await shutdown()


@pytest.mark.asyncio()
async def test_paper_trading_simulation_recovers_after_api_failure(tmp_path) -> None:
    call_counter = {"count": 0}

    async def handler(_request: web.Request) -> web.Response:
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return web.Response(status=502, text="gateway failure")
        return web.json_response({"order_id": f"sim-{call_counter['count']}"})

    base_url, shutdown, captured = await _start_paper_server(handler)

    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id="bootstrap-strategy",
        tactic_id="bootstrap-strategy",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "qa"),
        evidence_id="bootstrap-evidence",
    )

    diary_path = tmp_path / "diary.json"

    extras = {
        "PAPER_TRADING_API_URL": base_url,
        "PAPER_TRADING_ORDER_ENDPOINT": "/orders",
        "PAPER_TRADING_ORDER_ID_FIELD": "order_id",
        "PAPER_TRADING_DEFAULT_STAGE": PolicyLedgerStage.LIMITED_LIVE.value,
        "PAPER_TRADING_ORDER_TIMEOUT": "5",
        "POLICY_LEDGER_PATH": str(ledger_path),
        "DECISION_DIARY_PATH": str(diary_path),
        "BOOTSTRAP_TICK_INTERVAL": "0.0",
        "BOOTSTRAP_MAX_TICKS": "6",
        "BOOTSTRAP_BUY_THRESHOLD": "0.0",
        "BOOTSTRAP_SELL_THRESHOLD": "1.0",
        "BOOTSTRAP_MIN_CONFIDENCE": "0.0",
        "BOOTSTRAP_MIN_LIQ_CONF": "0.0",
        "BOOTSTRAP_ORDER_SIZE": "1",
    }

    config = SystemConfig(connection_protocol=ConnectionProtocol.paper, extras=extras)
    bus = EventBus()
    runtime, cleanups = _build_bootstrap_runtime(config, bus)

    try:
        await runtime.start()

        for _ in range(120):
            if len(captured) >= 2:
                break
            await asyncio.sleep(0.05)

        await runtime.stop()

        assert len(captured) >= 2, "Runtime did not retry after API failure"
        assert call_counter["count"] >= 2

        diary_data = json.loads(diary_path.read_text())
        entries = diary_data.get("entries", [])
        assert entries, "Decision diary did not capture any outcomes"

        failure_entries = [
            entry
            for entry in entries
            if entry.get("outcomes", {})
            .get("execution", {})
            .get("last_error")
        ]

        if failure_entries:
            failure_entry = failure_entries[0]
            error_payload = (
                failure_entry.get("outcomes", {})
                .get("execution", {})
                .get("last_error", {})
            )
            assert error_payload.get("stage") == "broker_submission"
            assert error_payload.get("exception_type") == "PaperTradingApiError"
            assert "gateway failure" in error_payload.get("exception", "")
            broker_submission = error_payload.get("broker_submission", {})
            assert broker_submission.get("response", {}).get("status") == 502
        else:
            latest_entry = entries[-1]
            execution_outcome = latest_entry.get("outcomes", {}).get("execution", {})
            assert execution_outcome.get("last_order")
            assert execution_outcome.get("last_error") in (None, {})
        metrics_snapshot = (
            entries[-1]
            .get("outcomes", {})
            .get("execution", {})
            .get("paper_metrics", {})
        )
        assert isinstance(metrics_snapshot, dict)
        if metrics_snapshot.get("total_orders", 0):
            assert metrics_snapshot.get("success_ratio") is not None
            assert metrics_snapshot.get("failure_ratio") is not None
            assert pytest.approx(
                metrics_snapshot.get("success_ratio", 0.0)
                + metrics_snapshot.get("failure_ratio", 0.0)
            ) == 1.0
        state = runtime.trading_manager.portfolio_monitor.get_state()
        assert isinstance(state, dict)
        summary = runtime.trading_manager.get_strategy_execution_summary()
        summary_entry = summary.get("bootstrap-strategy", {})
        assert summary_entry.get("executed", 0) >= 1
    finally:
        await runtime.stop()
        for cleanup in cleanups:
            result = cleanup()
            if asyncio.iscoroutine(result):
                await result
        await shutdown()


@pytest.mark.asyncio()
async def test_paper_trading_simulation_handles_persistent_api_failure(tmp_path) -> None:
    call_counter = {"count": 0}

    async def handler(_request: web.Request) -> web.Response:
        call_counter["count"] += 1
        return web.Response(status=500, text="paper broker meltdown")

    base_url, shutdown, captured = await _start_paper_server(handler)

    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id="bootstrap-strategy",
        tactic_id="bootstrap-strategy",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "qa"),
        evidence_id="paper-sim-failure",
    )

    diary_path = tmp_path / "diary.json"

    extras = {
        "PAPER_TRADING_API_URL": base_url,
        "PAPER_TRADING_ORDER_ENDPOINT": "/orders",
        "PAPER_TRADING_ORDER_ID_FIELD": "order_id",
        "PAPER_TRADING_ORDER_TIMEOUT": "0.5",
        "PAPER_TRADING_RETRY_ATTEMPTS": "2",
        "PAPER_TRADING_RETRY_BACKOFF": "0.0",
        "PAPER_TRADING_DEFAULT_STAGE": PolicyLedgerStage.LIMITED_LIVE.value,
        "POLICY_LEDGER_PATH": str(ledger_path),
        "DECISION_DIARY_PATH": str(diary_path),
        "BOOTSTRAP_TICK_INTERVAL": "0.0",
        "BOOTSTRAP_MAX_TICKS": "4",
        "BOOTSTRAP_BUY_THRESHOLD": "0.0",
        "BOOTSTRAP_SELL_THRESHOLD": "1.0",
        "BOOTSTRAP_MIN_CONFIDENCE": "0.0",
        "BOOTSTRAP_MIN_LIQ_CONF": "0.0",
        "BOOTSTRAP_ORDER_SIZE": "1",
    }

    config = SystemConfig(connection_protocol=ConnectionProtocol.paper, extras=extras)

    try:
        report = await run_paper_trading_simulation(
            config,
            min_orders=1,
            max_runtime=5.0,
            poll_interval=0.1,
            stop_when_complete=False,
        )
    finally:
        await shutdown()

    assert call_counter["count"] >= 1
    assert captured, "Paper trading API was never invoked"
    assert not report.orders

    assert report.errors, "Simulation did not capture broker failures"
    failure = report.errors[-1]
    assert failure.get("stage") == "broker_submission"
    assert failure.get("exception_type") == "PaperTradingApiError"
    assert "500" in failure.get("exception", "")

    metrics = report.paper_metrics or {}
    assert metrics.get("failed_orders", 0) >= 1
    assert metrics.get("successful_orders", 0) == 0
    assert metrics.get("success_ratio") == 0.0
    assert metrics.get("failure_ratio") == pytest.approx(1.0)

    assert report.decisions >= 1
    assert report.diary_entries >= 1

    diary_payload = json.loads(diary_path.read_text())
    entries = diary_payload.get("entries", [])
    assert entries, "Decision diary did not record failure entry"
    last_entry = entries[-1]
    execution_outcome = last_entry.get("outcomes", {}).get("execution", {})
    assert execution_outcome.get("last_order") in (None, {})
    error_snapshot = execution_outcome.get("last_error", {})
    assert error_snapshot.get("stage") == "broker_submission"
    assert error_snapshot.get("exception_type") == "PaperTradingApiError"
    broker_submission = error_snapshot.get("broker_submission", {})
    assert broker_submission.get("response", {}).get("status") == 500
