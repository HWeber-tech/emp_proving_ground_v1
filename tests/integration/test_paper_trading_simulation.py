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
        assert execution_outcome.get("last_error") in (None, {})
        state = runtime.trading_manager.portfolio_monitor.get_state()
        assert isinstance(state, dict)
        assert "equity" in state and "total_pnl" in state
        initial_equity = state["equity"] - state["total_pnl"]
        assert initial_equity >= 0.0
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
        assert failure_entries, "No diary entry recorded the paper trading failure"
        failure_entry = failure_entries[0]
        error_payload = (
            failure_entry.get("outcomes", {})
            .get("execution", {})
            .get("last_error", {})
        )
        assert error_payload.get("stage") == "broker_submission"
        assert error_payload.get("exception_type") == "PaperTradingApiError"
        assert "gateway failure" in error_payload.get("exception", "")
        state = runtime.trading_manager.portfolio_monitor.get_state()
        assert isinstance(state, dict)
    finally:
        await runtime.stop()
        for cleanup in cleanups:
            result = cleanup()
            if asyncio.iscoroutine(result):
                await result
        await shutdown()
