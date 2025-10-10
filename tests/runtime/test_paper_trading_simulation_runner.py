from typing import Any, Awaitable, Callable

import pytest
from aiohttp import web

from src.governance.policy_ledger import PolicyLedgerStage, PolicyLedgerStore
from src.governance.system_config import ConnectionProtocol, SystemConfig
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
async def test_run_paper_trading_simulation_executes_orders(tmp_path) -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.json_response({"order_id": "stub-001"})

    base_url, shutdown, captured = await _start_paper_server(handler)

    ledger_path = tmp_path / "ledger.json"
    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id="bootstrap-strategy",
        tactic_id="bootstrap-strategy",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "qa"),
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
        "BOOTSTRAP_MAX_TICKS": "5",
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
            poll_interval=0.05,
        )
    finally:
        await shutdown()

    assert captured, "Paper trading API did not receive any orders"
    assert report.orders, "Simulation did not capture any broker orders"
    assert report.orders[0]["symbol"] == "EURUSD"
    assert report.paper_broker and report.paper_broker["base_url"] == base_url
    assert report.diary_entries >= 1
    assert report.decisions >= 1
    assert report.errors == []
    assert report.portfolio_state is not None
    assert report.performance is not None
    assert report.performance.get("equity") is not None
    assert "roi" in report.performance

