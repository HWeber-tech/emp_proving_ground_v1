import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from aiohttp import web

from src.governance.policy_ledger import PolicyLedgerRecord, PolicyLedgerStage
from src.governance.system_config import ConnectionProtocol, SystemConfig
from src.runtime.predator_app import _build_bootstrap_runtime
from src.trading.execution.paper_broker_adapter import PaperBrokerExecutionAdapter
from src.trading.execution.release_router import ReleaseAwareExecutionRouter


class _StubBus:
    async def publish(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def publish_from_sync(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def subscribe(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def is_running(self) -> bool:
        return False


async def _start_server(handler: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> tuple[str, Callable[[], Awaitable[None]], list[dict[str, Any]]]:
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
    sockets = site._server.sockets  # type: ignore[attr-defined]
    assert sockets
    port = sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    async def shutdown() -> None:
        await runner.cleanup()

    return base_url, shutdown, captured


@pytest.mark.asyncio
async def test_bootstrap_runtime_routes_live_stage_to_paper_api(tmp_path: Any) -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.json_response({"order_id": "REST-999"})

    base_url, shutdown, captured = await _start_server(handler)

    ledger_path = tmp_path / "ledger.json"
    record = PolicyLedgerRecord(
        policy_id="bootstrap-strategy",
        tactic_id="bootstrap-strategy",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("qa", "risk"),
        evidence_id="evidence-123",
    )
    ledger_path.write_text(json.dumps({"records": {"bootstrap-strategy": record.as_dict()}}))

    extras = {
        "PAPER_TRADING_API_URL": base_url,
        "PAPER_TRADING_ORDER_ENDPOINT": "/orders",
        "PAPER_TRADING_ORDER_ID_FIELD": "order_id",
        "PAPER_TRADING_DEFAULT_STAGE": PolicyLedgerStage.LIMITED_LIVE.value,
        "PAPER_TRADING_ORDER_TIMEOUT": "3",
        "PAPER_TRADING_FAILOVER_THRESHOLD": "2",
        "PAPER_TRADING_FAILOVER_COOLDOWN": "0.5",
        "POLICY_LEDGER_PATH": str(ledger_path),
    }
    config = SystemConfig(connection_protocol=ConnectionProtocol.paper, extras=extras)

    runtime, cleanups = _build_bootstrap_runtime(config, _StubBus())

    try:
        router = runtime.trading_manager.execution_engine
        assert isinstance(router, ReleaseAwareExecutionRouter)
        assert isinstance(router.live_engine, PaperBrokerExecutionAdapter)

        intent = {
            "policy_id": "bootstrap-strategy",
            "symbol": "EURUSD",
            "side": "buy",
            "quantity": 0.5,
        }

        order_id = await router.process_order(intent)
        assert order_id == "REST-999"
        assert captured, "Paper trading API did not receive the order"
        assert captured[0]["symbol"] == "EURUSD"
        assert captured[0]["side"] == "buy"
    finally:
        for cleanup in cleanups:
            result = cleanup()
            if asyncio.iscoroutine(result):
                await result
        await shutdown()
