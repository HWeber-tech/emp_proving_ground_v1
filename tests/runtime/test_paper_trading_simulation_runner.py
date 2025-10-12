import json
from pathlib import Path
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


def _build_extras(
    base_url: str,
    ledger_path: Path,
    diary_path: Path,
    *,
    max_ticks: int = 3,
) -> dict[str, str]:
    return {
        "PAPER_TRADING_API_URL": base_url,
        "PAPER_TRADING_ORDER_ENDPOINT": "/orders",
        "PAPER_TRADING_ORDER_ID_FIELD": "order_id",
        "PAPER_TRADING_DEFAULT_STAGE": PolicyLedgerStage.LIMITED_LIVE.value,
        "PAPER_TRADING_ORDER_TIMEOUT": "5",
        "POLICY_LEDGER_PATH": str(ledger_path),
        "DECISION_DIARY_PATH": str(diary_path),
        "BOOTSTRAP_TICK_INTERVAL": "0.0",
        "BOOTSTRAP_MAX_TICKS": str(max_ticks),
        "BOOTSTRAP_BUY_THRESHOLD": "0.0",
        "BOOTSTRAP_SELL_THRESHOLD": "1.0",
        "BOOTSTRAP_MIN_CONFIDENCE": "0.0",
        "BOOTSTRAP_MIN_LIQ_CONF": "0.0",
        "BOOTSTRAP_ORDER_SIZE": "1",
    }


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

    extras = _build_extras(base_url, ledger_path, diary_path, max_ticks=5)

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
    assert "placed_at" in report.orders[0]
    assert report.paper_broker and report.paper_broker["base_url"] == base_url
    assert report.paper_metrics is not None
    assert report.paper_metrics.get("success_ratio") == pytest.approx(1.0)
    assert report.paper_metrics.get("failure_ratio") == pytest.approx(0.0)
    assert report.diary_entries >= 1
    assert report.decisions >= 1
    assert report.errors == []
    assert report.portfolio_state is not None
    assert report.performance is not None
    assert report.performance.get("equity") is not None
    assert "roi" in report.performance
    assert report.execution_stats is not None
    assert report.execution_stats.get("resource_usage") is not None
    assert report.performance_health is not None
    assert report.performance_health.get("throughput") is not None
    assert report.strategy_summary is not None
    summary = report.strategy_summary.get("bootstrap-strategy")
    assert summary is not None
    assert summary["executed"] >= 1
    assert report.release is not None
    posture = report.release.get("posture") if report.release else None
    assert posture is not None
    assert posture.get("stage") == PolicyLedgerStage.LIMITED_LIVE.value
    execution = report.release.get("execution") if report.release else None
    assert execution is not None
    assert execution.get("default_stage") == PolicyLedgerStage.LIMITED_LIVE.value


@pytest.mark.asyncio()
async def test_run_paper_trading_simulation_respects_stop_when_complete(tmp_path) -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.json_response({"order_id": "stub-keep-running"})

    base_url, shutdown, _captured = await _start_paper_server(handler)

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

    extras = _build_extras(base_url, ledger_path, diary_path, max_ticks=50)

    config = SystemConfig(connection_protocol=ConnectionProtocol.paper, extras=extras)

    try:
        report = await run_paper_trading_simulation(
            config,
            min_orders=1,
            max_runtime=0.3,
            poll_interval=0.05,
            stop_when_complete=False,
        )
    finally:
        await shutdown()

    assert report.orders, "Simulation did not capture any broker orders"
    assert report.runtime_seconds >= 0.2
    assert report.strategy_summary is not None
    assert report.release is not None
    assert report.paper_metrics is not None
    assert report.paper_metrics.get("success_ratio", 0.0) >= 0.0
    assert report.paper_metrics.get("failure_ratio", 0.0) >= 0.0


@pytest.mark.asyncio()
async def test_run_paper_trading_simulation_writes_report(tmp_path) -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.json_response({"order_id": "stub-persist"})

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

    extras = _build_extras(base_url, ledger_path, diary_path, max_ticks=4)

    config = SystemConfig(connection_protocol=ConnectionProtocol.paper, extras=extras)
    report_path = tmp_path / "reports" / "paper" / "summary.json"

    try:
        report = await run_paper_trading_simulation(
            config,
            min_orders=1,
            max_runtime=2.0,
            poll_interval=0.05,
            report_path=report_path,
        )
    finally:
        await shutdown()

    assert captured, "Paper trading API did not receive any orders"
    assert report.orders, "Simulation did not capture any broker orders"
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload.get("orders")
    assert payload.get("decisions") == report.decisions
    assert "paper_metrics" in payload
    metrics_payload = payload.get("paper_metrics", {})
    assert metrics_payload.get("success_ratio", 0.0) >= 0.0
    assert metrics_payload.get("failure_ratio", 0.0) >= 0.0
    assert "execution_stats" in payload
    assert "performance_health" in payload
    assert "strategy_summary" in payload
    assert "release" in payload


@pytest.mark.asyncio()
async def test_run_paper_trading_simulation_handles_broker_failure(tmp_path) -> None:
    async def handler(request: web.Request) -> web.Response:
        await request.json()
        return web.Response(status=503, text="gateway down")

    base_url, shutdown, _captured = await _start_paper_server(handler)

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

    extras = _build_extras(base_url, ledger_path, diary_path, max_ticks=4)

    config = SystemConfig(connection_protocol=ConnectionProtocol.paper, extras=extras)

    try:
        report = await run_paper_trading_simulation(
            config,
            min_orders=0,
            max_runtime=1.5,
            poll_interval=0.05,
        )
    finally:
        await shutdown()

    assert report.orders == []
    if report.errors:
        error_snapshot = report.errors[0]
        assert error_snapshot.get("stage") == "broker_submission"
        exception_text = str(error_snapshot.get("exception"))
        assert "503" in exception_text or "gateway" in exception_text.lower()
    else:
        assert report.paper_metrics is not None
        assert report.paper_metrics.get("failed_orders", 0) >= 1
    assert report.decisions >= 1
    assert report.paper_broker and report.paper_broker.get("base_url") == base_url
    assert report.paper_metrics is not None
    assert report.paper_metrics.get("success_ratio") == 0.0
    assert report.paper_metrics.get("failure_ratio") == 1.0
    assert report.execution_stats is not None
    assert report.performance_health is not None
    assert report.strategy_summary is not None
    assert report.release is not None
