from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

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


def _prepare_ledger(ledger_path: Path) -> None:
    store = PolicyLedgerStore(ledger_path)
    store.upsert(
        policy_id="bootstrap-strategy",
        tactic_id="bootstrap-strategy",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "qa"),
        evidence_id="replay-evidence",
    )


def _build_extras(
    base_url: str,
    ledger_path: Path,
    diary_path: Path,
    *,
    max_ticks: int = 5,
) -> dict[str, str]:
    extras = {
        "PAPER_TRADING_API_URL": base_url,
        "PAPER_TRADING_ORDER_ENDPOINT": "/orders",
        "PAPER_TRADING_ORDER_ID_FIELD": "order_id",
        "PAPER_TRADING_DEFAULT_STAGE": PolicyLedgerStage.LIMITED_LIVE.value,
        "PAPER_TRADING_ORDER_TIMEOUT": "5",
        "PAPER_TRADING_FAILOVER_THRESHOLD": "2",
        "PAPER_TRADING_FAILOVER_COOLDOWN": "0.5",
        "POLICY_LEDGER_PATH": str(ledger_path),
        "DECISION_DIARY_PATH": str(diary_path),
        "BOOTSTRAP_TICK_INTERVAL": "0.0",
        "BOOTSTRAP_MAX_TICKS": str(max_ticks),
        "BOOTSTRAP_BUY_THRESHOLD": "0.0",
        "BOOTSTRAP_SELL_THRESHOLD": "1.0",
        "BOOTSTRAP_MIN_CONFIDENCE": "0.0",
        "BOOTSTRAP_MIN_LIQ_CONF": "0.0",
        "BOOTSTRAP_ORDER_SIZE": "1",
        "TRADE_THROTTLE_ENABLED": "true",
        "TRADE_THROTTLE_NAME": "paper_guardrail",
        "TRADE_THROTTLE_MAX_TRADES": "10",
        "TRADE_THROTTLE_WINDOW_SECONDS": "60",
        "TRADE_THROTTLE_MIN_SPACING_SECONDS": "0.1",
        "TRADE_THROTTLE_COOLDOWN_SECONDS": "0",
        "TRADE_THROTTLE_SCOPE_FIELDS": "strategy_id",
        "INCIDENT_REQUIRED_RUNBOOKS": "paper_broker_outage",
        "INCIDENT_AVAILABLE_RUNBOOKS": "paper_broker_outage",
        "INCIDENT_MIN_PRIMARY_RESPONDERS": "1",
        "INCIDENT_PRIMARY_RESPONDERS": "alice",
        "INCIDENT_MIN_SECONDARY_RESPONDERS": "1",
        "INCIDENT_SECONDARY_RESPONDERS": "bob",
        "INCIDENT_TRAINING_INTERVAL_DAYS": "30",
        "INCIDENT_TRAINING_AGE_DAYS": "5",
        "INCIDENT_DRILL_INTERVAL_DAYS": "45",
        "INCIDENT_DRILL_AGE_DAYS": "10",
        "INCIDENT_POSTMORTEM_SLA_HOURS": "48",
        "INCIDENT_CHATOPS_READY": "true",
    }
    extras.update(
        {
            "RNG_SEED": "424242",
            "REPLAY_TAPE_ANCHOR": "2024-01-01T00:00:00Z",
        }
    )
    return extras


def _normalise_diary_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    decision = entry.get("decision") if isinstance(entry, Mapping) else None
    regime_state = entry.get("regime_state") if isinstance(entry, Mapping) else None
    notes = entry.get("notes") if isinstance(entry, Mapping) else None
    outcomes = entry.get("outcomes") if isinstance(entry, Mapping) else None
    normalised: dict[str, Any] = {
        "policy_id": entry.get("policy_id"),
        "decision": decision,
        "regime_state": regime_state,
    }
    if isinstance(notes, list):
        normalised["notes"] = list(notes)
    if isinstance(outcomes, Mapping):
        normalised["outcomes"] = {k: outcomes[k] for k in sorted(outcomes)}
    return normalised


def _load_diary_entries(path: Path) -> Sequence[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    return [_normalise_diary_entry(entry) for entry in entries if isinstance(entry, Mapping)]


@pytest.mark.asyncio()
async def test_paper_simulation_replay_is_deterministic(tmp_path) -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.json_response({"order_id": "deterministic-run"})

    base_url, shutdown, _captured = await _start_paper_server(handler)

    reports = []
    diaries: list[Sequence[dict[str, Any]]] = []

    try:
        for label in ("first", "second"):
            run_dir = tmp_path / label
            run_dir.mkdir()
            ledger_path = run_dir / "ledger.json"
            diary_path = run_dir / "diary.json"

            _prepare_ledger(ledger_path)

            extras = _build_extras(base_url, ledger_path, diary_path)
            config = SystemConfig(
                connection_protocol=ConnectionProtocol.paper,
                extras=extras,
            )

            report = await run_paper_trading_simulation(
                config,
                min_orders=1,
                max_runtime=5.0,
                poll_interval=0.05,
            )
            reports.append(report)
            diaries.append(_load_diary_entries(diary_path))
    finally:
        await shutdown()

    assert len(diaries[0]) == len(diaries[1])
    assert diaries[0] == diaries[1]

    perf_a = reports[0].performance
    perf_b = reports[1].performance
    assert perf_a is not None and perf_b is not None
    assert perf_a == perf_b
