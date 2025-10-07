from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.governance.policy_ledger import (
    LedgerReleaseManager,
    PolicyLedgerStage,
    PolicyLedgerStore,
)
from src.trading.execution.release_router import ReleaseAwareExecutionRouter


class StubEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[Any] = []

    async def process_order(self, intent: Any) -> str:
        self.calls.append(intent)
        return f"{self.name}-ok"


@pytest.mark.asyncio()
async def test_release_router_routes_live_stage(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    release_manager = LedgerReleaseManager(store)

    paper_engine = StubEngine("paper")
    live_engine = StubEngine("live")

    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
        live_engine=live_engine,
    )

    intent: dict[str, Any] = {"strategy_id": "alpha", "metadata": {}}
    first = await router.process_order(intent)
    assert first == "paper-ok"
    assert len(paper_engine.calls) == 1
    assert not live_engine.calls
    assert intent["metadata"]["release_stage"] == PolicyLedgerStage.EXPERIMENT.value
    assert intent["metadata"]["release_execution_route"] == "paper"

    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="diary-alpha",
    )

    second = await router.process_order(intent)
    assert second == "live-ok"
    assert len(live_engine.calls) == 1
    last_route = router.last_route()
    assert last_route is not None
    assert last_route["stage"] == PolicyLedgerStage.LIMITED_LIVE.value
    assert last_route["route"] == "live"


@pytest.mark.asyncio()
async def test_release_router_falls_back_to_paper(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "ledger.json")
    release_manager = LedgerReleaseManager(store)

    paper_engine = StubEngine("paper")
    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=paper_engine,
    )

    release_manager.promote(
        policy_id="beta",
        tactic_id="beta",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk",),
        evidence_id="diary-beta",
    )

    result = await router.process_order({"strategy_id": "beta"})
    assert result == "paper-ok"
    assert len(paper_engine.calls) == 1
    last_route = router.last_route()
    assert last_route is not None
    assert last_route["route"] == "paper"
