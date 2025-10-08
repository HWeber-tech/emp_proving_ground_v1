from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.config.risk.risk_config import RiskConfig
from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage, PolicyLedgerStore
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.execution.release_router import ReleaseAwareExecutionRouter
from src.trading.trading_manager import TradingManager


class _StubPortfolioMonitor:
    def __init__(self) -> None:
        self.releases: list[tuple[str, float]] = []
        self.reports: list[object] = []

    def release_position(self, symbol: str, quantity: float) -> None:
        self.releases.append((symbol, float(quantity)))

    async def on_execution_report(self, report: object) -> None:
        await asyncio.sleep(0)
        self.reports.append(report)


class _CompliantManager:
    def __init__(self) -> None:
        self._risk_config = RiskConfig()


class _InvalidManager:
    def get_risk_status(self) -> dict[str, object]:
        return {"risk_config": {"max_risk_per_trade_pct": -1}}


@pytest.mark.asyncio()
async def test_immediate_fill_adapter_captures_risk_metadata() -> None:
    monitor = _StubPortfolioMonitor()
    manager = _CompliantManager()
    adapter = ImmediateFillExecutionAdapter(
        monitor,
        risk_context_provider=lambda: manager,
    )

    await adapter.process_order({"symbol": "EURUSD", "quantity": 1.0, "side": "buy"})

    context = adapter.describe_risk_context()
    assert context["runbook"].endswith("risk_api_contract.md")
    metadata = context.get("metadata")
    assert metadata is not None
    assert metadata["max_risk_per_trade_pct"] > 0


@pytest.mark.asyncio()
async def test_immediate_fill_adapter_records_risk_error() -> None:
    monitor = _StubPortfolioMonitor()
    adapter = ImmediateFillExecutionAdapter(
        monitor,
        risk_context_provider=_InvalidManager,
    )

    await adapter.process_order({"symbol": "GBPUSD", "quantity": 2.0, "side": "sell"})

    context = adapter.describe_risk_context()
    error = context.get("error")
    assert error is not None
    assert error.get("runbook", "").endswith("risk_api_contract.md")


class _StubEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.provider = None

    async def process_order(self, intent: object) -> str:
        return f"{self.name}-processed"

    def set_risk_context_provider(self, provider):  # type: ignore[override]
        self.provider = provider


class _StubEventBus:
    def subscribe(self, *args, **kwargs) -> int:
        return 1

    def publish_from_sync(self, *args, **kwargs) -> int:
        return 1

    async def publish(self, *args, **kwargs) -> None:  # pragma: no cover - async compatibility
        return None


class _StubStrategyRegistry:
    def get_strategy(self, strategy_id: str) -> dict[str, str]:
        return {"strategy_id": strategy_id, "status": "active"}


@pytest.mark.asyncio()
async def test_release_router_describe_includes_risk_metadata(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "router.json")
    release_manager = LedgerReleaseManager(store)
    release_manager.promote(
        policy_id="alpha",
        tactic_id="alpha",
        stage=PolicyLedgerStage.EXPERIMENT,
        approvals=("ops",),
        evidence_id="ledger",
    )

    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=_StubEngine("paper"),
        live_engine=_StubEngine("live"),
        risk_context_provider=_CompliantManager,
    )

    await router.process_order({"strategy_id": "alpha"})
    context = router.describe()["risk_context"]

    assert context["runbook"].endswith("risk_api_contract.md")
    assert context.get("metadata")


@pytest.mark.asyncio()
async def test_release_router_records_risk_error(tmp_path: Path) -> None:
    store = PolicyLedgerStore(tmp_path / "router_error.json")
    release_manager = LedgerReleaseManager(store)

    router = ReleaseAwareExecutionRouter(
        release_manager=release_manager,
        paper_engine=_StubEngine("paper"),
        risk_context_provider=_InvalidManager,
    )

    await router.process_order({"strategy_id": "beta"})
    context = router.describe()["risk_context"]

    assert context["runbook"].endswith("risk_api_contract.md")
    assert context.get("error")


def test_trading_manager_configures_execution_risk_context(tmp_path: Path) -> None:
    event_bus = _StubEventBus()
    base_engine = _StubEngine("base")
    release_manager = LedgerReleaseManager(PolicyLedgerStore(tmp_path / "release.json"))

    manager = TradingManager(
        event_bus=event_bus,
        strategy_registry=_StubStrategyRegistry(),
        execution_engine=base_engine,
        release_manager=release_manager,
        risk_config=RiskConfig(),
    )

    provider = base_engine.provider
    assert callable(provider)
    assert provider() is manager

    router = manager.install_release_execution_router(live_engine=_StubEngine("live"))
    summary = manager.describe_release_execution()

    assert router.risk_context_provider is not None
    assert summary is not None
    assert summary["risk_context"]["runbook"].endswith("risk_api_contract.md")
