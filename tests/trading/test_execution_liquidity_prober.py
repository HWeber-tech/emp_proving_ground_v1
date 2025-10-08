import asyncio
from contextlib import suppress

import pytest

from src.config.risk.risk_config import RiskConfig
from src.runtime.task_supervisor import TaskSupervisor
from src.trading.execution.liquidity_prober import LiquidityProber
from src.trading.risk.risk_api import RISK_API_RUNBOOK


class _SupervisedBroker:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, float | str]] = {}
        self._allow_fill = asyncio.Event()

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> str:
        order_id = f"ORD-{len(self.orders) + 1}"
        self.orders[order_id] = {
            "status": "PENDING",
            "filled_qty": 0.0,
            "target_qty": float(quantity),
        }

        async def _complete() -> None:
            await self._allow_fill.wait()
            order = self.orders[order_id]
            order["status"] = "FILLED"
            order["filled_qty"] = order["target_qty"]

        asyncio.create_task(_complete())
        return order_id

    def get_order_status(self, order_id: str) -> dict[str, float | str] | None:
        return self.orders.get(order_id)

    def release(self) -> None:
        self._allow_fill.set()


class _InstantBroker:
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> str:
        order_id = f"ORD-{hash((symbol, side, quantity)) & 0xFFFF}"
        self._status = {
            "status": "FILLED",
            "filled_qty": float(quantity),
        }
        return order_id

    def get_order_status(self, order_id: str) -> dict[str, float | str] | None:
        return getattr(self, "_status", None)


class _CompliantManager:
    def __init__(self) -> None:
        self._risk_config = RiskConfig()


class _MissingRiskManager:
    def get_risk_status(self) -> dict[str, object]:
        return {}


@pytest.mark.asyncio()
async def test_liquidity_prober_uses_task_supervisor_for_probes() -> None:
    broker = _SupervisedBroker()
    supervisor = TaskSupervisor(namespace="test-liquidity-prober")
    prober = LiquidityProber(
        broker,
        config={"max_concurrent_probes": 2, "timeout_seconds": 1.0},
        task_supervisor=supervisor,
    )

    probe_task = asyncio.create_task(
        prober.probe_liquidity("EURUSD", [1.1010, 1.1015, 1.1020], "buy")
    )
    try:
        await asyncio.sleep(0.05)
        assert supervisor.active_count >= 3
        broker.release()
        await probe_task
        await asyncio.sleep(0)
        assert supervisor.active_count == 0
    finally:
        broker.release()
        with suppress(asyncio.CancelledError):
            if not probe_task.done():
                await probe_task
        await supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_liquidity_prober_captures_risk_metadata() -> None:
    broker = _InstantBroker()
    manager = _CompliantManager()
    prober = LiquidityProber(
        broker,
        risk_context_provider=lambda: manager,
    )

    result = await prober.probe_liquidity("EURUSD", [1.1012], "buy")
    assert result

    context = prober.describe_risk_context()
    assert context["runbook"].endswith("risk_api_contract.md")
    assert context["risk_api_runbook"] == RISK_API_RUNBOOK
    metadata = context.get("metadata")
    assert metadata is not None
    assert metadata["max_risk_per_trade_pct"] > 0


@pytest.mark.asyncio()
async def test_liquidity_prober_records_risk_error_on_failure() -> None:
    broker = _InstantBroker()
    prober = LiquidityProber(
        broker,
        risk_context_provider=_MissingRiskManager,
    )

    await prober.probe_liquidity("EURUSD", [1.1010], "sell")
    context = prober.describe_risk_context()
    error = context.get("error")
    assert error is not None
    assert error["runbook"] == RISK_API_RUNBOOK
