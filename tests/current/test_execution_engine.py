"""Regression tests for src.trading.execution.execution_engine.ExecutionEngine."""

from __future__ import annotations

import pytest

from src.config.risk.risk_config import RiskConfig
from src.trading.execution.execution_engine import ExecutionEngine
from src.trading.models.order import OrderStatus
from src.trading.risk.risk_api import RISK_API_RUNBOOK


@pytest.mark.asyncio
async def test_partial_fill_and_completion_updates_positions() -> None:
    engine = ExecutionEngine()
    order_id = await engine.send_order("EURUSD", "BUY", 100_000, price=1.1)

    engine.record_fill(order_id, 25_000, 1.1005)
    order = engine.get_order(order_id)

    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == pytest.approx(25_000.0)

    position = await engine.get_position("EURUSD")
    assert position is not None
    assert position.quantity == pytest.approx(25_000.0)

    engine.record_fill(order_id, 75_000, 1.1010)
    order = engine.get_order(order_id)

    assert order.status is OrderStatus.FILLED
    assert order.filled_quantity == pytest.approx(100_000.0)

    position = await engine.get_position("EURUSD")
    assert position is not None
    assert position.quantity == pytest.approx(100_000.0)
    assert position.average_price == pytest.approx(1.100875)


@pytest.mark.asyncio
async def test_retry_flow_tracks_attempts_and_remaining() -> None:
    engine = ExecutionEngine()
    order_id = await engine.send_order("EURUSD", "BUY", 80_000, price=1.2)

    engine.record_fill(order_id, 30_000, 1.2005)
    engine.mark_rejected(order_id, "timeout")

    retried = engine.retry_order(order_id, note="gateway retry")
    assert retried is True

    order = engine.get_order(order_id)
    assert order.status is OrderStatus.PENDING

    summary = engine.reconcile()
    open_orders = {entry["order_id"]: entry for entry in summary["open_orders"]}

    assert open_orders[order_id]["attempts"] == 2
    assert open_orders[order_id]["remaining_quantity"] == pytest.approx(50_000.0)
    assert open_orders[order_id]["last_error"] == "gateway retry"


@pytest.mark.asyncio
async def test_reconcile_captures_cancellations_and_realized_pnl() -> None:
    engine = ExecutionEngine()
    buy_id = await engine.send_order("EURUSD", "BUY", 100_000, price=1.19)
    engine.record_fill(buy_id, 100_000, 1.19)

    sell_id = await engine.send_order("EURUSD", "SELL", 50_000, price=1.23)
    engine.record_fill(sell_id, 50_000, 1.23)

    cancel_id = await engine.send_order("GBPUSD", "BUY", 10_000, price=1.3)
    cancelled = await engine.cancel_order(cancel_id)

    assert cancelled is True

    summary = engine.reconcile()

    assert {entry["order_id"] for entry in summary["filled_orders"]} == {buy_id, sell_id}
    assert {entry["order_id"] for entry in summary["cancelled_orders"]} == {cancel_id}

    positions = summary["positions"]
    assert "EURUSD" in positions
    assert positions["EURUSD"]["quantity"] == pytest.approx(50_000.0)
    assert positions["EURUSD"]["realized_pnl"] == pytest.approx((1.23 - 1.19) * 50_000)

    # Remaining open quantity for the buy leg should be reflected in the position snapshot.
    assert positions["EURUSD"]["average_price"] == pytest.approx(1.19)


class _CompliantManager:
    def __init__(self) -> None:
        self._risk_config = RiskConfig()


class _BrokenManager:
    def get_risk_status(self) -> dict[str, object]:
        return {"risk_config": {"max_risk_per_trade_pct": -1}}


@pytest.mark.asyncio
async def test_execution_engine_captures_risk_context_metadata() -> None:
    engine = ExecutionEngine()
    manager = _CompliantManager()
    engine.set_risk_context_provider(lambda: manager)

    await engine.send_order("EURUSD", "BUY", 10_000, price=1.105)

    context = engine.describe_risk_context()
    assert context["runbook"].endswith("risk_api_contract.md")
    assert context["risk_api_runbook"] == RISK_API_RUNBOOK
    metadata = context.get("metadata")
    assert metadata is not None
    assert metadata["max_risk_per_trade_pct"] > 0

    reconciliation = engine.reconcile()
    risk_summary = reconciliation.get("risk_context")
    assert isinstance(risk_summary, dict)
    assert risk_summary.get("runbook", "").endswith("risk_api_contract.md")


@pytest.mark.asyncio
async def test_execution_engine_records_risk_error_on_metadata_failure() -> None:
    engine = ExecutionEngine()
    engine.set_risk_context_provider(_BrokenManager)

    await engine.send_order("EURUSD", "BUY", 5_000, price=1.101)

    context = engine.describe_risk_context()
    error = context.get("error")
    assert error is not None
    assert error.get("runbook", "").endswith("risk_api_contract.md")
