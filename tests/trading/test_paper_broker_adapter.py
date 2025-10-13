import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping

import pytest

from src.config.risk.risk_config import RiskConfig
from src.governance.policy_ledger import PolicyLedgerStage
from src.trading.execution.paper_broker_adapter import (
    PaperBrokerError,
    PaperBrokerExecutionAdapter,
)
from src.trading.execution.release_router import ReleaseAwareExecutionRouter
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.monitoring.portfolio_monitor import PortfolioMonitor
from src.trading.trading_manager import TradingManager


class _StubBus:
    def __init__(self) -> None:
        self.subscriptions: dict[str, list[Any]] = {}

    def subscribe(self, topic: str, callback: Any) -> None:
        self.subscriptions.setdefault(topic, []).append(callback)

    async def publish(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - stub
        return None

    def publish_from_sync(self, _event: Any) -> None:  # pragma: no cover - stub
        return None

    def is_running(self) -> bool:  # pragma: no cover - stub
        return False


class _StubBroker:
    def __init__(self, *, order_id: str = "BROKER-001") -> None:
        self.calls: list[tuple[str, str, float]] = []
        self._order_id = order_id

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> str:
        self.calls.append((symbol, side, quantity))
        await asyncio.sleep(0)
        return self._order_id


class _DescribingBroker(_StubBroker):
    def __init__(
        self,
        *,
        order_id: str = "BROKER-001",
        description: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(order_id=order_id)
        self._description = description or {}

    def describe(self) -> Mapping[str, Any]:
        return dict(self._description)


class _TelemetryBroker(_StubBroker):
    def __init__(self, *, order_id: str = "BROKER-100") -> None:
        super().__init__(order_id=order_id)
        self._last_submission: dict[str, Any] | None = None

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> str:
        order_id = await super().place_market_order(symbol, side, quantity)
        self._last_submission = {
            "request": {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
            "response": {
                "order_id": order_id,
            },
        }
        return order_id

    def describe_last_submission(self) -> Mapping[str, Any]:
        if self._last_submission is None:
            return {}
        return {key: value for key, value in self._last_submission.items()}


class _FailingBroker:
    async def place_market_order(self, _symbol: str, _side: str, _quantity: float) -> None:
        await asyncio.sleep(0)
        return None


class _AlwaysActiveRegistry:
    def get_strategy(self, strategy_id: str) -> dict[str, str]:
        return {"strategy_id": strategy_id, "status": "active"}


class _StubReleaseManager:
    def __init__(self, stage: PolicyLedgerStage = PolicyLedgerStage.LIMITED_LIVE) -> None:
        self._stage = stage

    def resolve_stage(self, _policy_id: str | None) -> PolicyLedgerStage:
        return self._stage

    def resolve_thresholds(self, _policy_id: str | None) -> dict[str, Any]:
        return {}

    def describe(self, _policy_id: str | None) -> dict[str, Any]:
        return {"stage": self._stage.value, "thresholds": {}}


@pytest.mark.asyncio
async def test_paper_broker_adapter_submits_order_and_records_metadata() -> None:
    bus = _StubBus()
    monitor = PortfolioMonitor(bus)
    broker = _StubBroker(order_id="FIX-42")
    adapter = PaperBrokerExecutionAdapter(
        broker_interface=broker,
        portfolio_monitor=monitor,
        order_timeout=None,
    )

    order_id = await adapter.process_order(
        {
            "symbol": "EURUSD",
            "side": "buy",
            "quantity": 1.25,
            "metadata": {"policy_id": "alpha"},
        }
    )

    assert order_id == "FIX-42"
    assert broker.calls == [("EURUSD", "BUY", 1.25)]
    last_order = adapter.describe_last_order()
    assert last_order is not None
    assert last_order.get("symbol") == "EURUSD"
    assert last_order.get("order_id") == "FIX-42"
    assert last_order.get("latency_s") is not None
    assert "placed_at" in last_order
    assert datetime.fromisoformat(last_order["placed_at"])  # validates ISO timestamp
    assert adapter.describe_last_error() is None
    metrics = adapter.describe_metrics()
    assert metrics["total_orders"] == 1
    assert metrics["successful_orders"] == 1
    assert metrics["failed_orders"] == 0
    assert metrics["latency_samples"] == 1
    assert metrics["avg_latency_s"] is not None
    assert metrics["avg_latency_s"] >= 0.0
    assert metrics["success_ratio"] == pytest.approx(1.0)
    assert metrics["failure_ratio"] == pytest.approx(0.0)
    assert "first_order_at" in metrics
    assert "last_order_at" in metrics
    datetime.fromisoformat(metrics["first_order_at"])
    datetime.fromisoformat(metrics["last_order_at"])
    assert "last_error_at" not in metrics


@pytest.mark.asyncio
async def test_paper_broker_adapter_rejects_unsupported_order_type() -> None:
    bus = _StubBus()
    monitor = PortfolioMonitor(bus)
    broker = _StubBroker()
    adapter = PaperBrokerExecutionAdapter(
        broker_interface=broker,
        portfolio_monitor=monitor,
    )

    with pytest.raises(PaperBrokerError):
        await adapter.process_order(
            {
                "symbol": "EURUSD",
                "side": "BUY",
                "quantity": 1.0,
                "order_type": "limit",
                "price": 1.1,
            }
        )

    error_snapshot = adapter.describe_last_error()
    assert error_snapshot is not None
    assert error_snapshot["stage"] == "validation"
    assert error_snapshot["message"].startswith("Order type")
    metrics = adapter.describe_metrics()
    assert metrics["total_orders"] == 1
    assert metrics["successful_orders"] == 0
    assert metrics["failed_orders"] == 1
    assert metrics["latency_samples"] == 0
    assert metrics["avg_latency_s"] is None
    assert metrics["success_ratio"] == 0.0
    assert metrics["failure_ratio"] == 1.0
    assert "last_error_at" in metrics
    datetime.fromisoformat(metrics["last_error_at"])
    assert "last_error_at" in metrics
    datetime.fromisoformat(metrics["last_error_at"])


@pytest.mark.asyncio
async def test_paper_broker_adapter_consumes_order_history() -> None:
    bus = _StubBus()
    monitor = PortfolioMonitor(bus)
    broker = _StubBroker(order_id="BROKER-001")
    adapter = PaperBrokerExecutionAdapter(
        broker_interface=broker,
        portfolio_monitor=monitor,
        order_timeout=None,
    )

    await adapter.process_order({"symbol": "EURUSD", "side": "BUY", "quantity": 1.0})
    broker._order_id = "BROKER-002"
    await adapter.process_order({"symbol": "GBPUSD", "side": "SELL", "quantity": 2.0})

    history = adapter.consume_order_history()
    assert {entry["order_id"] for entry in history} == {"BROKER-001", "BROKER-002"}
    assert adapter.consume_order_history() == []


@pytest.mark.asyncio
async def test_paper_broker_adapter_consumes_error_history() -> None:
    bus = _StubBus()
    monitor = PortfolioMonitor(bus)
    broker = _StubBroker()
    adapter = PaperBrokerExecutionAdapter(
        broker_interface=broker,
        portfolio_monitor=monitor,
    )

    with pytest.raises(PaperBrokerError):
        await adapter.process_order(
            {
                "symbol": "EURUSD",
                "side": "BUY",
                "quantity": 0.0,
            }
        )

    errors = adapter.consume_error_history()
    assert errors and errors[0]["stage"] == "validation"
    assert "recorded_at" in errors[0]
    assert datetime.fromisoformat(errors[0]["recorded_at"])
    assert adapter.consume_error_history() == []


@pytest.mark.asyncio
async def test_paper_broker_adapter_raises_when_broker_returns_none() -> None:
    bus = _StubBus()
    monitor = PortfolioMonitor(bus)
    broker = _FailingBroker()
    adapter = PaperBrokerExecutionAdapter(
        broker_interface=broker,
        portfolio_monitor=monitor,
        order_timeout=None,
    )

    with pytest.raises(PaperBrokerError):
        await adapter.process_order(
            {"symbol": "EURUSD", "side": "SELL", "quantity": 2.0}
        )

    error_snapshot = adapter.describe_last_error()
    assert error_snapshot is not None
    assert error_snapshot["stage"] == "broker_submission"
    assert "empty order identifier" in error_snapshot["message"]
    metrics = adapter.describe_metrics()
    assert metrics["total_orders"] == 1
    assert metrics["successful_orders"] == 0
    assert metrics["failed_orders"] == 1
    assert metrics["latency_samples"] == 0
    assert metrics["avg_latency_s"] is None
    assert metrics["success_ratio"] == 0.0
    assert metrics["failure_ratio"] == 1.0


@pytest.mark.asyncio
async def test_paper_broker_adapter_describes_broker_interface() -> None:
    bus = _StubBus()
    monitor = PortfolioMonitor(bus)
    broker = _DescribingBroker(
        description={
            "base_url": "http://paper.example",
            "order_endpoint": "/orders",
            "order_id_field": "order_id",
        }
    )
    adapter = PaperBrokerExecutionAdapter(
        broker_interface=broker,
        portfolio_monitor=monitor,
    )

    summary = adapter.describe_broker()
    assert summary is not None
    assert summary.get("base_url") == "http://paper.example"
    assert summary.get("order_endpoint") == "/orders"
    assert summary.get("order_id_field") == "order_id"


@pytest.mark.asyncio
async def test_paper_broker_adapter_captures_broker_submission_metadata() -> None:
    bus = _StubBus()
    monitor = PortfolioMonitor(bus)
    broker = _TelemetryBroker(order_id="BROKER-777")
    adapter = PaperBrokerExecutionAdapter(
        broker_interface=broker,
        portfolio_monitor=monitor,
        order_timeout=None,
    )

    order_id = await adapter.process_order(
        {
            "symbol": "USDJPY",
            "side": "buy",
            "quantity": 4.0,
        }
    )

    assert order_id == "BROKER-777"
    last_order = adapter.describe_last_order()
    assert last_order is not None
    submission = last_order.get("broker_submission")
    assert isinstance(submission, Mapping)
    assert submission.get("response", {}).get("order_id") == "BROKER-777"
    assert submission.get("request", {}).get("symbol") == "USDJPY"


@pytest.mark.asyncio
async def test_trading_manager_routes_live_stage_to_paper_broker_adapter() -> None:
    bus = _StubBus()
    release_manager = _StubReleaseManager()
    manager = TradingManager(
        event_bus=bus,
        strategy_registry=_AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=100_000.0,
        redis_client=None,
        risk_config=RiskConfig(
            max_risk_per_trade_pct=Decimal("0.02"),
            max_total_exposure_pct=Decimal("0.5"),
            max_leverage=Decimal("4"),
            max_drawdown_pct=Decimal("0.1"),
            min_position_size=1,
        ),
        release_manager=release_manager,
    )

    # Install the immediate fill adapter as the paper engine
    manager.execution_engine = ImmediateFillExecutionAdapter(manager.portfolio_monitor)

    broker = _StubBroker(order_id="LIVE-7")
    router = manager.attach_live_broker_adapter(broker)

    assert isinstance(router, ReleaseAwareExecutionRouter)
    assert router.live_engine is not None

    intent = {
        "policy_id": "alpha",
        "symbol": "GBPUSD",
        "side": "sell",
        "quantity": 3.0,
    }

    order_id = await manager.execution_engine.process_order(intent)
    assert order_id == "LIVE-7"
    assert broker.calls == [("GBPUSD", "SELL", 3.0)]

    last_route = router.last_route()
    assert last_route is not None
    assert last_route.get("route") == "live"
