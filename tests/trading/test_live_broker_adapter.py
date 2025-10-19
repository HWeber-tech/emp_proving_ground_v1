import asyncio
from decimal import Decimal
from typing import Any

import pytest

from src.config.risk.risk_config import RiskConfig
from src.trading.execution.live_broker_adapter import (
    LiveBrokerError,
    LiveBrokerExecutionAdapter,
)
from src.trading.monitoring.portfolio_monitor import PortfolioMonitor
from src.trading.risk.risk_policy import RiskPolicy


class _StubBus:
    def __init__(self) -> None:
        self._subscriptions: dict[str, list[Any]] = {}

    def subscribe(self, topic: str, callback: Any) -> None:
        self._subscriptions.setdefault(topic, []).append(callback)

    async def emit(self, topic: str, payload: Any) -> None:  # pragma: no cover - stubbed side effects
        for callback in self._subscriptions.get(topic, []):
            if asyncio.iscoroutinefunction(callback):
                await callback(payload)
            else:
                callback(payload)


class _StubBroker:
    def __init__(self, *, order_id: str = "LIVE-001") -> None:
        self._order_id = order_id
        self.calls: list[tuple[str, str, float]] = []

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> str:
        self.calls.append((symbol, side, quantity))
        await asyncio.sleep(0)
        return self._order_id


class _RiskGatewayStub:
    def __init__(self, *, approve: bool = True) -> None:
        self.approve = approve
        self.calls: list[dict[str, Any]] = []
        self._decision: dict[str, Any] = {"status": "idle"}
        self._policy_decision: Any = None
        self._policy_snapshot: Any = None

    async def validate_trade_intent(self, intent: Any, portfolio_state: Any) -> Any | None:
        symbol = None
        if isinstance(intent, dict):
            symbol = intent.get("symbol")
        self.calls.append({"symbol": symbol})
        if self.approve:
            self._decision = {"status": "approved", "symbol": symbol}
            return intent
        self._decision = {"status": "rejected", "reason": "risk_gateway_rejected"}
        return None

    def get_last_decision(self) -> dict[str, Any]:
        return dict(self._decision)

    def get_last_policy_decision(self) -> Any:
        return self._policy_decision

    def set_policy_decision(self, decision: Any) -> None:
        self._policy_decision = decision

    def get_last_policy_snapshot(self) -> Any:
        return self._policy_snapshot

    def set_policy_snapshot(self, snapshot: Any) -> None:
        self._policy_snapshot = snapshot


@pytest.mark.asyncio
async def test_live_broker_adapter_revalidates_and_submits() -> None:
    bus = _StubBus()
    monitor = PortfolioMonitor(bus, redis_client=None)
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        max_leverage=Decimal("4"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
        max_position_size=1000,
        mandatory_stop_loss=False,
    )
    risk_policy = RiskPolicy.from_config(config)
    risk_gateway = _RiskGatewayStub(approve=True)
    broker = _StubBroker(order_id="LIVE-777")

    adapter = LiveBrokerExecutionAdapter(
        broker_interface=broker,
        portfolio_monitor=monitor,
        risk_gateway=risk_gateway,
        risk_policy=risk_policy,
        order_timeout=None,
        failover_threshold=2,
        failover_cooldown_seconds=0.2,
        risk_block_cooldown_seconds=0.0,
    )

    intent = {"symbol": "EURUSD", "side": "buy", "quantity": 10.0}
    order_id = await adapter.process_order(intent)

    assert order_id == "LIVE-777"
    assert broker.calls == [("EURUSD", "BUY", 10.0)]
    assert risk_gateway.calls, "risk gateway should receive validation calls"

    metrics = adapter.describe_metrics()
    assert metrics["risk_blocks"] == 0
    snapshot = adapter.describe_policy_snapshot()
    assert snapshot is not None
    assert snapshot.get("symbol") == "EURUSD"
    assert snapshot.get("side") == "BUY"


@pytest.mark.asyncio
async def test_live_broker_adapter_blocks_risk_policy_violation() -> None:
    bus = _StubBus()
    monitor = PortfolioMonitor(bus, redis_client=None)
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        max_leverage=Decimal("4"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=5,
        max_position_size=1000,
        mandatory_stop_loss=False,
    )
    risk_policy = RiskPolicy.from_config(config)
    risk_gateway = _RiskGatewayStub(approve=True)
    broker = _StubBroker()

    adapter = LiveBrokerExecutionAdapter(
        broker_interface=broker,
        portfolio_monitor=monitor,
        risk_gateway=risk_gateway,
        risk_policy=risk_policy,
        order_timeout=None,
        failover_threshold=1,
        failover_cooldown_seconds=0.1,
        risk_block_cooldown_seconds=0.1,
    )

    intent = {"symbol": "EURUSD", "side": "buy", "quantity": 1.0}

    with pytest.raises(LiveBrokerError):
        await adapter.process_order(intent)

    assert broker.calls == []
    metrics = adapter.describe_metrics()
    assert metrics["risk_blocks"] == 1
    block_payload = adapter.should_block_orders(intent)
    assert block_payload is not None
    assert block_payload.get("reason") == "policy.min_position_size"
    assert block_payload.get("risk_blocks") == 1
    assert adapter.describe_policy_snapshot() is not None


class _MinSizeRiskGateway(_RiskGatewayStub):
    async def validate_trade_intent(self, intent: Any, portfolio_state: Any) -> Any | None:
        symbol = intent.get("symbol") if isinstance(intent, dict) else None
        self.calls.append({"symbol": symbol})
        self._decision = {
            "status": "rejected",
            "reason": "policy.min_position_size",
            "symbol": symbol,
        }
        return None


@pytest.mark.asyncio
async def test_live_broker_adapter_delegates_min_size_near_miss() -> None:
    bus = _StubBus()
    monitor = PortfolioMonitor(bus, redis_client=None)
    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.5"),
        max_leverage=Decimal("4"),
        max_drawdown_pct=Decimal("0.1"),
        min_position_size=1,
        max_position_size=1000,
        mandatory_stop_loss=False,
    )
    risk_policy = RiskPolicy.from_config(config)
    risk_gateway = _MinSizeRiskGateway()
    broker = _StubBroker(order_id="LIVE-002")

    adapter = LiveBrokerExecutionAdapter(
        broker_interface=broker,
        portfolio_monitor=monitor,
        risk_gateway=risk_gateway,
        risk_policy=risk_policy,
        order_timeout=None,
        failover_threshold=1,
        failover_cooldown_seconds=0.1,
        risk_block_cooldown_seconds=0.1,
    )

    intent = {"symbol": "EURUSD", "side": "buy", "quantity": 0.1}
    order_id = await adapter.process_order(intent)

    assert order_id == "LIVE-002"
    assert broker.calls == [("EURUSD", "BUY", 0.1)]
    metrics = adapter.describe_metrics()
    assert metrics["risk_blocks"] == 0
    assert adapter.should_block_orders(intent) is None

    snapshot = adapter.describe_policy_snapshot()
    assert snapshot is not None
    assert snapshot.get("severity") == "near_miss"
    assert snapshot.get("delegated_route") == "paper"
    assert snapshot.get("reason") == "policy.min_position_size"
