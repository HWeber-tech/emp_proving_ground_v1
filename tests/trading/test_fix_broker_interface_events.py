from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping

import pytest
import simplefix

from src.config.risk.risk_config import RiskConfig
from src.trading.integration.fix_broker_interface import FIXBrokerInterface
from src.trading.risk.risk_api import RISK_API_RUNBOOK, TradingRiskInterface

class DummyRiskGateway:
    def __init__(self, *, approve: bool, adjusted_quantity: Decimal | None = None):
        self.approve = approve
        self.adjusted_quantity = adjusted_quantity
        self.last_intent: Any | None = None
        self.last_state: Mapping[str, Any] | None = None
        self._last_decision: Mapping[str, Any] | None = None
        self._last_policy_snapshot: Mapping[str, Any] | None = None
        self._risk_limits: Mapping[str, Any] = {
            "limits": {"max_open_positions": 5, "risk_per_trade": 0.01},
            "risk_config_summary": {"max_risk_per_trade_pct": 0.02},
            "runbook": RISK_API_RUNBOOK,
        }
        self._risk_reference: Mapping[str, Any] = {
            "risk_api_runbook": RISK_API_RUNBOOK,
            "risk_config_summary": dict(self._risk_limits["risk_config_summary"]),
            "limits": dict(self._risk_limits["limits"]),
        }

    async def validate_trade_intent(
        self, intent: Any, portfolio_state: Mapping[str, Any] | None
    ) -> Any | None:
        self.last_intent = intent
        self.last_state = portfolio_state
        if not self.approve:
            symbol = getattr(intent, "symbol", None)
            if symbol is None and isinstance(intent, Mapping):
                symbol = intent.get("symbol")
            self._last_decision = {
                "reason": "policy_violation",
                "symbol": symbol,
                "risk_reference": dict(self._risk_reference),
            }
            self._last_policy_snapshot = {
                "symbol": symbol,
                "approved": False,
                "violations": ["policy_violation"],
            }
            return None

        if self.adjusted_quantity is not None:
            if isinstance(intent, Mapping):
                intent = dict(intent)
                intent["quantity"] = self.adjusted_quantity
            else:
                intent.quantity = self.adjusted_quantity
        symbol = getattr(intent, "symbol", None)
        if symbol is None and isinstance(intent, Mapping):
            symbol = intent.get("symbol")
        self._last_decision = {
            "status": "approved",
            "symbol": symbol,
            "risk_reference": dict(self._risk_reference),
        }
        self._last_policy_snapshot = {
            "symbol": symbol,
            "approved": True,
            "violations": [],
        }
        return intent

    def get_last_decision(self) -> Mapping[str, Any] | None:
        return self._last_decision

    def get_last_policy_snapshot(self) -> Mapping[str, Any] | None:
        return self._last_policy_snapshot

    def get_risk_limits(self) -> Mapping[str, Any]:
        return dict(self._risk_limits)


class DummyEventBus:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, topic: str, payload: dict[str, Any]) -> None:
        self.emitted.append((topic, payload))


class DummyFixInitiator:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def send_message(self, message: Any) -> None:
        self.messages.append(message)


class _StubRiskManager:
    def __init__(self) -> None:
        self._risk_config = RiskConfig(
            max_risk_per_trade_pct=Decimal("0.02"),
            max_total_exposure_pct=Decimal("0.40"),
            mandatory_stop_loss=True,
        )


@pytest.mark.asyncio
async def test_fix_interface_emits_structured_order_events() -> None:
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    interface = FIXBrokerInterface(DummyEventBus(), trade_queue, fix_initiator=None)

    # Pre-populate with order metadata to mirror placement stage
    interface.orders["ORD-1"] = {
        "symbol": "EURUSD",
        "side": "BUY",
        "quantity": 10,
        "status": "PENDING",
        "timestamp": datetime.utcnow(),
    }

    observed: list[tuple[str, dict[str, Any]]] = []

    interface.add_event_listener(
        "acknowledged", lambda order_id, payload: observed.append((order_id, payload))
    )
    interface.add_event_listener(
        "partial_fill", lambda order_id, payload: observed.append((order_id, payload))
    )
    interface.add_event_listener(
        "filled", lambda order_id, payload: observed.append((order_id, payload))
    )

    # Ack message
    ack_msg = simplefix.FixMessage()
    ack_msg.append_pair(11, "ORD-1")
    ack_msg.append_pair(150, "0")
    await interface._handle_execution_report(ack_msg)  # type: ignore[attr-defined]

    # Partial fill
    partial_msg = simplefix.FixMessage()
    partial_msg.append_pair(11, "ORD-1")
    partial_msg.append_pair(150, "1")
    partial_msg.append_pair(32, "4")
    partial_msg.append_pair(31, "101.0")
    partial_msg.append_pair(14, "4")
    await interface._handle_execution_report(partial_msg)  # type: ignore[attr-defined]

    # Final fill
    fill_msg = simplefix.FixMessage()
    fill_msg.append_pair(11, "ORD-1")
    fill_msg.append_pair(150, "2")
    fill_msg.append_pair(32, "6")
    fill_msg.append_pair(31, "102.0")
    fill_msg.append_pair(14, "10")
    await interface._handle_execution_report(fill_msg)  # type: ignore[attr-defined]

    assert [event for event, _ in observed] == ["ORD-1", "ORD-1", "ORD-1"]
    statuses = [payload["status"] for _, payload in observed]
    assert statuses == ["ACKNOWLEDGED", "PARTIALLY_FILLED", "FILLED"]
    filled_qty = [payload.get("filled_qty") for _, payload in observed]
    assert filled_qty[-1] == pytest.approx(10)


@pytest.mark.asyncio
async def test_fix_interface_publishes_lifecycle_bus_events() -> None:
    bus = DummyEventBus()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    interface = FIXBrokerInterface(bus, trade_queue, fix_initiator=None)

    interface.orders["ORD-LC"] = {
        "symbol": "EURUSD",
        "side": "BUY",
        "quantity": 5,
        "status": "PENDING",
        "timestamp": datetime.utcnow(),
    }

    ack_msg = simplefix.FixMessage()
    ack_msg.append_pair(11, "ORD-LC")
    ack_msg.append_pair(150, "0")

    await interface._handle_execution_report(ack_msg)  # type: ignore[attr-defined]

    lifecycle_events = [
        payload for topic, payload in bus.emitted if topic == "trading.order.lifecycle"
    ]
    assert lifecycle_events
    payload = lifecycle_events[-1]
    assert payload["order_id"] == "ORD-LC"
    assert payload["event"] == "acknowledged"
    assert payload["status"] == "ACKNOWLEDGED"
    assert payload["symbol"] == "EURUSD"
    assert "T" in payload["timestamp"]
    assert payload["source"] == "fix_broker_interface"


@pytest.mark.asyncio
async def test_fix_interface_emits_lifecycle_event_for_cancel_reject() -> None:
    bus = DummyEventBus()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    interface = FIXBrokerInterface(bus, trade_queue, fix_initiator=None)

    interface.orders["ORD-CNCL"] = {
        "symbol": "EURUSD",
        "side": "SELL",
        "quantity": 2,
        "status": "PENDING",
        "timestamp": datetime.utcnow(),
    }

    message = simplefix.FixMessage()
    message.append_pair(11, "ORD-CNCL")
    message.append_pair(58, "Cannot cancel in terminal state")

    await interface._handle_order_cancel_reject(message)  # type: ignore[attr-defined]

    lifecycle_events = [
        payload for topic, payload in bus.emitted if topic == "trading.order.lifecycle"
    ]
    assert lifecycle_events
    payload = lifecycle_events[-1]
    assert payload["order_id"] == "ORD-CNCL"
    assert payload["event"] == "cancel_rejected"
    assert payload["reason"] == "Cannot cancel in terminal state"


@pytest.mark.asyncio
async def test_async_event_listener_is_awaited() -> None:
    bus = DummyEventBus()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    interface = FIXBrokerInterface(bus, trade_queue, fix_initiator=None)

    interface.orders["ORD-ASYNC"] = {
        "symbol": "GBPUSD",
        "side": "BUY",
        "quantity": 3,
        "status": "PENDING",
        "timestamp": datetime.utcnow(),
    }

    triggered = asyncio.Event()

    async def listener(order_id: str, _: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        if order_id == "ORD-ASYNC":
            triggered.set()

    interface.add_event_listener("acknowledged", listener)

    msg = simplefix.FixMessage()
    msg.append_pair(11, "ORD-ASYNC")
    msg.append_pair(150, "0")

    await interface._handle_execution_report(msg)  # type: ignore[attr-defined]

    assert triggered.is_set()


@pytest.mark.asyncio
async def test_fix_interface_blocks_when_risk_gateway_rejects() -> None:
    bus = DummyEventBus()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    risk_gateway = DummyRiskGateway(approve=False)

    def state_provider(symbol: str) -> Mapping[str, Any]:
        return {
            "symbol": symbol,
            "equity": 250_000.0,
            "current_price": 1.2345,
        }

    interface = FIXBrokerInterface(
        bus,
        trade_queue,
        DummyFixInitiator(),
        risk_gateway=risk_gateway,
        portfolio_state_provider=state_provider,
    )

    order_id = await interface.place_market_order("EURUSD", "BUY", 100_000.0)

    assert order_id is None
    assert risk_gateway.last_state == state_provider("EURUSD")
    assert bus.emitted
    topic, payload = bus.emitted[-1]
    assert topic == "telemetry.risk.intent_rejected"
    assert payload["reason"] == "policy_violation"
    assert payload["symbol"] == "EURUSD"
    assert payload["side"] == "BUY"
    assert payload["quantity"] == pytest.approx(100_000.0)
    assert payload.get("policy_snapshot", {}).get("approved") is False
    assert payload["runbook"].endswith("manual_fix_order_risk_block.md")
    assert payload["policy_violation"] is True
    assert payload["severity"] == "critical"
    assert payload["violations"] == ["policy_violation"]
    reference = payload.get("risk_reference")
    assert reference is not None
    assert reference["risk_api_runbook"] == RISK_API_RUNBOOK
    assert reference["limits"]["max_open_positions"] == 5
    assert reference["risk_config_summary"]["max_risk_per_trade_pct"] == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_fix_interface_enriches_risk_reference_from_provider() -> None:
    bus = DummyEventBus()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()

    config = RiskConfig(
        max_risk_per_trade_pct=Decimal("0.02"),
        max_total_exposure_pct=Decimal("0.50"),
        mandatory_stop_loss=True,
    )

    interface = TradingRiskInterface(
        config=config,
        status={"policy_limits": {"max_leverage": 7}},
    )

    def provider() -> TradingRiskInterface:
        return interface

    broker = FIXBrokerInterface(
        bus,
        trade_queue,
        DummyFixInitiator(),
        risk_interface_provider=provider,
    )

    await broker._publish_risk_rejection("EURUSD", "buy", 10_000.0, None)

    assert bus.emitted
    topic, payload = bus.emitted[-1]
    assert topic == "telemetry.risk.intent_rejected"
    reference = payload["risk_reference"]
    assert reference["risk_api_runbook"] == RISK_API_RUNBOOK
    summary = reference["risk_config_summary"]
    assert summary["max_risk_per_trade_pct"] == pytest.approx(0.02)
    assert summary["max_total_exposure_pct"] == pytest.approx(0.50)
    status = reference.get("risk_interface_status")
    assert status is not None
    assert status["policy_limits"]["max_leverage"] == pytest.approx(7)
    config_payload = reference.get("risk_config")
    assert config_payload is not None
    assert config_payload["mandatory_stop_loss"] is True


@pytest.mark.asyncio
async def test_fix_interface_updates_risk_reference_via_setter() -> None:
    bus = DummyEventBus()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    fix = DummyFixInitiator()

    interface = FIXBrokerInterface(bus, trade_queue, fix)

    def provider() -> Mapping[str, Any]:
        return {
            "summary": {
                "max_risk_per_trade_pct": 0.01,
                "max_total_exposure_pct": 0.5,
            },
        }

    interface.set_risk_interface_provider(provider)

    await interface._publish_risk_rejection("EURUSD", "buy", 10_000.0, None)

    assert bus.emitted
    _, payload = bus.emitted[-1]
    reference = payload["risk_reference"]
    summary = reference.get("risk_config_summary")
    assert summary is not None
    assert summary["max_risk_per_trade_pct"] == pytest.approx(0.01)
    assert summary["max_total_exposure_pct"] == pytest.approx(0.5)
    assert reference["risk_api_runbook"] == RISK_API_RUNBOOK


@pytest.mark.asyncio
async def test_fix_interface_uses_risk_gateway_adjustments() -> None:
    bus = DummyEventBus()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    fix = DummyFixInitiator()
    risk_gateway = DummyRiskGateway(approve=True, adjusted_quantity=Decimal("25000"))

    def state_provider(symbol: str) -> Mapping[str, Any]:
        return {
            "symbol": symbol,
            "equity": 500_000.0,
            "current_price": 1.1111,
        }

    interface = FIXBrokerInterface(
        bus,
        trade_queue,
        fix,
        risk_gateway=risk_gateway,
        portfolio_state_provider=state_provider,
    )

    order_id = await interface.place_market_order("GBPUSD", "SELL", 10_000.0)

    assert order_id is not None
    assert not bus.emitted
    assert fix.messages
    qty_field = fix.messages[0].get(38)
    if isinstance(qty_field, bytes):
        qty_field = qty_field.decode()
    assert float(qty_field) == pytest.approx(25_000.0)

    order_record = interface.get_order_status(order_id)
    assert order_record is not None
    assert order_record.get("quantity") == pytest.approx(25_000.0)
    metadata = order_record.get("metadata")
    assert metadata is not None
    assert metadata.get("reference_price") == pytest.approx(1.1111)
    risk_decision = order_record.get("risk_decision")
    assert risk_decision is not None
    assert risk_decision.get("status") == "approved"
    policy_snapshot = order_record.get("policy_snapshot")
    assert policy_snapshot is not None
    assert policy_snapshot.get("approved") is True


@pytest.mark.asyncio
async def test_fix_interface_captures_risk_context_metadata() -> None:
    bus = DummyEventBus()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    fix = DummyFixInitiator()
    manager = _StubRiskManager()

    broker = FIXBrokerInterface(
        bus,
        trade_queue,
        fix,
        risk_context_provider=lambda: manager,
    )

    order_id = await broker.place_market_order("EURUSD", "buy", 5_000.0)

    assert order_id is not None
    context = broker.describe_risk_context()
    assert context["runbook"].endswith("risk_api_contract.md")
    metadata = context.get("metadata")
    assert metadata is not None
    assert metadata["max_risk_per_trade_pct"] == pytest.approx(0.02)

    order_record = broker.get_order_status(order_id)
    assert order_record is not None
    risk_context = order_record.get("risk_context")
    assert risk_context is not None
    assert risk_context["metadata"]["max_total_exposure_pct"] == pytest.approx(0.40)


@pytest.mark.asyncio
async def test_fix_interface_includes_risk_context_on_rejection() -> None:
    bus = DummyEventBus()
    trade_queue: asyncio.Queue[Any] = asyncio.Queue()
    risk_gateway = DummyRiskGateway(approve=False)
    manager = _StubRiskManager()

    broker = FIXBrokerInterface(
        bus,
        trade_queue,
        DummyFixInitiator(),
        risk_gateway=risk_gateway,
        portfolio_state_provider=lambda symbol: {"symbol": symbol},
        risk_context_provider=lambda: manager,
    )

    order_id = await broker.place_market_order("EURUSD", "buy", 10_000.0)

    assert order_id is None
    assert bus.emitted
    _, payload = bus.emitted[-1]
    risk_context = payload.get("risk_context")
    assert risk_context is not None
    assert risk_context["runbook"].endswith("risk_api_contract.md")
    assert risk_context.get("metadata", {}).get("max_total_exposure_pct") == pytest.approx(0.40)
