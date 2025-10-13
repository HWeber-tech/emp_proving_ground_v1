from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pytest

from src.config.risk.risk_config import RiskConfig
from src.trading.trading_manager import TradingManager


class AlwaysActiveRegistry:
    def get_strategy(self, strategy_id: str) -> dict[str, str]:
        return {"status": "active"}


class DummyBus:
    def __init__(self) -> None:
        self.subscriptions: dict[str, list[object]] = {}

    def subscribe(self, topic: str, callback: object) -> None:
        self.subscriptions.setdefault(topic, []).append(callback)

    async def publish(self, _event: object) -> None:  # pragma: no cover - stub
        return None

    def publish_from_sync(self, _event: object) -> None:  # pragma: no cover - stub
        return None

    def is_running(self) -> bool:  # pragma: no cover - stub
        return False


class RecordingExecutionEngine:
    def __init__(self) -> None:
        self.calls = 0

    async def process_order(self, intent: object) -> str:
        self.calls += 1
        return "ok"


@dataclass
class ConfidenceIntent:
    symbol: str
    quantity: float
    price: float
    confidence: float
    strategy_id: str = "alpha"
    created_at: datetime | None = None


def _silence_trading_manager_publishers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("src.trading.trading_manager.publish_risk_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_roi_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_snapshot", _noop)
    monkeypatch.setattr("src.trading.trading_manager.publish_policy_violation", _noop)
    monkeypatch.setattr(
        "src.trading.trading_manager.publish_risk_interface_snapshot", _noop
    )
    monkeypatch.setattr("src.trading.trading_manager.publish_risk_interface_error", _noop)


@pytest.mark.asyncio()
async def test_four_hour_simulation_has_zero_guardrail_violations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_trading_manager_publishers(monkeypatch)

    logging.disable(logging.INFO)
    try:
        manager = TradingManager(
            event_bus=DummyBus(),
            strategy_registry=AlwaysActiveRegistry(),
            execution_engine=RecordingExecutionEngine(),
            risk_config=RiskConfig(
                max_risk_per_trade_pct=Decimal("0.02"),
                max_total_exposure_pct=Decimal("0.5"),
                max_leverage=Decimal("10"),
                max_drawdown_pct=Decimal("0.1"),
                min_position_size=1,
                mandatory_stop_loss=True,
                research_mode=False,
            ),
        )

        total_ticks = 4 * 60
        for index in range(total_ticks):
            intent = ConfidenceIntent(
                symbol="EURUSD",
                quantity=1.0,
                price=1.10,
                confidence=0.8,
            )
            setattr(intent, "event_id", f"acceptance-{index}")

            outcome = await manager.on_trade_intent(intent)
            assert outcome.status not in {"rejected", "blocked"}, outcome.status

        stats = manager.get_execution_stats()
        assert stats["guardrail_violations"] == 0
    finally:
        logging.disable(logging.NOTSET)
