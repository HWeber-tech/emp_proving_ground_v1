"""Utility script to capture a trading performance baseline snapshot."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.config.risk.risk_config import RiskConfig
from src.trading.execution.performance_baseline import collect_performance_baseline
from src.trading.execution.performance_monitor import ThroughputMonitor
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.trading_manager import TradingManager


UTC = timezone.utc


class _InlineBus:
    """Minimal event bus that records published events."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def subscribe(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - stub
        return None

    async def publish(self, event: Any) -> None:
        self.events.append(event)


class _AlwaysActiveRegistry:
    def get_strategy(self, strategy_id: str) -> dict[str, str]:
        return {"status": "active", "strategy_id": strategy_id}


@dataclass
class _DemoIntent:
    symbol: str
    quantity: float
    price: float
    confidence: float
    strategy_id: str
    event_id: str
    created_at: datetime


async def _run(trades: int = 3) -> dict[str, Any]:
    """Execute a short trading burst and return the performance baseline."""

    bus = _InlineBus()

    manager = TradingManager(
        event_bus=bus,
        strategy_registry=_AlwaysActiveRegistry(),
        execution_engine=None,
        initial_equity=25_000.0,
        risk_config=RiskConfig(
            min_position_size=1,
            mandatory_stop_loss=False,
            research_mode=True,
        ),
        throughput_monitor=ThroughputMonitor(window=32),
        trade_throttle={"max_trades": 1, "window_seconds": 60.0},
    )
    manager.execution_engine = ImmediateFillExecutionAdapter(manager.portfolio_monitor)

    async def _noop_publisher(*_args: Any, **_kwargs: Any) -> None:
        return None

    # Silence telemetry publishers to keep the script output focused on the baseline.
    import src.trading.trading_manager as trading_manager_module

    trading_manager_module.publish_risk_snapshot = _noop_publisher
    trading_manager_module.publish_roi_snapshot = _noop_publisher
    trading_manager_module.publish_policy_snapshot = _noop_publisher
    trading_manager_module.publish_policy_violation = _noop_publisher
    trading_manager_module.publish_risk_interface_snapshot = _noop_publisher
    trading_manager_module.publish_risk_interface_error = _noop_publisher

    async def _validate(intent: _DemoIntent, **_kwargs: Any) -> _DemoIntent:
        return intent

    manager.risk_gateway.validate_trade_intent = _validate  # type: ignore[assignment]

    for idx in range(trades):
        now = datetime.now(tz=UTC)
        intent = _DemoIntent(
            symbol="EURUSD",
            quantity=1.0,
            price=1.2000 + 0.0005 * idx,
            confidence=0.9,
            strategy_id="alpha",
            event_id=f"baseline-{idx}",
            created_at=now,
        )
        await manager.on_trade_intent(intent)

    baseline = collect_performance_baseline(manager)
    return baseline


def main() -> None:
    baseline = asyncio.run(_run())
    print("\n=== Baseline Snapshot ===")
    print(json.dumps(baseline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
