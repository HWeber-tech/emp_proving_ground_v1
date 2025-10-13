"""Utility script to capture a trading performance baseline snapshot."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config.risk.risk_config import RiskConfig
from src.trading.execution.performance_baseline import collect_performance_baseline
from src.trading.execution.performance_monitor import ThroughputMonitor
from src.trading.execution.paper_execution import ImmediateFillExecutionAdapter
from src.trading.trading_manager import TradingManager


UTC = timezone.utc


@dataclass
class BaselineOptions:
    """Runtime configuration for the performance baseline drill."""

    trades: int = 3
    throttle_max_trades: int = 1
    throttle_window_seconds: float = 60.0
    throttle_cooldown_seconds: float | None = None
    throttle_min_spacing_seconds: float | None = None
    throttle_multiplier: float | None = None
    throttle_scope_fields: tuple[str, ...] = field(default_factory=tuple)
    backlog_threshold_ms: float | None = None
    max_processing_ms: float = 250.0
    max_lag_ms: float = 250.0
    max_cpu_percent: float | None = None
    max_memory_mb: float | None = None
    max_memory_percent: float | None = None
    output: Path | None = None
    indent: int = 2

    def throttle_config(self) -> Mapping[str, Any]:
        config: dict[str, Any] = {
            "max_trades": self.throttle_max_trades,
            "window_seconds": self.throttle_window_seconds,
        }
        if self.throttle_cooldown_seconds is not None:
            config["cooldown_seconds"] = self.throttle_cooldown_seconds
        if self.throttle_min_spacing_seconds is not None:
            config["min_spacing_seconds"] = self.throttle_min_spacing_seconds
        if self.throttle_multiplier is not None:
            config["multiplier"] = self.throttle_multiplier
        if self.throttle_scope_fields:
            config["scope_fields"] = self.throttle_scope_fields
        return config

    def to_metadata(self) -> Mapping[str, Any]:
        return {
            "trades": self.trades,
            "trade_throttle": dict(self.throttle_config()),
            "max_processing_ms": self.max_processing_ms,
            "max_lag_ms": self.max_lag_ms,
            "backlog_threshold_ms": self.backlog_threshold_ms,
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_mb": self.max_memory_mb,
            "max_memory_percent": self.max_memory_percent,
        }


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


def parse_args(argv: Sequence[str] | None = None) -> BaselineOptions:
    parser = argparse.ArgumentParser(description=__doc__ or "performance baseline")
    parser.add_argument("--trades", type=int, default=3, help="number of intents to replay")
    parser.add_argument(
        "--throttle-max-trades",
        type=int,
        default=1,
        help="trade quota permitted within the throttle window",
    )
    parser.add_argument(
        "--throttle-window-seconds",
        type=float,
        default=60.0,
        help="rolling window duration in seconds",
    )
    parser.add_argument(
        "--throttle-cooldown-seconds",
        type=float,
        help="cooldown duration applied after hitting the rate limit",
    )
    parser.add_argument(
        "--throttle-min-spacing-seconds",
        type=float,
        help="required spacing between trades in seconds",
    )
    parser.add_argument(
        "--throttle-multiplier",
        type=float,
        help="optional throttle multiplier applied to position size",
    )
    parser.add_argument(
        "--throttle-scope-field",
        action="append",
        default=[],
        help="metadata fields used to scope the throttle (repeatable)",
    )
    parser.add_argument(
        "--backlog-threshold-ms",
        type=float,
        help="override backlog lag threshold in milliseconds",
    )
    parser.add_argument(
        "--max-processing-ms",
        type=float,
        default=250.0,
        help="processing latency budget for throughput assessment",
    )
    parser.add_argument(
        "--max-lag-ms",
        type=float,
        default=250.0,
        help="ingest lag budget for throughput assessment",
    )
    parser.add_argument(
        "--max-cpu-percent",
        type=float,
        help="CPU utilisation limit for the resource posture",
    )
    parser.add_argument(
        "--max-memory-mb",
        type=float,
        help="memory usage limit in megabytes",
    )
    parser.add_argument(
        "--max-memory-percent",
        type=float,
        help="memory usage limit as percentage of system RAM",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path to write the baseline JSON payload",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="indentation level for JSON output",
    )

    args = parser.parse_args(argv)

    trades = max(1, args.trades)
    throttle_max_trades = max(1, args.throttle_max_trades)
    window_seconds = max(args.throttle_window_seconds, 0.001)

    scope_fields = tuple(field.strip() for field in args.throttle_scope_field if field.strip())

    output = args.output
    if output is not None:
        output = output.expanduser().resolve()

    return BaselineOptions(
        trades=trades,
        throttle_max_trades=throttle_max_trades,
        throttle_window_seconds=window_seconds,
        throttle_cooldown_seconds=args.throttle_cooldown_seconds,
        throttle_min_spacing_seconds=args.throttle_min_spacing_seconds,
        throttle_multiplier=args.throttle_multiplier,
        throttle_scope_fields=scope_fields,
        backlog_threshold_ms=args.backlog_threshold_ms,
        max_processing_ms=args.max_processing_ms,
        max_lag_ms=args.max_lag_ms,
        max_cpu_percent=args.max_cpu_percent,
        max_memory_mb=args.max_memory_mb,
        max_memory_percent=args.max_memory_percent,
        output=output,
        indent=max(0, args.indent),
    )


async def _run(options: BaselineOptions) -> dict[str, Any]:
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
        trade_throttle=options.throttle_config(),
        backlog_threshold_ms=options.backlog_threshold_ms,
    )
    manager.execution_engine = ImmediateFillExecutionAdapter(manager.portfolio_monitor)

    async def _noop_publisher(*_args: Any, **_kwargs: Any) -> None:
        return None

    import src.trading.trading_manager as trading_manager_module

    publishers = (
        "publish_risk_snapshot",
        "publish_roi_snapshot",
        "publish_policy_snapshot",
        "publish_policy_violation",
        "publish_risk_interface_snapshot",
        "publish_risk_interface_error",
    )
    original_publishers: dict[str, Any] = {}

    for name in publishers:
        original_publishers[name] = getattr(trading_manager_module, name)
        setattr(trading_manager_module, name, _noop_publisher)

    async def _validate(intent: _DemoIntent, **_kwargs: Any) -> _DemoIntent:
        return intent

    manager.risk_gateway.validate_trade_intent = _validate  # type: ignore[assignment]

    try:
        for idx in range(options.trades):
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
    finally:
        for name, func in original_publishers.items():
            setattr(trading_manager_module, name, func)

    baseline = collect_performance_baseline(
        manager,
        max_processing_ms=options.max_processing_ms,
        max_lag_ms=options.max_lag_ms,
        backlog_threshold_ms=options.backlog_threshold_ms,
        max_cpu_percent=options.max_cpu_percent,
        max_memory_mb=options.max_memory_mb,
        max_memory_percent=options.max_memory_percent,
    )
    baseline = dict(baseline)
    baseline["options"] = options.to_metadata()
    return baseline


def _write_output(baseline: Mapping[str, Any], options: BaselineOptions) -> None:
    payload = json.dumps(baseline, indent=options.indent, sort_keys=True)
    print("\n=== Baseline Snapshot ===")
    print(payload)

    if options.output is None:
        return

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(payload + "\n", encoding="utf-8")
    print(f"\nBaseline written to {options.output}")


def main(argv: Sequence[str] | None = None) -> None:
    options = parse_args(argv)
    baseline = asyncio.run(_run(options))
    _write_output(baseline, options)


if __name__ == "__main__":
    main()
