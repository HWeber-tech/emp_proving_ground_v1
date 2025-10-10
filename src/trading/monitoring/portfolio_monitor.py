"""
Portfolio Monitor - Ticket TRADING-06
Stateful portfolio management with Redis persistence
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime
from typing import Any, Mapping, MutableMapping, Optional, Protocol, TypedDict, cast
from typing import runtime_checkable
from src.core.event_bus import Event, EventBus, get_global_bus
from src.core.coercion import coerce_float
from src.data_foundation.cache import (
    InMemoryRedis,
    ManagedRedisCache,
    RedisCachePolicy,
    wrap_managed_cache,
)

logger = logging.getLogger(__name__)

__all__ = ["PortfolioMonitor", "RedisLike", "InMemoryRedis"]


def _as_float(value: object, default: float = 0.0) -> float:
    return coerce_float(value, default=default)


class Position(TypedDict, total=False):
    quantity: float
    avg_price: float
    current_value: float
    last_price: float


class PortfolioState(TypedDict, total=False):
    cash: float
    open_positions: dict[str, Position]
    daily_pnl: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    last_updated: str
    open_positions_count: int
    equity: float
    current_daily_drawdown: float
    peak_equity: float
    daily_equity_start: float
    daily_reset_date: str


@runtime_checkable
class ExecutionReport(Protocol):
    symbol: object
    side: object
    quantity: object
    price: object


class RedisLike(Protocol):
    def get(self, key: str) -> object | None: ...

    def set(self, key: str, value: object) -> object | None: ...

    def metrics(self, *, reset: bool = False) -> Mapping[str, object]: ...


class PortfolioMonitor:
    """Stateful portfolio monitor with Redis-backed persistence and telemetry."""

    def __init__(
        self,
        event_bus: EventBus,
        redis_client: RedisLike | None = None,
        *,
        cache_policy: RedisCachePolicy | None = None,
    ) -> None:
        self.event_bus = event_bus

        if cache_policy is None and redis_client is None:
            policy = RedisCachePolicy.bootstrap_defaults()
            managed = wrap_managed_cache(None, policy=policy, bootstrap=True)
        else:
            policy = cache_policy or RedisCachePolicy.institutional_defaults()
            managed = wrap_managed_cache(redis_client, policy=policy)

        self.redis_client: ManagedRedisCache = managed
        self.cache_policy = managed.policy
        self.redis_key = "emp:portfolio_state"
        self._cache_metrics_topic = "telemetry.cache"

        self.portfolio: PortfolioState = self._load_initial_state()
        self._synthetic_position_holds = 0
        self._daily_baseline_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._last_reset_date: date = datetime.utcnow().date()

        self._ensure_state_defaults()
        self._update_pnl()
        self._publish_cache_metrics(reason="initial_load")

        # Subscribe to execution reports
        self.event_bus.subscribe("execution.report", self.on_execution_report)

        logger.info(f"PortfolioMonitor initialized with state: {self.portfolio}")

    def _load_initial_state(self) -> PortfolioState:
        """Load portfolio state from Redis or initialize defaults"""
        try:
            state_json = self.redis_client.get(self.redis_key)
            if state_json:
                # Debug: observe unexpected coroutine-like values from client stubs
                try:
                    is_coro = asyncio.iscoroutine(state_json)
                except Exception:
                    is_coro = False
                logger.debug("Redis get returned type=%s, is_coro=%s", type(state_json), is_coro)

                # Decode bytes/bytearray to str for json.loads
                state_str = (
                    state_json.decode("utf-8")
                    if isinstance(state_json, (bytes, bytearray))
                    else str(state_json)
                )
                state = cast(PortfolioState, json.loads(state_str))
                logger.info("Loaded portfolio state from Redis")
                return state
        except Exception as e:
            logger.warning(f"Failed to load state from Redis: {e}")

        # Default initial state
        default_state: PortfolioState = {
            "cash": 100000.0,
            "open_positions": {},
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "last_updated": datetime.now().isoformat(),
        }
        logger.info("Initialized with default portfolio state")
        return default_state

    def _publish_cache_metrics(self, *, reason: str) -> None:
        metrics_fn = getattr(self.redis_client, "metrics", None)
        if not callable(metrics_fn):
            return

        try:
            snapshot = metrics_fn(reset=True)
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to retrieve cache metrics", exc_info=True)
            return

        if not isinstance(snapshot, Mapping):
            snapshot = dict(snapshot or {})

        payload: dict[str, object] = dict(snapshot)
        payload["cache_key"] = self.redis_key
        payload["reason"] = reason

        def _normalise_int(name: str, default: int = 0) -> int:
            value = payload.get(name, default)
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            try:
                return int(float(str(value).strip()))
            except (TypeError, ValueError, AttributeError):
                return default

        for metric_name in ("hits", "misses", "evictions", "expirations", "invalidations", "sets"):
            payload[metric_name] = _normalise_int(metric_name)

        total_requests = payload["hits"] + payload["misses"]
        if "hit_rate" not in payload:
            payload["hit_rate"] = (payload["hits"] / total_requests) if total_requests > 0 else None

        if "keys" in snapshot:
            payload["keys"] = _normalise_int("keys")

        backing = getattr(self.redis_client, "raw_client", self.redis_client)
        payload.setdefault("backing", type(backing).__name__)
        payload.setdefault("namespace", self.cache_policy.namespace)
        payload.setdefault("configured", not isinstance(backing, InMemoryRedis))

        payload.setdefault(
            "policy",
            {
                "ttl_seconds": self.cache_policy.ttl_seconds,
                "max_keys": self.cache_policy.max_keys,
                "namespace": self.cache_policy.namespace,
                "invalidate_prefixes": list(self.cache_policy.invalidate_prefixes),
            },
        )

        event = Event(
            type=self._cache_metrics_topic,
            payload=payload,
            source="portfolio_monitor",
        )

        published = False
        publish_from_sync = getattr(self.event_bus, "publish_from_sync", None)
        is_running = getattr(self.event_bus, "is_running", lambda: False)
        if callable(publish_from_sync) and callable(is_running) and is_running():
            try:
                result = publish_from_sync(event)
                published = result is not None
            except Exception:  # pragma: no cover
                logger.debug("Local event bus publish failed", exc_info=True)

        if not published:
            try:
                topic_bus = get_global_bus()
                topic_bus.publish_sync(
                    self._cache_metrics_topic,
                    payload,
                    source="portfolio_monitor",
                )
                published = True
            except Exception:  # pragma: no cover - background bus optional
                logger.debug("Global cache telemetry publish failed", exc_info=True)

        if not published:
            logger.debug("Cache metrics emission skipped for reason %s", reason)

    def _ensure_state_defaults(self) -> None:
        """Normalise persisted state and seed derived metrics."""

        cash = _as_float(self.portfolio.get("cash"), default=100000.0)
        self.portfolio["cash"] = cash

        raw_positions = self.portfolio.get("open_positions") or {}
        normalised_positions: dict[str, Position] = {}
        if isinstance(raw_positions, Mapping):
            for raw_symbol, payload in raw_positions.items():
                symbol = str(raw_symbol)
                if not isinstance(payload, Mapping):
                    continue
                quantity = _as_float(payload.get("quantity"), default=0.0)
                avg_price = _as_float(payload.get("avg_price"), default=0.0)
                last_price = _as_float(payload.get("last_price"), default=avg_price)
                normalised_positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": avg_price,
                    "last_price": last_price,
                    "current_value": quantity * last_price,
                }
        self.portfolio["open_positions"] = normalised_positions

        now = datetime.utcnow()
        reset_marker = self.portfolio.get("daily_reset_date")
        if isinstance(reset_marker, str):
            try:
                self._last_reset_date = datetime.fromisoformat(reset_marker).date()
            except ValueError:
                self._last_reset_date = now.date()
        else:
            self._last_reset_date = now.date()
        self.portfolio["daily_reset_date"] = self._last_reset_date.isoformat()

        equity, unrealized = self._revalue_positions()
        realized = _as_float(self.portfolio.get("realized_pnl"), default=0.0)

        self.portfolio["realized_pnl"] = float(realized)
        self.portfolio["unrealized_pnl"] = float(unrealized)
        self.portfolio["equity"] = float(equity)

        baseline = _as_float(self.portfolio.get("daily_equity_start"), default=equity)
        self._daily_baseline_equity = baseline
        self.portfolio["daily_equity_start"] = float(baseline)

        peak_equity = _as_float(self.portfolio.get("peak_equity"), default=equity)
        self._peak_equity = max(peak_equity, equity)
        self.portfolio["peak_equity"] = float(self._peak_equity)

        self.portfolio.setdefault("daily_pnl", 0.0)
        self.portfolio["total_pnl"] = float(realized + unrealized)
        self.portfolio.setdefault("current_daily_drawdown", 0.0)

    def _ensure_daily_reset(self, *, now: datetime | None = None) -> None:
        """Reset daily metrics when the calendar day rolls over."""

        current_time = now or datetime.utcnow()
        current_date = current_time.date()
        if self._last_reset_date == current_date:
            return

        equity, _ = self._revalue_positions()
        self._daily_baseline_equity = equity
        self._peak_equity = equity
        self._last_reset_date = current_date
        self.portfolio.update(
            {
                "daily_pnl": 0.0,
                "daily_equity_start": float(equity),
                "daily_reset_date": current_date.isoformat(),
                "peak_equity": float(equity),
            }
        )

    def _revalue_positions(self) -> tuple[float, float]:
        """Return (equity, unrealized_pnl) using the latest known prices."""

        cash = _as_float(self.portfolio.get("cash"), default=0.0)
        total_value = cash
        unrealized = 0.0
        positions = self.portfolio.get("open_positions") or {}
        if not isinstance(positions, MutableMapping):
            return cash, 0.0

        for symbol, position in list(positions.items()):
            if not isinstance(position, MutableMapping):
                continue
            quantity = _as_float(position.get("quantity"), default=0.0)
            avg_price = _as_float(position.get("avg_price"), default=0.0)
            if abs(quantity) <= 1e-12:
                position["quantity"] = 0.0
                position["current_value"] = 0.0
                position["last_price"] = avg_price
                continue
            last_price = _as_float(position.get("last_price"), default=avg_price)
            position["last_price"] = last_price
            position_value = quantity * last_price
            position["current_value"] = position_value
            total_value += position_value
            unrealized += (last_price - avg_price) * quantity

        return total_value, unrealized

    def _save_state_to_redis(self) -> None:
        """Persist current portfolio state to Redis"""
        try:
            self.portfolio["last_updated"] = datetime.now().isoformat()
            state_json = json.dumps(self.portfolio)
            self.redis_client.set(self.redis_key, state_json)
            logger.debug("Portfolio state saved to Redis")
        except Exception as e:
            logger.error(f"Failed to save state to Redis: {e}")
        else:
            self._publish_cache_metrics(reason="persist_state")

    async def on_execution_report(self, event: ExecutionReport) -> None:
        """Handle execution reports and update portfolio state"""
        try:
            logger.info(f"Processing execution report: {event}")

            self._ensure_daily_reset()

            side = str(getattr(event, "side", "BUY")).upper()
            quantity = _as_float(getattr(event, "quantity", 0.0), default=0.0)
            price = _as_float(getattr(event, "price", 0.0), default=0.0)
            symbol = str(getattr(event, "symbol", "UNKNOWN") or "UNKNOWN")

            if quantity <= 0:
                logger.info("Ignoring execution report with non-positive quantity")
                self._update_pnl()
                self._save_state_to_redis()
                return

            positions = self.portfolio.setdefault("open_positions", {})
            position = positions.get(symbol)
            if position is None:
                position = {
                    "quantity": 0.0,
                    "avg_price": 0.0,
                    "current_value": 0.0,
                    "last_price": price,
                }
                positions[symbol] = position

            existing_qty = _as_float(position.get("quantity"), default=0.0)
            avg_price = _as_float(position.get("avg_price"), default=price)

            if side == "BUY":
                executed_qty = quantity
                cash = _as_float(self.portfolio.get("cash"), default=0.0) - executed_qty * price
                self.portfolio["cash"] = cash
                total_value = existing_qty * avg_price + executed_qty * price
                new_qty = existing_qty + executed_qty
                if new_qty > 0:
                    position["quantity"] = new_qty
                    position["avg_price"] = total_value / new_qty
                else:
                    position["quantity"] = 0.0
                    position["avg_price"] = price
                position["last_price"] = price
                position["current_value"] = position["quantity"] * price
            else:
                executed_qty = quantity if existing_qty <= 0 else min(existing_qty, quantity)
                cash = _as_float(self.portfolio.get("cash"), default=0.0) + executed_qty * price
                self.portfolio["cash"] = cash
                realized_delta = (price - avg_price) * executed_qty
                realized_total = _as_float(self.portfolio.get("realized_pnl"), default=0.0)
                self.portfolio["realized_pnl"] = float(realized_total + realized_delta)

                new_qty = existing_qty - executed_qty
                if new_qty <= 0:
                    positions.pop(symbol, None)
                else:
                    position["quantity"] = new_qty
                    position["last_price"] = price
                    position["current_value"] = new_qty * price

            # Update derived metrics and persist state
            self._update_pnl()
            self._save_state_to_redis()

            logger.info(f"Portfolio updated: {self.portfolio}")

        except Exception as e:
            logger.error(f"Error processing execution report: {e}")

    def _update_pnl(self) -> None:
        """Recompute derived P&L and drawdown statistics."""

        equity, unrealized = self._revalue_positions()
        realized = _as_float(self.portfolio.get("realized_pnl"), default=0.0)

        self._peak_equity = max(self._peak_equity, equity)

        if self._daily_baseline_equity <= 0.0:
            daily_pnl = 0.0
            daily_drawdown = 0.0
        else:
            daily_pnl = equity - self._daily_baseline_equity
            daily_drawdown = max(
                0.0, (self._daily_baseline_equity - equity) / self._daily_baseline_equity
            )

        self.portfolio.update(
            {
                "equity": float(equity),
                "unrealized_pnl": float(unrealized),
                "total_pnl": float(realized + unrealized),
                "daily_pnl": float(daily_pnl),
                "current_daily_drawdown": float(daily_drawdown),
                "daily_equity_start": float(self._daily_baseline_equity),
                "peak_equity": float(self._peak_equity),
            }
        )

    def get_state(self) -> PortfolioState:
        """Return a defensive copy of the portfolio state with derived metrics."""

        self._ensure_daily_reset()
        self._update_pnl()

        snapshot = cast(PortfolioState, json.loads(json.dumps(self.portfolio)))
        snapshot["open_positions"] = cast(
            dict[str, Position], dict(self.portfolio.get("open_positions", {}))
        )
        snapshot["last_updated"] = self.portfolio.get("last_updated", datetime.utcnow().isoformat())
        snapshot["cash"] = float(self.portfolio.get("cash", 0.0))
        snapshot["daily_pnl"] = float(self.portfolio.get("daily_pnl", 0.0))
        snapshot["total_pnl"] = float(self.portfolio.get("total_pnl", 0.0))
        snapshot["realized_pnl"] = float(self.portfolio.get("realized_pnl", 0.0))
        snapshot["unrealized_pnl"] = float(self.portfolio.get("unrealized_pnl", 0.0))
        snapshot["equity"] = float(self.portfolio.get("equity", 0.0))
        snapshot["open_positions_count"] = self.get_open_position_count()
        snapshot["current_daily_drawdown"] = float(
            self.portfolio.get("current_daily_drawdown", 0.0)
        )
        snapshot["peak_equity"] = float(self.portfolio.get("peak_equity", snapshot["equity"]))
        snapshot["daily_equity_start"] = float(
            self.portfolio.get("daily_equity_start", self._daily_baseline_equity)
        )
        snapshot["daily_reset_date"] = self.portfolio.get(
            "daily_reset_date", self._last_reset_date.isoformat()
        )
        return snapshot

    # Backwards compatible alias
    def get_portfolio(self) -> dict[str, object]:
        return cast(dict[str, object], self.get_state())

    def get_position(self, symbol: str) -> Optional[dict[str, object]]:
        """Get position for a specific symbol"""
        return cast(Optional[dict[str, object]], self.portfolio["open_positions"].get(symbol))

    def get_total_value(self) -> float:
        """Calculate total portfolio value (cash + positions)"""
        self._ensure_daily_reset()
        self._update_pnl()
        return float(self.portfolio.get("equity", 0.0))

    def get_open_position_count(self) -> int:
        """Return open position count including reserved synthetic slots."""

        return len(self.portfolio["open_positions"]) + self._synthetic_position_holds

    def reserve_position(self, symbol: str, quantity: float, price: float | None = None) -> None:
        """Reserve or extend an open position prior to execution."""

        self._ensure_daily_reset()

        position = self.portfolio["open_positions"].setdefault(
            symbol,
            {
                "quantity": 0.0,
                "avg_price": price or 0.0,
                "current_value": 0.0,
                "last_price": price or 0.0,
            },
        )
        existing_qty = _as_float(position.get("quantity"), default=0.0)
        new_qty = existing_qty + float(quantity)
        position["quantity"] = new_qty

        reference_price = price if price is not None else _as_float(position.get("avg_price"), 0.0)
        if reference_price and new_qty > 0:
            prior_value = existing_qty * _as_float(position.get("avg_price"), reference_price)
            position["avg_price"] = (prior_value + float(quantity) * reference_price) / new_qty
            position["current_value"] = new_qty * reference_price
            position["last_price"] = reference_price

        self._update_pnl()
        self._save_state_to_redis()

    def release_position(self, symbol: str, quantity: float) -> None:
        """Release reserved quantity after fills or cancellations."""

        position = self.portfolio["open_positions"].get(symbol)
        if not position:
            return

        position["quantity"] = float(position.get("quantity", 0.0)) - float(quantity)
        if position["quantity"] <= 0:
            self.portfolio["open_positions"].pop(symbol, None)

        self._update_pnl()
        self._save_state_to_redis()

    def increment_positions(self) -> None:
        """Legacy compatibility shim used by older TradingManager code."""

        self._synthetic_position_holds += 1
        self._save_state_to_redis()

    def reset_portfolio(self) -> None:
        """Reset portfolio to initial state"""
        self.portfolio = {
            "cash": 100000.0,
            "open_positions": {},
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "last_updated": datetime.now().isoformat(),
        }
        self._daily_baseline_equity = 100000.0
        self._peak_equity = 100000.0
        self._last_reset_date = datetime.utcnow().date()
        self.portfolio.update(
            {
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "equity": 100000.0,
                "current_daily_drawdown": 0.0,
                "daily_equity_start": 100000.0,
                "daily_reset_date": self._last_reset_date.isoformat(),
                "peak_equity": 100000.0,
            }
        )
        self._save_state_to_redis()
        logger.info("Portfolio reset to initial state")


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_portfolio_monitor() -> None:
        # Setup Redis connection (in-memory for demonstration)
        redis_client = InMemoryRedis()

        # Create event bus
        event_bus = EventBus()

        # Create portfolio monitor
        monitor = PortfolioMonitor(event_bus, redis_client)

        # Test execution reports
        test_events = [
            ExecutionReport(
                symbol="AAPL", side="BUY", quantity=100, price=150.0, order_id="test_001"
            ),
            ExecutionReport(
                symbol="AAPL", side="SELL", quantity=50, price=155.0, order_id="test_002"
            ),
        ]

        for event in test_events:
            await monitor.on_execution_report(event)
            print(f"Portfolio after {event.side}: {monitor.get_portfolio()}")

        # Test state persistence
        print("Testing state persistence...")
        new_monitor = PortfolioMonitor(event_bus, redis_client)
        print(f"Loaded state: {new_monitor.get_portfolio()}")

    asyncio.run(test_portfolio_monitor())
