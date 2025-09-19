"""Trading Manager v1.0 - Risk-Aware Trade Execution Coordinator."""

import logging
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional, cast

try:  # pragma: no cover - redis optional in bootstrap deployments
    import redis
except Exception:  # pragma: no cover
    redis = None  # type: ignore

from src.trading.monitoring.portfolio_monitor import InMemoryRedis, PortfolioMonitor
from src.trading.risk.risk_gateway import RiskGateway

try:
    from src.core.events import TradeIntent  # legacy
except Exception:  # pragma: no cover
    TradeIntent = TradeRejected = object
try:
    from src.core.risk.position_sizing import position_size as _PositionSizer  # canonical
except Exception:  # pragma: no cover
    _PositionSizer = None  # type: ignore[assignment]
# Provide precise callable typing for the sizer (Optional at runtime)
PositionSizer = cast(Optional[Callable[[Decimal, Decimal, Decimal], Decimal]], _PositionSizer)

logger = logging.getLogger(__name__)


class TradingManager:
    """
    Coordinates trade execution with integrated risk management.

    This component acts as the central hub for trade processing, ensuring
    all trades pass through the RiskGateway before reaching the execution engine.
    """

    def __init__(
        self,
        event_bus: Any,
        strategy_registry: Any,
        execution_engine: Any,
        initial_equity: float = 10000.0,
        risk_per_trade: float = 0.01,
        max_open_positions: int = 5,
        max_daily_drawdown: float = 0.05,
        *,
        redis_client: Any | None = None,
        liquidity_prober: Any | None = None,
        min_intent_confidence: float = 0.2,
        min_liquidity_confidence: float = 0.3,
    ) -> None:
        """
        Initialize the TradingManager with risk management components.

        Args:
            event_bus: Event bus for publishing/subscribing to events
            strategy_registry: Registry for strategy status checking
            execution_engine: Engine for executing validated trades
            initial_equity: Starting account equity
            risk_per_trade: Percentage of equity to risk per trade
            max_open_positions: Maximum allowed open positions
            max_daily_drawdown: Maximum daily drawdown percentage
        """
        self.event_bus = event_bus
        self.strategy_registry = strategy_registry
        self.execution_engine = execution_engine

        # Initialize risk management components
        resolved_client = self._resolve_redis_client(redis_client)
        self.portfolio_monitor = PortfolioMonitor(event_bus, resolved_client)
        self.position_sizer = PositionSizer
        risk_per_trade_decimal = Decimal(str(risk_per_trade))
        self.risk_gateway = RiskGateway(
            strategy_registry=strategy_registry,
            position_sizer=self.position_sizer,
            portfolio_monitor=self.portfolio_monitor,
            risk_per_trade=risk_per_trade_decimal,
            max_open_positions=max_open_positions,
            max_daily_drawdown=max_daily_drawdown,
            liquidity_prober=liquidity_prober,
            min_intent_confidence=min_intent_confidence,
            min_liquidity_confidence=min_liquidity_confidence,
        )

        logger.info(
            f"TradingManager initialized with equity={initial_equity}, "
            f"risk_per_trade={risk_per_trade*100}%, "
            f"max_open_positions={max_open_positions}, "
            f"max_daily_drawdown={max_daily_drawdown*100}%"
        )

    async def on_trade_intent(self, event: TradeIntent) -> None:
        """
        Handle incoming trade intents with integrated risk management.

        This method replaces the direct execution flow with a risk-validated
        pipeline. All trade intents must pass through the RiskGateway before
        reaching the execution engine.

        Args:
            event: The trade intent event to process
        """
        event_id = getattr(event, "event_id", getattr(event, "id", "unknown"))
        try:
            logger.info(f"Received trade intent: {event_id}")

            # Get current portfolio state
            portfolio_state = cast(Any, self.portfolio_monitor).get_state()

            # Validate trade intent through RiskGateway
            validated_intent = await self.risk_gateway.validate_trade_intent(
                intent=event, portfolio_state=portfolio_state
            )

            if validated_intent:
                # Trade passed validation - proceed to execution
                logger.info(
                    f"Trade intent {event_id} validated successfully. "
                    f"Calculated size: {self._extract_quantity(validated_intent)}"
                )

                symbol = self._extract_symbol(validated_intent)
                quantity = self._extract_quantity(validated_intent)
                price = self._extract_price(validated_intent, portfolio_state)
                cast(Any, self.portfolio_monitor).reserve_position(
                    symbol, float(quantity), price
                )

                # Send to execution engine
                await self.execution_engine.process_order(validated_intent)

            else:
                # Trade was rejected by RiskGateway
                logger.warning(f"Trade intent {event_id} was rejected by the Risk Gateway")

        except Exception as e:
            logger.error(f"Error processing trade intent {event_id}: {e}")

    async def start(self) -> None:
        """Start the TradingManager and subscribe to trade intents."""
        logger.info("Starting TradingManager...")
        # In real implementation, would subscribe to event bus
        # await self.event_bus.subscribe("trade_intent", self.on_trade_intent)

    async def stop(self) -> None:
        """Stop the TradingManager."""
        logger.info("Stopping TradingManager...")

    def get_risk_status(self) -> dict[str, object]:
        """
        Get current risk management status.

        Returns:
            Dictionary with current risk configuration and portfolio state
        """
        return {
            "risk_limits": cast(Any, self.risk_gateway).get_risk_limits(),
            "portfolio_state": cast(Any, self.portfolio_monitor).get_state(),
        }

    def _resolve_redis_client(self, provided: Any | None) -> Any:
        if provided is not None:
            return provided

        if redis is not None:
            try:
                client = redis.Redis(host="localhost", port=6379, db=0)
                client.ping()
                return client
            except Exception:
                logger.warning(
                    "Redis connection unavailable, reverting to in-memory portfolio store"
                )
        return InMemoryRedis()

    @staticmethod
    def _extract_symbol(intent: Any) -> str:
        for attr in ("symbol", "instrument", "asset"):
            value = getattr(intent, attr, None)
            if value:
                return str(value)
        if isinstance(intent, dict) and "symbol" in intent:
            return str(intent["symbol"])
        return "UNKNOWN"

    @staticmethod
    def _extract_quantity(intent: Any) -> Decimal:
        for attr in ("quantity", "size", "volume"):
            value = getattr(intent, attr, None)
            if value is not None:
                return Decimal(str(value))
        if isinstance(intent, dict):
            for key in ("quantity", "size", "volume"):
                if key in intent:
                    return Decimal(str(intent[key]))
        return Decimal("0")

    @staticmethod
    def _extract_price(intent: Any, portfolio_state: Mapping[str, Any]) -> float:
        for attr in ("price", "limit_price", "entry_price"):
            value = getattr(intent, attr, None)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    continue
        if isinstance(intent, dict):
            for key in ("price", "limit_price", "entry_price"):
                if key in intent:
                    try:
                        return float(intent[key])
                    except Exception:
                        continue
        try:
            return float(portfolio_state.get("current_price", 0.0))
        except Exception:
            return 0.0
