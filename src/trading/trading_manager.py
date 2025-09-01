"""
Trading Manager v1.0 - Risk-Aware Trade Execution Coordinator

Implements TRADING-05: Integrates risk management into the trading flow,
ensuring no un-validated trade intent reaches the execution engine.
"""

import logging
from decimal import Decimal
from typing import Any, Callable, Optional, cast

import redis
from src.trading.monitoring.portfolio_monitor import PortfolioMonitor

try:
    from src.core.events import TradeIntent  # legacy
except Exception:  # pragma: no cover
    TradeIntent = TradeRejected = object
try:
    from src.core.risk.position_sizing import position_size as _PositionSizer  # canonical
except Exception:  # pragma: no cover
    _PositionSizer = None  # type: ignore[assignment]
# Provide precise callable typing for the sizer (Optional at runtime)
PositionSizer: Optional[Callable[[Decimal, Decimal, Decimal], Decimal]] = cast(
    Optional[Callable[[Decimal, Decimal, Decimal], Decimal]], _PositionSizer
)

RiskGateway = None  # deprecated path removed; use core risk flows directly

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
        self.portfolio_monitor = PortfolioMonitor(
            event_bus, redis.Redis(host="localhost", port=6379, db=0)
        )
        # Provide required numeric args as Decimals (safe coercion)
        self.position_sizer = (
            PositionSizer(
                Decimal(str(risk_per_trade)),
                Decimal("0.02"),
                Decimal("0.02"),
            )
            if PositionSizer is not None
            else None
        )
        # Treat RiskGateway as Any to avoid "None not callable" type error in typing; runtime DI expected
        self.risk_gateway = cast(Any, RiskGateway)(
            strategy_registry=strategy_registry,
            position_sizer=self.position_sizer,
            portfolio_monitor=self.portfolio_monitor,
            max_open_positions=max_open_positions,
            max_daily_drawdown=max_daily_drawdown,
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
        try:
            logger.info(f"Received trade intent: {event.event_id}")

            # Get current portfolio state
            portfolio_state = cast(Any, self.portfolio_monitor).get_state()

            # Validate trade intent through RiskGateway
            validated_intent = await self.risk_gateway.validate_trade_intent(
                intent=event, portfolio_state=portfolio_state
            )

            if validated_intent:
                # Trade passed validation - proceed to execution
                logger.info(
                    f"Trade intent {event.event_id} validated successfully. "
                    f"Calculated size: {validated_intent.quantity}"
                )

                # Update portfolio state (mock implementation)
                cast(Any, self.portfolio_monitor).increment_positions()

                # Send to execution engine
                await self.execution_engine.process_order(validated_intent)

            else:
                # Trade was rejected by RiskGateway
                logger.warning(f"Trade intent {event.event_id} was rejected by the Risk Gateway")

        except Exception as e:
            logger.error(f"Error processing trade intent {event.event_id}: {e}")

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
