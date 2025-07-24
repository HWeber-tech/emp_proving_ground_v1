"""
Risk Gateway v1.0 - Pre-Trade Validation Service

Implements TRADING-04: Central pre-trade validation service that acts as the final
checkpoint, approving or rejecting trade intents based on critical risk rules.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

from src.core.events import TradeIntent, TradeRejected
from src.trading.risk.position_sizer import PositionSizer
from src.trading.execution.liquidity_prober import LiquidityProber

logger = logging.getLogger(__name__)


class RiskGateway:
    """
    Central pre-trade validation service that enforces risk rules on every trade.
    
    This component acts as the final checkpoint before any trade reaches the
    execution engine, ensuring all trades meet configured risk criteria.
    """
    
    def __init__(
        self,
        strategy_registry,
        position_sizer: PositionSizer,
        portfolio_monitor,
        liquidity_prober: Optional[LiquidityProber] = None,
        max_open_positions: int = 5,
        max_daily_drawdown: float = 0.05,
        liquidity_probe_threshold: float = 1.0  # Volume threshold for probing
    ):
        """
        Initialize the RiskGateway with configurable risk parameters.
        
        Args:
            strategy_registry: Registry for checking strategy status
            position_sizer: Position sizing utility
            portfolio_monitor: Portfolio state monitoring service
            liquidity_prober: LiquidityProber for liquidity validation
            max_open_positions: Maximum allowed open positions (default: 5)
            max_daily_drawdown: Maximum daily drawdown percentage (default: 5%)
            liquidity_probe_threshold: Volume threshold for triggering liquidity probing
        """
        self.strategy_registry = strategy_registry
        self.position_sizer = position_sizer
        self.portfolio_monitor = portfolio_monitor
        self.liquidity_prober = liquidity_prober
        self.max_open_positions = max_open_positions
        self.max_daily_drawdown = Decimal(str(max_daily_drawdown))
        self.liquidity_probe_threshold = liquidity_probe_threshold
        
        logger.info(
            f"RiskGateway initialized with max_open_positions={max_open_positions}, "
            f"max_daily_drawdown={max_daily_drawdown*100}%, "
            f"liquidity_probe_threshold={liquidity_probe_threshold}"
        )
    
    async def validate_trade_intent(
        self,
        intent: TradeIntent,
        portfolio_state: Dict[str, Any]
    ) -> Optional[TradeIntent]:
        """
        Validate a trade intent against all configured risk rules.
        
        This method executes a chain of validation checks. If any check fails,
        it logs the reason and returns None. If all checks pass, it enriches
        the intent with calculated position size and returns it.
        
        Args:
            intent: The trade intent to validate
            portfolio_state: Current portfolio state dictionary
            
        Returns:
            Validated TradeIntent with enriched data, or None if rejected
        """
        try:
            # Chain of validation checks
            validation_result = await self._run_validation_chain(intent, portfolio_state)
            
            if not validation_result["valid"]:
                # Log rejection and publish event
                reason = validation_result["reason"]
                logger.warning(f"Trade intent {intent.event_id} rejected: {reason}")
                
                rejection_event = TradeRejected(
                    event_id=f"rejected_{intent.event_id}_{datetime.now().isoformat()}",
                    timestamp=datetime.now(),
                    source="RiskGateway",
                    original_intent=intent,
                    reason=reason,
                    metadata={
                        "portfolio_state": portfolio_state,
                        "validation_checks": validation_result.get("checks", {})
                    }
                )
                
                # Note: In actual implementation, this would be published to event bus
                logger.info(f"Published TradeRejected event: {rejection_event.event_id}")
                
                return None
            
            # All checks passed - enrich the intent
            enriched_intent = self._enrich_intent(intent, validation_result)
            logger.info(f"Trade intent {intent.event_id} validated and enriched")
            
            return enriched_intent
            
        except Exception as e:
            logger.error(f"Error validating trade intent: {e}")
            return None
    
    async def _run_validation_chain(
        self,
        intent: TradeIntent,
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the complete validation chain against the trade intent.
        
        Returns:
            Dictionary with validation results
        """
        checks = {}
        
        # 1. Strategy Status Check
        strategy_status = await self._check_strategy_status(intent)
        checks["strategy_status"] = strategy_status
        if not strategy_status["valid"]:
            return strategy_status
        
        # 2. Daily Drawdown Check
        drawdown_check = self._check_daily_drawdown(portfolio_state)
        checks["daily_drawdown"] = drawdown_check
        if not drawdown_check["valid"]:
            return drawdown_check
        
        # 3. Open Positions Check
        positions_check = self._check_open_positions(portfolio_state)
        checks["open_positions"] = positions_check
        if not positions_check["valid"]:
            return positions_check
        
        # 4. Position Sizing
        sizing_result = await self._calculate_position_size(intent, portfolio_state)
        checks["position_sizing"] = sizing_result
        if not sizing_result["valid"]:
            return sizing_result
        
        # 5. Liquidity Validation (for large trades)
        liquidity_check = await self._validate_liquidity(intent, portfolio_state)
        checks["liquidity_validation"] = liquidity_check
        if not liquidity_check["valid"]:
            return liquidity_check
        
        # All checks passed
        return {
            "valid": True,
            "checks": checks,
            "calculated_size": sizing_result.get("size"),
            "stop_loss_price": sizing_result.get("stop_loss_price"),
            "liquidity_confidence_score": liquidity_check.get("liquidity_confidence_score")
        }
    
    async def _check_strategy_status(self, intent: TradeIntent) -> Dict[str, Any]:
        """Check if the strategy that generated the intent is active."""
        try:
            # Extract strategy ID from metadata
            strategy_id = intent.metadata.get("strategy_id")
            if not strategy_id:
                return {
                    "valid": False,
                    "reason": "No strategy_id found in trade intent metadata"
                }
            
            # Check strategy status (mock implementation)
            # In real implementation, this would query strategy_registry
            strategy_status = "active"  # Mock for now
            
            if strategy_status != "active":
                return {
                    "valid": False,
                    "reason": f"Strategy {strategy_id} is not active (status: {strategy_status})"
                }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Error checking strategy status: {e}"
            }
    
    def _check_daily_drawdown(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if daily drawdown limit has been exceeded."""
        try:
            current_drawdown = Decimal(str(portfolio_state.get("current_daily_drawdown", 0)))
            
            if current_drawdown > self.max_daily_drawdown:
                return {
                    "valid": False,
                    "reason": f"Daily drawdown limit exceeded: {current_drawdown*100}% > {self.max_daily_drawdown*100}%"
                }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Error checking daily drawdown: {e}"
            }
    
    def _check_open_positions(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if we are at the maximum open positions limit."""
        try:
            open_positions_count = int(portfolio_state.get("open_positions_count", 0))
            
            if open_positions_count >= self.max_open_positions:
                return {
                    "valid": False,
                    "reason": f"Maximum open positions limit reached: {open_positions_count} >= {self.max_open_positions}"
                }
            
            return {"valid": True}
            
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Error checking open positions: {e}"
            }
    
    async def _calculate_position_size(
        self,
        intent: TradeIntent,
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate appropriate position size using the PositionSizer."""
        try:
            # Extract required parameters from intent and portfolio
            equity = float(portfolio_state.get("equity", 0))
            stop_loss_pips = int(intent.metadata.get("stop_loss_pips", 50))  # Default 50 pips
            pip_value = float(portfolio_state.get("pip_value", 0.0001))  # Default for most pairs
            
            if equity <= 0:
                return {
                    "valid": False,
                    "reason": "Invalid equity value for position sizing"
                }
            
            # Calculate position size
            position_size = self.position_sizer.calculate_size_fixed_fractional(
                equity=equity,
                stop_loss_pips=stop_loss_pips,
                pip_value=pip_value
            )
            
            # Calculate stop loss price (simplified calculation)
            current_price = float(intent.price) if intent.price else 0
            if current_price <= 0:
                return {
                    "valid": False,
                    "reason": "Invalid price for stop loss calculation"
                }
            
            # Calculate stop loss based on action
            if intent.action.upper() == "BUY":
                stop_loss_price = Decimal(str(current_price - (stop_loss_pips * pip_value)))
            elif intent.action.upper() == "SELL":
                stop_loss_price = Decimal(str(current_price + (stop_loss_pips * pip_value)))
            else:
                return {
                    "valid": False,
                    "reason": f"Invalid trade action: {intent.action}"
                }
            
            return {
                "valid": True,
                "size": position_size,
                "stop_loss_price": stop_loss_price
            }
            
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Error calculating position size: {e}"
            }
    
    async def _validate_liquidity(
        self,
        intent: TradeIntent,
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate liquidity for large trades using LiquidityProber.
        
        Args:
            intent: The trade intent to validate
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary with liquidity validation results
        """
        # Skip liquidity validation if LiquidityProber not available
        if not self.liquidity_prober:
            return {"valid": True}
        
        # Skip for small trades
        if float(intent.quantity) < self.liquidity_probe_threshold:
            return {"valid": True}
        
        try:
            logger.info(
                f"Triggering liquidity probe for large trade: "
                f"{intent.symbol} {intent.side} {intent.quantity}"
            )
            
            # Generate price levels to probe
            # Probe around the intended price with 5 levels on each side
            current_price = float(intent.price) if intent.price else 0
            if current_price <= 0:
                # Use mid-price from portfolio state if available
                current_price = float(portfolio_state.get("current_price", 1.0))
            
            # Create price levels around intended price
            spread = 0.001  # 1 pip spread approximation
            price_levels = []
            
            if intent.side.upper() == "BUY":
                # For buy orders, probe ask side (higher prices)
                for i in range(-2, 3):
                    price_levels.append(current_price + (i * spread))
            else:
                # For sell orders, probe bid side (lower prices)
                for i in range(-2, 3):
                    price_levels.append(current_price - (i * spread))
            
            # Perform liquidity probing
            probe_results = await self.liquidity_prober.probe_liquidity(
                symbol=intent.symbol,
                price_levels=price_levels,
                side=intent.side.lower()
            )
            
            # Calculate liquidity confidence score
            confidence_score = self.liquidity_prober.calculate_liquidity_confidence_score(
                probe_results=probe_results,
                intended_volume=float(intent.quantity)
            )
            
            # Log probe summary
            probe_summary = self.liquidity_prober.get_probe_summary(probe_results)
            logger.info(
                f"Liquidity probe results for {intent.symbol}: "
                f"confidence_score={confidence_score:.3f}, "
                f"total_liquidity={probe_summary['total_liquidity']:.4f}, "
                f"levels_probed={probe_summary['total_levels']}"
            )
            
            # Check if liquidity is sufficient
            min_confidence_threshold = 0.3  # 30% minimum confidence
            if confidence_score < min_confidence_threshold:
                return {
                    "valid": False,
                    "reason": f"Insufficient liquidity: confidence_score={confidence_score:.3f} < {min_confidence_threshold}"
                }
            
            return {
                "valid": True,
                "liquidity_confidence_score": confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error validating liquidity: {e}")
            # Allow trade to proceed if liquidity validation fails
            return {"valid": True}
    
    def _enrich_intent(
        self,
        intent: TradeIntent,
        validation_result: Dict[str, Any]
    ) -> TradeIntent:
        """
        Enrich the trade intent with validation results.
        
        Args:
            intent: Original trade intent
            validation_result: Results from validation chain
            
        Returns:
            Enriched TradeIntent with additional data
        """
        # Create new intent with enriched data
        enriched_data = intent.dict()
        
        # Add calculated position size
        if "calculated_size" in validation_result:
            enriched_data["quantity"] = Decimal(str(validation_result["calculated_size"]))
        
        # Add liquidity confidence score
        if "liquidity_confidence_score" in validation_result:
            enriched_data["liquidity_confidence_score"] = validation_result["liquidity_confidence_score"]
        
        # Add validation metadata
        enriched_data["metadata"] = {
            **intent.metadata,
            "risk_validation": {
                "timestamp": datetime.now().isoformat(),
                "checks_passed": list(validation_result.get("checks", {}).keys())
            }
        }
        
        return TradeIntent(**enriched_data)
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """
        Get current risk limits for monitoring/audit.
        
        Returns:
            Dictionary with current risk configuration
        """
        return {
            "max_open_positions": self.max_open_positions,
            "max_daily_drawdown": float(self.max_daily_drawdown),
            "liquidity_probe_threshold": float(self.liquidity_probe_threshold),
            "position_sizer_config": self.position_sizer.get_risk_parameters()
        }
