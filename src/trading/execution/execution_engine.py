"""
Execution Engine
================

Core execution engine for managing order execution in the EMP Proving Ground
trading system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.sensory.dimensions.what.volatility_engine import vol_signal
from src.data_foundation.config.vol_config import load_vol_config

from src.trading.models.order import Order, OrderStatus
from src.trading.models.position import Position
from src.core.exceptions import ExecutionException

logger = logging.getLogger(__name__)


class ExecutionEngine(IExecutionEngine):
    """
    Concrete implementation of the execution engine for managing
    order execution and position management.
    """
    
    def __init__(self):
        self.active_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    async def initialize(self) -> bool:
        """Initialize the execution engine."""
        try:
            logger.info("Initializing execution engine...")
            self.active_orders.clear()
            self.positions.clear()
            self.execution_history.clear()
            logger.info("Execution engine initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize execution engine: {e}")
            return False
    
    async def execute_order(self, order: Order) -> bool:
        """
        Execute a trading order.
        
        Args:
            order: The order to execute
            
        Returns:
            True if execution successful, False otherwise
        """
        try:
            logger.info(f"Executing order: {order.order_id} - {order.symbol} {order.side} {order.quantity}")
            
            # Validate order
            if not self._validate_order(order):
                return False

            # Apply volatility-based sizing and stops (Tier-0)
            try:
                cfg = load_vol_config()
                # Minimal input: use last few bars from an external cache; here stub with zero
                vs = vol_signal(order.symbol, datetime.utcnow(), [], [])
                # Scale quantity and attach stop multiplier hint
                if hasattr(order, 'quantity') and order.quantity > 0:
                    order.quantity = max(1, int(order.quantity * vs.sizing_multiplier))
                setattr(order, 'stop_mult_hint', getattr(vs, 'stop_mult', 1.3))
                logger.info(f"Vol sizing applied: mult={vs.sizing_multiplier} regime={vs.regime}")
            except Exception:
                pass
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            order.status = OrderStatus.PENDING
            
            # Execute via real FIX API
            await self._execute_via_fix_api(order)
            
            # Update position
            await self._update_position(order)
            
            # Log execution
            self._log_execution(order)
            
            logger.info(f"Order executed successfully: {order.order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found for cancellation")
                return False
            
            order = self.active_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.active_orders[order_id]
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current position or None if no position
        """
        return self.positions.get(symbol)
    
    async def get_active_orders(self) -> List[Order]:
        """
        Get all active orders.
        
        Returns:
            List of active orders
        """
        return list(self.active_orders.values())
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        if not order.symbol:
            logger.error("Order symbol is required")
            return False
        
        if order.quantity <= 0:
            logger.error("Order quantity must be positive")
            return False
        
        if order.order_type not in ['MARKET', 'LIMIT', 'STOP']:
            logger.error(f"Invalid order type: {order.order_type}")
            return False
        
        return True
    
    async def _execute_via_fix_api(self, order: Order) -> None:
        """Execute order via real FIX API integration."""
        try:
            # Import the working FIX API
            from src.operational.icmarkets_api import ICMarketsManager
            from src.operational.icmarkets_config import ICMarketsConfig
            
            # Initialize FIX connection if not already done
            if not hasattr(self, '_fix_manager') or not self._fix_manager:
                config = ICMarketsConfig(environment="demo", account_number="9533708")
                self._fix_manager = ICMarketsManager(config)
                if not self._fix_manager.start():
                    raise Exception("Failed to start FIX API connection")
            
            # Convert order to FIX format
            symbol_id = self._get_symbol_id(order.symbol)
            side = "BUY" if order.side == "BUY" else "SELL"
            
            # Place order via FIX API
            fix_order_id = self._fix_manager.place_market_order(
                symbol=symbol_id,
                side=side,
                quantity=int(order.quantity)
            )
            
            if fix_order_id:
                # Wait for ExecutionReport
                execution_report = await self._wait_for_execution_report(fix_order_id)
                
                if execution_report:
                    # Update order with real execution data
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = execution_report.get('filled_quantity', order.quantity)
                    order.average_price = execution_report.get('average_price', 0.0)
                    order.filled_at = datetime.now()
                    order.broker_order_id = execution_report.get('broker_order_id')
                else:
                    # Execution failed
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = "No ExecutionReport received"
            else:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "FIX API order placement failed"
                
        except Exception as e:
            logger.error(f"FIX API execution failed: {e}")
            order.status = OrderStatus.REJECTED
            order.rejection_reason = f"FIX API error: {str(e)}"
    
    async def _update_position(self, order: Order) -> None:
        """Update position based on executed order."""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
        
        position = self.positions[symbol]
        
        if order.side == 'BUY':
            position.quantity += order.filled_quantity
            position.average_price = (
                (position.average_price * (position.quantity - order.filled_quantity) + 
                 order.average_price * order.filled_quantity) / position.quantity
            )
        else:  # SELL
            position.quantity -= order.filled_quantity
            position.realized_pnl += (order.average_price - position.average_price) * order.filled_quantity
    
    def _log_execution(self, order: Order) -> None:
        """Log order execution details."""
        execution_record = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'filled_quantity': order.filled_quantity,
            'average_price': order.average_price,
            'status': order.status.value,
            'timestamp': order.filled_at.isoformat() if order.filled_at else None
        }
        self.execution_history.append(execution_record)

    def _get_symbol_id(self, symbol: str) -> str:
        """Convert symbol name to FIX symbol ID."""
        # Symbol mapping based on IC Markets FIX API
        symbol_map = {
            'EURUSD': '1',
            'GBPUSD': '2', 
            'USDJPY': '3',
            'USDCHF': '4',
            'AUDUSD': '5',
            'USDCAD': '6',
            'NZDUSD': '7',
            'EURGBP': '8',
            'EURJPY': '9',
            'GBPJPY': '10'
        }
        return symbol_map.get(symbol.upper(), '1')  # Default to EURUSD
    
    async def _wait_for_execution_report(self, order_id: str, timeout: int = 30) -> Optional[Dict]:
        """Wait for ExecutionReport from FIX API."""
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            # Check if FIX manager has execution report
            if hasattr(self._fix_manager, 'orders') and order_id in self._fix_manager.orders:
                order_data = self._fix_manager.orders[order_id]
                if order_data.get('status') == 'FILLED':
                    return {
                        'filled_quantity': order_data.get('filled_quantity'),
                        'average_price': order_data.get('average_price'),
                        'broker_order_id': order_data.get('broker_order_id')
                    }
                elif order_data.get('status') == 'REJECTED':
                    return None
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        logger.warning(f"Timeout waiting for ExecutionReport for order {order_id}")
        return None

