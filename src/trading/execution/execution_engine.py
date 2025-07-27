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

from src.core.interfaces import IExecutionEngine
from src.core.exceptions import ExecutionException
from src.trading.models.order import Order, OrderStatus
from src.trading.models.position import Position

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
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            order.status = OrderStatus.PENDING
            
            # Simulate execution (in real implementation, this would use FIX API)
            await self._simulate_execution(order)
            
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
    
    async def _simulate_execution(self, order: Order) -> None:
        """Simulate order execution (placeholder for FIX API integration)."""
        # In real implementation, this would use FIX API
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Simulate successful execution at current market price
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = order.price or 100.0  # Placeholder price
        order.filled_at = datetime.now()
    
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
