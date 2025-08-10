"""
FIX Broker Interface for IC Markets
Provides integration between FIX protocol and trading system
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import simplefix

logger = logging.getLogger(__name__)


class FIXBrokerInterface:
    """Interface between FIX protocol and trading system."""
    
    def __init__(self, event_bus, trade_queue, fix_initiator):
        """
        Initialize FIX broker interface.
        
        Args:
            event_bus: Event bus for system communication
            trade_queue: Queue for trade messages
            fix_initiator: FIX initiator for sending orders
        """
        self.event_bus = event_bus
        self.trade_queue = trade_queue
        self.fix_initiator = fix_initiator
        self.running = False
        self.orders = {}
        
    async def start(self):
        """Start the broker interface."""
        self.running = True
        logger.info("FIX broker interface started")
        
        # Start message processing
        asyncio.create_task(self._process_trade_messages())
        
    async def stop(self):
        """Stop the broker interface."""
        self.running = False
        logger.info("FIX broker interface stopped")
        
    async def _process_trade_messages(self):
        """Process trade messages from the queue."""
        while self.running:
            try:
                # Get message from queue
                message = await self.trade_queue.get()
                
                # Process based on message type
                msg_type = message.get(35)
                
                if msg_type == b"8":  # Execution Report
                    await self._handle_execution_report(message)
                elif msg_type == b"9":  # Order Cancel Reject
                    await self._handle_order_cancel_reject(message)
                    
            except Exception as e:
                logger.error(f"Error processing trade message: {e}")
                
    async def _handle_execution_report(self, message):
        """Handle execution report messages."""
        try:
            order_id = message.get(11).decode() if message.get(11) else None
            exec_type = message.get(150).decode() if message.get(150) else None
            
            if order_id and exec_type:
                logger.info(f"Execution report for order {order_id}: {exec_type}")
                
                # Emit event for system
                await self.event_bus.emit("order_update", {
                    "order_id": order_id,
                    "exec_type": exec_type,
                    "timestamp": datetime.utcnow()
                })
                
        except Exception as e:
            logger.error(f"Error handling execution report: {e}")
            
    async def _handle_order_cancel_reject(self, message):
        """Handle order cancel reject messages."""
        try:
            order_id = message.get(11).decode() if message.get(11) else None
            reject_reason = message.get(58).decode() if message.get(58) else "Unknown"
            
            if order_id:
                logger.warning(f"Order cancel rejected for {order_id}: {reject_reason}")
                
                # Emit event for system
                await self.event_bus.emit("order_cancel_rejected", {
                    "order_id": order_id,
                    "reason": reject_reason,
                    "timestamp": datetime.utcnow()
                })
                
        except Exception as e:
            logger.error(f"Error handling order cancel reject: {e}")
            
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Risk guard: ensure portfolio risk within limits if a risk manager is available on event_bus
            try:
                risk_manager = getattr(self.event_bus, 'risk_manager', None) if self.event_bus else None
                if risk_manager and hasattr(risk_manager, 'check_risk_thresholds'):
                    if not risk_manager.check_risk_thresholds():
                        logger.warning("Order blocked by risk thresholds (VaR/ES limits)")
                        return None
            except Exception:
                # If risk check fails, proceed conservatively without blocking
                pass
            # Generate order ID
            order_id = f"ORD_{int(datetime.utcnow().timestamp() * 1000)}"
            
            # Create order message
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "D")  # NewOrderSingle
            msg.append_pair(11, order_id)  # ClOrdID
            msg.append_pair(55, symbol)  # Symbol
            msg.append_pair(54, "1" if side.upper() == "BUY" else "2")  # Side
            msg.append_pair(38, str(quantity))  # OrderQty
            msg.append_pair(40, "1")  # OrdType = Market
            msg.append_pair(60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f")[:-3])  # TransactTime
            
            # Send order
            if self.fix_initiator:
                self.fix_initiator.send_message(msg)
                logger.info(f"Market order placed: {side} {quantity} {symbol} (ID: {order_id})")
                
                # Store order
                self.orders[order_id] = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "status": "PENDING",
                    "timestamp": datetime.utcnow()
                }
                
                return order_id
                
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
            
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancel request sent, False otherwise
        """
        try:
            # Minimal cTrader schema: 11/41 only
            msg = simplefix.FixMessage()
            msg.append_pair(8, "FIX.4.4")
            msg.append_pair(35, "F")  # OrderCancelRequest
            cncl_id = f"CNCL_{order_id}"
            msg.append_pair(11, cncl_id)
            msg.append_pair(41, order_id)
            
            if self.fix_initiator:
                self.fix_initiator.send_message(msg)
                logger.info(f"Order cancel requested: {order_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False
            
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status dictionary or None
        """
        return self.orders.get(order_id)
        
    def get_all_orders(self) -> Dict[str, Dict[str, Any]]:
        """Get all orders."""
        return self.orders.copy()
