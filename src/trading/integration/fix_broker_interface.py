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
    """Interface between FIX protocol and trading system.

    This adapter exposes a minimal API to place/cancel orders and tracks
    lightweight in-memory order state. It can optionally notify registered
    listeners when execution reports arrive so higher-level components
    (e.g., order lifecycle/position trackers) can react to updates.
    """
    
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
        self._order_update_listeners = []  # callbacks taking (order_id: str, update: Dict[str, Any])
        
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
        """Handle execution report messages from FIX.

        Expected tags (best-effort):
          - 11 ClOrdID (bytes)
          - 150 ExecType (bytes) [0=New, 1=PartialFill, 2=Fill, 4=Cancelled, 8=Rejected]
          - 32 LastQty (bytes/str)
          - 31 LastPx (bytes/str)
        """
        try:
            order_id = message.get(11).decode() if message.get(11) else None
            exec_type = message.get(150).decode() if message.get(150) else None
            last_qty_raw = message.get(32)
            last_px_raw = message.get(31)
            last_qty = None
            last_px = None
            try:
                if last_qty_raw is not None:
                    last_qty = float(last_qty_raw.decode() if isinstance(last_qty_raw, (bytes, bytearray)) else last_qty_raw)
                if last_px_raw is not None:
                    last_px = float(last_px_raw.decode() if isinstance(last_px_raw, (bytes, bytearray)) else last_px_raw)
            except Exception:
                # Non-fatal; continue without qty/px
                pass

            if not order_id or not exec_type:
                return

            logger.info(f"Execution report for order {order_id}: {exec_type}")

            # Update in-memory order state
            order_state = self.orders.get(order_id, {
                "symbol": None,
                "side": None,
                "quantity": 0.0,
                "status": "UNKNOWN",
                "timestamp": datetime.utcnow(),
                "filled_qty": 0.0,
                "avg_px": None,
            })

            # Map ExecType to status string
            status_map = {
                "0": "ACKNOWLEDGED",
                "1": "PARTIALLY_FILLED",
                "2": "FILLED",
                "4": "CANCELLED",
                "8": "REJECTED",
            }
            new_status = status_map.get(exec_type, order_state.get("status", "UNKNOWN"))
            order_state["status"] = new_status

            if last_qty is not None and last_qty > 0:
                # Update running filled quantity and average price
                prev_filled = float(order_state.get("filled_qty", 0.0) or 0.0)
                prev_avg = order_state.get("avg_px")
                new_filled = prev_filled + float(last_qty)
                if last_px is not None:
                    if prev_avg is None or prev_filled <= 0:
                        new_avg = float(last_px)
                    else:
                        total_value = prev_avg * prev_filled + float(last_px) * float(last_qty)
                        new_avg = total_value / new_filled if new_filled > 0 else prev_avg
                    order_state["avg_px"] = new_avg
                order_state["filled_qty"] = new_filled

            self.orders[order_id] = order_state

            update_payload = {
                "order_id": order_id,
                "exec_type": exec_type,
                "status": order_state.get("status"),
                "filled_qty": order_state.get("filled_qty"),
                "avg_px": order_state.get("avg_px"),
                "symbol": order_state.get("symbol"),
                "side": order_state.get("side"),
                "timestamp": datetime.utcnow(),
            }

            # Emit event for system (if compatible bus provided)
            try:
                if self.event_bus and hasattr(self.event_bus, "emit"):
                    await self.event_bus.emit("order_update", update_payload)
            except Exception as emit_err:
                logger.debug(f"Event bus emit failed: {emit_err}")

            # Notify local listeners
            for callback in list(self._order_update_listeners):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(order_id, update_payload)
                    else:
                        callback(order_id, update_payload)
                except Exception as cb_err:
                    logger.warning(f"Order update listener error: {cb_err}")

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

    # --- Listener registration -------------------------------------------------
    def add_order_update_listener(self, callback):
        """Register a callback to receive order update notifications.

        Callback signature: (order_id: str, update: Dict[str, Any]) -> None | Awaitable
        """
        try:
            self._order_update_listeners.append(callback)
            return True
        except Exception:
            return False

    def remove_order_update_listener(self, callback):
        """Unregister a previously added callback."""
        try:
            if callback in self._order_update_listeners:
                self._order_update_listeners.remove(callback)
                return True
            return False
        except Exception:
            return False
