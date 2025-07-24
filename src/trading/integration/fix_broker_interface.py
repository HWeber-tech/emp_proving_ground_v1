"""
FIX Broker Interface - Institutional-grade trade execution
Implements the predator's new claws for IC Markets cTrader FIX protocol
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from decimal import Decimal
import simplefix

from src.core.events import ExecutionReportEvent, OrderStatus
from src.operational.event_bus import EventBus

log = logging.getLogger(__name__)


class FIXBrokerInterface:
    """
    Institutional-grade FIX broker interface for IC Markets cTrader
    Manages trade execution via FIX 4.4 protocol
    """
    
    def __init__(self, event_bus: EventBus, trade_session, session_id: str):
        """
        Initialize the FIX broker interface
        
        Args:
            event_bus: NATS event bus for internal communication
            trade_session: FIX trade session initiator
            session_id: FIX session identifier
        """
        self.event_bus = event_bus
        self.trade_session = trade_session
        self.session_id = session_id
        self.message_queue = asyncio.Queue()
        self.order_tracker = {}  # Track active orders
        self.symbol_mapping = {}  # Map internal symbols to FIX symbols
        
        log.info("FIXBrokerInterface initialized for institutional trading")
    
    async def start(self):
        """Start the message processing loop"""
        log.info("Starting FIX broker message processing")
        asyncio.create_task(self._message_processing_loop())
    
    async def stop(self):
        """Stop the message processing loop"""
        log.info("Stopping FIX broker message processing")
        await self.message_queue.put(None)  # Sentinel to stop loop
    
    def place_order(self, trade_intent: Dict[str, Any]) -> str:
        """
        Place a new order via FIX protocol
        
        Args:
            trade_intent: Trade intent with order details
            
        Returns:
            Order ID for tracking
        """
        try:
            # Generate unique order ID
            order_id = self._generate_order_id()
            
            # Create NewOrderSingle message
            order_msg = self._create_new_order_single(trade_intent, order_id)
            
            # Send order via FIX
            self._send_fix_message(order_msg)
            
            # Track order
            self.order_tracker[order_id] = {
                'intent': trade_intent,
                'status': 'PENDING',
                'created_at': datetime.utcnow()
            }
            
            log.info(f"Order placed: {order_id} - {trade_intent}")
            return order_id
            
        except Exception as e:
            log.error(f"Failed to place order: {e}")
            raise
    
    def modify_order(self, order_id: str, new_params: Dict[str, Any]) -> str:
        """
        Modify an existing order via OrderCancelReplaceRequest
        
        Args:
            order_id: Original order ID
            new_params: New order parameters
            
        Returns:
            New order ID for the modified order
        """
        try:
            if order_id not in self.order_tracker:
                raise ValueError(f"Order {order_id} not found")
            
            # Generate new order ID for modification
            new_order_id = self._generate_order_id()
            
            # Create OrderCancelReplaceRequest
            modify_msg = self._create_order_cancel_replace(order_id, new_order_id, new_params)
            
            # Send modification via FIX
            self._send_fix_message(modify_msg)
            
            # Track modification
            self.order_tracker[new_order_id] = {
                'original_order': order_id,
                'modification': new_params,
                'status': 'MODIFY_PENDING',
                'created_at': datetime.utcnow()
            }
            
            log.info(f"Order modification requested: {order_id} -> {new_order_id}")
            return new_order_id
            
        except Exception as e:
            log.error(f"Failed to modify order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> str:
        """
        Cancel an existing order via OrderCancelRequest
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancel request ID
        """
        try:
            if order_id not in self.order_tracker:
                raise ValueError(f"Order {order_id} not found")
            
            # Generate cancel request ID
            cancel_id = self._generate_order_id(prefix="CANCEL")
            
            # Create OrderCancelRequest
            cancel_msg = self._create_order_cancel_request(order_id, cancel_id)
            
            # Send cancellation via FIX
            self._send_fix_message(cancel_msg)
            
            # Track cancellation
            self.order_tracker[order_id]['status'] = 'CANCEL_PENDING'
            self.order_tracker[order_id]['cancel_id'] = cancel_id
            
            log.info(f"Order cancellation requested: {order_id}")
            return cancel_id
            
        except Exception as e:
            log.error(f"Failed to cancel order: {e}")
            raise
    
    def _create_new_order_single(self, trade_intent: Dict[str, Any], order_id: str) -> simplefix.FixMessage:
        """Create NewOrderSingle FIX message"""
        msg = simplefix.FixMessage()
        
        # Standard FIX headers
        msg.append_pair(35, "D")  # NewOrderSingle
        msg.append_pair(11, order_id)  # ClOrdID
        msg.append_pair(21, "1")  # HandlInst = Automated
        msg.append_pair(55, trade_intent['symbol'])  # Symbol
        msg.append_pair(54, "1" if trade_intent['side'] == "BUY" else "2")  # Side
        msg.append_pair(38, str(trade_intent['quantity']))  # OrderQty
        msg.append_pair(40, str(trade_intent['order_type']))  # OrdType
        
        # Order type specific fields
        if trade_intent['order_type'] == "2":  # Limit
            msg.append_pair(44, str(trade_intent['price']))  # Price
        elif trade_intent['order_type'] == "4":  # Stop
            msg.append_pair(99, str(trade_intent['stop_price']))  # StopPx
        
        # Time in force
        if 'time_in_force' in trade_intent:
            msg.append_pair(59, str(trade_intent['time_in_force']))
        
        # Account
        if 'account' in trade_intent:
            msg.append_pair(1, str(trade_intent['account']))
        
        # Transact time
        msg.append_pair(60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))
        
        return msg
    
    def _create_order_cancel_replace(self, orig_order_id: str, new_order_id: str, new_params: Dict[str, Any]) -> simplefix.FixMessage:
        """Create OrderCancelReplaceRequest FIX message"""
        msg = simplefix.FixMessage()
        
        msg.append_pair(35, "G")  # OrderCancelReplaceRequest
        msg.append_pair(11, new_order_id)  # ClOrdID
        msg.append_pair(41, orig_order_id)  # OrigClOrdID
        
        # Update fields
        if 'quantity' in new_params:
            msg.append_pair(38, str(new_params['quantity']))
        if 'price' in new_params:
            msg.append_pair(44, str(new_params['price']))
        if 'stop_price' in new_params:
            msg.append_pair(99, str(new_params['stop_price']))
        
        msg.append_pair(60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))
        
        return msg
    
    def _create_order_cancel_request(self, order_id: str, cancel_id: str) -> simplefix.FixMessage:
        """Create OrderCancelRequest FIX message"""
        msg = simplefix.FixMessage()
        
        msg.append_pair(35, "F")  # OrderCancelRequest
        msg.append_pair(11, cancel_id)  # ClOrdID
        msg.append_pair(41, order_id)  # OrigClOrdID
        
        msg.append_pair(60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))
        
        return msg
    
    def _send_fix_message(self, message: simplefix.FixMessage):
        """Send FIX message via trade session"""
        try:
            # Convert to bytes and send
            message_data = message.encode()
            # In a real implementation, this would use the FIX session
            log.debug(f"Sending FIX message: {message_data}")
            
            # For now, log the message (actual FIX sending will be implemented)
            log.info(f"FIX message prepared for sending: {message}")
            
        except Exception as e:
            log.error(f"Failed to send FIX message: {e}")
            raise
    
    def _generate_order_id(self, prefix: str = "ORDER") -> str:
        """Generate unique order ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}-{timestamp}-{unique_id}"
    
    async def _message_processing_loop(self):
        """Async message processing loop for thread-safe FIX message handling"""
        log.info("Starting FIX message processing loop")
        
        while True:
            try:
                message = await self.message_queue.get()
                if message is None:  # Sentinel to stop loop
                    break
                
                await self._process_fix_message(message)
                
            except Exception as e:
                log.error(f"Error in message processing loop: {e}")
    
    async def _process_fix_message(self, message: simplefix.FixMessage):
        """Process incoming FIX messages"""
        try:
            msg_type = message.get(35)
            
            if msg_type == b"8":  # ExecutionReport
                await self._process_execution_report(message)
            elif msg_type == b"9":  # OrderCancelReject
                await self._process_order_cancel_reject(message)
            elif msg_type == b"3":  # Reject
                await self._process_reject(message)
                
        except Exception as e:
            log.error(f"Error processing FIX message: {e}")
    
    async def _process_execution_report(self, message: simplefix.FixMessage):
        """Process ExecutionReport message"""
        try:
            # Parse execution report
            exec_report = self._parse_execution_report(message)
            
            # Update order tracker
            order_id = exec_report.get('cl_ord_id')
            if order_id in self.order_tracker:
                self.order_tracker[order_id]['status'] = exec_report['status']
                self.order_tracker[order_id]['last_update'] = datetime.utcnow()
            
            # Create internal event
            event = ExecutionReportEvent(
                order_id=exec_report['cl_ord_id'],
                symbol=exec_report['symbol'],
                side=exec_report['side'],
                quantity=exec_report['quantity'],
                filled_quantity=exec_report['filled_quantity'],
                remaining_quantity=exec_report['remaining_quantity'],
                price=exec_report['price'],
                filled_price=exec_report['filled_price'],
                status=exec_report['status'],
                timestamp=exec_report['timestamp']
            )
            
            # Publish to event bus
            await self.event_bus.publish('execution_report', event)
            log.info(f"Execution report processed: {event}")
            
        except Exception as e:
            log.error(f"Error processing execution report: {e}")
    
    async def _process_order_cancel_reject(self, message: simplefix.FixMessage):
        """Process OrderCancelReject message"""
        try:
            reject_data = {
                'cl_ord_id': message.get(11, b'').decode(),
                'orig_cl_ord_id': message.get(41, b'').decode(),
                'ord_status': message.get(39, b'').decode(),
                'text': message.get(58, b'').decode()
            }
            
            log.warning(f"Order cancel rejected: {reject_data}")
            
            # Update order tracker
            order_id = reject_data['orig_cl_ord_id']
            if order_id in self.order_tracker:
                self.order_tracker[order_id]['status'] = 'CANCEL_REJECTED'
            
        except Exception as e:
            log.error(f"Error processing order cancel reject: {e}")
    
    async def _process_reject(self, message: simplefix.FixMessage):
        """Process Reject message"""
        try:
            reject_data = {
                'ref_seq_num': message.get(45, b'').decode(),
                'text': message.get(58, b'').decode()
            }
            
            log.error(f"Order rejected: {reject_data}")
            
        except Exception as e:
            log.error(f"Error processing reject: {e}")
    
    def _parse_execution_report(self, message: simplefix.FixMessage) -> Dict[str, Any]:
        """Parse ExecutionReport message into clean data structure"""
        return {
            'cl_ord_id': message.get(11, b'').decode(),
            'order_id': message.get(37, b'').decode(),
            'exec_id': message.get(17, b'').decode(),
            'symbol': message.get(55, b'').decode(),
            'side': 'BUY' if message.get(54, b'').decode() == '1' else 'SELL',
            'quantity': Decimal(message.get(38, b'0').decode()),
            'filled_quantity': Decimal(message.get(14, b'0').decode()),
            'remaining_quantity': Decimal(message.get(151, b'0').decode()),
            'price': Decimal(message.get(44, b'0').decode()),
            'filled_price': Decimal(message.get(31, b'0').decode()),
            'status': self._map_fix_status(message.get(39, b'').decode()),
            'timestamp': datetime.utcnow()
        }
    
    def _map_fix_status(self, fix_status: str) -> OrderStatus:
        """Map FIX status codes to internal OrderStatus"""
        status_map = {
            '0': OrderStatus.NEW,
            '1': OrderStatus.PARTIALLY_FILLED,
            '2': OrderStatus.FILLED,
            '3': OrderStatus.DONE_FOR_DAY,
            '4': OrderStatus.CANCELED,
            '5': OrderStatus.REPLACED,
            '6': OrderStatus.PENDING_CANCEL,
            '7': OrderStatus.STOPPED,
            '8': OrderStatus.REJECTED
        }
        return status_map.get(fix_status, OrderStatus.UNKNOWN)
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get current order status"""
        return self.order_tracker.get(order_id)
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get all active orders"""
        return [
            {'order_id': oid, **data} 
            for oid, data in self.order_tracker.items()
            if data['status'] in ['PENDING', 'NEW', 'PARTIALLY_FILLED', 'MODIFY_PENDING', 'CANCEL_PENDING']
        ]
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """Get order history"""
        return [
            {'order_id': oid, **data} 
            for oid, data in self.order_tracker.items()
        ]
