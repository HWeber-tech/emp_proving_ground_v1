"""
CTrader Broker Interface - Live Trade Execution
Provides integration with cTrader Open API for order placement and management
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from decimal import Decimal

from src.core.events import TradeIntent, ExecutionReport
from src.governance.system_config import config

logger = logging.getLogger(__name__)


class CTraderBrokerInterface:
    """
    Live broker interface that connects to cTrader Open API
    Handles order placement and management
    """
    
    def __init__(self, event_bus, system_config=None):
        """
        Initialize CTrader Broker Interface
        
        Args:
            event_bus: Internal event bus for publishing execution reports
            system_config: System configuration
        """
        self.event_bus = event_bus
        self.config = system_config or config
        self.client: Optional[Any] = None
        self.connected = False
        self.data_organ = None
        
        self._try_import_ctrader()
    
    def _try_import_ctrader(self):
        """Try to import cTrader library gracefully"""
        try:
            from ctrader_open_api import Client, Protobuf, Messages
            from ctrader_open_api.enums import ProtoOAPayloadType
            
            self.Client = Client
            self.Protobuf = Protobuf
            self.Messages = Messages
            self.ProtoOAPayloadType = ProtoOAPayloadType
            self.C_TRADER_AVAILABLE = True
            
        except ImportError:
            self.Client = None
            self.Protobuf = None
            self.Messages = None
            self.ProtoOAPayloadType = None
            self.C_TRADER_AVAILABLE = False
            logger.warning("cTrader Open API library not available. Install with: pip install ctrader-open-api-py")
    
    def set_data_organ(self, data_organ):
        """Set the data organ to share the same client connection"""
        self.data_organ = data_organ
    
    async def place_order(self, intent: TradeIntent) -> None:
        """
        Place an order via cTrader API
        
        Args:
            intent: TradeIntent containing order details
        """
        if not self.C_TRADER_AVAILABLE:
            logger.error("cTrader library not available")
            return
            
        try:
            # Convert volume from lots to cTrader units (hundredths of a lot)
            volume_units = int(intent.quantity * 100)
            
            # Convert price from Decimal to integer (for 5-digit symbols)
            price_int = int(intent.price * 100000) if intent.price else None
            
            # Get symbol ID from symbol name
            symbol_id = None
            if self.data_organ:
                symbol_id = self.data_organ.get_symbol_id(intent.symbol)
            
            if symbol_id is None:
                logger.error(f"Symbol {intent.symbol} not found in cTrader symbols")
                return
            
            # Create new order request
            new_order_req = self.Messages.ProtoOANewOrderReq()
            new_order_req.ctidTraderAccountId = self.config.ctrader_account_id
            new_order_req.symbolId = symbol_id
            new_order_req.orderType = 1  # MARKET order
            new_order_req.tradeSide = 1 if intent.side == "BUY" else 2  # 1=BUY, 2=SELL
            new_order_req.volume = volume_units
            
            if intent.order_type == "LIMIT":
                new_order_req.orderType = 2
                new_order_req.limitPrice = price_int
            elif intent.order_type == "STOP":
                new_order_req.orderType = 3
                new_order_req.stopPrice = price_int
            
            # Send order
            if self.data_organ and self.data_organ.client:
                await self.data_organ.client.send(new_order_req)
                logger.info(f"Order sent to cTrader: {intent}")
            else:
                logger.error("No client connection available")
                
        except Exception as e:
            logger.error(f"Error placing order: {e}", exc_info=True)
    
    async def close_position(self, position_id: str, symbol: str) -> None:
        """
        Close a position via cTrader API
        
        Args:
            position_id: Position ID to close
            symbol: Symbol name
        """
        if not self.C_TRADER_AVAILABLE:
            logger.error("cTrader library not available")
            return
            
        try:
            # Get symbol ID
            symbol_id = None
            if self.data_organ:
                symbol_id = self.data_organ.get_symbol_id(symbol)
            
            if symbol_id is None:
                logger.error(f"Symbol {symbol} not found in cTrader symbols")
                return
            
            # Create close position request
            close_position_req = self.Messages.ProtoOAClosePositionReq()
            close_position_req.ctidTraderAccountId = self.config.ctrader_account_id
            close_position_req.positionId = int(position_id)
            
            # Send close request
            if self.data_organ and self.data_organ.client:
                await self.data_organ.client.send(close_position_req)
                logger.info(f"Position close sent to cTrader: {position_id}")
            else:
                logger.error("No client connection available")
                
        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from cTrader
        
        Returns:
            Dictionary containing account information
        """
        if not self.C_TRADER_AVAILABLE:
            return {"error": "cTrader library not available"}
            
        try:
            # This would typically involve sending a ProtoOAAccountInfoReq
            # For now, return basic info
            return {
                "account_id": self.config.ctrader_account_id,
                "connected": self.connected,
                "available": self.C_TRADER_AVAILABLE
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {"error": str(e)}
    
    async def start(self) -> None:
        """Start the broker interface"""
        if not self.C_TRADER_AVAILABLE:
            logger.error("cTrader library not available")
            return
            
        logger.info("CTrader broker interface ready")
    
    async def stop(self) -> None:
        """Stop the broker interface"""
        logger.info("CTrader broker interface stopped")
