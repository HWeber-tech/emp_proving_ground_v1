"""
CTrader Data Organ - Live Market Data Integration
Provides real-time market data from cTrader Open API
"""

import asyncio
import logging
from typing import Dict, Optional, Any, TYPE_CHECKING
from decimal import Decimal

from src.governance.system_config import config

if TYPE_CHECKING:
    from ctrader_open_api import Client, Protobuf, Messages
    from ctrader_open_api.enums import ProtoOAPayloadType

logger = logging.getLogger(__name__)


class CTraderDataOrgan:
    """
    Live market data organ that connects to cTrader Open API
    Provides real-time price feeds and execution reports
    """
    
    def __init__(self, event_bus, system_config=None):
        """
        Initialize CTrader Data Organ
        
        Args:
            event_bus: Internal event bus for publishing market data
            system_config: System configuration
        """
        self.event_bus = event_bus
        self.config = system_config or config
        self.client: Optional['Client'] = None
        self.connected = False
        self.symbol_mapping: Dict[str, int] = {}  # Maps symbol names to symbol IDs
        self.reverse_mapping: Dict[int, str] = {}  # Maps symbol IDs to symbol names
        
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
    
    def on_message_received(self, message: Any) -> None:
        """Handle incoming messages from cTrader server"""
        if not self.C_TRADER_AVAILABLE:
            return
            
        try:
            ProtoOAPayloadType = self.ProtoOAPayloadType
            
            if message.payloadType == ProtoOAPayloadType.PROTO_OA_SPOT_EVENT:
                self._handle_spot_event(message)
                
            elif message.payloadType == ProtoOAPayloadType.PROTO_OA_ERROR_RES:
                self._handle_error(message)
                
            elif message.payloadType == ProtoOAPayloadType.PROTO_OA_HEARTBEAT_EVENT:
                logger.debug("Heartbeat received from cTrader server")
                
            elif message.payloadType == ProtoOAPayloadType.PROTO_OA_EXECUTION_EVENT:
                self._handle_execution_event(message)
                
            elif message.payloadType == ProtoOAPayloadType.PROTO_OA_SYMBOLS_LIST_RES:
                self._handle_symbols_list(message)
                
        except Exception as e:
            logger.error(f"Error handling cTrader message: {e}", exc_info=True)
    
    def _handle_spot_event(self, message: Any) -> None:
        """Handle price updates"""
        try:
            Messages = self.Messages
            ProtoOASpotEvent = Messages.ProtoOASpotEvent
            
            spot_event = ProtoOASpotEvent()
            spot_event.ParseFromString(message.payload)
            
            symbol_id = spot_event.symbolId
            if symbol_id not in self.reverse_mapping:
                logger.warning(f"Received price update for unknown symbol ID: {symbol_id}")
                return
            
            symbol = self.reverse_mapping[symbol_id]
            
            # Convert integer prices to decimal
            bid_price: Optional[Decimal] = None
            ask_price: Optional[Decimal] = None
            
            if spot_event.HasField("bid"):
                bid_price = Decimal(str(spot_event.bid)) / Decimal('100000')
            if spot_event.HasField("ask"):
                ask_price = Decimal(str(spot_event.ask)) / Decimal('100000')
            
            # Create market understanding event
            from src.core.events import MarketUnderstanding
            market_data = MarketUnderstanding(
                symbol=symbol,
                bid=bid_price,
                ask=ask_price,
                timestamp=spot_event.timestamp,
                source="ctrader"
            )
            
            # Publish to internal event bus
            asyncio.create_task(self.event_bus.publish(market_data))
            
            logger.debug(f"PRICE UPDATE {symbol}: Bid={bid_price}, Ask={ask_price}")
            
        except Exception as e:
            logger.error(f"Error handling spot event: {e}", exc_info=True)
    
    def _handle_execution_event(self, message: Any) -> None:
        """Handle trade execution events"""
        try:
            Messages = self.Messages
            ProtoOAExecutionEvent = Messages.ProtoOAExecutionEvent
            
            execution_event = ProtoOAExecutionEvent()
            execution_event.ParseFromString(message.payload)
            
            # Create execution report
            from src.core.events import ExecutionReport
            execution_report = ExecutionReport(
                order_id=str(execution_event.order.orderId),
                symbol=self.reverse_mapping.get(execution_event.order.symbolName, execution_event.order.symbolName),
                side=execution_event.order.tradeSide,
                volume=Decimal(str(execution_event.order.volume)) / Decimal('100'),
                price=Decimal(str(execution_event.order.price)) / Decimal('100000'),
                timestamp=execution_event.timestamp,
                status="FILLED" if execution_event.order.orderType == 1 else "REJECTED"
            )
            
            # Publish to internal event bus
            asyncio.create_task(self.event_bus.publish(execution_report))
            
            logger.info(f"EXECUTION: {execution_report}")
            
        except Exception as e:
            logger.error(f"Error handling execution event: {e}", exc_info=True)
    
    def _handle_error(self, message: Any) -> None:
        """Handle error messages"""
        try:
            Messages = self.Messages
            ProtoOAErrorRes = Messages.ProtoOAErrorRes
            
            error_res = ProtoOAErrorRes()
            error_res.ParseFromString(message.payload)
            logger.error(f"cTrader Error: {error_res.description} (Code: {error_res.errorCode})")
        except Exception as e:
            logger.error(f"Error handling error response: {e}", exc_info=True)
    
    def _handle_symbols_list(self, message: Any) -> None:
        """Handle symbols list response"""
        try:
            Messages = self.Messages
            ProtoOASymbolsListRes = Messages.ProtoOASymbolsListRes
            
            symbols_res = ProtoOASymbolsListRes()
            symbols_res.ParseFromString(message.payload)
            
            for symbol in symbols_res.symbol:
                self.symbol_mapping[symbol.symbolName] = symbol.symbolId
                self.reverse_mapping[symbol.symbolId] = symbol.symbolName
                
            logger.info(f"Loaded {len(self.symbol_mapping)} symbols from cTrader")
            
        except Exception as e:
            logger.error(f"Error handling symbols list: {e}", exc_info=True)
    
    async def start(self) -> None:
        """Start the cTrader connection and subscribe to market data"""
        if not self.C_TRADER_AVAILABLE:
            logger.error("cTrader library not available")
            return
            
        try:
            # Initialize client
            self.client = self.Client(
                self.config.ctrader_demo_host,
                self.config.ctrader_port,
                on_message=self.on_message_received
            )
            
            await self.client.start()
            self.connected = True
            logger.info("Connected to cTrader server")
            
            # Authenticate application
            app_auth_req = self.Messages.ProtoOAApplicationAuthReq()
            app_auth_req.clientId = self.config.ctrader_client_id
            app_auth_req.clientSecret = self.config.ctrader_client_secret
            await self.client.send(app_auth_req)
            logger.info("Application authentication sent")
            
            # Wait for application auth response
            await asyncio.sleep(2)
            
            # Authenticate account
            account_auth_req = self.Messages.ProtoOAAccountAuthReq()
            account_auth_req.ctidTraderAccountId = self.config.ctrader_account_id
            account_auth_req.accessToken = self.config.ctrader_access_token
            await self.client.send(account_auth_req)
            logger.info("Account authentication sent")
            
            # Wait for account auth response
            await asyncio.sleep(2)
            
            # Get symbols list
            symbols_req = self.Messages.ProtoOASymbolsListReq()
            symbols_req.ctidTraderAccountId = self.config.ctrader_account_id
            await self.client.send(symbols_req)
            logger.info("Requested symbols list")
            
            # Wait for symbols response
            await asyncio.sleep(2)
            
            # Subscribe to price feeds for configured symbols
            await self._subscribe_to_symbols()
            
            logger.info("CTrader data organ started successfully")
            
        except Exception as e:
            logger.error(f"Error starting cTrader data organ: {e}", exc_info=True)
            self.connected = False
    
    async def _subscribe_to_symbols(self) -> None:
        """Subscribe to price feeds for configured symbols"""
        if not self.symbol_mapping:
            logger.warning("No symbols loaded yet, waiting...")
            await asyncio.sleep(5)
        
        symbols_to_subscribe = []
        for symbol in self.config.default_symbols:
            if symbol in self.symbol_mapping:
                symbols_to_subscribe.append(self.symbol_mapping[symbol])
            else:
                logger.warning(f"Symbol {symbol} not found in cTrader symbols")
        
        if symbols_to_subscribe:
            subscribe_req = self.Messages.ProtoOASubscribeSpotsReq()
            subscribe_req.ctidTraderAccountId = self.config.ctrader_account_id
            subscribe_req.symbolId.extend(symbols_to_subscribe)
            await self.client.send(subscribe_req)
            logger.info(f"Subscribed to {len(symbols_to_subscribe)} symbols")
    
    async def stop(self) -> None:
        """Stop the cTrader connection"""
        if self.client:
            await self.client.stop()
            self.connected = False
            logger.info("CTrader data organ stopped")
    
    def get_symbol_id(self, symbol: str) -> Optional[int]:
        """Get cTrader symbol ID for a symbol name"""
        return self.symbol_mapping.get(symbol)
    
    def get_symbol_name(self, symbol_id: int) -> Optional[str]:
        """Get symbol name for a cTrader symbol ID"""
        return self.reverse_mapping.get(symbol_id)
