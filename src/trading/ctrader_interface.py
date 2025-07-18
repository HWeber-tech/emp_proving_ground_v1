#!/usr/bin/env python3
"""
IC Markets cTrader OpenAPI Trading Interface

This module implements a complete trading interface for IC Markets cTrader
using the official OpenAPI with proper authentication, connection management,
and trading operations.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from ctrader_open_api import Client, Protobuf, Messages
from ctrader_open_api.enums import ProtoOAPayloadType

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode enumeration."""
    DEMO = "demo"
    LIVE = "live"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradingConfig:
    """Trading configuration."""
    client_id: str
    client_secret: str
    access_token: str
    refresh_token: str
    account_id: int
    mode: TradingMode = TradingMode.DEMO
    host: str = "demo.ctraderapi.com"
    port: int = 5035
    max_retries: int = 3
    heartbeat_interval: int = 10


@dataclass
class MarketData:
    """Market data structure."""
    symbol_id: int
    symbol_name: str
    bid: float
    ask: float
    timestamp: datetime
    digits: int = 5


@dataclass
class Order:
    """Order structure."""
    order_id: str
    symbol_id: int
    order_type: OrderType
    side: OrderSide
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: str = "pending"
    timestamp: datetime = None


@dataclass
class Position:
    """Position structure."""
    position_id: str
    symbol_id: int
    side: OrderSide
    volume: float
    entry_price: float
    current_price: float
    profit_loss: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = None


class CTraderInterface:
    """
    IC Markets cTrader OpenAPI Trading Interface.
    
    This class handles all trading operations including:
    - Authentication and connection management
    - Market data subscription
    - Order placement and management
    - Position tracking
    - Real-time updates
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize the cTrader interface.
        
        Args:
            config: Trading configuration
        """
        self.config = config
        self.client = None
        self.connected = False
        self.authenticated = False
        self.symbols_cache = {}
        self.market_data = {}
        self.orders = {}
        self.positions = {}
        self.callbacks = {
            'price_update': [],
            'order_update': [],
            'position_update': [],
            'error': []
        }
        
        # Set host based on mode
        if config.mode == TradingMode.LIVE:
            self.config.host = "live.ctraderapi.com"
        
        logger.info(f"cTrader interface initialized for {config.mode.value} mode")
    
    async def connect(self) -> bool:
        """
        Connect to cTrader and authenticate.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to cTrader...")
            
            # Create client
            self.client = Client(self.config.host, self.config.port, on_message=self._on_message_received)
            
            # Start connection
            await self.client.start()
            logger.info("Client connected, authenticating...")
            
            # Authenticate application
            if not await self._authenticate_application():
                logger.error("Application authentication failed")
                return False
            
            # Authenticate account
            if not await self._authenticate_account():
                logger.error("Account authentication failed")
                return False
            
            # Load symbols
            if not await self._load_symbols():
                logger.error("Failed to load symbols")
                return False
            
            self.connected = True
            self.authenticated = True
            
            logger.info("Successfully connected and authenticated to cTrader")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from cTrader."""
        if self.client:
            await self.client.stop()
            self.connected = False
            self.authenticated = False
            logger.info("Disconnected from cTrader")
    
    async def _authenticate_application(self) -> bool:
        """Authenticate the application."""
        try:
            app_auth_req = Messages.ProtoOAApplicationAuthReq()
            app_auth_req.clientId = self.config.client_id
            app_auth_req.clientSecret = self.config.client_secret
            
            await self.client.send(app_auth_req)
            logger.info("Application auth request sent")
            
            # Wait for response
            await asyncio.sleep(2)
            return True
            
        except Exception as e:
            logger.error(f"Application authentication error: {e}")
            return False
    
    async def _authenticate_account(self) -> bool:
        """Authenticate the trading account."""
        try:
            account_auth_req = Messages.ProtoOAAccountAuthReq()
            account_auth_req.ctidTraderAccountId = self.config.account_id
            account_auth_req.accessToken = self.config.access_token
            
            await self.client.send(account_auth_req)
            logger.info("Account auth request sent")
            
            # Wait for response
            await asyncio.sleep(2)
            return True
            
        except Exception as e:
            logger.error(f"Account authentication error: {e}")
            return False
    
    async def _load_symbols(self) -> bool:
        """Load available symbols."""
        try:
            symbols_req = Messages.ProtoOASymbolsListReq()
            symbols_req.ctidTraderAccountId = self.config.account_id
            
            await self.client.send(symbols_req)
            logger.info("Symbols list request sent")
            
            # Wait for response
            await asyncio.sleep(2)
            return True
            
        except Exception as e:
            logger.error(f"Symbols loading error: {e}")
            return False
    
    async def subscribe_to_symbol(self, symbol_name: str) -> bool:
        """
        Subscribe to market data for a symbol.
        
        Args:
            symbol_name: Symbol name (e.g., "EURUSD")
            
        Returns:
            True if subscription successful, False otherwise
        """
        try:
            # Find symbol ID
            symbol_id = self._get_symbol_id(symbol_name)
            if symbol_id is None:
                logger.error(f"Symbol {symbol_name} not found")
                return False
            
            # Subscribe to spot prices
            subscribe_req = Messages.ProtoOASubscribeSpotsReq()
            subscribe_req.ctidTraderAccountId = self.config.account_id
            subscribe_req.symbolId.append(symbol_id)
            
            await self.client.send(subscribe_req)
            logger.info(f"Subscribed to {symbol_name} (ID: {symbol_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"Symbol subscription error: {e}")
            return False
    
    async def place_order(self, symbol_name: str, order_type: OrderType, side: OrderSide, 
                         volume: float, price: Optional[float] = None, 
                         stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Optional[str]:
        """
        Place a trading order.
        
        Args:
            symbol_name: Symbol name
            order_type: Type of order
            side: Buy or sell
            volume: Order volume in lots
            price: Order price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            symbol_id = self._get_symbol_id(symbol_name)
            if symbol_id is None:
                logger.error(f"Symbol {symbol_name} not found")
                return None
            
            # Convert volume to integer (hundredths of lots)
            volume_int = int(volume * 100)
            
            # Create order request
            if order_type == OrderType.MARKET:
                order_req = Messages.ProtoOANewOrderReq()
                order_req.ctidTraderAccountId = self.config.account_id
                order_req.symbolId = symbol_id
                order_req.orderType = self._get_proto_order_type(order_type)
                order_req.tradeSide = self._get_proto_trade_side(side)
                order_req.volume = volume_int
                
                if stop_loss:
                    order_req.stopLoss = int(stop_loss * (10 ** self.symbols_cache[symbol_id]['digits']))
                if take_profit:
                    order_req.takeProfit = int(take_profit * (10 ** self.symbols_cache[symbol_id]['digits']))
                
            else:  # Limit/Stop orders
                if price is None:
                    logger.error("Price required for limit/stop orders")
                    return None
                
                order_req = Messages.ProtoOANewOrderReq()
                order_req.ctidTraderAccountId = self.config.account_id
                order_req.symbolId = symbol_id
                order_req.orderType = self._get_proto_order_type(order_type)
                order_req.tradeSide = self._get_proto_trade_side(side)
                order_req.volume = volume_int
                order_req.price = int(price * (10 ** self.symbols_cache[symbol_id]['digits']))
                
                if stop_loss:
                    order_req.stopLoss = int(stop_loss * (10 ** self.symbols_cache[symbol_id]['digits']))
                if take_profit:
                    order_req.takeProfit = int(take_profit * (10 ** self.symbols_cache[symbol_id]['digits']))
            
            await self.client.send(order_req)
            logger.info(f"Order request sent: {side.value} {volume} {symbol_name}")
            
            # Return a temporary order ID (will be updated when we get the response)
            temp_order_id = f"temp_{int(time.time())}"
            return temp_order_id
            
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation request sent, False otherwise
        """
        try:
            # Find the order
            if order_id not in self.orders:
                logger.error(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            
            # Create cancel request
            cancel_req = Messages.ProtoOACancelOrderReq()
            cancel_req.ctidTraderAccountId = self.config.account_id
            cancel_req.orderId = int(order_id)
            
            await self.client.send(cancel_req)
            logger.info(f"Cancel request sent for order {order_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False
    
    async def close_position(self, position_id: str) -> bool:
        """
        Close an existing position.
        
        Args:
            position_id: Position ID to close
            
        Returns:
            True if close request sent, False otherwise
        """
        try:
            # Find the position
            if position_id not in self.positions:
                logger.error(f"Position {position_id} not found")
                return False
            
            position = self.positions[position_id]
            
            # Create close request
            close_req = Messages.ProtoOAClosePositionReq()
            close_req.ctidTraderAccountId = self.config.account_id
            close_req.positionId = int(position_id)
            close_req.volume = int(position.volume * 100)  # Convert to integer
            
            await self.client.send(close_req)
            logger.info(f"Close request sent for position {position_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Position close error: {e}")
            return False
    
    def get_market_data(self, symbol_name: str) -> Optional[MarketData]:
        """
        Get current market data for a symbol.
        
        Args:
            symbol_name: Symbol name
            
        Returns:
            Market data or None if not available
        """
        symbol_id = self._get_symbol_id(symbol_name)
        if symbol_id and symbol_id in self.market_data:
            return self.market_data[symbol_id]
        return None
    
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())
    
    def get_orders(self) -> List[Order]:
        """Get all pending orders."""
        return list(self.orders.values())
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        Add a callback for events.
        
        Args:
            event_type: Type of event ('price_update', 'order_update', 'position_update', 'error')
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def _on_message_received(self, message: Protobuf):
        """Handle incoming messages from cTrader."""
        try:
            if message.payloadType == ProtoOAPayloadType.PROTO_OA_SPOT_EVENT:
                self._handle_spot_event(message)
            elif message.payloadType == ProtoOAPayloadType.PROTO_OA_EXECUTION_EVENT:
                self._handle_execution_event(message)
            elif message.payloadType == ProtoOAPayloadType.PROTO_OA_ERROR_RES:
                self._handle_error_event(message)
            elif message.payloadType == ProtoOAPayloadType.PROTO_OA_SYMBOLS_LIST_RES:
                self._handle_symbols_list(message)
            elif message.payloadType == ProtoOAPayloadType.PROTO_OA_HEARTBEAT_EVENT:
                # Heartbeat handled by library
                pass
            else:
                logger.debug(f"Unhandled message type: {message.payloadType}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _handle_spot_event(self, message: Protobuf):
        """Handle spot price events."""
        try:
            spot_event = Messages.ProtoOASpotEvent()
            spot_event.ParseFromString(message.payload)
            
            symbol_id = spot_event.symbolId
            symbol_name = self._get_symbol_name(symbol_id)
            
            if symbol_name:
                # Convert prices from integers
                digits = self.symbols_cache.get(symbol_id, {}).get('digits', 5)
                divisor = 10 ** digits
                
                bid = spot_event.bid / divisor if spot_event.HasField("bid") else None
                ask = spot_event.ask / divisor if spot_event.HasField("ask") else None
                
                if bid and ask:
                    market_data = MarketData(
                        symbol_id=symbol_id,
                        symbol_name=symbol_name,
                        bid=bid,
                        ask=ask,
                        timestamp=datetime.now(),
                        digits=digits
                    )
                    
                    self.market_data[symbol_id] = market_data
                    
                    # Trigger callbacks
                    for callback in self.callbacks['price_update']:
                        try:
                            callback(market_data)
                        except Exception as e:
                            logger.error(f"Price update callback error: {e}")
                    
                    logger.debug(f"Price update: {symbol_name} Bid={bid}, Ask={ask}")
                    
        except Exception as e:
            logger.error(f"Error handling spot event: {e}")
    
    def _handle_execution_event(self, message: Protobuf):
        """Handle execution events (order fills, position updates)."""
        try:
            execution_event = Messages.ProtoOAExecutionEvent()
            execution_event.ParseFromString(message.payload)
            
            # Handle order execution
            if execution_event.HasField("orderId"):
                order_id = str(execution_event.orderId)
                # Update order status
                if order_id in self.orders:
                    self.orders[order_id].status = "filled"
                    
                    # Trigger callbacks
                    for callback in self.callbacks['order_update']:
                        try:
                            callback(self.orders[order_id])
                        except Exception as e:
                            logger.error(f"Order update callback error: {e}")
            
            # Handle position updates
            if execution_event.HasField("positionId"):
                position_id = str(execution_event.positionId)
                # Update position data
                # This would need more detailed implementation based on position events
                
        except Exception as e:
            logger.error(f"Error handling execution event: {e}")
    
    def _handle_error_event(self, message: Protobuf):
        """Handle error events."""
        try:
            error_res = Messages.ProtoOAErrorRes()
            error_res.ParseFromString(message.payload)
            
            error_msg = f"cTrader Error: {error_res.description} (Code: {error_res.errorCode})"
            logger.error(error_msg)
            
            # Trigger callbacks
            for callback in self.callbacks['error']:
                try:
                    callback(error_msg)
                except Exception as e:
                    logger.error(f"Error callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling error event: {e}")
    
    def _handle_symbols_list(self, message: Protobuf):
        """Handle symbols list response."""
        try:
            symbols_res = Messages.ProtoOASymbolsListRes()
            symbols_res.ParseFromString(message.payload)
            
            for symbol in symbols_res.symbol:
                self.symbols_cache[symbol.symbolId] = {
                    'name': symbol.symbolName,
                    'digits': symbol.digits,
                    'pip_position': symbol.pipPosition
                }
            
            logger.info(f"Loaded {len(self.symbols_cache)} symbols")
            
        except Exception as e:
            logger.error(f"Error handling symbols list: {e}")
    
    def _get_symbol_id(self, symbol_name: str) -> Optional[int]:
        """Get symbol ID by name."""
        for symbol_id, symbol_data in self.symbols_cache.items():
            if symbol_data['name'] == symbol_name:
                return symbol_id
        return None
    
    def _get_symbol_name(self, symbol_id: int) -> Optional[str]:
        """Get symbol name by ID."""
        if symbol_id in self.symbols_cache:
            return self.symbols_cache[symbol_id]['name']
        return None
    
    def _get_proto_order_type(self, order_type: OrderType) -> int:
        """Convert order type to protocol buffer enum."""
        mapping = {
            OrderType.MARKET: 1,  # MARKET
            OrderType.LIMIT: 2,   # LIMIT
            OrderType.STOP: 3,    # STOP
            OrderType.STOP_LIMIT: 4  # STOP_LIMIT
        }
        return mapping.get(order_type, 1)
    
    def _get_proto_trade_side(self, side: OrderSide) -> int:
        """Convert trade side to protocol buffer enum."""
        mapping = {
            OrderSide.BUY: 1,   # BUY
            OrderSide.SELL: 2   # SELL
        }
        return mapping.get(side, 1)


class TokenManager:
    """Manages OAuth tokens for cTrader API."""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = "https://connect.icmarkets.com/api/v2/oauth/token"
    
    def get_authorization_url(self, redirect_uri: str = "http://localhost/") -> str:
        """Generate authorization URL."""
        return f"https://connect.icmarkets.com/oauth/authorize?client_id={self.client_id}&redirect_uri={redirect_uri}&scope=trading"
    
    def exchange_code_for_token(self, auth_code: str, redirect_uri: str = "http://localhost/") -> Optional[Dict]:
        """Exchange authorization code for access token."""
        try:
            response = requests.post(
                self.token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": auth_code,
                    "redirect_uri": redirect_uri,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[Dict]:
        """Refresh access token using refresh token."""
        try:
            response = requests.post(
                self.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None
    
    def get_trading_accounts(self, access_token: str) -> Optional[List[Dict]]:
        """Get available trading accounts."""
        try:
            response = requests.get(
                "https://connect.icmarkets.com/api/v2/tradingaccounts",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if response.status_code == 200:
                return response.json().get('data', [])
            else:
                logger.error(f"Failed to get trading accounts: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Get trading accounts error: {e}")
            return None


def main():
    """Test the cTrader interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test cTrader interface")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to subscribe to")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    config = TradingConfig(**config_data)
    
    # Create interface
    interface = CTraderInterface(config)
    
    # Add callbacks
    def on_price_update(market_data):
        print(f"Price Update: {market_data.symbol_name} Bid={market_data.bid}, Ask={market_data.ask}")
    
    def on_error(error_msg):
        print(f"Error: {error_msg}")
    
    interface.add_callback('price_update', on_price_update)
    interface.add_callback('error', on_error)
    
    # Run test
    async def test():
        if await interface.connect():
            await interface.subscribe_to_symbol(args.symbol)
            print(f"Subscribed to {args.symbol}. Waiting for price updates...")
            await asyncio.sleep(30)
        await interface.disconnect()
    
    asyncio.run(test())


if __name__ == "__main__":
    main() 