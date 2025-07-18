"""
Real IC Markets cTrader OpenAPI Trading Interface

This module provides a real implementation of the cTrader OpenAPI
for live trading with IC Markets.

Features:
- OAuth 2.0 authentication
- Real market data subscription
- Live order placement and execution
- Real position tracking and P&L calculation
- WebSocket connection for real-time updates
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import websockets

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by cTrader."""

    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"


class OrderSide(Enum):
    """Order sides."""

    BUY = "Buy"
    SELL = "Sell"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "Pending"
    FILLED = "Filled"
    PARTIALLY_FILLED = "PartiallyFilled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"


@dataclass
class TradingConfig:
    """Configuration for cTrader connection."""

    client_id: str
    client_secret: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    account_id: Optional[str] = None
    demo_account: bool = True
    host: str = "demo.ctrader.com"  # demo.ctrader.com or live.ctrader.com
    port: int = 443
    timeout: int = 30


@dataclass
class MarketData:
    """Real-time market data."""

    symbol_name: str
    symbol_id: int
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    timestamp: datetime
    spread: float = field(init=False)

    def __post_init__(self):
        self.spread = self.ask - self.bid


@dataclass
class Order:
    """Order information."""

    order_id: str
    symbol_name: str
    symbol_id: int
    order_type: OrderType
    side: OrderSide
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_volume: float = 0.0
    average_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Position information."""

    position_id: str
    symbol_name: str
    symbol_id: int
    side: OrderSide
    volume: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    profit_loss: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class TokenManager:
    """Manages OAuth 2.0 tokens for cTrader API."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.access_token = config.access_token
        self.refresh_token = config.refresh_token
        self.token_expiry = None
        self.base_url = f"https://{config.host}"

    async def get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary."""
        if (
            self.access_token
            and self.token_expiry
            and datetime.now() < self.token_expiry
        ):
            return self.access_token

        if self.refresh_token:
            await self.refresh_access_token()
        else:
            raise ValueError("No access token or refresh token available")

        if not self.access_token:
            raise ValueError("Failed to obtain access token")

        return self.access_token

    async def refresh_access_token(self):
        """Refresh the access token using refresh token."""
        url = f"{self.base_url}/connect/token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data["access_token"]
                    self.refresh_token = token_data.get(
                        "refresh_token", self.refresh_token
                    )
                    self.token_expiry = datetime.now() + timedelta(
                        seconds=token_data["expires_in"] - 60
                    )
                    logger.info("Access token refreshed successfully")
                else:
                    raise Exception(f"Failed to refresh token: {response.status}")

    async def exchange_code_for_token(
        self, authorization_code: str, redirect_uri: str
    ) -> Dict[str, str]:
        """Exchange authorization code for access token."""
        url = f"{self.base_url}/connect/token"
        data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": redirect_uri,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data["access_token"]
                    self.refresh_token = token_data["refresh_token"]
                    self.token_expiry = datetime.now() + timedelta(
                        seconds=token_data["expires_in"] - 60
                    )
                    logger.info("Authorization code exchanged for tokens successfully")
                    return token_data
                else:
                    raise Exception(
                        f"Failed to exchange code for token: {response.status}"
                    )


class RealCTraderInterface:
    """
    Real IC Markets cTrader OpenAPI Trading Interface.

    This class provides real trading functionality including:
    - OAuth 2.0 authentication
    - Real market data subscription
    - Live order placement and execution
    - Real position tracking and P&L calculation
    """

    def __init__(self, config: TradingConfig):
        """
        Initialize the real cTrader interface.

        Args:
            config: Trading configuration with OAuth credentials
        """
        self.config = config
        self.token_manager = TokenManager(config)
        self.session = None
        self.websocket = None
        self.connected = False
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.market_data: Dict[str, MarketData] = {}
        self.symbol_map: Dict[str, int] = {}
        self.account_info = None

        # Callbacks
        self.on_price_update: Optional[Callable[[MarketData], None]] = None
        self.on_order_update: Optional[Callable[[Order], None]] = None
        self.on_position_update: Optional[Callable[[Position], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        logger.info(f"Initialized real cTrader interface for {config.host}")

    async def connect(self) -> bool:
        """Connect to cTrader API."""
        try:
            # Create session
            self.session = aiohttp.ClientSession()

            # Get access token
            await self.token_manager.get_access_token()

            # Get account information
            await self._get_trading_accounts()

            # Get symbol information
            await self._get_symbols()

            # Connect WebSocket for real-time data
            await self._connect_websocket()

            self.connected = True
            logger.info("Successfully connected to cTrader API")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to cTrader API: {e}")
            if self.session:
                await self.session.close()
            return False

    async def disconnect(self):
        """Disconnect from cTrader API."""
        self.connected = False

        if self.websocket:
            await self.websocket.close()

        if self.session:
            await self.session.close()

        logger.info("Disconnected from cTrader API")

    async def _get_trading_accounts(self):
        """Get trading account information."""
        url = f"https://{self.config.host}/connect/api/accounts"
        headers = {
            "Authorization": f"Bearer {await self.token_manager.get_access_token()}"
        }

        if not self.session:
            raise Exception("Session not initialized")

        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                accounts = await response.json()
                if accounts:
                    self.account_info = accounts[0]  # Use first account
                    self.config.account_id = str(
                        self.account_info["ctidTraderAccountId"]
                    )
                    logger.info(f"Using account: {self.account_info['accountName']}")
                else:
                    raise Exception("No trading accounts found")
            else:
                raise Exception(f"Failed to get accounts: {response.status}")

    async def _get_symbols(self):
        """Get available symbols."""
        url = f"https://{self.config.host}/connect/api/symbols"
        headers = {
            "Authorization": f"Bearer {await self.token_manager.get_access_token()}"
        }

        if not self.session:
            raise Exception("Session not initialized")

        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                symbols = await response.json()
                for symbol in symbols:
                    self.symbol_map[symbol["symbolName"]] = symbol["symbolId"]
                logger.info(f"Loaded {len(self.symbol_map)} symbols")
            else:
                raise Exception(f"Failed to get symbols: {response.status}")

    async def _connect_websocket(self):
        """Connect to WebSocket for real-time data."""
        ws_url = f"wss://{self.config.host}/connect/api/streaming"
        headers = {
            "Authorization": f"Bearer {await self.token_manager.get_access_token()}"
        }

        try:
            self.websocket = await websockets.connect(ws_url, extra_headers=headers)
            logger.info("WebSocket connected for real-time data")

            # Start listening for messages
            asyncio.create_task(self._listen_websocket())

        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise

    async def _listen_websocket(self):
        """Listen for WebSocket messages."""
        try:
            async for message in self.websocket:
                await self._handle_websocket_message(json.loads(message))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if self.on_error:
                self.on_error(f"WebSocket error: {e}")

    async def _handle_websocket_message(self, message: Dict[str, Any]):
        """Handle WebSocket message."""
        msg_type = message.get("type")

        if msg_type == "price":
            await self._handle_price_update(message)
        elif msg_type == "order":
            await self._handle_order_update(message)
        elif msg_type == "position":
            await self._handle_position_update(message)
        else:
            logger.debug(f"Unknown message type: {msg_type}")

    async def _handle_price_update(self, message: Dict[str, Any]):
        """Handle price update message."""
        try:
            market_data = MarketData(
                symbol_name=message["symbolName"],
                symbol_id=message["symbolId"],
                bid=float(message["bid"]),
                ask=float(message["ask"]),
                bid_volume=float(message.get("bidVolume", 0)),
                ask_volume=float(message.get("askVolume", 0)),
                timestamp=datetime.fromisoformat(message["timestamp"]),
            )

            self.market_data[market_data.symbol_name] = market_data

            if self.on_price_update:
                self.on_price_update(market_data)

        except Exception as e:
            logger.error(f"Error handling price update: {e}")

    async def _handle_order_update(self, message: Dict[str, Any]):
        """Handle order update message."""
        try:
            order = Order(
                order_id=str(message["orderId"]),
                symbol_name=message["symbolName"],
                symbol_id=message["symbolId"],
                order_type=OrderType(message["orderType"]),
                side=OrderSide(message["side"]),
                volume=float(message["volume"]),
                price=float(message.get("price", 0)) if message.get("price") else None,
                stop_loss=(
                    float(message.get("stopLoss", 0))
                    if message.get("stopLoss")
                    else None
                ),
                take_profit=(
                    float(message.get("takeProfit", 0))
                    if message.get("takeProfit")
                    else None
                ),
                status=OrderStatus(message["status"]),
                filled_volume=float(message.get("filledVolume", 0)),
                average_price=(
                    float(message.get("averagePrice", 0))
                    if message.get("averagePrice")
                    else None
                ),
                timestamp=datetime.fromisoformat(message["timestamp"]),
            )

            self.orders[order.order_id] = order

            if self.on_order_update:
                self.on_order_update(order)

        except Exception as e:
            logger.error(f"Error handling order update: {e}")

    async def _handle_position_update(self, message: Dict[str, Any]):
        """Handle position update message."""
        try:
            position = Position(
                position_id=str(message["positionId"]),
                symbol_name=message["symbolName"],
                symbol_id=message["symbolId"],
                side=OrderSide(message["side"]),
                volume=float(message["volume"]),
                entry_price=float(message["entryPrice"]),
                current_price=float(message["currentPrice"]),
                stop_loss=(
                    float(message.get("stopLoss", 0))
                    if message.get("stopLoss")
                    else None
                ),
                take_profit=(
                    float(message.get("takeProfit", 0))
                    if message.get("takeProfit")
                    else None
                ),
                profit_loss=float(message["profitLoss"]),
                swap=float(message.get("swap", 0)),
                commission=float(message.get("commission", 0)),
                timestamp=datetime.fromisoformat(message["timestamp"]),
            )

            self.positions[position.position_id] = position

            if self.on_position_update:
                self.on_position_update(position)

        except Exception as e:
            logger.error(f"Error handling position update: {e}")

    async def subscribe_market_data(self, symbols: List[str]) -> bool:
        """Subscribe to market data for symbols."""
        if not self.websocket:
            logger.error("WebSocket not connected")
            return False

        try:
            subscribe_message = {"type": "subscribe", "symbols": symbols}

            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to market data for {len(symbols)} symbols")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {e}")
            return False

    async def place_order(
        self,
        symbol_name: str,
        order_type: OrderType,
        side: OrderSide,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[str]:
        """Place a new order."""
        if not self.connected or not self.session:
            logger.error("Not connected to cTrader API")
            return None

        try:
            symbol_id = self.symbol_map.get(symbol_name)
            if not symbol_id:
                logger.error(f"Symbol {symbol_name} not found")
                return None

            url = f"https://{self.config.host}/connect/api/orders"
            headers = {
                "Authorization": f"Bearer {await self.token_manager.get_access_token()}"
            }

            data = {
                "symbolId": symbol_id,
                "orderType": order_type.value,
                "side": side.value,
                "volume": volume,
                "ctidTraderAccountId": self.config.account_id,
            }

            if price:
                data["price"] = price
            if stop_loss:
                data["stopLoss"] = stop_loss
            if take_profit:
                data["takeProfit"] = take_profit

            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    order_id = str(result["orderId"])
                    logger.info(f"Order placed successfully: {order_id}")
                    return order_id
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to place order: {response.status} - {error_text}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if not self.connected:
            logger.error("Not connected to cTrader API")
            return False

        try:
            url = f"https://{self.config.host}/connect/api/orders/{order_id}"
            headers = {
                "Authorization": f"Bearer {await self.token_manager.get_access_token()}"
            }

            async with self.session.delete(url, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"Order {order_id} cancelled successfully")
                    return True
                else:
                    logger.error(
                        f"Failed to cancel order {order_id}: {response.status}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Modify position stop loss or take profit."""
        if not self.connected:
            logger.error("Not connected to cTrader API")
            return False

        try:
            url = f"https://{self.config.host}/connect/api/positions/{position_id}"
            headers = {
                "Authorization": f"Bearer {await self.token_manager.get_access_token()}"
            }

            data = {}
            if stop_loss is not None:
                data["stopLoss"] = stop_loss
            if take_profit is not None:
                data["takeProfit"] = take_profit

            if not data:
                logger.warning("No modifications specified")
                return False

            async with self.session.put(url, headers=headers, json=data) as response:
                if response.status == 200:
                    logger.info(f"Position {position_id} modified successfully")
                    return True
                else:
                    logger.error(
                        f"Failed to modify position {position_id}: {response.status}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return False

    async def close_position(self, position_id: str) -> bool:
        """Close a position."""
        if not self.connected:
            logger.error("Not connected to cTrader API")
            return False

        try:
            url = f"https://{self.config.host}/connect/api/positions/{position_id}"
            headers = {
                "Authorization": f"Bearer {await self.token_manager.get_access_token()}"
            }

            async with self.session.delete(url, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"Position {position_id} closed successfully")
                    return True
                else:
                    logger.error(
                        f"Failed to close position {position_id}: {response.status}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    def get_positions(self) -> List[Position]:
        """Get current positions."""
        return list(self.positions.values())

    def get_orders(self) -> List[Order]:
        """Get current orders."""
        return list(self.orders.values())

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get market data for a symbol."""
        return self.market_data.get(symbol)

    def _get_symbol_name(self, symbol_id: int) -> Optional[str]:
        """Get symbol name from symbol ID."""
        for name, sid in self.symbol_map.items():
            if sid == symbol_id:
                return name
        return None

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        return self.account_info


# Convenience function to create demo config
def create_demo_config(client_id: str, client_secret: str) -> TradingConfig:
    """Create a demo trading configuration."""
    return TradingConfig(
        client_id=client_id,
        client_secret=client_secret,
        demo_account=True,
        host="demo.ctrader.com",
    )


# Convenience function to create live config
def create_live_config(client_id: str, client_secret: str) -> TradingConfig:
    """Create a live trading configuration."""
    return TradingConfig(
        client_id=client_id,
        client_secret=client_secret,
        demo_account=False,
        host="live.ctrader.com",
    )
