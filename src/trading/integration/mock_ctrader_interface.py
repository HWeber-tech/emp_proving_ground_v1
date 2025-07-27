#!/usr/bin/env python3
"""
Mock cTrader Interface for Testing
from src.core.market_data import MarketData

This module provides a mock implementation of the cTrader OpenAPI
for testing purposes when the real library is not available.
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

:{self.port}")
    
    async def stop(self):
        """Stop the mock client."""
        self.connected = False
        self.running = False
        logger.info("Mock cTrader client disconnected")
    
    async def send(self, message):
        """Send a mock message."""
        logger.info(f"Mock message sent: {type(message).__name__}")
        
        # Simulate message processing
        await asyncio.sleep(0.1)
        
        # Generate mock response
        if hasattr(message, 'clientId'):
            # Application auth response
            mock_response = {
                'payloadType': 'PROTO_OA_APPLICATION_AUTH_RES',
                'payload': {'status': 'OK'}
            }
        elif hasattr(message, 'ctidTraderAccountId'):
            # Account auth response
            mock_response = {
                'payloadType': 'PROTO_OA_ACCOUNT_AUTH_RES',
                'payload': {'status': 'OK'}
            }
        else:
            mock_response = {
                'payloadType': 'PROTO_OA_GENERIC_RES',
                'payload': {'status': 'OK'}
            }
        
        # Simulate message received
        if self.on_message:
            await self.on_message(mock_response)

class MockCTraderInterface:
    """
    Mock IC Markets cTrader OpenAPI Trading Interface.
    
    This class simulates all trading operations for testing purposes.
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize the mock cTrader interface.
        
        Args:
            config: Trading configuration
        """
        self.config = config
        self.client = None
        self.connected = False
        self.authenticated = False
        self.symbols_cache = {
            1: {'name': 'EURUSD', 'digits': 5, 'pip_position': 4},
            2: {'name': 'GBPUSD', 'digits': 5, 'pip_position': 4},
            3: {'name': 'USDJPY', 'digits': 3, 'pip_position': 2}
        }
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
        
        logger.info(f"Mock cTrader interface initialized for {config.mode.value} mode")
    
    async def connect(self) -> bool:
        """
        Connect to mock cTrader and authenticate.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to mock cTrader...")
            
            # Create mock client
            self.client = MockCTraderClient(self.config.host, self.config.port, self._on_message_received)
            
            # Start connection
            await self.client.start()
            logger.info("Mock client connected, authenticating...")
            
            # Simulate authentication
            await asyncio.sleep(1)
            self.authenticated = True
            
            self.connected = True
            
            logger.info("Successfully connected and authenticated to mock cTrader")
            return True
            
        except Exception as e:
            logger.error(f"Mock connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from mock cTrader."""
        if self.client:
            await self.client.stop()
            self.connected = False
            self.authenticated = False
            logger.info("Disconnected from mock cTrader")
    
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
            
            # Simulate subscription
            await asyncio.sleep(0.1)
            logger.info(f"Subscribed to {symbol_name} (ID: {symbol_id})")
            
            # Generate mock market data
            self._generate_mock_market_data(symbol_id, symbol_name)
            
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
            
            # Generate order ID
            order_id = f"order_{int(time.time())}"
            
            # Create order
            order = Order(
                order_id=order_id,
                symbol_id=symbol_id,
                order_type=order_type,
                side=side,
                volume=volume,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status="pending",
                timestamp=datetime.now()
            )
            
            self.orders[order_id] = order
            
            # Simulate order processing
            await asyncio.sleep(0.1)
            
            # Simulate order fill for market orders
            if order_type == OrderType.MARKET:
                await self._simulate_order_fill(order_id, symbol_name)
            
            logger.info(f"Order placed: {side.value} {volume} {symbol_name} (ID: {order_id})")
            return order_id
            
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
            order.status = "cancelled"
            
            logger.info(f"Order cancelled: {order_id}")
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
            
            # Remove position
            del self.positions[position_id]
            
            logger.info(f"Position closed: {position_id}")
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
    
    def _on_message_received(self, message):
        """Handle incoming messages from mock cTrader."""
        logger.debug(f"Mock message received: {message}")
    
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
    
    def _generate_mock_market_data(self, symbol_id: int, symbol_name: str):
        """Generate mock market data for a symbol."""
        import random
        
        # Generate realistic prices
        base_price = 1.07123 if symbol_name == "EURUSD" else 1.25123 if symbol_name == "GBPUSD" else 150.123
        
        bid = base_price + random.uniform(-0.0005, 0.0005)
        ask = bid + random.uniform(0.0001, 0.0003)
        
        market_data = MarketData(
            symbol_id=symbol_id,
            symbol_name=symbol_name,
            bid=bid,
            ask=ask,
            timestamp=datetime.now(),
            digits=self.symbols_cache[symbol_id]['digits']
        )
        
        self.market_data[symbol_id] = market_data
        
        # Trigger callbacks
        for callback in self.callbacks['price_update']:
            try:
                callback(market_data)
            except Exception as e:
                logger.error(f"Price update callback error: {e}")
    
    async def _simulate_order_fill(self, order_id: str, symbol_name: str):
        """Simulate order fill and position creation."""
        order = self.orders[order_id]
        order.status = "filled"
        
        # Create position
        position_id = f"position_{int(time.time())}"
        current_price = (self.market_data[order.symbol_id].bid + self.market_data[order.symbol_id].ask) / 2
        
        position = Position(
            position_id=position_id,
            symbol_id=order.symbol_id,
            side=order.side,
            volume=order.volume,
            entry_price=current_price,
            current_price=current_price,
            profit_loss=0.0,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            timestamp=datetime.now()
        )
        
        self.positions[position_id] = position
        
        # Trigger callbacks
        for callback in self.callbacks['order_update']:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Order update callback error: {e}")
        
        for callback in self.callbacks['position_update']:
            try:
                callback(position)
            except Exception as e:
                logger.error(f"Position update callback error: {e}")

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
            # Mock token exchange for testing
            mock_tokens = {
                "access_token": "mock_access_token_12345",
                "refresh_token": "mock_refresh_token_67890",
                "expires_in": 3600,
                "token_type": "Bearer"
            }
            
            logger.info("Mock token exchange completed")
            return mock_tokens
                
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[Dict]:
        """Refresh access token using refresh token."""
        try:
            # Mock token refresh for testing
            mock_tokens = {
                "access_token": "mock_refreshed_access_token_12345",
                "refresh_token": refresh_token,
                "expires_in": 3600,
                "token_type": "Bearer"
            }
            
            logger.info("Mock token refresh completed")
            return mock_tokens
                
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None
    
    def get_trading_accounts(self, access_token: str) -> Optional[List[Dict]]:
        """Get available trading accounts."""
        try:
            # Mock trading accounts for testing
            mock_accounts = [
                {
                    "ctidTraderAccountId": 12345678,
                    "accountNumber": "12345678",
                    "isLive": False,
                    "broker": "IC Markets"
                }
            ]
            
            logger.info("Mock trading accounts retrieved")
            return mock_accounts
                
        except Exception as e:
            logger.error(f"Get trading accounts error: {e}")
            return None

# Alias for compatibility
CTraderInterface = MockCTraderInterface

def main():
    """Test the mock cTrader interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test mock cTrader interface")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to subscribe to")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_data = json.load(f)
    
    config = TradingConfig(**config_data)
    
    # Create interface
    interface = MockCTraderInterface(config)
    
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
            await asyncio.sleep(10)
        await interface.disconnect()
    
    asyncio.run(test())

if __name__ == "__main__":
    main() 