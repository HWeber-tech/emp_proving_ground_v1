"""
Legacy: cTrader Broker Interface (OpenAPI) - Disabled in FIX-only build
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime

logger = logging.getLogger(__name__)


class CTraderBrokerInterface:
    """Disabled. Use FIXBrokerInterface instead."""

    def __init__(self, event_bus, config):
        raise ImportError("CTrader OpenAPI is disabled. Use FIXBrokerInterface.")
        
    async def start(self):
        return False
        
    async def stop(self):
        return None
        
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        """
        Place a market order via cTrader.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            
        Returns:
            Order ID if successful, None otherwise
        """
        return None
            
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancel request sent, False otherwise
        """
        return False
            
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status dictionary or None
        """
        return None
        
    def get_all_orders(self) -> Dict[str, Dict[str, Any]]:
        return {}
        
    async def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance information.
        
        Returns:
            Account balance dictionary
        """
        return {}
            
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get open positions.
        
        Returns:
            List of position dictionaries
        """
        return []
