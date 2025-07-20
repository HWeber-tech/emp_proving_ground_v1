"""
Portfolio Monitor - Ticket TRADING-06
Stateful portfolio management with Redis persistence
"""

import json
import asyncio
import redis
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from src.core.events import EventBus, ExecutionReport

logger = logging.getLogger(__name__)


class PortfolioMonitor:
    """
    Stateful portfolio monitor that persists state to Redis
    Provides crash-resilient portfolio management
    """
    
    def __init__(self, event_bus: EventBus, redis_client: redis.Redis):
        self.event_bus = event_bus
        self.redis_client = redis_client
        self.redis_key = "emp:portfolio_state"
        self.portfolio = self._load_initial_state()
        
        # Subscribe to execution reports
        self.event_bus.subscribe("execution.report", self.on_execution_report)
        
        logger.info(f"PortfolioMonitor initialized with state: {self.portfolio}")
    
    def _load_initial_state(self) -> Dict[str, Any]:
        """Load portfolio state from Redis or initialize defaults"""
        try:
            state_json = self.redis_client.get(self.redis_key)
            if state_json:
                state = json.loads(state_json)
                logger.info("Loaded portfolio state from Redis")
                return state
        except Exception as e:
            logger.warning(f"Failed to load state from Redis: {e}")
        
        # Default initial state
        default_state = {
            'cash': 100000.0,
            'open_positions': {},
            'daily_pnl': 0.0,
            'total_pnl': 0.0,
            'last_updated': datetime.now().isoformat()
        }
        logger.info("Initialized with default portfolio state")
        return default_state
    
    def _save_state_to_redis(self) -> None:
        """Persist current portfolio state to Redis"""
        try:
            self.portfolio['last_updated'] = datetime.now().isoformat()
            state_json = json.dumps(self.portfolio)
            self.redis_client.set(self.redis_key, state_json)
            logger.debug("Portfolio state saved to Redis")
        except Exception as e:
            logger.error(f"Failed to save state to Redis: {e}")
    
    async def on_execution_report(self, event: ExecutionReport) -> None:
        """Handle execution reports and update portfolio state"""
        try:
            logger.info(f"Processing execution report: {event}")
            
            # Update cash based on trade
            if event.side == "BUY":
                self.portfolio['cash'] -= event.quantity * event.price
            else:  # SELL
                self.portfolio['cash'] += event.quantity * event.price
            
            # Update positions
            symbol = event.symbol
            if symbol not in self.portfolio['open_positions']:
                self.portfolio['open_positions'][symbol] = {
                    'quantity': 0,
                    'avg_price': 0.0,
                    'current_value': 0.0
                }
            
            position = self.portfolio['open_positions'][symbol]
            
            if event.side == "BUY":
                # Calculate new average price for buys
                total_value = (position['quantity'] * position['avg_price']) + (event.quantity * event.price)
                position['quantity'] += event.quantity
                position['avg_price'] = total_value / position['quantity'] if position['quantity'] > 0 else 0.0
            else:  # SELL
                position['quantity'] -= event.quantity
                if position['quantity'] <= 0:
                    # Position closed
                    del self.portfolio['open_positions'][symbol]
                else:
                    # Update position value
                    position['current_value'] = position['quantity'] * event.price
            
            # Update P&L calculations
            self._update_pnl()
            
            # Persist state
            self._save_state_to_redis()
            
            logger.info(f"Portfolio updated: {self.portfolio}")
            
        except Exception as e:
            logger.error(f"Error processing execution report: {e}")
    
    def _update_pnl(self) -> None:
        """Update P&L calculations"""
        # This is a simplified P&L calculation
        # In a real system, this would use current market prices
        self.portfolio['daily_pnl'] = 0.0  # Placeholder
        self.portfolio['total_pnl'] = 0.0  # Placeholder
    
    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio state"""
        return self.portfolio.copy()
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol"""
        return self.portfolio['open_positions'].get(symbol)
    
    def get_total_value(self) -> float:
        """Calculate total portfolio value (cash + positions)"""
        total_value = self.portfolio['cash']
        
        # Add position values (using last known prices)
        for symbol, position in self.portfolio['open_positions'].items():
            total_value += position.get('current_value', 0.0)
        
        return total_value
    
    def reset_portfolio(self) -> None:
        """Reset portfolio to initial state"""
        self.portfolio = {
            'cash': 100000.0,
            'open_positions': {},
            'daily_pnl': 0.0,
            'total_pnl': 0.0,
            'last_updated': datetime.now().isoformat()
        }
        self._save_state_to_redis()
        logger.info("Portfolio reset to initial state")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_portfolio_monitor():
        # Setup Redis connection
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        # Create event bus
        event_bus = EventBus()
        
        # Create portfolio monitor
        monitor = PortfolioMonitor(event_bus, redis_client)
        
        # Test execution reports
        test_events = [
            ExecutionReport(
                symbol="AAPL",
                side="BUY",
                quantity=100,
                price=150.0,
                order_id="test_001"
            ),
            ExecutionReport(
                symbol="AAPL",
                side="SELL",
                quantity=50,
                price=155.0,
                order_id="test_002"
            )
        ]
        
        for event in test_events:
            await monitor.on_execution_report(event)
            print(f"Portfolio after {event.side}: {monitor.get_portfolio()}")
        
        # Test state persistence
        print("Testing state persistence...")
        new_monitor = PortfolioMonitor(event_bus, redis_client)
        print(f"Loaded state: {new_monitor.get_portfolio()}")
    
    asyncio.run(test_portfolio_monitor())
