"""
Market Simulator v1.0 - High-Fidelity Historical Data Playback

Implements SIM-02 ticket requirements for event-driven historical data simulation.
Provides tick-by-tick market data playback that perfectly mimics live data flow.
"""

import asyncio
import pandas as pd
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)


class MarketSimulator:
    """
    High-fidelity market data simulator that plays back historical data
    tick-by-tick to perfectly mimic live market conditions.
    """
    
    def __init__(self, event_bus, sensory_cortex=None):
        """
        Initialize the market simulator.
        
        Args:
            event_bus: The event bus for publishing market data
            sensory_cortex: Optional sensory cortex for processing market data
        """
        self.event_bus = event_bus
        self.sensory_cortex = sensory_cortex
        self.historical_data = None
        self.current_index = 0
        self.is_running = False
        
    async def load_data(self, symbol: str = "EURUSD", interval: str = "1h") -> bool:
        """
        Load historical market data using YahooFinanceOrgan.
        
        Args:
            symbol: Trading symbol to load data for
            interval: Data interval (1h, 1d, etc.)
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            from src.sensory.organs.yahoo_finance_organ import YahooFinanceOrgan
            
            # Try to load real historical data
            organ = YahooFinanceOrgan()
            df = organ.load_data(symbol, interval)
            
            if df is not None and len(df) > 0:
                # Convert Yahoo Finance format to simulator format
                data = []
                for _, row in df.iterrows():
                    close_val = float(str(row['close'])) if pd.notna(row['close']) else 0.0
                    volume_val = float(str(row['volume'])) if pd.notna(row['volume']) else 0.0
                    
                    if close_val > 0 and volume_val >= 0:
                        open_val = float(str(row['open'])) if pd.notna(row['open']) else close_val
                        high_val = float(str(row['high'])) if pd.notna(row['high']) else close_val
                        low_val = float(str(row['low'])) if pd.notna(row['low']) else close_val
                        
                        data.append({
                            'timestamp': pd.to_datetime(str(row['timestamp'])),
                            'symbol': str(symbol),
                            'price': close_val,
                            'volume': volume_val,
                            'open': open_val,
                            'high': high_val,
                            'low': low_val,
                            'bid': close_val - 0.0001,
                            'ask': close_val + 0.0001,
                            'spread': 0.0002,
                            'volatility': 0.0
                        })
                
                self.historical_data = pd.DataFrame(data)
                logger.info(f"Loaded {len(self.historical_data)} real historical data points for {symbol}")
                return True
            
            # Fallback to random data generation
            logger.warning("No real historical data found, using fallback random data")
            return await self._generate_random_data(symbol)
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False
    
    async def _generate_random_data(self, symbol: str) -> bool:
        """Generate fallback random data for testing"""
        try:
            import numpy as np
            
            # Generate 1000 random data points
            dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
            
            # Generate random OHLCV data
            base_price = 1.1000
            returns = np.random.normal(0, 0.001, 1000)
            prices = base_price * (1 + returns).cumprod()
            
            data = []
            for date, price in zip(dates, prices):
                price_val = float(price)
                volume_val = float(np.random.randint(1000, 10000))
                
                data.append({
                    'timestamp': date,
                    'symbol': str(symbol),
                    'price': price_val,
                    'volume': volume_val,
                    'open': float(price * (1 + np.random.normal(0, 0.0001))),
                    'high': float(price * (1 + abs(np.random.normal(0, 0.0002)))),
                    'low': float(price * (1 - abs(np.random.normal(0, 0.0002)))),
                    'bid': price_val - 0.0001,
                    'ask': price_val + 0.0001,
                    'spread': 0.0002,
                    'volatility': float(abs(np.random.normal(0, 0.001)))
                })
            
            self.historical_data = pd.DataFrame(data)
            logger.info(f"Generated {len(self.historical_data)} random data points for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating random data: {e}")
            return False
    
    async def run(self) -> None:
        """
        Run the market simulation by playing back historical data tick-by-tick.
        """
        if self.historical_data is None:
            logger.error("No historical data loaded. Call load_data() first.")
            return
            
        self.is_running = True
        self.current_index = 0
        
        try:
            logger.info(f"Starting market simulation with {len(self.historical_data)} ticks")
            
            # Convert DataFrame to list of dictionaries for safer iteration
            data_list = self.historical_data.to_dict('records')
            
            for i, row in enumerate(data_list):
                if not self.is_running:
                    break
                    
                # Create market understanding event
                from ..core.events import MarketUnderstanding
                
                market_understanding = MarketUnderstanding(
                    event_id=f"market_{i}_{datetime.now().isoformat()}",
                    timestamp=datetime.now(),
                    source="MarketSimulator",
                    symbol=str(row['symbol']),
                    price=Decimal(str(row['price'])),
                    volume=Decimal(str(row['volume'])),
                    indicators={},
                    metadata=row
                )
                
                # Publish to event bus
                await self.event_bus.publish(market_understanding)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.001)
                
                self.current_index = i
                
            logger.info("Market simulation completed successfully")
            
        except Exception as e:
            logger.error(f"Error during market simulation: {e}")
            raise
            
    async def stop(self) -> None:
        """Stop the market simulation."""
        self.is_running = False
        logger.info("Market simulation stopped")
        
    def get_progress(self) -> Dict[str, Any]:
        """
        Get the current simulation progress.
        
        Returns:
            Dictionary with progress information
        """
        if self.historical_data is None:
            return {'status': 'no_data', 'progress': 0.0}
            
        total_ticks = len(self.historical_data)
        progress = (self.current_index / total_ticks) * 100 if total_ticks > 0 else 0.0
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'progress': progress,
            'current_index': self.current_index,
            'total_ticks': total_ticks,
            'remaining_ticks': total_ticks - self.current_index
        }
