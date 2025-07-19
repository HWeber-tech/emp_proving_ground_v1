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
from pathlib import Path
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
        self.historical_data: Optional[pd.DataFrame] = None
        self.current_index = 0
        self.is_running = False
        
    async def load_data(self, data_path: str) -> bool:
        """
        Load historical market data from file.
        
        Args:
            data_path: Path to the historical data file (CSV or Parquet)
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            data_path_obj = Path(data_path)
            
            if not data_path_obj.exists():
                logger.error(f"Data file not found: {data_path}")
                return False
                
            if data_path_obj.suffix.lower() == '.csv':
                self.historical_data = pd.read_csv(data_path)
            elif data_path_obj.suffix.lower() == '.parquet':
                self.historical_data = pd.read_parquet(data_path)
            else:
                logger.error(f"Unsupported file format: {data_path_obj.suffix}")
                return False
                
            # Ensure required columns exist
            required_columns = ['timestamp', 'symbol', 'price', 'volume']
            missing_columns = [col for col in required_columns if col not in self.historical_data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            # Sort by timestamp
            self.historical_data = self.historical_data.sort_values('timestamp')
            self.historical_data['timestamp'] = pd.to_datetime(self.historical_data['timestamp'])
            
            logger.info(f"Loaded {len(self.historical_data)} historical data points from {data_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
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
            
            for index, row in self.historical_data.iterrows():
                if not self.is_running:
                    break
                    
                # Create market data dictionary with safe type handling
                market_data = {
                    'timestamp': str(row['timestamp']),
                    'symbol': str(row['symbol']),
                    'price': float(row['price']) if pd.notna(row['price']) else 0.0,
                    'volume': float(row['volume']) if pd.notna(row['volume']) else 0.0,
                    'bid': float(row.get('bid', row['price'])) if pd.notna(row.get('bid', row['price'])) else 0.0,
                    'ask': float(row.get('ask', row['price'])) if pd.notna(row.get('ask', row['price'])) else 0.0,
                    'spread': float(row.get('spread', 0.0)) if pd.notna(row.get('spread', 0.0)) else 0.0,
                    'volatility': float(row.get('volatility', 0.0)) if pd.notna(row.get('volatility', 0.0)) else 0.0
                }
                
                # Add any additional columns
                for col in row.index:
                    col_str = str(col)
                    if col_str not in market_data and col_str != 'timestamp':
                        value = row[col]
                        if pd.notna(value):
                            market_data[col_str] = value
                
                # Create market understanding event
                from ..core.events import MarketUnderstanding
                
                market_understanding = MarketUnderstanding(
                    event_id=f"market_{index}_{datetime.now().isoformat()}",
                    timestamp=datetime.now(),
                    source="MarketSimulator",
                    symbol=market_data['symbol'],
                    price=Decimal(str(market_data['price'])),
                    volume=Decimal(str(market_data['volume'])),
                    indicators={},
                    metadata=market_data
                )
                
                # Publish to event bus
                await self.event_bus.publish(market_understanding)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.001)
                
                self.current_index = int(index)
                
            # Signal simulation complete
            from ..core.events import SimulationComplete
            
            complete_event = SimulationComplete(
                event_id=f"sim_complete_{datetime.now().isoformat()}",
                timestamp=datetime.now(),
                source="MarketSimulator",
                total_ticks=len(self.historical_data),
                final_index=int(self.current_index)
            )
            await self.event_bus.publish(complete_event)
            
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
            return {'status': '
