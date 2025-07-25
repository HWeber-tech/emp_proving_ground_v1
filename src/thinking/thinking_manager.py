#!/usr/bin/env python3
"""
Thinking Manager - Epic 4: Fusing Foresight
Enhanced thinking manager that integrates predictive intelligence into the cognitive loop.
"""

import asyncio
from collections import deque
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
import logging

from core.events import ContextPacket, MarketForecast
from thinking.prediction.predictive_modeler import PredictiveMarketModeler
from config.system_config import SystemConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThinkingManager:
    """Enhanced thinking manager with predictive intelligence."""
    
    def __init__(self, system_config: SystemConfig):
        """Initialize the thinking manager with predictive capabilities."""
        self.system_config = system_config
        self.predictive_modeler: Optional[PredictiveMarketModeler] = None
        self.market_data_buffer = deque(maxlen=100)  # Rolling buffer for recent market data
        
        # Initialize predictive modeler if configured
        self._initialize_predictive_modeler()
    
    def _initialize_predictive_modeler(self):
        """Initialize the predictive modeler from system config."""
        model_run_id = self.system_config.get('mlflow', {}).get('model_run_id')
        mlflow_uri = self.system_config.get('mlflow', {}).get('uri', 'http://localhost:5000')
        
        if model_run_id:
            try:
                self.predictive_modeler = PredictiveMarketModeler(
                    model_run_id=model_run_id,
                    mlflow_uri=mlflow_uri
                )
                logger.info(f"Predictive modeler initialized with model: {model_run_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize predictive modeler: {e}")
                self.predictive_modeler = None
        else:
            logger.info("No model_run_id configured, predictive features disabled")
    
    async def on_market_understanding(self, market_data: Dict[str, Any]) -> ContextPacket:
        """
        Handle market understanding event and generate context with forecast.
        
        Args:
            market_data: Raw market data from sensors
            
        Returns:
            ContextPacket with predictive intelligence
        """
        logger.debug("Processing market understanding event")
        
        # Add to buffer
        self.market_data_buffer.append(market_data)
        
        # Create base context packet
        context = ContextPacket(
            timestamp=datetime.utcnow().isoformat(),
            market_data=market_data,
            risk_metrics=self._calculate_risk_metrics(market_data),
            sequence_id=f"{datetime.utcnow().timestamp()}",
            market_forecast=None
        )
        
        # Generate forecast if predictive modeler is available
        if self.predictive_modeler and len(self.market_data_buffer) >= 20:
            try:
                # Convert buffer to DataFrame for prediction
                df = self._buffer_to_dataframe()
                
                # Generate forecast
                forecast = await self.predictive_modeler.forecast(df)
                
                # Create MarketForecast object
                market_forecast = MarketForecast(
                    prob_up=forecast["prob_up"],
                    prob_down=forecast["prob_down"],
                    prob_flat=forecast["prob_flat"],
                    model_version=forecast["model_version"],
                    timestamp=forecast["timestamp"]
                )
                
                # Add to context
                context.market_forecast = market_forecast
                logger.info(f"Generated forecast: UP={forecast['prob_up']:.2f}, DOWN={forecast['prob_down']:.2f}, FLAT={forecast['prob_flat']:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to generate forecast: {e}")
        
        return context
    
    def _calculate_risk_metrics(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate basic risk metrics from market data."""
        # Placeholder for risk calculation
        return {
            "volatility": 0.02,
            "max_drawdown": 0.05,
            "sharpe_ratio": 1.5
        }
    
    def _buffer_to_dataframe(self) -> pd.DataFrame:
        """Convert the market data buffer to a DataFrame."""
        if not self.market_data_buffer:
            return pd.DataFrame()
        
        # Convert deque to list
        data_list = list(self.market_data_buffer)
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # Try to map from common column names
                col_mapping = {
                    'open': ['Open', 'open_price'],
                    'high': ['High', 'high_price'],
                    'low': ['Low', 'low_price'],
                    'close': ['Close', 'close_price', 'price'],
                    'volume': ['Volume', 'volume']
                }
                
                for alt_col in col_mapping.get(col, []):
                    if alt_col in df.columns:
                        df = df.rename(columns={alt_col: col})
                        break
        
        return df
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the predictive model."""
        if self.predictive_modeler:
            return self.predictive_modeler.get_model_info()
        return {"status": "disabled", "reason": "no_model_configured"}
    
    def is_predictive_enabled(self) -> bool:
        """Check if predictive features are enabled."""
        return self.predictive_modeler is not None

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_thinking_manager():
        """Test the thinking manager with predictive integration."""
        
        # Mock system config
        class MockSystemConfig:
            def get(self, key, default=None):
                if key == 'mlflow':
                    return {'model_run_id': 'test_run_id', 'uri': 'http://localhost:5000'}
                return default
        
        config = MockSystemConfig()
        manager = ThinkingManager(config)
        
        # Test market data
        market_data = {
            'open': 1.1000,
            'high': 1.1005,
            'low': 1.0995,
            'close': 1.1002,
            'volume': 1000
        }
        
        # Generate context
        context = await manager.on_market_understanding(market_data)
        print("Generated context:")
        print(f"  Timestamp: {context.timestamp}")
        print(f"  Has forecast: {context.market_forecast is not None}")
        if context.market_forecast:
            print(f"  Forecast: UP={context.market_forecast.prob_up:.2f}, DOWN={context.market_forecast.prob_down:.2f}, FLAT={context.market_forecast.prob_flat:.2f}")
    
    # Run test
    asyncio.run(test_thinking_manager())
