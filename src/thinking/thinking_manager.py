#!/usr/bin/env python3
"""
Thinking Manager - Epic 4: Fusing Foresight
Enhanced thinking manager that integrates predictive intelligence into the cognitive loop.
"""

import asyncio
import logging
import math
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, cast

# Removed pandas dependency

from governance.system_config import SystemConfig
from core.context_packet import ContextPacket
from thinking.prediction.predictive_modeler import PredictiveMarketModeler
from operational.state_store import StateStore

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
        """Initialize the predictive modeler."""
        try:
            self.predictive_modeler = PredictiveMarketModeler(StateStore())
            logger.info("Predictive modeler initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize predictive modeler: {e}")
            self.predictive_modeler = None
    
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
            timestamp=datetime.utcnow(),
            symbol=str(market_data.get("symbol", "UNKNOWN")),
            current_price=float(market_data.get("close") or market_data.get("price") or 0.0),
            current_cvd=float(market_data.get("cvd") or market_data.get("current_cvd") or 0.0),
            metadata={
                "risk_metrics": self._calculate_risk_metrics(market_data),
                "sequence_id": f"{datetime.utcnow().timestamp()}",
            },
        )
        
        # Generate forecast if predictive modeler is available
        if self.predictive_modeler and len(self.market_data_buffer) >= 20:
            try:
                # Derive simple state from the rolling buffer without pandas
                state = self._compute_state_from_buffer()
                current_state = {"price": state["price"], "volatility": state["volatility"]}

                # Predict scenarios
                results = await self.predictive_modeler.predict_market_scenarios(
                    current_state=current_state, time_horizon=timedelta(days=1), num_scenarios=200
                )

                # Attach summary to context metadata
                context.metadata["predictions"] = {"count": len(results)}
                logger.info(f"Generated {len(results)} scenario predictions")

            except Exception as e:
                logger.warning(f"Failed to generate predictions: {e}")
        
        return context
    
    def _calculate_risk_metrics(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate basic risk metrics from market data."""
        return {
            "volatility": 0.02,
            "max_drawdown": 0.05,
            "sharpe_ratio": 1.5
        }
    
    def _compute_state_from_buffer(self) -> Dict[str, float]:
        """Compute current price and volatility from the rolling buffer without pandas."""
        if not self.market_data_buffer:
            return {"price": 0.0, "volatility": 0.02}

        # Extract numeric prices in order
        prices: list[float] = []
        for item in list(self.market_data_buffer):
            if isinstance(item, dict):
                val: Optional[Any] = item.get("close") if "close" in item else item.get("price")
                if val is None:
                    continue
                try:
                    prices.append(float(cast(Union[float, int, str], val)))
                except (TypeError, ValueError):
                    continue

        price = float(prices[-1]) if prices else 0.0

        # Compute simple volatility as std of simple returns
        if len(prices) >= 2:
            rets: list[float] = []
            for i in range(1, len(prices)):
                prev = prices[i - 1]
                curr = prices[i]
                if prev:
                    rets.append((curr - prev) / prev)
            if rets:
                mean = sum(rets) / len(rets)
                variance = sum((r - mean) ** 2 for r in rets) / len(rets)
                vol = float(math.sqrt(variance))
            else:
                vol = 0.02
        else:
            vol = 0.02

        return {"price": price, "volatility": vol}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the predictive model."""
        if self.predictive_modeler:
            return {"status": "enabled"}
        return {"status": "disabled", "reason": "no_model_configured"}
    
    def is_predictive_enabled(self) -> bool:
        """Check if predictive features are enabled."""
        return self.predictive_modeler is not None

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_thinking_manager():
        """Test the thinking manager with predictive integration."""
        
        config = SystemConfig()
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
        print(f"  Has predictions: {'predictions' in context.metadata}")
        if 'predictions' in context.metadata:
            print(f"  Predictions count: {context.metadata['predictions'].get('count', 0)}")
    
    # Run test
    asyncio.run(test_thinking_manager())
