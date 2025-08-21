"""
EMP Economic Organ v1.1

Economic data analysis organ for the sensory layer.
Processes economic indicators and macro data.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.core.base import MarketData, SensoryOrgan, SensoryReading

logger = logging.getLogger(__name__)


class EconomicOrgan(SensoryOrgan):
    """Economic data analysis organ."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("economic_organ", config or {})
        self.indicators = self.config.get("indicators", ["gdp", "inflation", "employment", "interest_rates"])
        
    async def process(self, market_data: MarketData) -> SensoryReading:
        """Process economic data."""
        try:
            # Mock economic analysis
            macro_trend, macro_strength = self._calculate_mock_economic_signals(market_data)
            
            reading = SensoryReading(
                organ_name=self.name,
                timestamp=market_data.timestamp,
                data={
                    "macro_trend": macro_trend,
                    "macro_strength": macro_strength,
                    "indicators_analyzed": self.indicators
                },
                metadata={
                    "organ_version": "1.1.0",
                    "processing_time_ms": 40
                }
            )
            
            logger.debug(f"Economic organ processed: trend={macro_trend}, strength={macro_strength:.3f}")
            return reading
            
        except Exception as e:
            logger.error(f"Error in economic organ processing: {e}")
            return self._create_error_reading(market_data.timestamp)
            
    def _calculate_mock_economic_signals(self, market_data: MarketData) -> tuple:
        """Calculate mock economic signals."""
        import random
        random.seed(hash(market_data.timestamp) % 1000)
        
        # Mock trend calculation
        trends = ["BULLISH", "BEARISH", "NEUTRAL"]
        trend = random.choice(trends)
        strength = random.uniform(0.3, 0.9)
        
        return trend, strength
        
    def _create_error_reading(self, timestamp: datetime) -> SensoryReading:
        """Create error reading when processing fails."""
        return SensoryReading(
            organ_name=self.name,
            timestamp=timestamp,
            data={
                "macro_trend": "NEUTRAL",
                "macro_strength": 0.5,
                "indicators_analyzed": []
            },
            metadata={
                "organ_version": "1.1.0",
                "error": "Processing failed"
            }
        ) 
