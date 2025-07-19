"""
EMP News Organ v1.1

News sentiment analysis organ for the sensory layer.
Processes news data and extracts sentiment signals.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.base import SensoryOrgan, MarketData, SensoryReading

logger = logging.getLogger(__name__)


class NewsOrgan(SensoryOrgan):
    """News sentiment analysis organ."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("news_organ", config or {})
        self.sentiment_cache: Dict[str, float] = {}
        self.keywords = self.config.get("keywords", ["inflation", "fed", "ecb", "gdp", "employment"])
        
    async def process(self, market_data: MarketData) -> SensoryReading:
        """Process news data and extract sentiment."""
        try:
            # In a real implementation, this would connect to news APIs
            # For now, return a mock sentiment reading
            sentiment_score = self._calculate_mock_sentiment(market_data)
            
            reading = SensoryReading(
                organ_name=self.name,
                timestamp=market_data.timestamp,
                data={
                    "sentiment_score": sentiment_score,
                    "confidence": 0.7,
                    "source_count": 5,
                    "keywords_found": self.keywords[:2]
                },
                metadata={
                    "organ_version": "1.1.0",
                    "processing_time_ms": 50
                }
            )
            
            logger.debug(f"News organ processed: sentiment={sentiment_score:.3f}")
            return reading
            
        except Exception as e:
            logger.error(f"Error in news organ processing: {e}")
            return self._create_error_reading(market_data.timestamp)
            
    def _calculate_mock_sentiment(self, market_data: MarketData) -> float:
        """Calculate mock sentiment based on price movement."""
        # Simple mock sentiment based on price volatility
        # In reality, this would analyze actual news content
        import random
        random.seed(hash(market_data.timestamp) % 1000)
        
        # Base sentiment on price movement
        if hasattr(market_data, 'close') and hasattr(market_data, 'open'):
            price_change = (market_data.close - market_data.open) / market_data.open
            base_sentiment = 0.5 + (price_change * 10)  # Scale price change to sentiment
            base_sentiment = max(0.0, min(1.0, base_sentiment))  # Clamp to [0,1]
        else:
            base_sentiment = 0.5
            
        # Add some noise
        noise = random.uniform(-0.1, 0.1)
        sentiment = base_sentiment + noise
        return max(0.0, min(1.0, sentiment))
        
    def _create_error_reading(self, timestamp: datetime) -> SensoryReading:
        """Create error reading when processing fails."""
        return SensoryReading(
            organ_name=self.name,
            timestamp=timestamp,
            data={
                "sentiment_score": 0.5,
                "confidence": 0.0,
                "source_count": 0,
                "keywords_found": []
            },
            metadata={
                "organ_version": "1.1.0",
                "error": "Processing failed"
            }
        ) 