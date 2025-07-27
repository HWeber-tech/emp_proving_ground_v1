"""
EMP Sentiment Organ v1.1

Market sentiment analysis organ for the sensory layer.
Processes social media and market sentiment data.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..core.base import SensoryOrgan, MarketData, SensoryReading

logger = logging.getLogger(__name__)


class SentimentOrgan(SensoryOrgan):
    """Market sentiment analysis organ."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("sentiment_organ", config or {})
        self.sentiment_threshold = self.config.get("sentiment_threshold", 0.6)
        
    async def process(self, market_data: MarketData) -> SensoryReading:
        """Process sentiment data."""
        try:
            # Mock sentiment analysis
            sentiment_score = self._calculate_mock_sentiment(market_data)
            
            reading = SensoryReading(
                organ_name=self.name,
                timestamp=market_data.timestamp,
                data={
                    "overall_sentiment": sentiment_score,
                    "confidence": 0.8,
                    "source": "social_media"
                },
                metadata={
                    "organ_version": "1.1.0",
                    "processing_time_ms": 30
                }
            )
            
            logger.debug(f"Sentiment organ processed: sentiment={sentiment_score:.3f}")
            return reading
            
        except Exception as e:
            logger.error(f"Error in sentiment organ processing: {e}")
            return self._create_error_reading(market_data.timestamp)
            
    def _calculate_mock_sentiment(self, market_data: MarketData) -> float:
        """Calculate mock sentiment."""
        import random
        random.seed(hash(market_data.timestamp) % 1000)
        return random.uniform(0.3, 0.8)
        
    def _create_error_reading(self, timestamp: datetime) -> SensoryReading:
        """Create error reading when processing fails."""
        return SensoryReading(
            organ_name=self.name,
            timestamp=timestamp,
            data={
                "overall_sentiment": 0.5,
                "confidence": 0.0,
                "source": "error"
            },
            metadata={
                "organ_version": "1.1.0",
                "error": "Processing failed"
            }
        ) 