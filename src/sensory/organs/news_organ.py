"""
EMP News Organ v1.1

News sentiment analysis organ for the sensory layer.
Processes news data and extracts sentiment signals.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.core.base import MarketData

if TYPE_CHECKING:
    from typing import Protocol

    class _SensoryOrganProto(Protocol):
        name: str
        config: dict[str, Any]

        def __init__(self, name: str, config: Optional[dict[str, Any]] = ...) -> None: ...

    class _SensoryReadingProto(Protocol):
        organ_name: str
        timestamp: datetime
        data: dict[str, Any]
        metadata: dict[str, Any]


# Minimal runtime placeholders preserving behavior
class SensoryOrgan:
    def __init__(self, name: str, config: Optional[dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}


class SensoryReading:
    def __init__(
        self, organ_name: str, timestamp: datetime, data: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        self.organ_name = organ_name
        self.timestamp = timestamp
        self.data = data
        self.metadata = metadata


logger = logging.getLogger(__name__)


class NewsOrgan(SensoryOrgan):
    """News sentiment analysis organ."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("news_organ", config or {})
        self.sentiment_cache: dict[str, float] = {}
        self.keywords = self.config.get(
            "keywords", ["inflation", "fed", "ecb", "gdp", "employment"]
        )

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
                    "keywords_found": self.keywords[:2],
                },
                metadata={"organ_version": "1.1.0", "processing_time_ms": 50},
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
        if hasattr(market_data, "close") and hasattr(market_data, "open"):
            price_change: float = (float(market_data.close) - float(market_data.open)) / float(
                market_data.open
            )
            base_sentiment: float = 0.5 + (price_change * 10)  # Scale price change to sentiment
            base_sentiment = max(0.0, min(1.0, base_sentiment))  # Clamp to [0,1]
        else:
            base_sentiment = 0.5

        # Add some noise
        noise: float = random.uniform(-0.1, 0.1)
        sentiment: float = float(base_sentiment + noise)
        return float(max(0.0, min(1.0, sentiment)))

    def _create_error_reading(self, timestamp: datetime) -> SensoryReading:
        """Create error reading when processing fails."""
        return SensoryReading(
            organ_name=self.name,
            timestamp=timestamp,
            data={
                "sentiment_score": 0.5,
                "confidence": 0.0,
                "source_count": 0,
                "keywords_found": [],
            },
            metadata={"organ_version": "1.1.0", "error": "Processing failed"},
        )
