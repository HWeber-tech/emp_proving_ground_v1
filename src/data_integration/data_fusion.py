"""
Cross-Source Data Fusion Engine - Phase 2 Implementation

This module provides intelligent data fusion capabilities that combine data from multiple sources,
handle conflicts, and provide confidence-weighted outputs.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Advanced Data Integration
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.sensory.core.base import MarketData

from .data_validation import MarketDataValidator, ValidationLevel

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Available data sources"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FRED = "fred"
    NEWSAPI = "newsapi"
    MOCK = "mock"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving data conflicts"""
    WEIGHTED_AVERAGE = "weighted_average"
    HIGHEST_CONFIDENCE = "highest_confidence"
    MOST_RECENT = "most_recent"
    CONSENSUS = "consensus"


@dataclass
class SourceWeight:
    """Weight configuration for data sources"""
    source: DataSource
    weight: float  # 0.0 to 1.0
    confidence_threshold: float = 0.7
    max_age_seconds: int = 300  # 5 minutes


@dataclass
class FusedDataPoint:
    """A fused data point from multiple sources"""
    timestamp: datetime
    bid: float
    ask: float
    volume: int
    volatility: float
    confidence: float
    sources_used: List[DataSource]
    conflicts_resolved: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionConfig:
    """Configuration for data fusion"""
    sources: List[SourceWeight]
    conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.WEIGHTED_AVERAGE
    min_confidence: float = 0.6
    max_age_seconds: int = 300
    enable_validation: bool = True
    validation_level: ValidationLevel = ValidationLevel.STRICT


class DataHarmonizer:
    """Harmonizes data from different sources to common format"""
    
    def __init__(self):
        self.validator = MarketDataValidator()
    
    def harmonize_market_data(self, data_points: List[Tuple[DataSource, MarketData]]) -> List[Tuple[DataSource, MarketData]]:
        """Harmonize market data from different sources"""
        harmonized = []
        
        for source, data in data_points:
            # Validate data quality
            validation = self.validator.validate_market_data(data, ValidationLevel.BASIC)
            
            if validation.is_valid:
                # Normalize timestamp to nearest minute
                normalized_timestamp = self._normalize_timestamp(data.timestamp)
                
                # Create harmonized data point
                harmonized_data = MarketData(
                    timestamp=normalized_timestamp,
                    bid=data.bid,
                    ask=data.ask,
                    volume=data.volume,
                    volatility=data.volatility
                )
                
                harmonized.append((source, harmonized_data))
                logger.debug(f"Harmonized data from {source.value}: {harmonized_data}")
            else:
                logger.warning(f"Invalid data from {source.value}: {validation.issues}")
        
        return harmonized
    
    def _normalize_timestamp(self, timestamp: datetime) -> datetime:
        """Normalize timestamp to nearest minute"""
        return timestamp.replace(second=0, microsecond=0)
    
    def align_timeframes(self, data_points: List[Tuple[DataSource, MarketData]], 
                        target_interval: timedelta = timedelta(minutes=1)) -> List[Tuple[DataSource, MarketData]]:
        """Align data points to common timeframe"""
        if not data_points:
            return []
        
        # Group by time intervals
        aligned_data = {}
        for source, data in data_points:
            interval_start = self._get_interval_start(data.timestamp, target_interval)
            
            if interval_start not in aligned_data:
                aligned_data[interval_start] = []
            
            aligned_data[interval_start].append((source, data))
        
        # For each interval, select the best data point
        result = []
        for interval_start, points in aligned_data.items():
            if len(points) == 1:
                result.append(points[0])
            else:
                # Multiple points in same interval - select most recent
                best_point = max(points, key=lambda x: x[1].timestamp)
                result.append(best_point)
        
        return result
    
    def _get_interval_start(self, timestamp: datetime, interval: timedelta) -> datetime:
        """Get the start of the interval containing the timestamp"""
        seconds = int(timestamp.timestamp())
        interval_seconds = int(interval.total_seconds())
        aligned_seconds = (seconds // interval_seconds) * interval_seconds
        return datetime.fromtimestamp(aligned_seconds)


class ConflictResolver:
    """Resolves conflicts between data from different sources"""
    
    def __init__(self, strategy: ConflictResolutionStrategy):
        self.strategy = strategy
    
    def resolve_conflicts(self, data_points: List[Tuple[DataSource, MarketData, float]], 
                         weights: Dict[DataSource, float]) -> Tuple[MarketData, List[str]]:
        """Resolve conflicts between data points"""
        if not data_points:
            return None, []
        
        if len(data_points) == 1:
            source, data, confidence = data_points[0]
            return data, []
        
        conflicts = []
        
        # Check for conflicts
        bids = [data.bid for _, data, _ in data_points]
        asks = [data.ask for _, data, _ in data_points]
        volumes = [data.volume for _, data, _ in data_points]
        
        # Detect conflicts
        bid_range = max(bids) - min(bids)
        ask_range = max(asks) - min(asks)
        volume_range = max(volumes) - min(volumes)
        
        if bid_range > 0.001:  # 1 pip for forex
            conflicts.append(f"Bid conflict: range={bid_range:.5f}")
        
        if ask_range > 0.001:
            conflicts.append(f"Ask conflict: range={ask_range:.5f}")
        
        if volume_range > 1000:
            conflicts.append(f"Volume conflict: range={volume_range}")
        
        # Resolve based on strategy
        if self.strategy == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_resolution(data_points, weights), conflicts
        elif self.strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            return self._highest_confidence_resolution(data_points), conflicts
        elif self.strategy == ConflictResolutionStrategy.MOST_RECENT:
            return self._most_recent_resolution(data_points), conflicts
        elif self.strategy == ConflictResolutionStrategy.CONSENSUS:
            return self._consensus_resolution(data_points), conflicts
        else:
            # Default to weighted average
            return self._weighted_average_resolution(data_points, weights), conflicts
    
    def _weighted_average_resolution(self, data_points: List[Tuple[DataSource, MarketData, float]], 
                                   weights: Dict[DataSource, float]) -> MarketData:
        """Resolve conflicts using weighted average"""
        total_weight = 0
        weighted_bid = 0
        weighted_ask = 0
        weighted_volume = 0
        weighted_volatility = 0
        
        for source, data, confidence in data_points:
            weight = weights.get(source, 0.5) * confidence
            total_weight += weight
            
            weighted_bid += data.bid * weight
            weighted_ask += data.ask * weight
            weighted_volume += data.volume * weight
            weighted_volatility += data.volatility * weight
        
        if total_weight == 0:
            # Fallback to simple average
            return self._simple_average_resolution(data_points)
        
        # Use most recent timestamp
        most_recent = max(data_points, key=lambda x: x[1].timestamp)
        
        return MarketData(
            timestamp=most_recent[1].timestamp,
            bid=weighted_bid / total_weight,
            ask=weighted_ask / total_weight,
            volume=int(weighted_volume / total_weight),
            volatility=weighted_volatility / total_weight
        )
    
    def _highest_confidence_resolution(self, data_points: List[Tuple[DataSource, MarketData, float]]) -> MarketData:
        """Resolve conflicts by selecting highest confidence data"""
        best_point = max(data_points, key=lambda x: x[2])
        return best_point[1]
    
    def _most_recent_resolution(self, data_points: List[Tuple[DataSource, MarketData, float]]) -> MarketData:
        """Resolve conflicts by selecting most recent data"""
        most_recent = max(data_points, key=lambda x: x[1].timestamp)
        return most_recent[1]
    
    def _consensus_resolution(self, data_points: List[Tuple[DataSource, MarketData, float]]) -> MarketData:
        """Resolve conflicts using consensus (median)"""
        bids = [data.bid for _, data, _ in data_points]
        asks = [data.ask for _, data, _ in data_points]
        volumes = [data.volume for _, data, _ in data_points]
        volatilities = [data.volatility for _, data, _ in data_points]
        
        # Use most recent timestamp
        most_recent = max(data_points, key=lambda x: x[1].timestamp)
        
        return MarketData(
            timestamp=most_recent[1].timestamp,
            bid=np.median(bids),
            ask=np.median(asks),
            volume=int(np.median(volumes)),
            volatility=np.median(volatilities)
        )
    
    def _simple_average_resolution(self, data_points: List[Tuple[DataSource, MarketData, float]]) -> MarketData:
        """Simple average resolution as fallback"""
        bids = [data.bid for _, data, _ in data_points]
        asks = [data.ask for _, data, _ in data_points]
        volumes = [data.volume for _, data, _ in data_points]
        volatilities = [data.volatility for _, data, _ in data_points]
        
        # Use most recent timestamp
        most_recent = max(data_points, key=lambda x: x[1].timestamp)
        
        return MarketData(
            timestamp=most_recent[1].timestamp,
            bid=np.mean(bids),
            ask=np.mean(asks),
            volume=int(np.mean(volumes)),
            volatility=np.mean(volatilities)
        )


class DataFusionEngine:
    """Main data fusion engine that combines data from multiple sources"""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.harmonizer = DataHarmonizer()
        self.resolver = ConflictResolver(config.conflict_strategy)
        self.validator = MarketDataValidator()
        
        # Create source weights dictionary
        self.source_weights = {sw.source: sw for sw in config.sources}
        
        logger.info(f"Data fusion engine initialized with {len(config.sources)} sources")
    
    async def fuse_market_data(self, symbol: str, 
                             data_providers: Dict[DataSource, Any]) -> Optional[FusedDataPoint]:
        """Fuse market data from multiple sources"""
        try:
            # Collect data from all sources
            raw_data_points = []
            
            for source, provider in data_providers.items():
                if source not in self.source_weights:
                    continue
                
                try:
                    # Get data from provider
                    data = await self._get_data_from_provider(provider, symbol, source)
                    if data:
                        raw_data_points.append((source, data))
                except Exception as e:
                    logger.warning(f"Error getting data from {source.value}: {e}")
            
            if not raw_data_points:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Harmonize data
            harmonized_data = self.harmonizer.harmonize_market_data(raw_data_points)
            
            if not harmonized_data:
                logger.warning(f"No valid data after harmonization for {symbol}")
                return None
            
            # Calculate confidence scores
            data_with_confidence = []
            for source, data in harmonized_data:
                confidence = self._calculate_confidence(source, data)
                data_with_confidence.append((source, data, confidence))
            
            # Filter by confidence threshold
            filtered_data = [
                (source, data, conf) for source, data, conf in data_with_confidence
                if conf >= self.config.min_confidence
            ]
            
            if not filtered_data:
                logger.warning(f"No data meets confidence threshold for {symbol}")
                return None
            
            # Resolve conflicts
            weights = {sw.source: sw.weight for sw in self.config.sources}
            resolved_data, conflicts = self.resolver.resolve_conflicts(filtered_data, weights)
            
            if not resolved_data:
                logger.warning(f"Failed to resolve conflicts for {symbol}")
                return None
            
            # Validate final result
            if self.config.enable_validation:
                validation = self.validator.validate_market_data(resolved_data, self.config.validation_level)
                if not validation.is_valid:
                    logger.warning(f"Fused data validation failed for {symbol}: {validation.issues}")
                    return None
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(filtered_data, weights)
            
            # Create fused data point
            fused_point = FusedDataPoint(
                timestamp=resolved_data.timestamp,
                bid=resolved_data.bid,
                ask=resolved_data.ask,
                volume=resolved_data.volume,
                volatility=resolved_data.volatility,
                confidence=overall_confidence,
                sources_used=[source for source, _, _ in filtered_data],
                conflicts_resolved=conflicts,
                metadata={
                    'symbol': symbol,
                    'fusion_strategy': self.config.conflict_strategy.value,
                    'data_points_count': len(filtered_data)
                }
            )
            
            logger.info(f"Fused data for {symbol}: confidence={overall_confidence:.3f}, "
                       f"sources={len(filtered_data)}, conflicts={len(conflicts)}")
            
            return fused_point
            
        except Exception as e:
            logger.error(f"Error fusing data for {symbol}: {e}")
            return None
    
    async def _get_data_from_provider(self, provider: Any, symbol: str, source: DataSource) -> Optional[MarketData]:
        """Get data from a specific provider"""
        try:
            if source == DataSource.YAHOO_FINANCE:
                return await provider.get_market_data(symbol)
            elif source == DataSource.ALPHA_VANTAGE:
                return await provider.get_real_time_quote(symbol)
            elif source == DataSource.MOCK:
                # Mock provider returns data directly
                return provider.get_market_data(symbol)
            else:
                logger.warning(f"Unknown provider type: {source}")
                return None
        except Exception as e:
            logger.error(f"Error getting data from {source.value}: {e}")
            return None
    
    def _calculate_confidence(self, source: DataSource, data: MarketData) -> float:
        """Calculate confidence score for a data point"""
        source_weight = self.source_weights.get(source)
        if not source_weight:
            return 0.0
        
        # Base confidence from source weight
        confidence = source_weight.weight
        
        # Adjust for data age
        age_seconds = (datetime.now() - data.timestamp).total_seconds()
        if age_seconds > source_weight.max_age_seconds:
            age_factor = max(0.1, 1.0 - (age_seconds - source_weight.max_age_seconds) / 300)
            confidence *= age_factor
        
        # Adjust for data quality
        validation = self.validator.validate_market_data(data, ValidationLevel.BASIC)
        if validation.is_valid:
            confidence *= validation.confidence
        else:
            confidence *= 0.5
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_overall_confidence(self, data_points: List[Tuple[DataSource, MarketData, float]], 
                                    weights: Dict[DataSource, float]) -> float:
        """Calculate overall confidence for fused data"""
        if not data_points:
            return 0.0
        
        total_weight = 0
        weighted_confidence = 0
        
        for source, _, confidence in data_points:
            weight = weights.get(source, 0.5)
            total_weight += weight
            weighted_confidence += confidence * weight
        
        if total_weight == 0:
            return np.mean([conf for _, _, conf in data_points])
        
        return weighted_confidence / total_weight
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion engine statistics"""
        return {
            'sources_configured': len(self.config.sources),
            'conflict_strategy': self.config.conflict_strategy.value,
            'min_confidence': self.config.min_confidence,
            'validation_enabled': self.config.enable_validation,
            'validation_level': self.config.validation_level.value
        }


# Example usage and testing
async def test_data_fusion():
    """Test data fusion engine"""
    # Create configuration
    config = FusionConfig(
        sources=[
            SourceWeight(DataSource.YAHOO_FINANCE, 0.6, 0.7, 300),
            SourceWeight(DataSource.ALPHA_VANTAGE, 0.3, 0.8, 300),
            SourceWeight(DataSource.MOCK, 0.1, 0.5, 300)
        ],
        conflict_strategy=ConflictResolutionStrategy.WEIGHTED_AVERAGE,
        min_confidence=0.6
    )
    
    # Create fusion engine
    engine = DataFusionEngine(config)
    
    # Test fusion (would need actual providers)
    print(f"Fusion engine stats: {engine.get_fusion_stats()}")


if __name__ == "__main__":
    asyncio.run(test_data_fusion()) 
