"""
EMP Sensory Cortex v1.1

Sensory cortex integration for the sensory layer.
Integrates sensory organs and publishes MarketUnderstanding events.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from ...core.events import MarketData, SensorySignal, MarketUnderstanding
from ...core.event_bus import publish_event, EventType

logger = logging.getLogger(__name__)


@dataclass
class SensoryOrgan:
    """Sensory organ configuration."""
    name: str
    organ_type: str
    enabled: bool = True
    weight: float = 1.0
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class SensoryCortex:
    """Sensory cortex integrating multiple sensory organs."""
    
    def __init__(self):
        self.organs: Dict[str, SensoryOrgan] = {}
        self.signal_history: List[SensorySignal] = []
        self.understanding_history: List[MarketUnderstanding] = []
        self.max_history: int = 1000
        
        logger.info("Sensory Cortex initialized")
        
    def register_organ(self, organ: SensoryOrgan):
        """Register a sensory organ."""
        self.organs[organ.name] = organ
        logger.info(f"Registered sensory organ: {organ.name} ({organ.organ_type})")
        
    def unregister_organ(self, organ_name: str):
        """Unregister a sensory organ."""
        if organ_name in self.organs:
            del self.organs[organ_name]
            logger.info(f"Unregistered sensory organ: {organ_name}")
            
    def enable_organ(self, organ_name: str):
        """Enable a sensory organ."""
        if organ_name in self.organs:
            self.organs[organ_name].enabled = True
            logger.info(f"Enabled sensory organ: {organ_name}")
            
    def disable_organ(self, organ_name: str):
        """Disable a sensory organ."""
        if organ_name in self.organs:
            self.organs[organ_name].enabled = False
            logger.info(f"Disabled sensory organ: {organ_name}")
            
    def set_organ_weight(self, organ_name: str, weight: float):
        """Set weight for a sensory organ."""
        if organ_name in self.organs:
            self.organs[organ_name].weight = weight
            logger.info(f"Set weight for {organ_name}: {weight}")
            
    async def process_market_data(self, market_data: MarketData) -> MarketUnderstanding:
        """Process market data through all sensory organs."""
        try:
            # Collect signals from all enabled organs
            signals = []
            
            for organ_name, organ in self.organs.items():
                if not organ.enabled:
                    continue
                    
                # Process through organ (simplified - in real implementation, 
                # each organ would have its own processing logic)
                signal = await self._process_through_organ(organ, market_data)
                if signal:
                    signals.append(signal)
                    
            # Calculate composite score
            composite_score = self._calculate_composite_score(signals)
            
            # Calculate confidence
            confidence = self._calculate_confidence(signals)
            
            # Create market understanding
            understanding = MarketUnderstanding(
                timestamp=datetime.now(),
                symbol=market_data.symbol,
                signals=signals,
                composite_score=composite_score,
                confidence=confidence,
                metadata={
                    "cortex_version": "1.1.0",
                    "active_organs": [name for name, org in self.organs.items() if org.enabled],
                    "signal_count": len(signals)
                }
            )
            
            # Store in history
            self._store_understanding(understanding)
            
            # Publish to event bus
            await publish_event(understanding)
            
            logger.debug(f"Processed market data for {market_data.symbol}: {composite_score:.3f} score")
            return understanding
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return self._create_default_understanding(market_data)
            
    async def _process_through_organ(self, organ: SensoryOrgan, 
                                   market_data: MarketData) -> Optional[SensorySignal]:
        """Process market data through a specific organ."""
        try:
            # This is a simplified implementation
            # In a real system, each organ would have its own processing logic
            
            if organ.organ_type == "price_organ":
                return await self._process_price_organ(organ, market_data)
            elif organ.organ_type == "volume_organ":
                return await self._process_volume_organ(organ, market_data)
            elif organ.organ_type == "orderbook_organ":
                return await self._process_orderbook_organ(organ, market_data)
            elif organ.organ_type == "news_organ":
                return await self._process_news_organ(organ, market_data)
            elif organ.organ_type == "sentiment_organ":
                return await self._process_sentiment_organ(organ, market_data)
            elif organ.organ_type == "economic_organ":
                return await self._process_economic_organ(organ, market_data)
            else:
                logger.warning(f"Unknown organ type: {organ.organ_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing through organ {organ.name}: {e}")
            return None
            
    async def _process_price_organ(self, organ: SensoryOrgan, 
                                 market_data: MarketData) -> SensorySignal:
        """Process through price organ."""
        # Simplified price analysis
        price_change = (market_data.close - market_data.open) / market_data.open
        price_volatility = (market_data.high - market_data.low) / market_data.open
        
        # Calculate signal value
        signal_value = price_change * organ.weight
        
        return SensorySignal(
            timestamp=datetime.now(),
            signal_type="price_movement",
            value=signal_value,
            confidence=0.8,
            metadata={
                "organ": organ.name,
                "price_change": price_change,
                "volatility": price_volatility
            }
        )
        
    async def _process_volume_organ(self, organ: SensoryOrgan, 
                                  market_data: MarketData) -> SensorySignal:
        """Process through volume organ."""
        # Simplified volume analysis
        volume_signal = 0.0  # Would be calculated based on volume patterns
        
        return SensorySignal(
            timestamp=datetime.now(),
            signal_type="volume_pattern",
            value=volume_signal,
            confidence=0.7,
            metadata={
                "organ": organ.name,
                "volume": market_data.volume
            }
        )
        
    async def _process_orderbook_organ(self, organ: SensoryOrgan, 
                                     market_data: MarketData) -> SensorySignal:
        """Process through orderbook organ."""
        # Simplified orderbook analysis
        if market_data.bid and market_data.ask:
            spread = (market_data.ask - market_data.bid) / market_data.bid
            spread_signal = -spread * 10  # Negative signal for wide spreads
        else:
            spread_signal = 0.0
            
        return SensorySignal(
            timestamp=datetime.now(),
            signal_type="orderbook_imbalance",
            value=spread_signal,
            confidence=0.6,
            metadata={
                "organ": organ.name,
                "bid": market_data.bid,
                "ask": market_data.ask
            }
        )
        
    async def _process_news_organ(self, organ: SensoryOrgan, 
                                market_data: MarketData) -> SensorySignal:
        """Process through news organ."""
        # Simplified news analysis
        news_signal = 0.0  # Would be calculated based on news sentiment
        
        return SensorySignal(
            timestamp=datetime.now(),
            signal_type="news_sentiment",
            value=news_signal,
            confidence=0.5,
            metadata={
                "organ": organ.name,
                "news_count": 0
            }
        )
        
    async def _process_sentiment_organ(self, organ: SensoryOrgan, 
                                     market_data: MarketData) -> SensorySignal:
        """Process through sentiment organ."""
        # Simplified sentiment analysis
        sentiment_signal = 0.0  # Would be calculated based on social sentiment
        
        return SensorySignal(
            timestamp=datetime.now(),
            signal_type="social_sentiment",
            value=sentiment_signal,
            confidence=0.4,
            metadata={
                "organ": organ.name,
                "sentiment_score": 0.0
            }
        )
        
    async def _process_economic_organ(self, organ: SensoryOrgan, 
                                    market_data: MarketData) -> SensorySignal:
        """Process through economic organ."""
        # Simplified economic analysis
        economic_signal = 0.0  # Would be calculated based on economic indicators
        
        return SensorySignal(
            timestamp=datetime.now(),
            signal_type="economic_indicator",
            value=economic_signal,
            confidence=0.6,
            metadata={
                "organ": organ.name,
                "economic_indicators": []
            }
        )
        
    def _calculate_composite_score(self, signals: List[SensorySignal]) -> float:
        """Calculate composite score from all signals."""
        if not signals:
            return 0.0
            
        # Weighted average of signal values
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in signals:
            # Get organ weight from metadata
            organ_name = signal.metadata.get("organ", "unknown")
            organ = self.organs.get(organ_name)
            weight = organ.weight if organ else 1.0
            
            weighted_sum += signal.value * weight * signal.confidence
            total_weight += weight * signal.confidence
            
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
            
    def _calculate_confidence(self, signals: List[SensorySignal]) -> float:
        """Calculate overall confidence from signals."""
        if not signals:
            return 0.0
            
        # Average confidence weighted by signal importance
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for signal in signals:
            organ_name = signal.metadata.get("organ", "unknown")
            organ = self.organs.get(organ_name)
            weight = organ.weight if organ else 1.0
            
            weighted_confidence += signal.confidence * weight
            total_weight += weight
            
        if total_weight > 0:
            return weighted_confidence / total_weight
        else:
            return 0.0
            
    def _store_understanding(self, understanding: MarketUnderstanding):
        """Store market understanding in history."""
        self.understanding_history.append(understanding)
        
        # Limit history size
        if len(self.understanding_history) > self.max_history:
            self.understanding_history.pop(0)
            
    def _create_default_understanding(self, market_data: MarketData) -> MarketUnderstanding:
        """Create default market understanding when processing fails."""
        return MarketUnderstanding(
            timestamp=datetime.now(),
            symbol=market_data.symbol,
            signals=[],
            composite_score=0.0,
            confidence=0.1,
            metadata={
                "cortex_version": "1.1.0",
                "error": "Processing failed",
                "active_organs": []
            }
        )
        
    def get_understanding_history(self, limit: Optional[int] = None) -> List[MarketUnderstanding]:
        """Get market understanding history."""
        if limit:
            return self.understanding_history[-limit:]
        return self.understanding_history.copy()
        
    def get_organ_status(self) -> Dict[str, Any]:
        """Get status of all sensory organs."""
        status = {}
        for name, organ in self.organs.items():
            status[name] = {
                "type": organ.organ_type,
                "enabled": organ.enabled,
                "weight": organ.weight,
                "parameters": organ.parameters
            }
        return status
        
    def get_cortex_statistics(self) -> Dict[str, Any]:
        """Get sensory cortex statistics."""
        return {
            "total_organs": len(self.organs),
            "enabled_organs": len([org for org in self.organs.values() if org.enabled]),
            "understanding_history_size": len(self.understanding_history),
            "signal_history_size": len(self.signal_history),
            "max_history": self.max_history
        }
        
    def clear_history(self):
        """Clear understanding and signal history."""
        self.understanding_history.clear()
        self.signal_history.clear()
        logger.info("Sensory cortex history cleared") 