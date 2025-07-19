"""
EMP Sensory Cortex v1.1

The sensory cortex integrates inputs from multiple sensory organs
to create a unified perception of the market environment.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from src.core.interfaces import MarketData, SensorySignal
from src.core.events import publish_sensory_event
from src.core.exceptions import SensoryException

logger = logging.getLogger(__name__)


@dataclass
class SensoryReading:
    """Unified sensory reading from multiple organs."""
    timestamp: datetime
    organs: Dict[str, SensorySignal]
    composite_signal: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]


class SensoryCortex:
    """Central sensory cortex for integrating multiple sensory organs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.organs: Dict[str, Any] = {}
        self.integration_weights: Dict[str, float] = self.config.get('integration_weights', {})
        self.running = False
        
    def register_organ(self, organ_id: str, organ: Any):
        """Register a sensory organ with the cortex."""
        self.organs[organ_id] = organ
        logger.info(f"Registered sensory organ: {organ_id}")
        
    def unregister_organ(self, organ_id: str):
        """Unregister a sensory organ from the cortex."""
        if organ_id in self.organs:
            del self.organs[organ_id]
            logger.info(f"Unregistered sensory organ: {organ_id}")
            
    async def process_market_data(self, data: MarketData) -> SensoryReading:
        """Process market data through all registered organs."""
        try:
            organ_signals = {}
            
            # Process through each organ
            for organ_id, organ in self.organs.items():
                try:
                    signal = organ.perceive(data)
                    organ_signals[organ_id] = signal
                    
                    # Publish individual organ signal
                    await publish_sensory_event(
                        signal={
                            'organ_id': organ_id,
                            'signal_type': signal.signal_type,
                            'value': signal.value,
                            'confidence': signal.confidence,
                            'metadata': signal.metadata
                        },
                        organ_id=organ_id
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing data in organ {organ_id}: {e}")
                    continue
                    
            # Create composite signal
            composite_signal = self._create_composite_signal(organ_signals)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(organ_signals)
            
            # Create unified reading
            reading = SensoryReading(
                timestamp=data.timestamp,
                organs=organ_signals,
                composite_signal=composite_signal,
                confidence=overall_confidence,
                metadata={
                    'organ_count': len(organ_signals),
                    'data_source': data.source,
                    'symbol': data.symbol
                }
            )
            
            logger.debug(f"Created sensory reading with {len(organ_signals)} organs")
            return reading
            
        except Exception as e:
            raise SensoryException(f"Error in sensory cortex processing: {e}")
            
    def _create_composite_signal(self, organ_signals: Dict[str, SensorySignal]) -> Dict[str, Any]:
        """Create a composite signal from multiple organ signals."""
        composite = {
            'trend': 0.0,
            'momentum': 0.0,
            'volatility': 0.0,
            'sentiment': 0.0,
            'risk': 0.0,
            'confidence': 0.0
        }
        
        total_weight = 0
        
        for organ_id, signal in organ_signals.items():
            weight = self.integration_weights.get(organ_id, 1.0)
            total_weight += weight
            
            # Map organ signals to composite dimensions
            if signal.signal_type == "price_composite":
                composite['trend'] += signal.value * weight * 0.4
                composite['momentum'] += signal.value * weight * 0.3
                composite['volatility'] += signal.value * weight * 0.3
                
            elif signal.signal_type == "volume_composite":
                composite['momentum'] += signal.value * weight * 0.5
                composite['confidence'] += signal.confidence * weight
                
            elif signal.signal_type == "sentiment_composite":
                composite['sentiment'] += signal.value * weight
                
            elif signal.signal_type == "risk_composite":
                composite['risk'] += signal.value * weight
                
        # Normalize by total weight
        if total_weight > 0:
            for key in composite:
                composite[key] /= total_weight
                
        return composite
        
    def _calculate_overall_confidence(self, organ_signals: Dict[str, SensorySignal]) -> float:
        """Calculate overall confidence from all organ signals."""
        if not organ_signals:
            return 0.0
            
        confidences = [signal.confidence for signal in organ_signals.values()]
        return sum(confidences) / len(confidences)
        
    async def start(self):
        """Start the sensory cortex."""
        self.running = True
        logger.info("Sensory cortex started")
        
    async def stop(self):
        """Stop the sensory cortex."""
        self.running = False
        logger.info("Sensory cortex stopped")
        
    def get_organ_status(self) -> Dict[str, Any]:
        """Get status of all registered organs."""
        status = {}
        for organ_id, organ in self.organs.items():
            status[organ_id] = {
                'registered': True,
                'calibrated': getattr(organ, 'calibrated', False),
                'type': type(organ).__name__
            }
        return status
        
    def calibrate_all_organs(self) -> bool:
        """Calibrate all registered organs."""
        success_count = 0
        total_count = len(self.organs)
        
        for organ_id, organ in self.organs.items():
            try:
                if hasattr(organ, 'calibrate'):
                    if organ.calibrate():
                        success_count += 1
                        logger.info(f"Organ {organ_id} calibrated successfully")
                    else:
                        logger.warning(f"Organ {organ_id} calibration failed")
                else:
                    logger.warning(f"Organ {organ_id} has no calibrate method")
            except Exception as e:
                logger.error(f"Error calibrating organ {organ_id}: {e}")
                
        logger.info(f"Calibrated {success_count}/{total_count} organs")
        return success_count == total_count 