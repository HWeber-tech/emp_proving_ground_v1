"""
EMP Cross-Modal Integration v1.1

Integrates sensory data across multiple modalities.
Combines different sensory inputs into unified perceptions.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.base import SensoryReading, MarketData

logger = logging.getLogger(__name__)


class CrossModalIntegrator:
    """Integrates sensory data across multiple modalities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.integration_weights = self.config.get('integration_weights', {
            'price': 0.4,
            'volume': 0.2,
            'sentiment': 0.2,
            'news': 0.1,
            'economic': 0.1
        })
        
    async def integrate_readings(self, readings: List[SensoryReading], 
                               market_data: MarketData) -> SensoryReading:
        """Integrate multiple sensory readings into a unified perception."""
        try:
            if not readings:
                return self._create_default_integration(market_data.timestamp)
                
            # Group readings by modality
            modality_readings = self._group_by_modality(readings)
            
            # Integrate each modality
            integrated_modalities = {}
            for modality, modality_readings_list in modality_readings.items():
                integrated_modalities[modality] = self._integrate_modality(
                    modality_readings_list
                )
                
            # Combine modalities
            unified_perception = self._combine_modalities(integrated_modalities)
            
            # Create integrated reading
            integrated_reading = SensoryReading(
                organ_name="cross_modal_integrator",
                timestamp=market_data.timestamp,
                data=unified_perception,
                metadata={
                    'modalities_integrated': list(modality_readings.keys()),
                    'reading_count': len(readings),
                    'integration_method': 'weighted_fusion'
                }
            )
            
            logger.debug(f"Cross-modal integration completed: {len(readings)} readings integrated")
            return integrated_reading
            
        except Exception as e:
            logger.error(f"Error in cross-modal integration: {e}")
            return self._create_error_integration(market_data.timestamp)
            
    def _group_by_modality(self, readings: List[SensoryReading]) -> Dict[str, List[SensoryReading]]:
        """Group readings by their modality."""
        modality_groups = {}
        
        for reading in readings:
            # Extract modality from organ name
            modality = self._extract_modality(reading.organ_name)
            
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(reading)
            
        return modality_groups
        
    def _extract_modality(self, organ_name: str) -> str:
        """Extract modality from organ name."""
        if 'price' in organ_name:
            return 'price'
        elif 'volume' in organ_name:
            return 'volume'
        elif 'sentiment' in organ_name:
            return 'sentiment'
        elif 'news' in organ_name:
            return 'news'
        elif 'economic' in organ_name:
            return 'economic'
        else:
            return 'unknown'
            
    def _integrate_modality(self, readings: List[SensoryReading]) -> Dict[str, Any]:
        """Integrate readings within a single modality."""
        if not readings:
            return {}
            
        # Extract key data points
        data_points = []
        confidences = []
        
        for reading in readings:
            if 'data' in reading.data:
                data_points.append(reading.data['data'])
            if 'confidence' in reading.data:
                confidences.append(reading.data['confidence'])
                
        # Calculate integrated values
        if data_points:
            integrated_data = self._calculate_weighted_average(data_points, confidences)
        else:
            integrated_data = {}
            
        # Calculate overall confidence
        overall_confidence = np.mean(confidences) if confidences else 0.5
        
        return {
            'integrated_data': integrated_data,
            'confidence': overall_confidence,
            'reading_count': len(readings)
        }
        
    def _calculate_weighted_average(self, data_points: List[Dict[str, Any]], 
                                  weights: List[float]) -> Dict[str, Any]:
        """Calculate weighted average of data points."""
        if not data_points or not weights:
            return {}
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        else:
            normalized_weights = [w / total_weight for w in weights]
            
        # Calculate weighted averages for numeric values
        integrated_data = {}
        
        # Get all keys from all data points
        all_keys = set()
        for data_point in data_points:
            all_keys.update(data_point.keys())
            
        for key in all_keys:
            values = []
            key_weights = []
            
            for i, data_point in enumerate(data_points):
                if key in data_point:
                    value = data_point[key]
                    if isinstance(value, (int, float)):
                        values.append(value)
                        key_weights.append(normalized_weights[i])
                        
            if values and key_weights:
                # Calculate weighted average
                weighted_sum = sum(v * w for v, w in zip(values, key_weights))
                integrated_data[key] = weighted_sum
            elif values:
                # Fallback to simple average
                integrated_data[key] = np.mean(values)
                
        return integrated_data
        
    def _combine_modalities(self, integrated_modalities: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine integrated modalities into unified perception."""
        unified_perception = {}
        
        # Combine data from all modalities
        for modality, modality_data in integrated_modalities.items():
            if 'integrated_data' in modality_data:
                for key, value in modality_data['integrated_data'].items():
                    unified_key = f"{modality}_{key}"
                    unified_perception[unified_key] = value
                    
        # Calculate overall confidence
        confidences = [modality_data.get('confidence', 0.5) 
                      for modality_data in integrated_modalities.values()]
        overall_confidence = np.mean(confidences) if confidences else 0.5
        
        # Add modality-specific weights
        for modality, modality_data in integrated_modalities.items():
            weight = self.integration_weights.get(modality, 0.1)
            modality_data['weight'] = weight
            
        unified_perception['overall_confidence'] = overall_confidence
        unified_perception['modality_weights'] = self.integration_weights
        unified_perception['modality_data'] = integrated_modalities
        
        return unified_perception
        
    def _create_default_integration(self, timestamp: datetime) -> SensoryReading:
        """Create default integration when no readings are available."""
        return SensoryReading(
            organ_name="cross_modal_integrator",
            timestamp=timestamp,
            data={
                'overall_confidence': 0.0,
                'modality_weights': self.integration_weights,
                'modality_data': {}
            },
            metadata={
                'modalities_integrated': [],
                'reading_count': 0,
                'integration_method': 'default'
            }
        )
        
    def _create_error_integration(self, timestamp: datetime) -> SensoryReading:
        """Create error integration when processing fails."""
        return SensoryReading(
            organ_name="cross_modal_integrator",
            timestamp=timestamp,
            data={
                'overall_confidence': 0.0,
                'modality_weights': self.integration_weights,
                'modality_data': {}
            },
            metadata={
                'modalities_integrated': [],
                'reading_count': 0,
                'integration_method': 'error',
                'error': 'Integration failed'
            }
        ) 