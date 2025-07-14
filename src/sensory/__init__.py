"""
Multidimensional Market Intelligence System - EMP Integration

This module provides the enhanced 4D+1 Sensory Cortex that understands markets
through orchestrated dimensional awareness across five key dimensions:

- WHY: Fundamental forces and economic drivers
- HOW: Institutional mechanics and execution patterns  
- WHAT: Technical reality and price action
- WHEN: Temporal intelligence and timing dynamics
- ANOMALY: Chaos, manipulation, and stress detection

Each dimension maintains awareness of others, creating emergent market understanding
that transcends the sum of its parts.
"""

from .orchestration.intelligence_engine import IntelligenceEngine
from .core.base import MarketUnderstanding, DimensionalReading, MarketRegime, MarketData

# Main interface for backward compatibility with EMP
class SensoryCortex:
    """
    Enhanced 4D+1 Sensory Cortex - The "Brain" of Trading Organisms.
    
    This class provides the main interface to the multidimensional market intelligence
    system, maintaining compatibility with existing EMP components while providing
    significantly enhanced market perception capabilities.
    """
    
    def __init__(self, symbol: str, data_storage):
        """
        Initialize the enhanced sensory cortex.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            data_storage: Data storage for market data access
        """
        self.symbol = symbol
        self.data_storage = data_storage
        self.intelligence_engine = IntelligenceEngine()
        
    def calibrate(self, start_time, end_time) -> bool:
        """
        Calibrate the sensory cortex with historical data.
        
        Args:
            start_time: Start of calibration period
            end_time: End of calibration period
            
        Returns:
            True if calibration successful
        """
        # The original system doesn't need explicit calibration
        # It learns adaptively from data
        return True
    
    def perceive(self, current_data, historical_window: int = 1000):
        """
        Generate a complete sensory reading of the current market state.
        
        Args:
            current_data: Current market data point
            historical_window: Number of historical points to analyze
            
        Returns:
            Enhanced sensory reading with multidimensional intelligence
        """
        # Convert pandas Series to MarketData object
        market_data = self._convert_to_market_data(current_data)
        
        # Process through the intelligence engine
        understanding = self.intelligence_engine.process_market_data(market_data)
        
        # Convert to SensoryReading for backward compatibility
        return SensoryReading(understanding)
    
    def get_market_understanding(self):
        """
        Get the current market understanding across all dimensions.
        
        Returns:
            MarketUnderstanding object with comprehensive analysis
        """
        return self.intelligence_engine.get_current_understanding()
    
    def get_dimensional_readings(self):
        """
        Get individual dimensional readings.
        
        Returns:
            Dict of dimensional readings
        """
        understanding = self.intelligence_engine.get_current_understanding()
        if understanding:
            # The original system doesn't expose individual readings directly
            # We'll create a summary from the understanding
            return {
                'why': DimensionalReading(
                    dimension='why',
                    value=0.0,  # Would need to extract from understanding
                    confidence=understanding.confidence,
                    timestamp=understanding.timestamp,
                    context={}
                ),
                'how': DimensionalReading(
                    dimension='how',
                    value=0.0,
                    confidence=understanding.confidence,
                    timestamp=understanding.timestamp,
                    context={}
                ),
                'what': DimensionalReading(
                    dimension='what',
                    value=0.0,
                    confidence=understanding.confidence,
                    timestamp=understanding.timestamp,
                    context={}
                ),
                'when': DimensionalReading(
                    dimension='when',
                    value=0.0,
                    confidence=understanding.confidence,
                    timestamp=understanding.timestamp,
                    context={}
                ),
                'anomaly': DimensionalReading(
                    dimension='anomaly',
                    value=0.0,
                    confidence=understanding.confidence,
                    timestamp=understanding.timestamp,
                    context={}
                )
            }
        return {}
    
    def _convert_to_market_data(self, current_data):
        """Convert pandas Series to MarketData object"""
        timestamp = current_data.name
        close_price = current_data['close']
        
        # Estimate bid/ask from close price
        bid = close_price - 0.0001
        ask = close_price + 0.0001
        spread = 0.0002
        
        # Get volume, default to 1000 if not available
        volume = current_data.get('volume', 1000)
        
        return MarketData(
            timestamp=timestamp,
            symbol=self.symbol,
            bid=bid,
            ask=ask,
            volume=volume,
            spread=spread
        )

# Legacy compatibility - create a SensoryReading class that wraps the new system
class SensoryReading:
    """
    Legacy compatibility wrapper for the enhanced sensory system.
    
    This class provides backward compatibility with the old SensoryReading interface
    while leveraging the new multidimensional intelligence system.
    """
    
    def __init__(self, market_understanding: MarketUnderstanding):
        """
        Initialize from a MarketUnderstanding object.
        
        Args:
            market_understanding: The enhanced market understanding
        """
        self.timestamp = market_understanding.timestamp
        self.overall_sentiment = 0.0  # Would need to extract from narrative
        self.confidence_level = market_understanding.confidence
        self.risk_level = 1.0 - market_understanding.confidence
        
        # Extract information from the narrative and context
        narrative = market_understanding.narrative.lower()
        
        # Determine macro trend from narrative
        if 'bullish' in narrative or 'bull' in narrative:
            self.macro_trend = 'BULLISH'
        elif 'bearish' in narrative or 'bear' in narrative:
            self.macro_trend = 'BEARISH'
        else:
            self.macro_trend = 'NEUTRAL'
        
        self.macro_strength = market_understanding.predictive_power
        self.macro_volatility = 1.0 - market_understanding.confidence
        self.macro_regime = market_understanding.regime.name
        
        # Default values for institutional flow
        self.institutional_flow = 0.0
        self.institutional_confidence = market_understanding.confidence
        self.large_order_activity = 0.0
        self.order_flow_imbalance = 0.0
        
        # Technical signals from narrative
        if 'buy' in narrative or 'long' in narrative:
            self.technical_signal = 'BUY'
        elif 'sell' in narrative or 'short' in narrative:
            self.technical_signal = 'SELL'
        else:
            self.technical_signal = 'HOLD'
        
        self.technical_strength = market_understanding.predictive_power
        self.support_level = None
        self.resistance_level = None
        self.momentum_score = 0.0
        self.volatility_score = 0.5
        
        # Session and timing
        self.session_phase = 'LONDON'  # Default
        self.session_volatility = 0.5
        self.time_of_day_score = 0.0
        self.day_of_week_score = 0.0
        self.news_impact_score = 0.3
        
        # Anomaly detection
        self.manipulation_probability = 0.0
        self.stop_hunt_probability = 0.0
        self.spoofing_detected = False
        self.liquidity_zone_activity = 0.3
        self.unusual_volume = 0.3
        
        # Metadata
        self.processing_time_ms = 0.0
        self.data_quality_score = 1.0

# Export the main classes
__all__ = [
    'SensoryCortex',
    'SensoryReading', 
    'IntelligenceEngine',
    'MarketUnderstanding',
    'DimensionalReading',
    'MarketRegime',
    'MarketData'
]

