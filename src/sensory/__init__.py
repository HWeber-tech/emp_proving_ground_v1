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
from .core.base import MarketUnderstanding, DimensionalReading, MarketRegime

# Main interface for backward compatibility
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
        self.intelligence_engine = IntelligenceEngine(symbol, data_storage)
        
    def calibrate(self, start_time, end_time) -> bool:
        """
        Calibrate the sensory cortex with historical data.
        
        Args:
            start_time: Start of calibration period
            end_time: End of calibration period
            
        Returns:
            True if calibration successful
        """
        return self.intelligence_engine.calibrate(start_time, end_time)
    
    def perceive(self, current_data, historical_window: int = 1000):
        """
        Generate a complete sensory reading of the current market state.
        
        Args:
            current_data: Current market data point
            historical_window: Number of historical points to analyze
            
        Returns:
            Enhanced sensory reading with multidimensional intelligence
        """
        return self.intelligence_engine.process_market_data(current_data)
    
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
        return self.intelligence_engine.get_dimensional_readings()

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
        self.overall_sentiment = market_understanding.consensus_direction
        self.confidence_level = market_understanding.confidence
        self.risk_level = 1.0 - market_understanding.confidence
        
        # Extract dimensional readings
        readings = market_understanding.all_readings
        why_reading = readings[0]  # WHY dimension
        how_reading = readings[1]  # HOW dimension
        what_reading = readings[2]  # WHAT dimension
        when_reading = readings[3]  # WHEN dimension
        anomaly_reading = readings[4]  # ANOMALY dimension
        
        # Map to legacy interface
        self.macro_trend = 'BULLISH' if why_reading.value > 0.3 else 'BEARISH' if why_reading.value < -0.3 else 'NEUTRAL'
        self.macro_strength = abs(why_reading.value)
        self.macro_volatility = 1.0 - why_reading.confidence
        self.macro_regime = market_understanding.regime.name
        
        self.institutional_flow = how_reading.value
        self.institutional_confidence = how_reading.confidence
        self.large_order_activity = how_reading.context.get('large_order_activity', 0.0)
        self.order_flow_imbalance = how_reading.context.get('order_flow_imbalance', 0.0)
        
        self.technical_signal = 'BUY' if what_reading.value > 0.3 else 'SELL' if what_reading.value < -0.3 else 'HOLD'
        self.technical_strength = abs(what_reading.value)
        self.support_level = what_reading.context.get('support_level')
        self.resistance_level = what_reading.context.get('resistance_level')
        self.momentum_score = what_reading.value
        self.volatility_score = what_reading.context.get('volatility_score', 0.5)
        
        self.session_phase = when_reading.context.get('session_phase', 'LONDON')
        self.session_volatility = when_reading.context.get('session_volatility', 0.5)
        self.time_of_day_score = when_reading.value
        self.day_of_week_score = when_reading.context.get('day_of_week_score', 0.0)
        self.news_impact_score = when_reading.context.get('news_impact_score', 0.3)
        
        self.manipulation_probability = anomaly_reading.value
        self.stop_hunt_probability = anomaly_reading.context.get('stop_hunt_probability', 0.0)
        self.spoofing_detected = anomaly_reading.context.get('spoofing_detected', False)
        self.liquidity_zone_activity = anomaly_reading.context.get('liquidity_activity', 0.3)
        self.unusual_volume = anomaly_reading.context.get('unusual_volume', 0.3)
        
        # Metadata
        self.processing_time_ms = market_understanding.context.get('processing_time_ms', 0.0)
        self.data_quality_score = market_understanding.context.get('data_quality_score', 1.0)

# Export the main classes
__all__ = [
    'SensoryCortex',
    'SensoryReading', 
    'IntelligenceEngine',
    'MarketUnderstanding',
    'DimensionalReading',
    'MarketRegime'
] 