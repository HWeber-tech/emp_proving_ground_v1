"""
Master Orchestrator for Sensory Cortex
=====================================

Provides unified interface to the sensory cortex system for external modules.
This acts as a facade pattern to simplify integration with the complex sensory system.

Author: EMP Development Team
Phase: 2 - Integration Layer
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.sensory.enhanced.integration.sensory_integration_orchestrator import (
    SensoryIntegrationOrchestrator, 
    UnifiedMarketIntelligence
)
from src.sensory.core.base import MarketData, InstrumentMeta

logger = logging.getLogger(__name__)


class MasterOrchestrator:
    """
    Master orchestrator providing unified access to the sensory cortex.
    
    This class serves as the primary interface for external modules to interact
    with the sensory system without needing to understand the internal complexity.
    """
    
    def __init__(self, instrument_meta: InstrumentMeta):
        """
        Initialize the master orchestrator.
        
        Args:
            instrument_meta: Instrument metadata for configuration
        """
        self.instrument_meta = instrument_meta
        self.sensory_orchestrator = SensoryIntegrationOrchestrator()
        self.initialized = True
        
        logger.info(f"Master orchestrator initialized for {instrument_meta.symbol}")
    
    async def get_sensory_reading(self, market_data: Dict[str, Any]) -> UnifiedMarketIntelligence:
        """
        Get unified sensory reading from all dimensions.
        
        Args:
            market_data: Market data dictionary containing price and other data
            
        Returns:
            UnifiedMarketIntelligence: Complete market intelligence
        """
        try:
            # Ensure we have symbol in market data
            if 'symbol' not in market_data:
                market_data['symbol'] = self.instrument_meta.symbol
            
            # Process market intelligence
            intelligence = await self.sensory_orchestrator.process_market_intelligence(
                market_data, 
                symbol=market_data.get('symbol')
            )
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Error getting sensory reading: {e}")
            return self._get_fallback_intelligence()
    
    async def analyze_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market conditions across all dimensions.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Dict containing analysis results
        """
        try:
            intelligence = await self.get_sensory_reading(market_data)
            
            return {
                'overall_confidence': intelligence.overall_confidence,
                'signal_strength': intelligence.signal_strength,
                'risk_assessment': intelligence.risk_assessment,
                'opportunity_score': intelligence.opportunity_score,
                'confluence_score': intelligence.confluence_score,
                'recommended_action': intelligence.recommended_action,
                'timestamp': intelligence.timestamp
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return self._get_fallback_analysis()
    
    def get_instrument_info(self) -> InstrumentMeta:
        """Get instrument metadata."""
        return self.instrument_meta
    
    def _get_fallback_intelligence(self) -> UnifiedMarketIntelligence:
        """Return fallback intelligence when processing fails."""
        intelligence = UnifiedMarketIntelligence(symbol=self.instrument_meta.symbol)
        intelligence.overall_confidence = 0.5
        intelligence.signal_strength = 0.5
        intelligence.risk_assessment = 0.5
        intelligence.opportunity_score = 0.0
        intelligence.confluence_score = 0.0
        intelligence.recommended_action = 'hold'
        intelligence.timestamp = datetime.now()
        return intelligence
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Return fallback analysis when processing fails."""
        return {
            'overall_confidence': 0.5,
            'signal_strength': 0.5,
            'risk_assessment': 0.5,
            'opportunity_score': 0.0,
            'confluence_score': 0.0,
            'recommended_action': 'hold',
            'timestamp': datetime.now()
        }


# Backward compatibility alias
class SensoryCortex(MasterOrchestrator):
    """Alias for backward compatibility"""
    pass


# Convenience function for quick access
async def get_market_intelligence(symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick access function to get market intelligence.
    
    Args:
        symbol: Trading symbol
        market_data: Market data dictionary
        
    Returns:
        Dict containing market intelligence
    """
    from src.sensory.core.base import InstrumentMeta
    
    # Create instrument metadata
    instrument_meta = InstrumentMeta(
        symbol=symbol,
        pip_size=0.0001,
        lot_size=100000,
        timezone="UTC",
        typical_spread=0.00015,
        avg_daily_range=0.01
    )
    
    # Create orchestrator
    orchestrator = MasterOrchestrator(instrument_meta)
    
    # Get intelligence
    return await orchestrator.analyze_market_conditions(market_data)


if __name__ == "__main__":
    import asyncio
    
    async def test_master_orchestrator():
        """Test the master orchestrator."""
        # Create test data
            'symbol': 'EURUSD'
        }
        
        # Test orchestrator
        from src.sensory.core.base import InstrumentMeta
        
        instrument_meta = InstrumentMeta(
            symbol='EURUSD',
            pip_size=0.0001,
            lot_size=100000,
            timezone="UTC",
            typical_spread=0.00015,
            avg_daily_range=0.01
        )
        
        orchestrator = MasterOrchestrator(instrument_meta)
        
        print("Test Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    asyncio.run(test_master_orchestrator())
