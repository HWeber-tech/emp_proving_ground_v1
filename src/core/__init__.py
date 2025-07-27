"""
Core Module
===========

Core implementations for the EMP trading system.
Provides concrete implementations of all critical interfaces.
"""

from .population_manager import PopulationManager
from .sensory_organ import SensoryOrgan, create_sensory_organ, WHAT_ORGAN, WHEN_ORGAN, ANOMALY_ORGAN, CHAOS_ORGAN
from .risk_manager import RiskManager, get_global_risk_manager as get_risk_manager
from .instrument import Instrument, get_instrument, get_all_instruments

__all__ = [
    # Population Manager
    'PopulationManager',
    
    # Sensory Organs
    'SensoryOrgan',
    'create_sensory_organ',
    'WHAT_ORGAN',
    'WHEN_ORGAN',
    'ANOMALY_ORGAN',
    'CHAOS_ORGAN',
    
    # Risk Manager
    'RiskManager',
    'get_risk_manager',
    
    # Instrument
    'Instrument',
    'get_instrument',
    'get_all_instruments',
]

# Re-export for convenience
from .population_manager import PopulationManager
from .sensory_organ import SensoryOrgan, create_sensory_organ, WHAT_ORGAN, WHEN_ORGAN, ANOMALY_ORGAN, CHAOS_ORGAN
from .risk_manager import RiskManager, get_global_risk_manager as get_risk_manager
from .instrument import Instrument, get_instrument, get_all_instruments
