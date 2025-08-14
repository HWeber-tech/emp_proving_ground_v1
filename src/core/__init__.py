"""
Core Module
===========

Core implementations for the EMP trading system.
Provides concrete implementations of all critical interfaces.
"""

from .population_manager import PopulationManager

try:
    from .sensory_organ import (  # type: ignore
        ANOMALY_ORGAN,
        CHAOS_ORGAN,
        WHAT_ORGAN,
        WHEN_ORGAN,
        SensoryOrgan,
        create_sensory_organ,
    )
except Exception:  # pragma: no cover
    # Legacy compatibility placeholders
    class SensoryOrgan:  # type: ignore
        pass
    def create_sensory_organ(*_args, **_kwargs):  # type: ignore
        return None
    WHAT_ORGAN = WHEN_ORGAN = ANOMALY_ORGAN = CHAOS_ORGAN = None  # type: ignore
from .instrument import Instrument, get_all_instruments, get_instrument
from .risk.manager import RiskManager  # consolidated SoT

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

try:
    from .sensory_organ import (  # type: ignore
        ANOMALY_ORGAN,
        CHAOS_ORGAN,
        WHAT_ORGAN,
        WHEN_ORGAN,
        SensoryOrgan,
        create_sensory_organ,
    )
except Exception:  # pragma: no cover
    pass
from .instrument import Instrument, get_all_instruments, get_instrument
from .risk.manager import RiskManager
