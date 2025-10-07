"""
Core Module
===========

Core implementations for the EMP trading system.
Provides concrete implementations of all critical interfaces.
"""

from __future__ import annotations

from .instrument import Instrument, get_all_instruments, get_instrument
from .population_manager import PopulationManager
from .risk.manager import RiskManager, get_risk_manager  # consolidated SoT
from .sensory_organ import (
    RealSensoryOrgan,
    SensoryDriftConfig,
    SensoryOrgan,
    create_sensory_organ,
)

__all__ = [
    # Population Manager
    "PopulationManager",
    # Sensory Organs
    "RealSensoryOrgan",
    "SensoryDriftConfig",
    "SensoryOrgan",
    "create_sensory_organ",
    # Risk Manager
    "RiskManager",
    "get_risk_manager",
    # Instrument
    "Instrument",
    "get_instrument",
    "get_all_instruments",
]

# Re-export for convenience
from .population_manager import PopulationManager
