"""
Core Module
===========

Core implementations for the EMP trading system.
Provides concrete implementations of all critical interfaces.
"""

from __future__ import annotations

from .population_manager import PopulationManager
from .sensory_organ import (
    ANOMALY_ORGAN,
    CHAOS_ORGAN,
    WHAT_ORGAN,
    WHEN_ORGAN,
    SensoryOrgan,
    create_sensory_organ,
)
from .instrument import Instrument, get_all_instruments, get_instrument
from .risk.manager import RiskManager, get_risk_manager  # consolidated SoT

__all__ = [
    # Population Manager
    "PopulationManager",
    # Sensory Organs
    "SensoryOrgan",
    "create_sensory_organ",
    "WHAT_ORGAN",
    "WHEN_ORGAN",
    "ANOMALY_ORGAN",
    "CHAOS_ORGAN",
    # Risk Manager
    "RiskManager",
    "get_risk_manager",
    # Instrument
    "Instrument",
    "get_instrument",
    "get_all_instruments",
]
