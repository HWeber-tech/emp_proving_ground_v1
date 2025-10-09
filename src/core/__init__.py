"""
Core Module
===========

Core implementations for the EMP trading system.
Provides concrete implementations of all critical interfaces.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .instrument import Instrument, get_all_instruments, get_instrument
from .population_manager import PopulationManager
from src.risk.manager import RiskManager  # canonical risk facade

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
    # Instrument
    "Instrument",
    "get_instrument",
    "get_all_instruments",
]

_SENSORY_EXPORTS = {
    "RealSensoryOrgan",
    "SensoryDriftConfig",
    "SensoryOrgan",
    "create_sensory_organ",
}


def _resolve_sensory_export(name: str) -> Any:
    module = import_module("src.core.sensory_organ")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in _SENSORY_EXPORTS:
        return _resolve_sensory_export(name)
    raise AttributeError(name)
