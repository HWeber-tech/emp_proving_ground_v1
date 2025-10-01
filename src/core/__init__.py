"""
Core Module
===========

Core implementations for the EMP trading system.
Provides concrete implementations of all critical interfaces.
"""

from __future__ import annotations

import logging

from .population_manager import PopulationManager

logger = logging.getLogger(__name__)

try:
    from .sensory_organ import (
        ANOMALY_ORGAN,
        CHAOS_ORGAN,
        WHAT_ORGAN,
        WHEN_ORGAN,
        SensoryOrgan,
        create_sensory_organ,
    )
except (ImportError, AttributeError) as exc:  # pragma: no cover - defensive import
    logger.warning(
        "Falling back to sensory organ stubs due to import failure: %s",
        exc,
        exc_info=True,
    )

    # Legacy compatibility placeholders
    class SensoryOrgan:  # type: ignore
        """Fallback sensory organ placeholder used when the real module is unavailable."""

        pass

    def create_sensory_organ(*_args, **_kwargs):  # type: ignore
        """Fallback creator returning ``None`` when the real implementation is missing."""

        return None

    WHAT_ORGAN = WHEN_ORGAN = ANOMALY_ORGAN = CHAOS_ORGAN = None  # type: ignore
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

# Re-export for convenience
from .population_manager import PopulationManager

