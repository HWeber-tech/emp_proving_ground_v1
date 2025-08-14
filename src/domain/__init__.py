"""
EMP Domain Models v1.1

Shared domain models used across all layers.
Separates domain concerns from infrastructure concerns.
"""

from src.core import RiskConfig as _DeprecatedRiskConfig  # legacy alias if needed

# InstrumentProvider and CurrencyConverter are now sourced from src.core
from src.core import RiskManager as _DeprecatedRiskManager  # legacy alias if needed
from src.core import configuration as _unused_configuration  # maintain package namespace
from src.core import evolution as _unused_evolution  # maintain package namespace
from src.core import performance as _unused_performance  # maintain package namespace
from src.core import sensory_organ as _unused_sensory_organ  # namespace continuity
from src.core import strategy as _unused_strategy  # maintain package namespace
from src.core.instrument import Instrument
from src.core.risk.manager import RiskConfig

from .models import ExecutionReport

__all__ = [
    'RiskConfig',
    'Instrument',
    'ExecutionReport'
]

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Domain Models - Shared Business Entities" 
