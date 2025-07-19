"""
EMP Domain Models v1.1

Shared domain models used across all layers.
Separates domain concerns from infrastructure concerns.
"""

from .models import RiskConfig, Instrument, InstrumentProvider, CurrencyConverter

__version__ = "1.1.0"
__author__ = "EMP System"
__description__ = "Domain Models - Shared Business Entities" 