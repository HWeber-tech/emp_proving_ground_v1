"""
Risk management module for EMP system.

Provides risk management facade implementation.
"""

from .risk_manager_impl import RiskManagerImpl, create_risk_manager

__all__ = ["RiskManagerImpl", "create_risk_manager"]
