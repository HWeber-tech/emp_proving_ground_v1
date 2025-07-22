"""
Risk management module for EMP system.

Provides real risk management implementations including Kelly Criterion
position sizing and portfolio risk metrics.
"""

from .real_risk_manager import RealRiskManager, RealRiskConfig

__all__ = ['RealRiskManager', 'RealRiskConfig']
