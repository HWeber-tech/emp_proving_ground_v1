"""
Integration Package
===================

This package provides system-wide component integration and management
for the EMP Proving Ground trading system.
"""

from __future__ import annotations

from .component_integrator import ComponentIntegrator
from .component_integrator_impl import ComponentIntegratorImpl

__all__ = ["ComponentIntegrator", "ComponentIntegratorImpl"]
