#!/usr/bin/env python3
"""
Shim for RealSensoryOrgan in dimensions namespace.

Canonical implementation resides at src/sensory/real_sensory_organ.py.
This module re-exports it to maintain backward compatibility.
"""

from __future__ import annotations

from src.sensory.real_sensory_organ import RealSensoryOrgan as RealSensoryOrgan

__all__ = ["RealSensoryOrgan"]
