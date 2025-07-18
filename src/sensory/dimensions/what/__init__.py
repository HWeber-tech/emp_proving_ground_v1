"""
What Sense - Technical Reality and Market Structure

This sense handles technical reality analysis, market structure, and price action.
It processes market data to understand WHAT the market is doing.

Sub-modules:
- technical_reality: Core technical analysis and market structure
- price_action: Pure price action analysis
- market_structure: Market structure and support/resistance
- regime_detection: Market regime detection and classification

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Sensory Cortex Refactoring
"""

# Import the main engine
from .what_engine import WhatEngine

__all__ = [
    'WhatEngine'
] 