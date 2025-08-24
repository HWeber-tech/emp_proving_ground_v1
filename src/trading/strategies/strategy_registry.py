"""
Deprecated local StrategyRegistry placeholder.
Use the consolidated SQLite-backed registry at src/governance/strategy_registry.py
"""

from __future__ import annotations

from src.governance.strategy_registry import StrategyRegistry  # re-export

__all__ = ["StrategyRegistry"]
