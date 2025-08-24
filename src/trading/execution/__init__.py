"""
Trading Execution Package
=========================

Order execution capabilities for the EMP Proving Ground system.

Note: Avoid importing heavy modules at package import time to keep
unit tests (that import submodules directly) lightweight.
"""

from __future__ import annotations

__all__ = ["ExecutionEngine"]


def __getattr__(name: str) -> object:
    if name == "ExecutionEngine":
        from .execution_engine import ExecutionEngine

        return ExecutionEngine
    raise AttributeError(name)
