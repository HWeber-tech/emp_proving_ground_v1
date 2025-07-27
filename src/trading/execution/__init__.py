"""
Trading Execution Package
=========================

This package provides order execution capabilities for the EMP Proving Ground
trading system, including FIX protocol integration and order management.
"""

from .execution_engine import ExecutionEngine
from .fix_executor import FIXExecutor

__all__ = ['ExecutionEngine', 'FIXExecutor']
