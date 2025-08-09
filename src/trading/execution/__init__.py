"""
Trading Execution Package
=========================

Order execution capabilities for the EMP Proving Ground system.

Note: Avoid importing heavy modules at package import time to keep
unit tests (that import submodules directly) lightweight.
"""

__all__ = ['ExecutionEngine', 'FIXExecutor']

def __getattr__(name):
    if name == 'ExecutionEngine':
        from .execution_engine import ExecutionEngine  # type: ignore
        return ExecutionEngine
    if name == 'FIXExecutor':
        from .fix_executor import FIXExecutor  # type: ignore
        return FIXExecutor
    raise AttributeError(name)
