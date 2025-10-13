"""
Trading Execution Package
=========================

Order execution capabilities for the EMP Proving Ground system.

Note: Avoid importing heavy modules at package import time to keep
unit tests (that import submodules directly) lightweight.
"""

from __future__ import annotations

__all__ = [
    "ExecutionEngine",
    "ReleaseAwareExecutionRouter",
    "PaperBrokerExecutionAdapter",
    "PaperBrokerError",
    "LiveBrokerExecutionAdapter",
    "LiveBrokerError",
]


def __getattr__(name: str) -> object:
    if name == "ExecutionEngine":
        from .execution_engine import ExecutionEngine

        return ExecutionEngine
    if name == "ReleaseAwareExecutionRouter":
        from .release_router import ReleaseAwareExecutionRouter

        return ReleaseAwareExecutionRouter
    if name == "PaperBrokerExecutionAdapter":
        from .paper_broker_adapter import PaperBrokerExecutionAdapter

        return PaperBrokerExecutionAdapter
    if name == "PaperBrokerError":
        from .paper_broker_adapter import PaperBrokerError

        return PaperBrokerError
    if name == "LiveBrokerExecutionAdapter":
        from .live_broker_adapter import LiveBrokerExecutionAdapter

        return LiveBrokerExecutionAdapter
    if name == "LiveBrokerError":
        from .live_broker_adapter import LiveBrokerError

        return LiveBrokerError
    raise AttributeError(name)
