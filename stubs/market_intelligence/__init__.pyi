from __future__ import annotations
from types import ModuleType as _ModuleType

# Stub package to prevent mypy from loading the same source under two module names.
# Canonical path is src.market_intelligence.*.

# Expose subpackages to satisfy __all__
dimensions: _ModuleType

__all__ = ["dimensions"]