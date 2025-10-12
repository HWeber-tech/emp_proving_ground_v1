"""Sensory dimensions - Multi-dimensional analysis organs.

The concrete organ implementations live in :mod:`.executable_organs`, but we
delay importing that module until an attribute is requested to avoid cyclical
imports during bootstrap.  This keeps the public surface stable for callers
while ensuring module initialisation works in isolation (e.g. unit tests that
import :mod:`src.runtime.paper_simulation` without pre-loading sensory organs).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

__all__ = [
    "AnomalySensoryOrgan",
    "HowSensoryOrgan",
    "WhatSensoryOrgan",
    "WhenSensoryOrgan",
    "WhySensoryOrgan",
]

_EXPORTS: Mapping[str, str] = {
    name: "src.sensory.organs.dimensions.executable_organs"
    for name in __all__
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'src.sensory.organs.dimensions' has no attribute '{name}'")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
