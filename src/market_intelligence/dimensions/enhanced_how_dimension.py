from __future__ import annotations

import importlib
from typing import Dict

# PEP 562 lazy forwarding to canonical sensory implementation
_LAZY_EXPORTS: Dict[str, str] = {
    "InstitutionalIntelligenceEngine": "src.sensory.enhanced.how_dimension:InstitutionalIntelligenceEngine",
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> object:
    target = _LAZY_EXPORTS.get(name)
    if not target:
        raise AttributeError(name)
    module, _, attr = target.partition(":")
    mod = importlib.import_module(module)
    obj = getattr(mod, attr)
    globals()[name] = obj  # cache for future attribute access
    return obj


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))