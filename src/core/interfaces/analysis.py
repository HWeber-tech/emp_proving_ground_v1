from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Protocol, TypeAlias, runtime_checkable


@runtime_checkable
class ThinkingPattern(Protocol):
    def learn(self, feedback: Mapping[str, object]) -> bool: ...


@runtime_checkable
class SensorySignal(Protocol):
    signal_type: str
    value: float
    confidence: float


AnalysisResult: TypeAlias = Dict[str, Any]


__all__ = ["ThinkingPattern", "SensorySignal", "AnalysisResult"]
