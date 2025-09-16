from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CounterLike(Protocol):
    def inc(self, amount: float = 1.0) -> None: ...

    def labels(self, **labels: str) -> "CounterLike": ...


@runtime_checkable
class GaugeLike(Protocol):
    def set(self, value: float) -> None: ...

    def inc(self, amount: float = 1.0) -> None: ...

    def dec(self, amount: float = 1.0) -> None: ...

    def labels(self, **labels: str) -> "GaugeLike": ...


@runtime_checkable
class HistogramLike(Protocol):
    def observe(self, value: float) -> None: ...

    def labels(self, **labels: str) -> "HistogramLike": ...


__all__ = ["CounterLike", "GaugeLike", "HistogramLike"]
