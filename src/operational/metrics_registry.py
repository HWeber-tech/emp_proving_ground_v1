"""
Lazy, thread-safe metrics registry that degrades to no-ops when prometheus_client
is unavailable. Never raises on import; safe to use in any environment.
"""

import logging
from threading import RLock
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Union, cast
from src.core.interfaces import CounterLike, GaugeLike, HistogramLike

_log = logging.getLogger(__name__)


# Protocols describing the subset of prometheus metric APIs we rely on.
# Import canonical Protocols from core interfaces to avoid duplication.


# No-op implementations when prometheus_client is absent
class NoOpCounter:
    def inc(self, amount: float = 1.0) -> None:
        return None

    def labels(self, **labels: str) -> "NoOpCounter":
        return self

    # convenience API not part of the protocol
    def with_labels(self, labels: Dict[str, str]) -> "NoOpCounter":
        return self


class NoOpGauge:
    def set(self, value: float) -> None:
        return None

    def inc(self, amount: float = 1.0) -> None:
        return None

    def dec(self, amount: float = 1.0) -> None:
        return None

    def labels(self, **labels: str) -> "NoOpGauge":
        return self

    # convenience API not part of the protocol
    def with_labels(self, labels: Dict[str, str]) -> "NoOpGauge":
        return self


class NoOpHistogram:
    def observe(self, value: float) -> None:
        return None

    def labels(self, **labels: str) -> "NoOpHistogram":
        return self

    # convenience API not part of the protocol
    def with_labels(self, labels: Dict[str, str]) -> "NoOpHistogram":
        return self


# Internal typing aliases
MetricLike = Union[CounterLike, GaugeLike, HistogramLike]
MemoKey = Tuple[str, str, Optional[Tuple[str, ...]]]


# Constructor Protocols for metric types
class CounterCtor(Protocol):
    def __call__(
        self, name: str, documentation: str, labelnames: Optional[Sequence[str]] = None
    ) -> CounterLike: ...


class GaugeCtor(Protocol):
    def __call__(
        self, name: str, documentation: str, labelnames: Optional[Sequence[str]] = None
    ) -> GaugeLike: ...


class HistogramCtor(Protocol):
    def __call__(
        self,
        name: str,
        documentation: str,
        buckets: Optional[Sequence[float]] = None,
        labelnames: Optional[Sequence[str]] = None,
    ) -> HistogramLike: ...


class MetricsRegistry:
    """
    Lazily imports prometheus_client and memoizes created metrics.

    Thread-safety:
      - Fast path: dict get without locking
      - Slow path on miss: acquire lock, re-check, create, store
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._enabled: Optional[bool] = None  # None=unknown; True/False determined at first use
        self._counter_ctor: Optional[CounterCtor] = None
        self._gauge_ctor: Optional[GaugeCtor] = None
        self._hist_ctor: Optional[HistogramCtor] = None
        self._memo: Dict[MemoKey, MetricLike] = {}
        self._logged_no_prom: bool = False

    def _import_prometheus(self) -> bool:
        try:
            from prometheus_client import Counter, Gauge, Histogram

            self._counter_ctor = cast(CounterCtor, Counter)
            self._gauge_ctor = cast(GaugeCtor, Gauge)
            self._hist_ctor = cast(HistogramCtor, Histogram)
            self._enabled = True
            return True
        except Exception:
            self._enabled = False
            if not self._logged_no_prom:
                _log.debug("prometheus_client unavailable; metrics will be no-op")
                self._logged_no_prom = True
            return False

    def _ensure_backend(self) -> None:
        if self._enabled is None:
            self._import_prometheus()

    @staticmethod
    def _key(
        kind: str, name: str, labelnames: Optional[List[str]]
    ) -> Tuple[str, str, Optional[Tuple[str, ...]]]:
        return kind, name, tuple(labelnames) if labelnames else None

    def get_counter(
        self, name: str, description: str, labelnames: Optional[List[str]] = None
    ) -> CounterLike:
        self._ensure_backend()
        if not self._enabled:
            return NoOpCounter()

        key = self._key("counter", name, labelnames)
        metric = self._memo.get(key)
        if metric is not None:
            return cast(CounterLike, metric)

        with self._lock:
            metric = self._memo.get(key)
            if metric is not None:
                return cast(CounterLike, metric)
            ctor = self._counter_ctor
            if ctor is None:
                return NoOpCounter()
            if labelnames:
                metric_c = ctor(name, description, labelnames=labelnames)
            else:
                metric_c = ctor(name, description)
            self._memo[key] = metric_c
            return metric_c

    def get_gauge(
        self, name: str, description: str, labelnames: Optional[List[str]] = None
    ) -> GaugeLike:
        self._ensure_backend()
        if not self._enabled:
            return NoOpGauge()

        key = self._key("gauge", name, labelnames)
        metric = self._memo.get(key)
        if metric is not None:
            return cast(GaugeLike, metric)

        with self._lock:
            metric = self._memo.get(key)
            if metric is not None:
                return cast(GaugeLike, metric)
            ctor = self._gauge_ctor
            if ctor is None:
                return NoOpGauge()
            if labelnames:
                metric_g = ctor(name, description, labelnames=labelnames)
            else:
                metric_g = ctor(name, description)
            self._memo[key] = metric_g
            return metric_g

    def get_histogram(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labelnames: Optional[List[str]] = None,
    ) -> HistogramLike:
        self._ensure_backend()
        if not self._enabled:
            return NoOpHistogram()

        key = self._key("hist", name, labelnames)
        metric = self._memo.get(key)
        if metric is not None:
            return cast(HistogramLike, metric)

        with self._lock:
            metric = self._memo.get(key)
            if metric is not None:
                return cast(HistogramLike, metric)
            ctor = self._hist_ctor
            if ctor is None:
                return NoOpHistogram()
            buckets_seq: Optional[Sequence[float]] = tuple(buckets) if buckets is not None else None
            labelnames_seq: Optional[Sequence[str]] = (
                tuple(labelnames) if labelnames is not None else None
            )
            metric_h = ctor(name, description, buckets=buckets_seq, labelnames=labelnames_seq)
            self._memo[key] = metric_h
            return metric_h


# Module-level singleton accessor
_REGISTRY_SINGLETON: Optional[MetricsRegistry] = None


def get_registry() -> MetricsRegistry:
    global _REGISTRY_SINGLETON
    if _REGISTRY_SINGLETON is None:
        _REGISTRY_SINGLETON = MetricsRegistry()
    return _REGISTRY_SINGLETON
