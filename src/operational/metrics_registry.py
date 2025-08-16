"""
Lazy, thread-safe metrics registry that degrades to no-ops when prometheus_client
is unavailable. Never raises on import; safe to use in any environment.
"""

import logging
from threading import RLock
from typing import Any, Dict, List, Optional, Protocol, Tuple, Self


_log = logging.getLogger(__name__)


# Protocols describing the subset of prometheus metric APIs we rely on.
class CounterLike(Protocol):
    def inc(self, amount: float = 1.0) -> None: ...
    def labels(self, **labels: str) -> "CounterLike": ...


class GaugeLike(Protocol):
    def set(self, value: float) -> None: ...
    def inc(self, amount: float = 1.0) -> None: ...
    def dec(self, amount: float = 1.0) -> None: ...
    def labels(self, **labels: str) -> "GaugeLike": ...


class HistogramLike(Protocol):
    def observe(self, value: float) -> None: ...
    def labels(self, **labels: str) -> "HistogramLike": ...


# No-op implementations when prometheus_client is absent
class NoOpCounter:
    def inc(self, amount: float = 1.0) -> None:
        return None

    def labels(self, **labels: str) -> "NoOpCounter":
        return self

    # convenience API not part of the protocol
    def with_labels(self, labels: Dict[str, str]) -> Self:
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
    def with_labels(self, labels: Dict[str, str]) -> Self:
        return self


class NoOpHistogram:
    def observe(self, value: float) -> None:
        return None

    def labels(self, **labels: str) -> "NoOpHistogram":
        return self

    # convenience API not part of the protocol
    def with_labels(self, labels: Dict[str, str]) -> Self:
        return self


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
        self._counter_cls: Any = None
        self._gauge_cls: Any = None
        self._hist_cls: Any = None
        self._memo: Dict[Tuple[str, str, Optional[Tuple[str, ...]]], Any] = {}
        self._logged_no_prom: bool = False

    def _import_prometheus(self) -> bool:
        try:
            from prometheus_client import Counter, Gauge, Histogram  # type: ignore
            self._counter_cls = Counter
            self._gauge_cls = Gauge
            self._hist_cls = Histogram
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
    def _key(kind: str, name: str, labelnames: Optional[List[str]]) -> Tuple[str, str, Optional[Tuple[str, ...]]]:
        return kind, name, tuple(labelnames) if labelnames else None

    def get_counter(self, name: str, description: str, labelnames: Optional[List[str]] = None) -> CounterLike:
        self._ensure_backend()
        if not self._enabled:
            return NoOpCounter()

        key = self._key("counter", name, labelnames)
        metric = self._memo.get(key)
        if metric is not None:
            return metric

        with self._lock:
            metric = self._memo.get(key)
            if metric is not None:
                return metric
            if labelnames:
                metric = self._counter_cls(name, description, labelnames=labelnames)
            else:
                metric = self._counter_cls(name, description)
            self._memo[key] = metric
            return metric

    def get_gauge(self, name: str, description: str, labelnames: Optional[List[str]] = None) -> GaugeLike:
        self._ensure_backend()
        if not self._enabled:
            return NoOpGauge()

        key = self._key("gauge", name, labelnames)
        metric = self._memo.get(key)
        if metric is not None:
            return metric

        with self._lock:
            metric = self._memo.get(key)
            if metric is not None:
                return metric
            if labelnames:
                metric = self._gauge_cls(name, description, labelnames=labelnames)
            else:
                metric = self._gauge_cls(name, description)
            self._memo[key] = metric
            return metric

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
            return metric

        with self._lock:
            metric = self._memo.get(key)
            if metric is not None:
                return metric
            kwargs: Dict[str, Any] = {}
            if buckets is not None:
                kwargs["buckets"] = tuple(buckets)
            if labelnames is not None:
                kwargs["labelnames"] = labelnames
            metric = self._hist_cls(name, description, **kwargs)
            self._memo[key] = metric
            return metric


# Module-level singleton accessor
_REGISTRY_SINGLETON: Optional[MetricsRegistry] = None


def get_registry() -> MetricsRegistry:
    global _REGISTRY_SINGLETON
    if _REGISTRY_SINGLETON is None:
        _REGISTRY_SINGLETON = MetricsRegistry()
    return _REGISTRY_SINGLETON