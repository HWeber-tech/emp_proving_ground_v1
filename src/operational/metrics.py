"""
Prometheus metrics for FIX connectivity and market data.

Safely degrades to no-ops if prometheus_client is unavailable.
"""

import os
import threading
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _METRICS_ENABLED = True
except Exception:  # pragma: no cover - fallback when lib not installed
    _METRICS_ENABLED = False

    class _Noop:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            return None
        def set(self, *args, **kwargs):
            return None
        def observe(self, *args, **kwargs):
            return None

    def Counter(*args, **kwargs):  # type: ignore
        return _Noop()
    def Gauge(*args, **kwargs):  # type: ignore
        return _Noop()
    def Histogram(*args, **kwargs):  # type: ignore
        return _Noop()
    def start_http_server(*args, **kwargs):  # type: ignore
        return None


_started_lock = threading.Lock()
_started = False


fix_messages_total = Counter(
    "fix_messages_total",
    "Total FIX messages received by type",
    ["session", "msg_type"],
)

fix_reconnect_attempts_total = Counter(
    "fix_reconnect_attempts_total",
    "Total reconnect attempts",
    ["session", "outcome"],  # outcome ∈ {success,failure}
)

fix_business_rejects_total = Counter(
    "fix_business_rejects_total",
    "Total business message rejects",
    ["ref_msg_type"],
)

fix_session_connected = Gauge(
    "fix_session_connected",
    "Session connectivity (1 connected, 0 disconnected)",
    ["session"],
)

fix_exec_report_latency_seconds = Histogram(
    "fix_exec_report_latency_seconds",
    "Latency from NewOrderSingle send to first ExecutionReport",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

fix_cancel_latency_seconds = Histogram(
    "fix_cancel_latency_seconds",
    "Latency from OrderCancelRequest send to Canceled ExecutionReport",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

fix_md_staleness_seconds = Gauge(
    "fix_md_staleness_seconds",
    "Seconds since last market data update",
    ["symbol"],
)


def start_metrics_server(port: Optional[int] = None) -> None:
    global _started
    if port is None:
        port = int(os.environ.get("EMP_METRICS_PORT", "8081"))
    with _started_lock:
        if _started:
            return
        try:
            start_http_server(port)
        except Exception:
            # No-op if failing
            return
        _started = True


def inc_message(session: str, msg_type: str) -> None:
    try:
        fix_messages_total.labels(session=session, msg_type=msg_type).inc()
    except Exception:
        pass


def set_session_connected(session: str, connected: bool) -> None:
    try:
        fix_session_connected.labels(session=session).set(1 if connected else 0)
    except Exception:
        pass


def inc_reconnect(session: str, outcome: str) -> None:
    try:
        fix_reconnect_attempts_total.labels(session=session, outcome=outcome).inc()
    except Exception:
        pass


def inc_business_reject(ref_msg_type: Optional[str]) -> None:
    try:
        fix_business_rejects_total.labels(ref_msg_type=str(ref_msg_type or "?")).inc()
    except Exception:
        pass


def observe_exec_latency(seconds: float) -> None:
    try:
        if seconds >= 0:
            fix_exec_report_latency_seconds.observe(seconds)
    except Exception:
        pass


def observe_cancel_latency(seconds: float) -> None:
    try:
        if seconds >= 0:
            fix_cancel_latency_seconds.observe(seconds)
    except Exception:
        pass


def set_md_staleness(symbol: str, seconds: float) -> None:
    try:
        fix_md_staleness_seconds.labels(symbol=symbol).set(max(0.0, seconds))
    except Exception:
        pass


