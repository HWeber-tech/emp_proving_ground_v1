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

fix_parity_mismatched_orders = Gauge(
    "fix_parity_mismatched_orders",
    "Count of orders with suspected parity mismatch",
)

fix_parity_mismatched_positions = Gauge(
    "fix_parity_mismatched_positions",
    "Count of symbols with suspected position parity mismatch",
)

fix_md_rejects_total = Counter(
    "fix_md_rejects_total",
    "Total Market Data Request rejects",
    ["reason"],
)


def inc_md_reject(reason: str) -> None:
    try:
        fix_md_rejects_total.labels(reason=str(reason or "?")).inc()
    except Exception:
        pass


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


# Heartbeat/TestRequest metrics
fix_heartbeat_interval_seconds = Histogram(
    "fix_heartbeat_interval_seconds",
    "Observed interval between incoming heartbeats",
    buckets=(1, 5, 10, 20, 30, 45, 60, 90, 120),
    labelnames=["session"],
)

fix_test_requests_total = Counter(
    "fix_test_requests_total",
    "Total TestRequests sent due to heartbeat delays",
    ["session"],
)

fix_missed_heartbeats_total = Counter(
    "fix_missed_heartbeats_total",
    "Total missed-heartbeat disconnects",
    ["session"],
)


def observe_heartbeat_interval(session: str, seconds: float) -> None:
    try:
        if seconds >= 0:
            fix_heartbeat_interval_seconds.labels(session=session).observe(seconds)
    except Exception:
        pass


def inc_test_request(session: str) -> None:
    try:
        fix_test_requests_total.labels(session=session).inc()
    except Exception:
        pass


def inc_missed_heartbeat(session: str) -> None:
    try:
        fix_missed_heartbeats_total.labels(session=session).inc()
    except Exception:
        pass


# Pre-trade risk denials
fix_pretrade_denials_total = Counter(
    "fix_pretrade_denials_total",
    "Total pre-trade risk denials",
    ["symbol", "reason"],
)


def inc_pretrade_denial(symbol: str, reason: str) -> None:
    try:
        fix_pretrade_denials_total.labels(symbol=symbol, reason=reason).inc()
    except Exception:
        pass


# Volatility engine metrics
vol_sigma_ann = Gauge(
    "vol_sigma_ann",
    "Annualized volatility estimate (per symbol)",
    ["symbol"],
)

vol_regime_total = Counter(
    "vol_regime_total",
    "Volatility regime observations",
    ["symbol", "regime"],
)

vol_rv_garch_divergence = Gauge(
    "vol_rv_garch_divergence",
    "Absolute divergence between RV and GARCH-ann vol",
    ["symbol"],
)


def set_vol_sigma(symbol: str, sigma_ann: float) -> None:
    try:
        vol_sigma_ann.labels(symbol=symbol).set(max(0.0, sigma_ann))
    except Exception:
        pass


def inc_vol_regime(symbol: str, regime: str) -> None:
    try:
        vol_regime_total.labels(symbol=symbol, regime=str(regime)).inc()
    except Exception:
        pass


def set_vol_divergence(symbol: str, divergence: float) -> None:
    try:
        vol_rv_garch_divergence.labels(symbol=symbol).set(max(0.0, divergence))
    except Exception:
        pass


