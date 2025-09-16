"""
Thin façade over a lazy metrics registry. Never raises on import and avoids
creating metrics at import time. All wrappers are non-raising.
"""

import logging
import os
from threading import Lock
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, TypeVar, cast

from src.core.interfaces import GaugeLike
from src.operational.metrics_registry import get_registry

_log = logging.getLogger(__name__)

# Internal state for exporter
_started_lock = Lock()
_started = False

_warned_lock = Lock()
_warned_metrics: set[str] = set()


class _NoopGauge:
    """Fallback gauge used when metrics backends are unavailable."""

    def set(self, value: float) -> None:  # pragma: no cover - no-op
        return None

    def inc(self, amount: float = 1.0) -> None:  # pragma: no cover - no-op
        return None

    def dec(self, amount: float = 1.0) -> None:  # pragma: no cover - no-op
        return None

    def labels(self, **labels: str) -> "_NoopGauge":  # pragma: no cover - no-op
        return self


_NOOP_GAUGE = _NoopGauge()

_T = TypeVar("_T")


def _log_metric_failure(metric: str, error: Exception, *, repeated_level: int = logging.DEBUG) -> None:
    """Log a metric failure once at warning level and thereafter at debug."""

    with _warned_lock:
        first = metric not in _warned_metrics
        if first:
            _warned_metrics.add(metric)

    if first:
        _log.warning("Failed to update metric '%s': %s", metric, error, exc_info=error)
    else:
        _log.log(repeated_level, "Repeated failure updating metric '%s': %s", metric, error)


def _call_metric(metric: str, action: Callable[[], _T], fallback: Callable[[], _T] | None = None) -> Optional[_T]:
    """Execute a metric action while logging failures once."""

    try:
        return action()
    except Exception as exc:  # pragma: no cover - exercised via tests
        _log_metric_failure(metric, exc)
        if fallback is not None:
            try:
                return fallback()
            except Exception as fallback_exc:  # pragma: no cover
                _log_metric_failure(f"{metric}.fallback", fallback_exc)
        return None


class LazyGaugeProxy:
    def __init__(self, name: str, description: str, labelnames: Optional[list[str]] = None) -> None:
        self._name = name
        self._desc = description
        self._labelnames = list(labelnames) if labelnames else None
        self._resolved: Optional[GaugeLike] = None
        self._lock = Lock()

    def resolve(self) -> GaugeLike:
        if self._resolved is None:
            with self._lock:
                if self._resolved is None:
                    self._resolved = get_registry().get_gauge(self._name, self._desc, self._labelnames)
        return self._resolved

    def set(self, value: float) -> None:
        _call_metric(f"{self._name}.set", lambda: self.resolve().set(float(value)))

    def inc(self, amount: float = 1.0) -> None:
        _call_metric(f"{self._name}.inc", lambda: self.resolve().inc(float(amount)))

    def dec(self, amount: float = 1.0) -> None:
        _call_metric(f"{self._name}.dec", lambda: self.resolve().dec(float(amount)))

    def labels(self, **labels: str) -> GaugeLike:
        result = _call_metric(f"{self._name}.labels", lambda: self.resolve().labels(**labels))
        if result is not None:
            return result
        fallback = _call_metric(f"{self._name}.labels_fallback", self.resolve)
        return cast(GaugeLike, fallback if fallback is not None else _NOOP_GAUGE)


# Legacy globals (lazy; no metrics created at import time)
fix_parity_mismatched_orders = LazyGaugeProxy("fix_parity_mismatched_orders", "Parity mismatched orders")
fix_parity_mismatched_positions = LazyGaugeProxy("fix_parity_mismatched_positions", "Parity mismatched positions")


# FIX/MD wrappers (all non-raising)
def inc_md_reject(reason: str) -> None:
    safe_reason = reason or "?"

    def _action() -> None:
        get_registry().get_counter(
            "fix_md_rejects_total",
            "Total Market Data Request rejects",
            ["reason"],
        ).labels(reason=safe_reason).inc()

    _call_metric("fix_md_rejects_total.inc", _action)


def inc_message(session: str, msg_type: str) -> None:
    safe_session = session or "?"
    safe_type = msg_type or "?"

    def _action() -> None:
        get_registry().get_counter(
            "fix_messages_total",
            "Total FIX messages received by type",
            ["session", "msg_type"],
        ).labels(session=safe_session, msg_type=safe_type).inc()

    _call_metric("fix_messages_total.inc", _action)


def set_session_connected(session: str, connected: bool) -> None:
    safe_session = session or "?"
    value = 1.0 if connected else 0.0

    def _action() -> None:
        get_registry().get_gauge(
            "fix_session_connected",
            "Session connectivity (1 connected, 0 disconnected)",
            ["session"],
        ).labels(session=safe_session).set(value)

    _call_metric("fix_session_connected.set", _action)


def inc_reconnect(session: str, outcome: str) -> None:
    safe_session = session or "?"
    safe_outcome = outcome or "?"

    def _action() -> None:
        get_registry().get_counter(
            "fix_reconnect_attempts_total",
            "Total reconnect attempts",
            ["session", "outcome"],
        ).labels(session=safe_session, outcome=safe_outcome).inc()

    _call_metric("fix_reconnect_attempts_total.inc", _action)


def inc_business_reject(ref_msg_type: Optional[str]) -> None:
    safe_type = ref_msg_type or "?"

    def _action() -> None:
        get_registry().get_counter(
            "fix_business_rejects_total",
            "Total business message rejects",
            ["ref_msg_type"],
        ).labels(ref_msg_type=safe_type).inc()

    _call_metric("fix_business_rejects_total.inc", _action)


def observe_exec_latency(seconds: float) -> None:
    seconds_value = float(seconds)
    if seconds_value < 0.0:
        return

    def _action() -> None:
        get_registry().get_histogram(
            "fix_exec_report_latency_seconds",
            "Latency from NewOrderSingle send to first ExecutionReport",
            [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
        ).observe(seconds_value)

    _call_metric("fix_exec_report_latency_seconds.observe", _action)


def observe_cancel_latency(seconds: float) -> None:
    seconds_value = float(seconds)
    if seconds_value < 0.0:
        return

    def _action() -> None:
        get_registry().get_histogram(
            "fix_cancel_latency_seconds",
            "Latency from OrderCancelRequest send to Canceled ExecutionReport",
            [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
        ).observe(seconds_value)

    _call_metric("fix_cancel_latency_seconds.observe", _action)


def set_md_staleness(symbol: str, seconds: float) -> None:
    safe_symbol = symbol or "?"
    seconds_value = max(0.0, float(seconds))

    def _action() -> None:
        get_registry().get_gauge(
            "fix_md_staleness_seconds",
            "Seconds since last market data update",
            ["symbol"],
        ).labels(symbol=safe_symbol).set(seconds_value)

    _call_metric("fix_md_staleness_seconds.set", _action)


# Heartbeat
def observe_heartbeat_interval(session: str, seconds: float) -> None:
    seconds_value = float(seconds)
    if seconds_value < 0.0:
        return
    safe_session = session or "?"

    def _action() -> None:
        get_registry().get_histogram(
            "fix_heartbeat_interval_seconds",
            "Observed interval between incoming heartbeats",
            [1, 5, 10, 20, 30, 45, 60, 90, 120],
            ["session"],
        ).labels(session=safe_session).observe(seconds_value)

    _call_metric("fix_heartbeat_interval_seconds.observe", _action)


def inc_test_request(session: str) -> None:
    safe_session = session or "?"

    def _action() -> None:
        get_registry().get_counter(
            "fix_test_requests_total",
            "Total TestRequests sent due to heartbeat delays",
            ["session"],
        ).labels(session=safe_session).inc()

    _call_metric("fix_test_requests_total.inc", _action)


def inc_missed_heartbeat(session: str) -> None:
    safe_session = session or "?"

    def _action() -> None:
        get_registry().get_counter(
            "fix_missed_heartbeats_total",
            "Total missed-heartbeat disconnects",
            ["session"],
        ).labels(session=safe_session).inc()

    _call_metric("fix_missed_heartbeats_total.inc", _action)


# Pre-trade
def inc_pretrade_denial(symbol: str, reason: str) -> None:
    safe_symbol = symbol or "?"
    safe_reason = reason or "?"

    def _action() -> None:
        get_registry().get_counter(
            "fix_pretrade_denials_total",
            "Total pre-trade risk denials",
            ["symbol", "reason"],
        ).labels(symbol=safe_symbol, reason=safe_reason).inc()

    _call_metric("fix_pretrade_denials_total.inc", _action)


# Vol
def set_vol_sigma(symbol: str, sigma_ann: float) -> None:
    safe_symbol = symbol or "?"
    sigma_value = max(0.0, float(sigma_ann))

    def _action() -> None:
        get_registry().get_gauge(
            "vol_sigma_ann",
            "Annualized volatility estimate (per symbol)",
            ["symbol"],
        ).labels(symbol=safe_symbol).set(sigma_value)

    _call_metric("vol_sigma_ann.set", _action)


def inc_vol_regime(symbol: str, regime: str) -> None:
    safe_symbol = symbol or "?"
    safe_regime = regime or "?"

    def _action() -> None:
        get_registry().get_counter(
            "vol_regime_total",
            "Volatility regime observations",
            ["symbol", "regime"],
        ).labels(symbol=safe_symbol, regime=safe_regime).inc()

    _call_metric("vol_regime_total.inc", _action)


def set_vol_divergence(symbol: str, divergence: float) -> None:
    safe_symbol = symbol or "?"
    divergence_value = max(0.0, float(divergence))

    def _action() -> None:
        get_registry().get_gauge(
            "vol_rv_garch_divergence",
            "Absolute divergence between RV and GARCH-ann vol",
            ["symbol"],
        ).labels(symbol=safe_symbol).set(divergence_value)

    _call_metric("vol_rv_garch_divergence.set", _action)


# WHY
def set_why_signal(symbol: str, value: float) -> None:
    safe_symbol = symbol or "?"
    value_float = float(value)

    def _action() -> None:
        get_registry().get_gauge(
            "why_composite_signal",
            "WHY composite signal strength",
            ["symbol"],
        ).labels(symbol=safe_symbol).set(value_float)

    _call_metric("why_composite_signal.set", _action)


def set_why_conf(symbol: str, value: float) -> None:
    safe_symbol = symbol or "?"
    bounded_value = max(0.0, min(1.0, float(value)))

    def _action() -> None:
        get_registry().get_gauge(
            "why_confidence",
            "WHY confidence",
            ["symbol"],
        ).labels(symbol=safe_symbol).set(bounded_value)

    _call_metric("why_confidence.set", _action)


def set_why_feature(name: str, value: float | bool, labels: Optional[Dict[str, str]] = None) -> None:
    labelnames = ["feature"] + (sorted(labels.keys()) if labels else [])
    merged_labels = {**(labels or {}), "feature": name}
    gauge_value = 1.0 if bool(value) else 0.0

    def _action() -> None:
        get_registry().get_gauge(
            "why_feature_available",
            "Availability (1/0) of WHY features (e.g., yields, macro)",
            labelnames,
        ).labels(**merged_labels).set(gauge_value)

    _call_metric("why_feature_available.set", _action)


# Exporter
def start_metrics_server(port: Optional[int] = None) -> None:
    """
    Start the Prometheus exporter HTTP server in-process (idempotent).
    Defaults to EMP_METRICS_PORT (8081). Silently no-ops if prometheus_client
    is unavailable or if the server fails to start.
    """
    global _started
    with _started_lock:
        if _started:
            return

        env_port = os.environ.get("EMP_METRICS_PORT", "8081")
        raw_port = port if port is not None else env_port
        try:
            effective_port = int(raw_port)
        except Exception as exc:  # pragma: no cover - hard to trigger deterministically
            _log.warning(
                "Invalid metrics port %r (from %s); defaulting to 8081",
                raw_port,
                "argument" if port is not None else "EMP_METRICS_PORT",
            )
            _log.debug("Port parsing failure", exc_info=exc)
            effective_port = 8081

        try:
            from prometheus_client import start_http_server
        except ImportError:
            _log.debug("prometheus_client not installed; metrics exporter disabled")
            return

        try:
            start_http_server(effective_port)
        except Exception as exc:  # pragma: no cover - depends on runtime environment
            _log.warning("Failed to start metrics exporter on port %s: %s", effective_port, exc, exc_info=exc)
            return

        _started = True
        _log.info("Prometheus metrics exporter started on port %s", effective_port)

# ---- Core telemetry sink adapter registration (ports/adapters) ----
# Provide static typing via TYPE_CHECKING while keeping runtime optional dependency.
if TYPE_CHECKING:  # pragma: no cover
    from src.core.telemetry import MetricsSink as _MetricsSinkBase
else:  # Runtime fallback base to avoid import-time failures
    class _MetricsSinkBase:
        pass

# Runtime import for the registration function
_rt_set_metrics_sink: Optional[Callable[[_MetricsSinkBase], None]] = None
try:
    from src.core.telemetry import set_metrics_sink as _rt_set_metrics_sink
except Exception:  # pragma: no cover
    _rt_set_metrics_sink = None

class _RegistryMetricsSink:
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        labelnames: Optional[List[str]] = sorted(labels.keys()) if labels else None

        def _action() -> None:
            gauge = get_registry().get_gauge(name, name, labelnames)
            if labels:
                gauge.labels(**labels).set(float(value))
            else:
                gauge.set(float(value))

        _call_metric(f"metrics_sink.{name}.set_gauge", _action)

    def inc_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        labelnames: Optional[List[str]] = sorted(labels.keys()) if labels else None

        def _action() -> None:
            counter = get_registry().get_counter(name, name, labelnames)
            if labels:
                counter.labels(**labels).inc(float(amount))
            else:
                counter.inc(float(amount))

        _call_metric(f"metrics_sink.{name}.inc_counter", _action)

    def observe_histogram(
        self,
        name: str,
        value: float,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        eff_buckets: List[float] = list(buckets) if buckets is not None else [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]

        def _action() -> None:
            if labels:
                labelnames: List[str] = sorted(labels.keys())
                histogram = get_registry().get_histogram(name, name, eff_buckets, labelnames)
                histogram.labels(**labels).observe(float(value))
            else:
                histogram = get_registry().get_histogram(name, name, eff_buckets)
                histogram.observe(float(value))

        _call_metric(f"metrics_sink.{name}.observe_histogram", _action)

# Register the sink at runtime if available
if _rt_set_metrics_sink is not None:  # pragma: no cover
    try:
        _rt_set_metrics_sink(cast(_MetricsSinkBase, _RegistryMetricsSink()))
    except Exception as exc:
        _log_metric_failure("metrics_sink.registration", exc)
