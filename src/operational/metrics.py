"""
Thin façade over a lazy metrics registry. Never raises on import and avoids
creating metrics at import time. All wrappers are non-raising.
"""

import logging
import os
from threading import Lock
from typing import Dict, Optional

from src.operational.metrics_registry import (
    GaugeLike,
    get_registry,
)

_log = logging.getLogger(__name__)

# Internal state for exporter
_started_lock = Lock()
_started = False


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
        try:
            self.resolve().set(float(value))
        except Exception:
            pass

    def inc(self, amount: float = 1.0) -> None:
        try:
            self.resolve().inc(float(amount))
        except Exception:
            pass

    def dec(self, amount: float = 1.0) -> None:
        try:
            self.resolve().dec(float(amount))
        except Exception:
            pass

    def labels(self, **labels: str) -> GaugeLike:
        try:
            return self.resolve().labels(**labels)
        except Exception:
            return self.resolve()


# Legacy globals (lazy; no metrics created at import time)
fix_parity_mismatched_orders = LazyGaugeProxy("fix_parity_mismatched_orders", "Parity mismatched orders")
fix_parity_mismatched_positions = LazyGaugeProxy("fix_parity_mismatched_positions", "Parity mismatched positions")


# FIX/MD wrappers (all non-raising)
def inc_md_reject(reason: str) -> None:
    try:
        get_registry().get_counter("fix_md_rejects_total", "Total Market Data Request rejects", ["reason"]).labels(
            reason=reason or "?"
        ).inc()
    except Exception:
        pass


def inc_message(session: str, msg_type: str) -> None:
    try:
        get_registry().get_counter("fix_messages_total", "Total FIX messages received by type", ["session", "msg_type"]).labels(
            session=session or "?", msg_type=msg_type or "?"
        ).inc()
    except Exception:
        pass


def set_session_connected(session: str, connected: bool) -> None:
    try:
        get_registry().get_gauge("fix_session_connected", "Session connectivity (1 connected, 0 disconnected)", ["session"]).labels(
            session=session or "?"
        ).set(1.0 if connected else 0.0)
    except Exception:
        pass


def inc_reconnect(session: str, outcome: str) -> None:
    try:
        get_registry().get_counter("fix_reconnect_attempts_total", "Total reconnect attempts", ["session", "outcome"]).labels(
            session=session or "?", outcome=outcome or "?"
        ).inc()
    except Exception:
        pass


def inc_business_reject(ref_msg_type: Optional[str]) -> None:
    try:
        get_registry().get_counter("fix_business_rejects_total", "Total business message rejects", ["ref_msg_type"]).labels(
            ref_msg_type=(ref_msg_type or "?")
        ).inc()
    except Exception:
        pass


def observe_exec_latency(seconds: float) -> None:
    try:
        if float(seconds) >= 0.0:
            get_registry().get_histogram(
                "fix_exec_report_latency_seconds",
                "Latency from NewOrderSingle send to first ExecutionReport",
                [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
            ).observe(float(seconds))
    except Exception:
        pass


def observe_cancel_latency(seconds: float) -> None:
    try:
        if float(seconds) >= 0.0:
            get_registry().get_histogram(
                "fix_cancel_latency_seconds",
                "Latency from OrderCancelRequest send to Canceled ExecutionReport",
                [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
            ).observe(float(seconds))
    except Exception:
        pass


def set_md_staleness(symbol: str, seconds: float) -> None:
    try:
        get_registry().get_gauge("fix_md_staleness_seconds", "Seconds since last market data update", ["symbol"]).labels(
            symbol=symbol or "?"
        ).set(max(0.0, float(seconds)))
    except Exception:
        pass


# Heartbeat
def observe_heartbeat_interval(session: str, seconds: float) -> None:
    try:
        if float(seconds) >= 0.0:
            get_registry().get_histogram(
                "fix_heartbeat_interval_seconds",
                "Observed interval between incoming heartbeats",
                [1, 5, 10, 20, 30, 45, 60, 90, 120],
                ["session"],
            ).labels(session=session or "?").observe(float(seconds))
    except Exception:
        pass


def inc_test_request(session: str) -> None:
    try:
        get_registry().get_counter("fix_test_requests_total", "Total TestRequests sent due to heartbeat delays", ["session"]).labels(
            session=session or "?"
        ).inc()
    except Exception:
        pass


def inc_missed_heartbeat(session: str) -> None:
    try:
        get_registry().get_counter("fix_missed_heartbeats_total", "Total missed-heartbeat disconnects", ["session"]).labels(
            session=session or "?"
        ).inc()
    except Exception:
        pass


# Pre-trade
def inc_pretrade_denial(symbol: str, reason: str) -> None:
    try:
        get_registry().get_counter("fix_pretrade_denials_total", "Total pre-trade risk denials", ["symbol", "reason"]).labels(
            symbol=symbol or "?", reason=reason or "?"
        ).inc()
    except Exception:
        pass


# Vol
def set_vol_sigma(symbol: str, sigma_ann: float) -> None:
    try:
        get_registry().get_gauge("vol_sigma_ann", "Annualized volatility estimate (per symbol)", ["symbol"]).labels(
            symbol=symbol or "?"
        ).set(max(0.0, float(sigma_ann)))
    except Exception:
        pass


def inc_vol_regime(symbol: str, regime: str) -> None:
    try:
        get_registry().get_counter("vol_regime_total", "Volatility regime observations", ["symbol", "regime"]).labels(
            symbol=symbol or "?", regime=regime or "?"
        ).inc()
    except Exception:
        pass


def set_vol_divergence(symbol: str, divergence: float) -> None:
    try:
        get_registry().get_gauge("vol_rv_garch_divergence", "Absolute divergence between RV and GARCH-ann vol", ["symbol"]).labels(
            symbol=symbol or "?"
        ).set(max(0.0, float(divergence)))
    except Exception:
        pass


# WHY
def set_why_signal(symbol: str, value: float) -> None:
    try:
        get_registry().get_gauge("why_composite_signal", "WHY composite signal strength", ["symbol"]).labels(
            symbol=symbol or "?"
        ).set(float(value))
    except Exception:
        pass


def set_why_conf(symbol: str, value: float) -> None:
    try:
        get_registry().get_gauge("why_confidence", "WHY confidence", ["symbol"]).labels(
            symbol=symbol or "?"
        ).set(max(0.0, min(1.0, float(value))))
    except Exception:
        pass


def set_why_feature(name: str, value: float | bool, labels: Optional[Dict[str, str]] = None) -> None:
    try:
        labelnames = ["feature"] + (sorted(labels.keys()) if labels else [])
        get_registry().get_gauge(
            "why_feature_available",
            "Availability (1/0) of WHY features (e.g., yields, macro)",
            labelnames,
        ).labels(**({**(labels or {}), "feature": name})).set(1.0 if bool(value) else 0.0)
    except Exception:
        pass


# Exporter
def start_metrics_server(port: Optional[int] = None) -> None:
    """
    Start the Prometheus exporter HTTP server in-process (idempotent).
    Defaults to EMP_METRICS_PORT (8081). Silently no-ops if prometheus_client
    is unavailable or if the server fails to start.
    """
    global _started
    try:
        with _started_lock:
            if _started:
                return
            try:
                effective_port = int(port) if port is not None else int(os.environ.get("EMP_METRICS_PORT", "8081"))
            except Exception:
                effective_port = 8081
            try:
                from prometheus_client import start_http_server  # type: ignore
            except ImportError:
                return
            try:
                start_http_server(effective_port)
                _started = True
            except Exception:
                return
    except Exception:
        return

# ---- Core telemetry sink adapter registration (ports/adapters) ----
try:
    # Register an adapter so domain code can emit metrics via core.telemetry.MetricsSink
    from src.core.telemetry import MetricsSink as _MetricsSink, set_metrics_sink as _set_metrics_sink  # type: ignore
except Exception:  # pragma: no cover
    _MetricsSink = None  # type: ignore

if _MetricsSink is not None:  # pragma: no cover
    from typing import Dict, List, Optional

    class _RegistryMetricsSink(_MetricsSink):  # type: ignore[misc]
        def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
            try:
                labelnames = sorted(labels.keys()) if labels else None
                g = get_registry().get_gauge(name, name, labelnames)  # type: ignore[arg-type]
                if labels:
                    g.labels(**labels).set(float(value))
                else:
                    g.set(float(value))
            except Exception:
                pass

        def inc_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
            try:
                labelnames = sorted(labels.keys()) if labels else None
                c = get_registry().get_counter(name, name, labelnames)  # type: ignore[arg-type]
                if labels:
                    c.labels(**labels).inc(float(amount))
                else:
                    c.inc(float(amount))
            except Exception:
                pass

        def observe_histogram(
            self,
            name: str,
            value: float,
            buckets: Optional[List[float]] = None,
            labels: Optional[Dict[str, str]] = None,
        ) -> None:
            try:
                eff_buckets = list(buckets) if buckets is not None else [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
                if labels:
                    labelnames = sorted(labels.keys())
                    h = get_registry().get_histogram(name, name, eff_buckets, labelnames)  # type: ignore[call-arg]
                    h.labels(**labels).observe(float(value))
                else:
                    h = get_registry().get_histogram(name, name, eff_buckets)  # type: ignore[call-arg]
                    h.observe(float(value))
            except Exception:
                pass

    try:
        _set_metrics_sink(_RegistryMetricsSink())  # type: ignore[operator]
    except Exception:
        pass
