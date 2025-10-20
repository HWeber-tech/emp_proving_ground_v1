"""
Thin façade over a lazy metrics registry. Never raises on import and avoids
creating metrics at import time. All wrappers are non-raising.
"""

import logging
import os
import ssl
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Type, TypeVar, cast

from socketserver import ThreadingMixIn

from src.core.interfaces import GaugeLike
from src.operational.metrics_registry import get_registry

_log = logging.getLogger(__name__)

# Internal state for exporter
_started_lock = Lock()
_started = False
_TLS_METRICS_PATHS = {"/metrics", "/metrics/"}


class _ThreadingTLSServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: Type[BaseHTTPRequestHandler],
        ssl_context: ssl.SSLContext,
    ) -> None:
        self._ssl_context = ssl_context
        super().__init__(server_address, request_handler_class)

    def get_request(self):  # type: ignore[override]
        raw_socket, addr = super().get_request()
        tls_socket = self._ssl_context.wrap_socket(raw_socket, server_side=True)
        return tls_socket, addr


def _make_metrics_server(
    port: int,
    handler_cls: Type[BaseHTTPRequestHandler],
    ssl_context: ssl.SSLContext,
) -> _ThreadingTLSServer:
    return _ThreadingTLSServer(("", port), handler_cls, ssl_context)


def _build_metrics_handler(
    *,
    generate_latest: Callable[..., bytes],
    registry,
    content_type: str,
) -> Type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - 3rd-party interface
            if self.path not in _TLS_METRICS_PATHS:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = generate_latest(registry)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_HEAD(self) -> None:  # noqa: N802 - 3rd-party interface
            if self.path not in _TLS_METRICS_PATHS:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = generate_latest(registry)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover - noise control
            _log.debug("Prometheus metrics request: " + fmt, *args)

    return _Handler

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


def _normalise_label(value: Optional[str], *, fallback: str = "unknown") -> str:
    """Coerce metric label values into stable strings."""

    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback

_T = TypeVar("_T")


def _log_metric_failure(
    metric: str, error: Exception, *, repeated_level: int = logging.DEBUG
) -> None:
    """Log a metric failure once at warning level and thereafter at debug."""

    with _warned_lock:
        first = metric not in _warned_metrics
        if first:
            _warned_metrics.add(metric)

    if first:
        _log.warning("Failed to update metric '%s': %s", metric, error, exc_info=error)
    else:
        _log.log(repeated_level, "Repeated failure updating metric '%s': %s", metric, error)


def _call_metric(
    metric: str, action: Callable[[], _T], fallback: Callable[[], _T] | None = None
) -> Optional[_T]:
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
                    self._resolved = get_registry().get_gauge(
                        self._name, self._desc, self._labelnames
                    )
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
fix_parity_mismatched_orders = LazyGaugeProxy(
    "fix_parity_mismatched_orders", "Parity mismatched orders"
)
fix_parity_mismatched_positions = LazyGaugeProxy(
    "fix_parity_mismatched_positions", "Parity mismatched positions"
)

# Understanding-loop SLO probes
_understanding_loop_latency_seconds = LazyGaugeProxy(
    "understanding_loop_latency_seconds",
    "Understanding loop latency in seconds by statistic",
    ["loop", "stat"],
)
_understanding_loop_latency_status = LazyGaugeProxy(
    "understanding_loop_latency_status",
    "Status level for understanding loop latency SLO (0=pass,1=warn,2=fail)",
    ["loop"],
)
_drift_alert_freshness_seconds = LazyGaugeProxy(
    "drift_alert_freshness_seconds",
    "Freshness of drift alert telemetry in seconds",
    ["alert"],
)
_drift_alert_freshness_status = LazyGaugeProxy(
    "drift_alert_freshness_status",
    "Status level for drift alert freshness SLO (0=pass,1=warn,2=fail)",
    ["alert"],
)
_replay_determinism_drift = LazyGaugeProxy(
    "replay_determinism_drift",
    "Drift score for replay determinism probes",
    ["probe"],
)
_replay_determinism_status = LazyGaugeProxy(
    "replay_determinism_status",
    "Status level for replay determinism SLO (0=pass,1=warn,2=fail)",
    ["probe"],
)
_replay_determinism_mismatches = LazyGaugeProxy(
    "replay_determinism_mismatches",
    "Count of mismatched fields detected by replay determinism probes",
    ["probe"],
)


# Compliance/governance telemetry gauges
_compliance_policy_breaches = LazyGaugeProxy(
    "compliance_policy_breaches",
    "Compliance-panel count of policy breaches intercepted",
)
_compliance_risk_warnings = LazyGaugeProxy(
    "compliance_risk_warnings",
    "Compliance-panel count of risk warnings observed",
)
_governance_actions_total = LazyGaugeProxy(
    "governance_actions",
    "Governance interventions recorded by compliance dashboard",
)
_governance_promotions_total = LazyGaugeProxy(
    "governance_promotions",
    "Governance promotions recorded by compliance dashboard",
)

# Security telemetry gauges
_security_failed_logins_last_hour = LazyGaugeProxy(
    "security_failed_logins_last_hour",
    "Failed login attempts recorded in the previous hour",
)

# Ingest telemetry gauges
_ingest_rejected_records_per_hour = LazyGaugeProxy(
    "ingest_rejected_records_per_hour",
    "Rejected ingest records per hour by dimension",
    ["dimension"],
)

# Evolution KPI gauges
_evolution_time_to_candidate_hours = LazyGaugeProxy(
    "evolution_time_to_candidate_hours",
    "Evolution time-to-candidate SLA metrics (hours) by statistic",
    ["stat"],
)
_evolution_time_to_candidate_total = LazyGaugeProxy(
    "evolution_time_to_candidate_total",
    "Total number of findings evaluated for candidate promotion",
)
_evolution_time_to_candidate_breaches = LazyGaugeProxy(
    "evolution_time_to_candidate_breaches",
    "Count of time-to-candidate SLA breaches detected",
)
_evolution_promotions_total = LazyGaugeProxy(
    "evolution_promotions_total",
    "Total promotions applied via policy ledger",
)
_evolution_demotions_total = LazyGaugeProxy(
    "evolution_demotions_total",
    "Total demotions applied via policy ledger",
)
_evolution_transitions_total = LazyGaugeProxy(
    "evolution_transitions_total",
    "Total stage transitions processed via policy ledger",
)
_evolution_promotion_rate = LazyGaugeProxy(
    "evolution_promotion_rate",
    "Share of stage transitions that resulted in promotions",
)
_evolution_budget_usage = LazyGaugeProxy(
    "evolution_budget_usage",
    "Exploration budget utilisation metrics by statistic",
    ["stat"],
)
_evolution_budget_blocked_total = LazyGaugeProxy(
    "evolution_budget_blocked_total",
    "Total exploration attempts blocked by budget",
)
_evolution_budget_forced_total = LazyGaugeProxy(
    "evolution_budget_forced_total",
    "Total forced exploration decisions due to budgeting",
)
_evolution_budget_samples_total = LazyGaugeProxy(
    "evolution_budget_samples_total",
    "Number of policy decisions contributing exploration budget snapshots",
)
_evolution_rollback_latency_hours = LazyGaugeProxy(
    "evolution_rollback_latency_hours",
    "Rollback latency metrics (hours) by statistic",
    ["stat"],
)
_evolution_rollback_events_total = LazyGaugeProxy(
    "evolution_rollback_events_total",
    "Number of rollback latency samples observed",
)


def set_compliance_policy_breaches(value: float) -> None:
    """Record the latest count of policy breaches intercepted by compliance."""

    _compliance_policy_breaches.set(float(value))


def set_compliance_risk_warnings(value: float) -> None:
    """Record the latest count of risk warnings observed by compliance."""

    _compliance_risk_warnings.set(float(value))


def set_governance_actions_total(value: float) -> None:
    """Record the latest count of governance interventions captured."""

    _governance_actions_total.set(float(value))


def set_governance_promotions_total(value: float) -> None:
    """Record the latest count of governance promotions captured."""

    _governance_promotions_total.set(float(value))


def set_security_failed_logins(value: float | int | None) -> None:
    """Record failed login attempts observed during the most recent hour."""

    if value is None:
        return

    _security_failed_logins_last_hour.set(max(float(value), 0.0))


def set_ingest_rejected_records_per_hour(
    dimension: str, value: float | int | None
) -> None:
    """Record the rejected ingest records rate for a dimension."""

    if value is None:
        return

    safe_dimension = _normalise_label(dimension)

    def _action() -> None:
        _ingest_rejected_records_per_hour.labels(dimension=safe_dimension).set(
            max(float(value), 0.0)
        )

    _call_metric("ingest_rejected_records_per_hour.set", _action)


def set_evolution_time_to_candidate_stat(stat: str, value: float | None) -> None:
    """Record a time-to-candidate SLA statistic in hours."""

    if value is None:
        return

    safe_stat = _normalise_label(stat)

    def _action() -> None:
        _evolution_time_to_candidate_hours.labels(stat=safe_stat).set(float(value))

    _call_metric("evolution_time_to_candidate_hours.set", _action)


def set_evolution_time_to_candidate_total(count: int) -> None:
    """Record the total number of evaluated findings for SLA tracking."""

    _evolution_time_to_candidate_total.set(float(count))


def set_evolution_time_to_candidate_breaches(value: float) -> None:
    """Record the number of SLA breaches detected for time-to-candidate."""

    _evolution_time_to_candidate_breaches.set(float(value))


def set_evolution_promotion_counts(promotions: float, demotions: float) -> None:
    """Record promotion and demotion totals observed in the ledger."""

    _evolution_promotions_total.set(float(promotions))
    _evolution_demotions_total.set(float(demotions))


def set_evolution_promotion_transitions(transitions: float) -> None:
    """Record the total number of stage transitions inspected."""

    _evolution_transitions_total.set(float(transitions))


def set_evolution_promotion_rate(rate: float) -> None:
    """Record the share of transitions that resulted in promotions."""

    _evolution_promotion_rate.set(float(rate))


def set_evolution_budget_usage(stat: str, value: float | None) -> None:
    """Record exploration budget usage statistics."""

    if value is None:
        return

    safe_stat = _normalise_label(stat)

    def _action() -> None:
        _evolution_budget_usage.labels(stat=safe_stat).set(float(value))

    _call_metric("evolution_budget_usage.set", _action)


def set_evolution_budget_blocked(value: float) -> None:
    """Record the number of exploration attempts blocked by budget limits."""

    _evolution_budget_blocked_total.set(float(value))


def set_evolution_budget_forced(value: float) -> None:
    """Record the number of forced exploration decisions due to budgeting."""

    _evolution_budget_forced_total.set(float(value))


def set_evolution_budget_samples(value: float) -> None:
    """Record how many decisions reported exploration budget snapshots."""

    _evolution_budget_samples_total.set(float(value))


def set_evolution_rollback_latency(stat: str, value: float | None) -> None:
    """Record rollback latency statistics in hours."""

    if value is None:
        return

    safe_stat = _normalise_label(stat)

    def _action() -> None:
        _evolution_rollback_latency_hours.labels(stat=safe_stat).set(float(value))

    _call_metric("evolution_rollback_latency_hours.set", _action)


def set_evolution_rollback_events(value: float) -> None:
    """Record the number of rollback latency samples observed."""

    _evolution_rollback_events_total.set(float(value))


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


def set_why_feature(
    name: str, value: float | bool, labels: Optional[Dict[str, str]] = None
) -> None:
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


def set_understanding_throttle_state(
    name: str,
    *,
    state: str,
    active: bool,
    multiplier: Optional[float],
    regime: Optional[str] = None,
    decision: Optional[str] = None,
) -> None:
    """Record understanding-loop throttle posture in Prometheus-compatible gauges."""

    labels = {
        "throttle": _normalise_label(name),
        "state": _normalise_label(state),
        "regime": _normalise_label(regime),
        "decision": _normalise_label(decision),
    }

    def _set_active() -> None:
        get_registry().get_gauge(
            "understanding_throttle_active",
            "Active state (1/0) for understanding loop throttles",
            ["throttle", "state", "regime", "decision"],
        ).labels(**labels).set(1.0 if active else 0.0)

    _call_metric("understanding_throttle_active.set", _set_active)

    multiplier_value = float(multiplier) if multiplier is not None else 0.0

    def _set_multiplier() -> None:
        get_registry().get_gauge(
            "understanding_throttle_multiplier",
            "Multiplier applied by understanding loop throttles",
            ["throttle", "state", "regime", "decision"],
        ).labels(**labels).set(multiplier_value)

    _call_metric("understanding_throttle_multiplier.set", _set_multiplier)


# Understanding-loop SLO probe exporters
def set_understanding_loop_latency(
    loop: str, stat: str, value: float | None
) -> None:
    if value is None:
        return

    labels = {
        "loop": _normalise_label(loop),
        "stat": _normalise_label(stat),
    }

    def _action() -> None:
        _understanding_loop_latency_seconds.labels(**labels).set(float(value))

    _call_metric("understanding_loop_latency_seconds.set", _action)


def set_understanding_loop_latency_status(loop: str, level: int) -> None:
    labels = {"loop": _normalise_label(loop)}

    def _action() -> None:
        _understanding_loop_latency_status.labels(**labels).set(float(level))

    _call_metric("understanding_loop_latency_status.set", _action)


def set_drift_alert_freshness(alert: str, freshness_seconds: float | None) -> None:
    if freshness_seconds is None:
        return

    labels = {"alert": _normalise_label(alert)}

    def _action() -> None:
        _drift_alert_freshness_seconds.labels(**labels).set(float(freshness_seconds))

    _call_metric("drift_alert_freshness_seconds.set", _action)


def set_drift_alert_status(alert: str, level: int) -> None:
    labels = {"alert": _normalise_label(alert)}

    def _action() -> None:
        _drift_alert_freshness_status.labels(**labels).set(float(level))

    _call_metric("drift_alert_freshness_status.set", _action)


def set_replay_determinism_drift(probe: str, drift: float | None) -> None:
    if drift is None:
        return

    labels = {"probe": _normalise_label(probe)}

    def _action() -> None:
        _replay_determinism_drift.labels(**labels).set(float(drift))

    _call_metric("replay_determinism_drift.set", _action)


def set_replay_determinism_status(probe: str, level: int) -> None:
    labels = {"probe": _normalise_label(probe)}

    def _action() -> None:
        _replay_determinism_status.labels(**labels).set(float(level))

    _call_metric("replay_determinism_status.set", _action)


def set_replay_determinism_mismatches(probe: str, count: int) -> None:
    labels = {"probe": _normalise_label(probe)}

    def _action() -> None:
        _replay_determinism_mismatches.labels(**labels).set(float(max(count, 0)))

    _call_metric("replay_determinism_mismatches.set", _action)


# Exporter
def start_metrics_server(
    port: Optional[int] = None,
    *,
    cert_path: Optional[str] = None,
    key_path: Optional[str] = None,
) -> None:
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
        except (TypeError, ValueError) as exc:  # pragma: no cover - hard to trigger deterministically
            _log.warning(
                "Invalid metrics port %r (from %s); defaulting to 8081",
                raw_port,
                "argument" if port is not None else "EMP_METRICS_PORT",
            )
            _log.debug("Port parsing failure", exc_info=exc)
            effective_port = 8081

        try:
            from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
        except ImportError:
            _log.debug("prometheus_client not installed; metrics exporter disabled")
            return

        resolved_cert_path = cert_path or os.environ.get("EMP_METRICS_TLS_CERT_PATH")
        resolved_key_path = key_path or os.environ.get("EMP_METRICS_TLS_KEY_PATH")
        if not resolved_cert_path or not resolved_key_path:
            raise ValueError(
                "Metrics exporter requires TLS configuration via cert_path/key_path or "
                "EMP_METRICS_TLS_CERT_PATH / EMP_METRICS_TLS_KEY_PATH"
            )

        try:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_context.load_cert_chain(
                certfile=os.path.expanduser(str(resolved_cert_path)),
                keyfile=os.path.expanduser(str(resolved_key_path)),
            )
        except (OSError, ValueError, ssl.SSLError) as exc:
            _log.warning("Failed to load TLS assets for metrics exporter: %s", exc, exc_info=exc)
            return

        handler_cls = _build_metrics_handler(
            generate_latest=generate_latest,
            registry=REGISTRY,
            content_type=CONTENT_TYPE_LATEST,
        )

        try:
            httpd = _make_metrics_server(effective_port, handler_cls, ssl_context)
        except OSError as exc:  # pragma: no cover - depends on runtime
            _log.warning(
                "Failed to start metrics exporter on port %s: %s", effective_port, exc, exc_info=exc
            )
            return

        thread = threading.Thread(
            target=httpd.serve_forever,
            name="metrics-exporter",
            daemon=True,
        )
        thread.start()

        _started = True
        bound_port = httpd.server_address[1]
        _log.info("Prometheus metrics exporter started with TLS on port %s", bound_port)


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
except ImportError as exc:  # pragma: no cover - optional dependency not present
    _log.debug("Telemetry metrics sink unavailable; registration skipped", exc_info=exc)
    _rt_set_metrics_sink = None
except Exception as exc:  # pragma: no cover - defensive guardrail during import
    _log.warning(
        "Failed to import telemetry metrics sink; registration skipped", exc_info=exc
    )
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

    def inc_counter(
        self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
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
        eff_buckets: List[float] = (
            list(buckets)
            if buckets is not None
            else [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]
        )

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
