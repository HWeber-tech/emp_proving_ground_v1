from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from src.operational import metrics


@pytest.fixture(autouse=True)
def _reset_metric_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "_warned_metrics", set())
    monkeypatch.setattr(metrics, "_started", False)


def test_log_metric_failure_escalates_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        metrics._log_metric_failure("test.metric", RuntimeError("boom"))
        metrics._log_metric_failure("test.metric", RuntimeError("boom"))

    assert any(
        record.levelno == logging.WARNING and "Failed to update metric" in record.getMessage()
        for record in caplog.records
    )
    repeated = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG and "Repeated failure" in record.getMessage()
    ]
    assert repeated, "expected repeated failure to be logged at debug level"


class _RecordingGauge:
    def __init__(self, name: str) -> None:
        self.name = name
        self.set_calls: List[float] = []
        self.labels_calls: List[Dict[str, str]] = []

    def set(self, value: float) -> None:
        self.set_calls.append(float(value))

    def inc(self, amount: float = 1.0) -> None:  # pragma: no cover - not used in tests
        self.set_calls.append(float(amount))

    def dec(self, amount: float = 1.0) -> None:  # pragma: no cover - not used in tests
        self.set_calls.append(-float(amount))

    def labels(self, **labels: str) -> "_RecordingGauge":
        self.labels_calls.append(dict(labels))
        return self


class _RecordingCounter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.inc_calls: List[float] = []
        self.labels_calls: List[Dict[str, str]] = []

    def inc(self, amount: float = 1.0) -> None:
        self.inc_calls.append(float(amount))

    def labels(self, **labels: str) -> "_RecordingCounter":
        self.labels_calls.append(dict(labels))
        return self


class _RecordingHistogram:
    def __init__(self, name: str, buckets: List[float]) -> None:
        self.name = name
        self.buckets = list(buckets)
        self.labels_calls: List[Dict[str, str]] = []
        self.observed: List[float] = []

    def observe(self, value: float) -> None:
        self.observed.append(float(value))

    def labels(self, **labels: str) -> "_RecordingHistogram":
        self.labels_calls.append(dict(labels))
        return self


class _RecordingRegistry:
    def __init__(self) -> None:
        self.gauge_requests: List[Tuple[str, Tuple[str, ...] | None]] = []
        self.counter_requests: List[Tuple[str, Tuple[str, ...] | None]] = []
        self.histogram_requests: List[Tuple[str, Tuple[float, ...], Tuple[str, ...] | None]] = []
        self.gauges: List[_RecordingGauge] = []
        self.counters: List[_RecordingCounter] = []
        self.histograms: List[_RecordingHistogram] = []

    def get_gauge(
        self, name: str, description: str, labelnames: List[str] | None = None
    ) -> _RecordingGauge:
        self.gauge_requests.append((name, tuple(labelnames) if labelnames else None))
        gauge = _RecordingGauge(name)
        self.gauges.append(gauge)
        return gauge

    def get_counter(
        self, name: str, description: str, labelnames: List[str] | None = None
    ) -> _RecordingCounter:
        self.counter_requests.append((name, tuple(labelnames) if labelnames else None))
        counter = _RecordingCounter(name)
        self.counters.append(counter)
        return counter

    def get_histogram(
        self,
        name: str,
        description: str,
        buckets: List[float] | None = None,
        labelnames: List[str] | None = None,
    ) -> _RecordingHistogram:
        eff_buckets = tuple(buckets or [])
        self.histogram_requests.append((name, eff_buckets, tuple(labelnames) if labelnames else None))
        histogram = _RecordingHistogram(name, list(eff_buckets))
        self.histograms.append(histogram)
        return histogram


def test_lazy_gauge_proxy_labels_recovers_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _RecordingRegistry()
    flaky_gauge = _RecordingGauge("lazy")

    def _get_gauge(name: str, description: str, labelnames: List[str] | None = None) -> _RecordingGauge:
        registry.gauge_requests.append((name, tuple(labelnames) if labelnames else None))
        return flaky_gauge

    monkeypatch.setattr(metrics, "get_registry", lambda: types.SimpleNamespace(get_gauge=_get_gauge))

    calls = 0

    def _flaky_labels(**labels: str) -> _RecordingGauge:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("labels failed")
        flaky_gauge.labels_calls.append(dict(labels))
        return flaky_gauge

    flaky_gauge.labels = _flaky_labels  # type: ignore[assignment]

    proxy = metrics.LazyGaugeProxy("lazy", "Lazy gauge")
    first = proxy.labels(tag="value")
    second = proxy.labels(tag="value")

    assert first is flaky_gauge
    assert second is flaky_gauge
    assert flaky_gauge.labels_calls == [{"tag": "value"}]


def test_lazy_gauge_proxy_labels_returns_noop_when_registry_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingRegistry:
        def get_gauge(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("registry offline")

    monkeypatch.setattr(metrics, "get_registry", lambda: _FailingRegistry())

    proxy = metrics.LazyGaugeProxy("missing", "Missing gauge")
    result = proxy.labels()

    assert result is metrics._NOOP_GAUGE


def test_start_metrics_server_uses_environment_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cert = (
        Path(__file__).resolve().parent.parent / "runtime" / "certs" / "server.pem"
    )
    key = (
        Path(__file__).resolve().parent.parent / "runtime" / "certs" / "server.key"
    )
    payload = b"metric 1\n"

    module = types.SimpleNamespace(
        CONTENT_TYPE_LATEST="text/plain",
        REGISTRY=object(),
        generate_latest=lambda _registry: payload,
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", module)
    monkeypatch.setenv("EMP_METRICS_PORT", "9123")
    monkeypatch.setenv("EMP_METRICS_TLS_CERT_PATH", str(cert))
    monkeypatch.setenv("EMP_METRICS_TLS_KEY_PATH", str(key))

    ports: list[int] = []

    class _Server:
        def __init__(self, port: int) -> None:
            self.server_address = ("", port)
            self.calls = 0

        def serve_forever(self) -> None:
            self.calls += 1
            ports.append(self.server_address[1])

    def _fake_make_metrics_server(port: int, _handler, _context) -> _Server:
        return _Server(port)

    class _Thread:
        def __init__(self, target, **_kwargs) -> None:
            self._target = target

        def start(self) -> None:
            self._target()

    monkeypatch.setattr(metrics, "_make_metrics_server", _fake_make_metrics_server)
    monkeypatch.setattr(metrics.threading, "Thread", _Thread)

    metrics.start_metrics_server()
    metrics.start_metrics_server()

    assert ports == [9123]


def test_start_metrics_server_warns_on_invalid_port(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cert = (
        Path(__file__).resolve().parent.parent / "runtime" / "certs" / "server.pem"
    )
    key = (
        Path(__file__).resolve().parent.parent / "runtime" / "certs" / "server.key"
    )
    module = types.SimpleNamespace(
        CONTENT_TYPE_LATEST="text/plain",
        REGISTRY=object(),
        generate_latest=lambda _registry: b"metric 1\n",
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", module)
    monkeypatch.setenv("EMP_METRICS_PORT", "invalid")
    monkeypatch.setenv("EMP_METRICS_TLS_CERT_PATH", str(cert))
    monkeypatch.setenv("EMP_METRICS_TLS_KEY_PATH", str(key))

    ports: list[int] = []

    class _Server:
        def __init__(self, port: int) -> None:
            self.server_address = ("", port)

        def serve_forever(self) -> None:
            ports.append(self.server_address[1])

    monkeypatch.setattr(metrics, "_make_metrics_server", lambda port, _handler, _context: _Server(port))

    class _Thread:
        def __init__(self, target, **_kwargs) -> None:
            self._target = target

        def start(self) -> None:
            self._target()

    monkeypatch.setattr(metrics.threading, "Thread", _Thread)

    with caplog.at_level(logging.WARNING):
        metrics.start_metrics_server()

    assert ports == [8081]
    assert any("defaulting to 8081" in record.getMessage() for record in caplog.records)


def test_registry_metrics_sink_routes_through_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _RecordingRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: registry)

    sink = metrics._RegistryMetricsSink()
    sink.set_gauge("gauge", 5.0, labels={"b": "2", "a": "1"})
    sink.inc_counter("counter", amount=3.0, labels={"x": "9"})
    sink.observe_histogram("hist", 0.5)

    assert registry.gauge_requests == [("gauge", ("a", "b"))]
    assert registry.counters[0].labels_calls == [{"x": "9"}]
    assert registry.counters[0].inc_calls == [3.0]
    assert registry.histogram_requests[0][0] == "hist"
    assert registry.histograms[0].observed == [0.5]


def test_call_metric_invokes_fallback_when_action_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fallback_calls: list[str] = []

    def _action() -> None:
        raise RuntimeError("boom")

    def _fallback() -> str:
        fallback_calls.append("called")
        return "fallback"

    with caplog.at_level(logging.WARNING):
        result = metrics._call_metric("test.metric", _action, fallback=_fallback)

    assert result == "fallback"
    assert fallback_calls == ["called"]
    assert any(
        "Failed to update metric" in record.getMessage() for record in caplog.records
    )


def test_fix_metrics_wrappers_sanitise_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _RecordingRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: registry)

    metrics.inc_md_reject("")
    metrics.inc_message("", "")
    metrics.set_session_connected("", True)
    metrics.inc_reconnect("", "")
    metrics.inc_business_reject(None)

    assert registry.counter_requests[0] == ("fix_md_rejects_total", ("reason",))
    assert registry.counters[0].labels_calls == [{"reason": "?"}]
    assert registry.counters[0].inc_calls == [1.0]

    assert registry.counter_requests[1] == ("fix_messages_total", ("session", "msg_type"))
    assert registry.counters[1].labels_calls == [{"session": "?", "msg_type": "?"}]

    assert registry.gauge_requests[0] == ("fix_session_connected", ("session",))
    assert registry.gauges[0].labels_calls == [{"session": "?"}]
    assert registry.gauges[0].set_calls == [1.0]

    assert registry.counter_requests[2] == ("fix_reconnect_attempts_total", ("session", "outcome"))
    assert registry.counters[2].labels_calls == [{"session": "?", "outcome": "?"}]

    assert registry.counter_requests[3] == ("fix_business_rejects_total", ("ref_msg_type",))
    assert registry.counters[3].labels_calls == [{"ref_msg_type": "?"}]


def test_latency_and_staleness_metrics_apply_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _RecordingRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: registry)

    metrics.observe_exec_latency(-0.5)
    metrics.observe_cancel_latency(-1.0)
    metrics.observe_heartbeat_interval("", -2.0)
    metrics.set_md_staleness("", -3.0)

    assert not registry.histograms
    assert registry.gauges[0].labels_calls == [{"symbol": "?"}]
    assert registry.gauges[0].set_calls == [0.0]

    metrics.observe_exec_latency(0.25)
    metrics.observe_cancel_latency(0.75)
    metrics.observe_heartbeat_interval("", 5.0)

    assert registry.histogram_requests[0] == (
        "fix_exec_report_latency_seconds",
        (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
        None,
    )
    assert registry.histograms[0].observed == [0.25]

    assert registry.histogram_requests[1] == (
        "fix_cancel_latency_seconds",
        (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
        None,
    )
    assert registry.histograms[1].observed == [0.75]

    assert registry.histogram_requests[2] == (
        "fix_heartbeat_interval_seconds",
        (1, 5, 10, 20, 30, 45, 60, 90, 120),
        ("session",),
    )
    assert registry.histograms[2].labels_calls == [{"session": "?"}]
    assert registry.histograms[2].observed == [5.0]


def test_why_feature_metric_sorts_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _RecordingRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: registry)

    metrics.set_why_feature("macro_bias", True, labels={"b": "2", "a": "1"})
    metrics.set_why_feature("macro_bias", False)

    assert registry.gauge_requests[-2] == (
        "why_feature_available",
        ("feature", "a", "b"),
    )
    assert registry.gauges[-2].labels_calls == [
        {"feature": "macro_bias", "a": "1", "b": "2"}
    ]
    assert registry.gauges[-2].set_calls == [1.0]

    assert registry.gauge_requests[-1] == ("why_feature_available", ("feature",))
    assert registry.gauges[-1].labels_calls == [{"feature": "macro_bias"}]
    assert registry.gauges[-1].set_calls == [0.0]


def test_set_understanding_throttle_state_exports_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _RecordingRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: registry)

    metrics.set_understanding_throttle_state(
        "volatility",
        state="reduced",
        active=True,
        multiplier=0.75,
        regime="balanced",
        decision="momentum_breakout",
    )
    metrics.set_understanding_throttle_state(
        "",
        state="",
        active=False,
        multiplier=None,
        regime=None,
        decision=None,
    )

    assert registry.gauge_requests[:2] == [
        ("understanding_throttle_active", ("throttle", "state", "regime", "decision")),
        ("understanding_throttle_multiplier", ("throttle", "state", "regime", "decision")),
    ]

    active_gauge = registry.gauges[0]
    multiplier_gauge = registry.gauges[1]
    assert active_gauge.labels_calls == [
        {
            "throttle": "volatility",
            "state": "reduced",
            "regime": "balanced",
            "decision": "momentum_breakout",
        }
    ]
    assert active_gauge.set_calls == [1.0]

    assert multiplier_gauge.labels_calls == [
        {
            "throttle": "volatility",
            "state": "reduced",
            "regime": "balanced",
            "decision": "momentum_breakout",
        }
    ]
    assert multiplier_gauge.set_calls == [pytest.approx(0.75)]

    inactive_gauge = registry.gauges[2]
    inactive_multiplier = registry.gauges[3]
    assert inactive_gauge.labels_calls == [
        {
            "throttle": "unknown",
            "state": "unknown",
            "regime": "unknown",
            "decision": "unknown",
        }
    ]
    assert inactive_gauge.set_calls == [0.0]

    assert inactive_multiplier.set_calls == [0.0]


def test_understanding_loop_slo_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _RecordingRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: registry)

    metrics.set_understanding_loop_latency("professional", "p95", 1.75)
    metrics.set_understanding_loop_latency_status("professional", 2)
    metrics.set_drift_alert_freshness("page_hinkley", 240.0)
    metrics.set_drift_alert_status("page_hinkley", 1)
    metrics.set_replay_determinism_drift("fast_weights", 0.08)
    metrics.set_replay_determinism_status("fast_weights", 0)
    metrics.set_replay_determinism_mismatches("fast_weights", 3)

    expected_gauges = {
        "understanding_loop_latency_seconds": ("loop", "stat"),
        "understanding_loop_latency_status": ("loop",),
        "drift_alert_freshness_seconds": ("alert",),
        "drift_alert_freshness_status": ("alert",),
        "replay_determinism_drift": ("probe",),
        "replay_determinism_status": ("probe",),
        "replay_determinism_mismatches": ("probe",),
    }

    recorded = {name: labels for name, labels in registry.gauge_requests}
    for gauge_name, labelnames in expected_gauges.items():
        assert recorded[gauge_name] == labelnames

    latency_gauge = registry.gauges[0]
    assert latency_gauge.labels_calls == [{"loop": "professional", "stat": "p95"}]
    assert latency_gauge.set_calls == [pytest.approx(1.75)]

    status_gauge = registry.gauges[1]
    assert status_gauge.labels_calls == [{"loop": "professional"}]
    assert status_gauge.set_calls == [2.0]

    drift_gauge = registry.gauges[2]
    assert drift_gauge.labels_calls == [{"alert": "page_hinkley"}]
    assert drift_gauge.set_calls == [pytest.approx(240.0)]

    replay_status_gauge = registry.gauges[5]
    assert replay_status_gauge.labels_calls == [{"probe": "fast_weights"}]
    assert replay_status_gauge.set_calls == [0.0]

    mismatches_gauge = registry.gauges[6]
    assert mismatches_gauge.set_calls == [3.0]
