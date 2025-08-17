import importlib
import sys
import types

import pytest

import src.operational.metrics as metrics


class _StubGauge:
    def __init__(self):
        self.set_calls = []
        self.inc_calls = []
        self.dec_calls = []
        self.labels_calls = []

    def set(self, v: float):
        self.set_calls.append(float(v))

    def inc(self, v: float = 1.0):
        self.inc_calls.append(float(v))

    def dec(self, v: float = 1.0):
        self.dec_calls.append(float(v))

    def labels(self, **labels: str):
        self.labels_calls.append(labels)
        return self


class _StubRegistry:
    def __init__(self):
        self.gauges = {}

    def get_gauge(self, name, desc, labelnames=None):
        g = _StubGauge()
        self.gauges[name] = g
        return g


def test_lazy_gauge_proxy_resolve_and_methods(monkeypatch: pytest.MonkeyPatch):
    stub = _StubRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: stub)

    g = metrics.LazyGaugeProxy("unit_test_gauge", "desc", ["x"])
    # set / inc / dec should not raise
    g.set(1.5)
    g.inc()
    g.inc(2.0)
    g.dec()
    g.dec(0.5)
    # labels should return a gauge-like object and not raise
    labeled = g.labels(x="A")
    assert labeled is stub.gauges["unit_test_gauge"]
    # Verify calls captured by stub
    sg = stub.gauges["unit_test_gauge"]
    assert sg.set_calls == [1.5]
    assert sg.inc_calls == [1.0, 2.0]
    assert sg.dec_calls == [1.0, 0.5]
    assert sg.labels_calls == [{"x": "A"}]


def test_start_metrics_server_import_present(monkeypatch: pytest.MonkeyPatch):
    # Reset started flag
    monkeypatch.setattr(metrics, "_started", False, raising=True)
    # Prepare dummy prometheus_client with call capture
    calls = []

    def _start_http_server(port: int):
        calls.append(port)

    dummy_module = types.SimpleNamespace(start_http_server=_start_http_server)
    monkeypatch.setitem(sys.modules, "prometheus_client", dummy_module)
    # Provide a specific port via env
    monkeypatch.setenv("EMP_METRICS_PORT", "9100")

    metrics.start_metrics_server()
    assert metrics._started is True
    assert calls == [9100], "Exporter should be started once with the configured port"

    # Idempotent: second call should not start again
    metrics.start_metrics_server()
    assert calls == [9100], "Exporter should not start twice"


def test_start_metrics_server_import_absent(monkeypatch: pytest.MonkeyPatch):
    # Reset started flag
    monkeypatch.setattr(metrics, "_started", False, raising=True)
    # Force ImportError by inserting a dummy module missing start_http_server
    dummy_missing = types.ModuleType("prometheus_client")
    monkeypatch.setitem(sys.modules, "prometheus_client", dummy_missing)

    # Should silently no-op without raising and without flipping _started
    metrics.start_metrics_server()
    assert metrics._started is False