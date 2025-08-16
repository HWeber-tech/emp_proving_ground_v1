from __future__ import annotations

import builtins
import sys
import types

import pytest

from src.operational import metrics as metrics


# ----- Stubs -----


class CounterStub:
    def __init__(self, name: str):
        self.name = name
        self.labels_calls: list[dict[str, str]] = []
        self.inc_calls: int = 0
        self.last_inc_amount: float | None = None

    def inc(self, amount: float = 1.0) -> None:
        self.inc_calls += 1
        self.last_inc_amount = amount

    def labels(self, **labels: str) -> "CounterStub":
        self.labels_calls.append(labels)
        return self


class GaugeStub:
    def __init__(self, name: str):
        self.name = name
        self.labels_calls: list[dict[str, str]] = []
        self.set_calls: int = 0
        self.set_values: list[float] = []
        self.inc_calls: int = 0
        self.dec_calls: int = 0
        self.last_inc_amount: float | None = None
        self.last_dec_amount: float | None = None

    def set(self, value: float) -> None:
        self.set_calls += 1
        self.set_values.append(float(value))

    def inc(self, amount: float = 1.0) -> None:
        self.inc_calls += 1
        self.last_inc_amount = amount

    def dec(self, amount: float = 1.0) -> None:
        self.dec_calls += 1
        self.last_dec_amount = amount

    def labels(self, **labels: str) -> "GaugeStub":
        self.labels_calls.append(labels)
        return self


class HistogramStub:
    def __init__(self, name: str):
        self.name = name
        self.labels_calls: list[dict[str, str]] = []
        self.observe_calls: int = 0
        self.observed_values: list[float] = []

    def observe(self, value: float) -> None:
        self.observe_calls += 1
        self.observed_values.append(float(value))

    def labels(self, **labels: str) -> "HistogramStub":
        self.labels_calls.append(labels)
        return self


class StubRegistry:
    def __init__(self) -> None:
        self.counters: dict[tuple[str, tuple[str, ...] | None], CounterStub] = {}
        self.gauges: dict[tuple[str, tuple[str, ...] | None], GaugeStub] = {}
        self.hists: dict[tuple[str, tuple[str, ...] | None], HistogramStub] = {}

    @staticmethod
    def _key(name: str, labelnames: list[str] | None) -> tuple[str, tuple[str, ...] | None]:
        return (name, tuple(labelnames) if labelnames else None)

    def get_counter(self, name: str, _desc: str, labelnames: list[str] | None = None) -> CounterStub:
        key = self._key(name, labelnames)
        if key not in self.counters:
            self.counters[key] = CounterStub(name)
        return self.counters[key]

    def get_gauge(self, name: str, _desc: str, labelnames: list[str] | None = None) -> GaugeStub:
        key = self._key(name, labelnames)
        if key not in self.gauges:
            self.gauges[key] = GaugeStub(name)
        return self.gauges[key]

    def get_histogram(
        self, name: str, _desc: str, _buckets: list[float] | None = None, labelnames: list[str] | None = None
    ) -> HistogramStub:
        key = self._key(name, labelnames)
        if key not in self.hists:
            self.hists[key] = HistogramStub(name)
        return self.hists[key]

    # helpers
    def find_counter(self, name: str) -> CounterStub:
        for stub in self.counters.values():
            if stub.name == name:
                return stub
        raise AssertionError(f"Counter {name} not recorded")

    def find_gauge(self, name: str) -> GaugeStub:
        for stub in self.gauges.values():
            if stub.name == name:
                return stub
        raise AssertionError(f"Gauge {name} not recorded")

    def find_hist(self, name: str) -> HistogramStub:
        for stub in self.hists.values():
            if stub.name == name:
                return stub
        raise AssertionError(f"Histogram {name} not recorded")


@pytest.fixture
def stub_registry(monkeypatch) -> StubRegistry:
    stub = StubRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: stub, raising=False)
    return stub


# ----- A) FIX/MD wrappers -----


def test_fix_md_and_message_wrappers_labels_and_calls(stub_registry: StubRegistry):
    # inc_message
    metrics.inc_message("S1", "A")
    metrics.inc_message(None, None)
    c = stub_registry.find_counter("fix_messages_total")
    assert c.inc_calls == 2
    assert c.labels_calls[-1] == {"session": "?", "msg_type": "?"}

    # inc_md_reject
    metrics.inc_md_reject("BAD")
    metrics.inc_md_reject(None)
    c = stub_registry.find_counter("fix_md_rejects_total")
    assert c.inc_calls == 2
    assert c.labels_calls[-1] == {"reason": "?"}

    # set_session_connected
    metrics.set_session_connected("S1", True)
    metrics.set_session_connected(None, False)
    g = stub_registry.find_gauge("fix_session_connected")
    assert g.set_calls == 2
    assert g.set_values[0] == 1.0 and g.set_values[-1] == 0.0
    assert g.labels_calls[-1] == {"session": "?"}

    # inc_reconnect
    metrics.inc_reconnect("S1", "OK")
    metrics.inc_reconnect(None, None)
    c = stub_registry.find_counter("fix_reconnect_attempts_total")
    assert c.inc_calls == 2
    assert c.labels_calls[-1] == {"session": "?", "outcome": "?"}

    # inc_business_reject
    metrics.inc_business_reject("D")
    metrics.inc_business_reject(None)
    c = stub_registry.find_counter("fix_business_rejects_total")
    assert c.inc_calls == 2
    assert c.labels_calls[-1] == {"ref_msg_type": "?"}


# ----- B) Histograms observe for non-negative only -----


def test_histograms_observe_only_non_negative(stub_registry: StubRegistry):
    metrics.observe_exec_latency(0.2)
    metrics.observe_exec_latency(-1.0)
    h = stub_registry.find_hist("fix_exec_report_latency_seconds")
    assert h.observe_calls == 1
    assert h.observed_values == [0.2]

    metrics.observe_cancel_latency(0.05)
    metrics.observe_cancel_latency(-0.5)
    h2 = stub_registry.find_hist("fix_cancel_latency_seconds")
    assert h2.observe_calls == 1
    assert h2.observed_values == [0.05]


# ----- C) MD staleness clamp -----


def test_set_md_staleness_clamps_negative(stub_registry: StubRegistry):
    metrics.set_md_staleness("SYM", -5.0)
    g = stub_registry.find_gauge("fix_md_staleness_seconds")
    assert g.set_calls >= 1
    assert g.set_values[-1] == 0.0
    assert g.labels_calls[-1] == {"symbol": "SYM"}


# ----- D) Heartbeat -----


def test_heartbeat_metrics(stub_registry: StubRegistry):
    metrics.observe_heartbeat_interval("S1", 1.5)
    metrics.observe_heartbeat_interval("S1", -2.0)  # no observe
    h = stub_registry.find_hist("fix_heartbeat_interval_seconds")
    assert h.labels_calls[-1] == {"session": "S1"}
    assert h.observe_calls == 1
    assert h.observed_values == [1.5]

    metrics.inc_test_request(None)
    c1 = stub_registry.find_counter("fix_test_requests_total")
    assert c1.inc_calls == 1
    assert c1.labels_calls[-1] == {"session": "?"}

    metrics.inc_missed_heartbeat("S9")
    c2 = stub_registry.find_counter("fix_missed_heartbeats_total")
    assert c2.inc_calls == 1
    assert c2.labels_calls[-1] == {"session": "S9"}


# ----- E) Pre-trade -----


def test_pretrade_denial_default_labels(stub_registry: StubRegistry):
    metrics.inc_pretrade_denial(None, None)
    c = stub_registry.find_counter("fix_pretrade_denials_total")
    assert c.inc_calls == 1
    assert c.labels_calls[-1] == {"symbol": "?", "reason": "?"}


# ----- F) Vol wrappers -----


def test_vol_wrappers(stub_registry: StubRegistry):
    metrics.set_vol_sigma("IBM", -0.2)
    g1 = stub_registry.find_gauge("vol_sigma_ann")
    assert g1.set_values[-1] == 0.0
    assert g1.labels_calls[-1] == {"symbol": "IBM"}

    metrics.inc_vol_regime(None, None)
    c = stub_registry.find_counter("vol_regime_total")
    assert c.inc_calls == 1
    assert c.labels_calls[-1] == {"symbol": "?", "regime": "?"}

    metrics.set_vol_divergence("AAPL", -1.0)
    g2 = stub_registry.find_gauge("vol_rv_garch_divergence")
    assert g2.set_values[-1] == 0.0
    assert g2.labels_calls[-1] == {"symbol": "AAPL"}


# ----- G) WHY wrappers -----


def test_why_wrappers(stub_registry: StubRegistry):
    metrics.set_why_signal("MSFT", -123.4)
    g_sig = stub_registry.find_gauge("why_composite_signal")
    assert g_sig.set_values[-1] == -123.4
    assert g_sig.labels_calls[-1] == {"symbol": "MSFT"}

    metrics.set_why_conf("MSFT", 2.0)  # clamp to 1
    metrics.set_why_conf("MSFT", -0.1)  # clamp to 0
    g_conf = stub_registry.find_gauge("why_confidence")
    assert g_conf.set_values[:2] == [1.0, 0.0]
    assert g_conf.labels_calls[-1] == {"symbol": "MSFT"}

    # set_why_feature with and without extra labels; 1.0 if truthy else 0.0
    metrics.set_why_feature("featA", True, labels={"k": "v"})
    # The registry keys by (name, labelnames), so different label sets create different stubs.
    # Verify the 'featA' call recorded correctly on some stub with matching labels.
    g_feat = stub_registry.find_gauge("why_feature_available")
    assert {"k": "v", "feature": "featA"} in g_feat.labels_calls
    assert g_feat.set_values[-1] == 1.0

    metrics.set_why_feature("featB", False, labels=None)
    # Find the stub that recorded only the 'feature' label for featB and validate it observed 0.0
    candidates = [
        s for s in stub_registry.gauges.values()
        if s.name == "why_feature_available" and {"feature": "featB"} in s.labels_calls
    ]
    assert candidates, "Expected why_feature_available stub for featB"
    g_feat_b = candidates[-1]
    assert g_feat_b.set_values[-1] == 0.0
    assert {"feature": "featB"} in g_feat_b.labels_calls


# ----- H) Legacy proxies -----


def test_legacy_gauge_proxies_delegate(stub_registry: StubRegistry):
    metrics.fix_parity_mismatched_orders.set(42.0)
    metrics.fix_parity_mismatched_orders.inc()
    metrics.fix_parity_mismatched_orders.dec()
    metrics.fix_parity_mismatched_orders.labels(source="X").set(1.0)

    g = stub_registry.find_gauge("fix_parity_mismatched_orders")
    assert g.set_calls >= 2  # direct set + labels().set
    assert g.inc_calls == 1
    assert g.dec_calls == 1
    assert g.labels_calls[-1] == {"source": "X"}

    metrics.fix_parity_mismatched_positions.inc()
    g2 = stub_registry.find_gauge("fix_parity_mismatched_positions")
    assert g2.inc_calls == 1


# ----- I) Exporter -----


def test_start_metrics_server_no_prom(monkeypatch):
    monkeypatch.setattr(metrics, "_started", False, raising=True)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # pragma: no cover - behavior validated by assertions below
        if name == "prometheus_client":
            raise ImportError("prometheus_client unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Should not raise even if prometheus_client is missing; idempotent
    metrics.start_metrics_server()
    metrics.start_metrics_server()  # second call no-op


def test_start_metrics_server_with_prom(monkeypatch):
    monkeypatch.setattr(metrics, "_started", False, raising=True)

    captured_ports: list[int] = []

    def start_http_server(port: int) -> None:
        captured_ports.append(int(port))

    fake_mod = types.SimpleNamespace(start_http_server=start_http_server)
    monkeypatch.setitem(sys.modules, "prometheus_client", fake_mod)

    metrics.start_metrics_server(port=9123)
    metrics.start_metrics_server(port=9123)  # idempotent

    assert captured_ports == [9123]