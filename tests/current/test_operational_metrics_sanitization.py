import pytest

import src.operational.metrics as metrics


class StubCounter:
    def __init__(self) -> None:
        self.labels_calls: list[dict[str, str]] = []
        self.inc_calls: list[float] = []

    def labels(self, **labels: str) -> "StubCounter":
        self.labels_calls.append(labels)
        return self

    def inc(self, amount: float = 1.0) -> None:
        self.inc_calls.append(float(amount))


class StubGauge:
    def __init__(self) -> None:
        self.labels_calls: list[dict[str, str]] = []
        self.set_calls: list[float] = []

    def labels(self, **labels: str) -> "StubGauge":
        self.labels_calls.append(labels)
        return self

    def set(self, value: float) -> None:
        self.set_calls.append(float(value))

    def inc(self, amount: float = 1.0) -> None:  # pragma: no cover - unused
        self.set_calls.append(float(amount))

    def dec(self, amount: float = 1.0) -> None:  # pragma: no cover - unused
        self.set_calls.append(-float(amount))


class StubHistogram:
    def __init__(self) -> None:
        self.labels_calls: list[dict[str, str]] = []
        self.observe_calls: list[float] = []
        self.buckets: tuple[float, ...] | None = None

    def labels(self, **labels: str) -> "StubHistogram":
        self.labels_calls.append(labels)
        return self

    def observe(self, value: float) -> None:
        self.observe_calls.append(float(value))


class StubRegistry:
    def __init__(self) -> None:
        self.counters: dict[tuple[str, tuple[str, ...]], StubCounter] = {}
        self.gauges: dict[tuple[str, tuple[str, ...]], StubGauge] = {}
        self.histograms: dict[tuple[str, tuple[str, ...]], StubHistogram] = {}

    @staticmethod
    def _key(name: str, labelnames: list[str] | None) -> tuple[str, tuple[str, ...]]:
        labels_tuple = tuple(labelnames) if labelnames else tuple()
        return name, labels_tuple

    def get_counter(
        self, name: str, description: str, labelnames: list[str] | None = None
    ) -> StubCounter:
        key = self._key(name, labelnames)
        counter = self.counters.get(key)
        if counter is None:
            counter = StubCounter()
            self.counters[key] = counter
        return counter

    def get_gauge(
        self, name: str, description: str, labelnames: list[str] | None = None
    ) -> StubGauge:
        key = self._key(name, labelnames)
        gauge = self.gauges.get(key)
        if gauge is None:
            gauge = StubGauge()
            self.gauges[key] = gauge
        return gauge

    def get_histogram(
        self,
        name: str,
        description: str,
        buckets: list[float] | None = None,
        labelnames: list[str] | None = None,
    ) -> StubHistogram:
        key = self._key(name, labelnames)
        histogram = self.histograms.get(key)
        if histogram is None:
            histogram = StubHistogram()
            self.histograms[key] = histogram
        if buckets is not None:
            histogram.buckets = tuple(buckets)
        return histogram


@pytest.fixture()
def stub_registry(monkeypatch: pytest.MonkeyPatch) -> StubRegistry:
    registry = StubRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: registry)
    return registry


def test_fix_metric_sanitization(stub_registry: StubRegistry) -> None:
    metrics.inc_md_reject("")
    metrics.inc_message("", "")
    metrics.set_session_connected("", True)
    metrics.inc_reconnect("", "")
    metrics.inc_business_reject(None)

    assert stub_registry.counters[("fix_md_rejects_total", ("reason",))].labels_calls[-1] == {
        "reason": "?"
    }
    assert stub_registry.counters[("fix_messages_total", ("session", "msg_type"))].labels_calls[
        -1
    ] == {
        "session": "?",
        "msg_type": "?",
    }
    gauge = stub_registry.gauges[("fix_session_connected", ("session",))]
    assert gauge.labels_calls[-1] == {"session": "?"}
    assert gauge.set_calls[-1] == pytest.approx(1.0)
    assert stub_registry.counters[
        ("fix_reconnect_attempts_total", ("session", "outcome"))
    ].labels_calls[-1] == {
        "session": "?",
        "outcome": "?",
    }
    assert stub_registry.counters[("fix_business_rejects_total", ("ref_msg_type",))].labels_calls[
        -1
    ] == {"ref_msg_type": "?"}


def test_fix_latency_and_staleness_guards(stub_registry: StubRegistry) -> None:
    metrics.observe_exec_latency(-1.0)
    metrics.observe_cancel_latency(-0.5)
    assert ("fix_exec_report_latency_seconds", tuple()) not in stub_registry.histograms
    assert ("fix_cancel_latency_seconds", tuple()) not in stub_registry.histograms

    metrics.observe_exec_latency(0.25)
    metrics.observe_cancel_latency(0.5)
    assert stub_registry.histograms[("fix_exec_report_latency_seconds", tuple())].observe_calls == [
        pytest.approx(0.25)
    ]
    assert stub_registry.histograms[("fix_cancel_latency_seconds", tuple())].observe_calls == [
        pytest.approx(0.5)
    ]

    metrics.set_md_staleness("", -2.0)
    staleness = stub_registry.gauges[("fix_md_staleness_seconds", ("symbol",))]
    assert staleness.labels_calls[-1] == {"symbol": "?"}
    assert staleness.set_calls[-1] == pytest.approx(0.0)


def test_fix_heartbeat_counters(stub_registry: StubRegistry) -> None:
    metrics.observe_heartbeat_interval("", -1)
    assert ("fix_heartbeat_interval_seconds", ("session",)) not in stub_registry.histograms

    metrics.observe_heartbeat_interval("trade", 15)
    heartbeat = stub_registry.histograms[("fix_heartbeat_interval_seconds", ("session",))]
    assert heartbeat.labels_calls[-1] == {"session": "trade"}
    assert heartbeat.observe_calls[-1] == pytest.approx(15.0)

    metrics.inc_test_request("")
    metrics.inc_missed_heartbeat("")
    assert stub_registry.counters[("fix_test_requests_total", ("session",))].labels_calls[-1] == {
        "session": "?"
    }
    assert stub_registry.counters[("fix_missed_heartbeats_total", ("session",))].labels_calls[
        -1
    ] == {"session": "?"}


def test_pretrade_and_vol_metrics_boundaries(stub_registry: StubRegistry) -> None:
    metrics.inc_pretrade_denial("", "")
    denial = stub_registry.counters[("fix_pretrade_denials_total", ("symbol", "reason"))]
    assert denial.labels_calls[-1] == {"symbol": "?", "reason": "?"}

    metrics.set_vol_sigma("", -2.5)
    sigma = stub_registry.gauges[("vol_sigma_ann", ("symbol",))]
    assert sigma.labels_calls[-1] == {"symbol": "?"}
    assert sigma.set_calls[-1] == pytest.approx(0.0)

    metrics.inc_vol_regime("", "")
    vol_regime = stub_registry.counters[("vol_regime_total", ("symbol", "regime"))]
    assert vol_regime.labels_calls[-1] == {"symbol": "?", "regime": "?"}

    metrics.set_vol_divergence("", -1.2)
    divergence = stub_registry.gauges[("vol_rv_garch_divergence", ("symbol",))]
    assert divergence.labels_calls[-1] == {"symbol": "?"}
    assert divergence.set_calls[-1] == pytest.approx(0.0)


def test_why_metrics_bounds_and_labels(stub_registry: StubRegistry) -> None:
    metrics.set_why_signal("", 0.4)
    signal = stub_registry.gauges[("why_composite_signal", ("symbol",))]
    assert signal.labels_calls[-1] == {"symbol": "?"}
    assert signal.set_calls[-1] == pytest.approx(0.4)

    metrics.set_why_conf("", 2.5)
    metrics.set_why_conf("", -1.0)
    confidence = stub_registry.gauges[("why_confidence", ("symbol",))]
    assert confidence.set_calls[-2:] == [pytest.approx(1.0), pytest.approx(0.0)]

    metrics.set_why_feature("yields", False, labels={"tenor": "2Y"})
    feature = stub_registry.gauges[("why_feature_available", ("feature", "tenor"))]
    assert feature.labels_calls[-1] == {"feature": "yields", "tenor": "2Y"}
    assert feature.set_calls[-1] == pytest.approx(0.0)
