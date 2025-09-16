import logging

import pytest

from src.operational import metrics


class _ExplodingRegistry:
    def get_gauge(self, name: str, description: str, labelnames: list[str] | None) -> object:
        raise RuntimeError("gauge boom")

    def get_counter(self, name: str, description: str, labelnames: list[str] | None) -> object:
        raise RuntimeError("counter boom")

    def get_histogram(
        self,
        name: str,
        description: str,
        buckets: list[float],
        labelnames: list[str] | None = None,
    ) -> object:
        raise RuntimeError("histogram boom")


@pytest.fixture(autouse=True)
def reset_metric_warning_state(monkeypatch):
    # Ensure each test sees a fresh warning tracker so expectations remain deterministic.
    monkeypatch.setattr(metrics, "_warned_metrics", set())
    return None


def test_lazy_gauge_proxy_logs_once(monkeypatch, caplog):
    registry = _ExplodingRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: registry)
    proxy = metrics.LazyGaugeProxy("test_metric", "desc")

    with caplog.at_level(logging.DEBUG):
        proxy.set(1.0)
        proxy.set(2.0)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG and "Repeated failure" in r.getMessage()]
    assert len(warnings) == 1
    assert any("test_metric.set" in r.getMessage() for r in warnings)
    assert debug, "Expected repeated failure log at debug level"


def test_registry_metrics_sink_logs(monkeypatch, caplog):
    registry = _ExplodingRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: registry)
    sink = metrics._RegistryMetricsSink()

    with caplog.at_level(logging.DEBUG):
        sink.set_gauge("foo", 1.0, labels={"l": "x"})
        sink.set_gauge("foo", 2.0, labels={"l": "x"})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG and "Repeated failure" in r.getMessage()]
    assert len(warnings) == 1
    assert any("metrics_sink.foo.set_gauge" in r.getMessage() for r in warnings)
    assert debug
