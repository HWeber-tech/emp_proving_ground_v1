import pytest

from src.operational.metrics_registry import MetricsRegistry

pytest.importorskip("prometheus_client")


def test_real_prom_metrics_basic_operations():
    reg = MetricsRegistry()

    # Counter with labels
    c = reg.get_counter("m4_real_counter", "desc", ["l"])
    child_c = c.labels(l="x")
    child_c.inc()
    child_c.inc(2.0)

    # Gauge with labels
    g = reg.get_gauge("m4_real_gauge", "desc", ["l"])
    child_g = g.labels(l="x")
    child_g.set(0.0)
    child_g.inc(1.0)
    child_g.dec(0.5)

    # Histogram with labels
    h = reg.get_histogram("m4_real_hist", "desc", [0.1, 1.0, 5.0], ["l"])
    child_h = h.labels(l="x")
    child_h.observe(0.2)

    # Memoization identity check
    same_c = reg.get_counter("m4_real_counter", "desc", ["l"])
    assert same_c is c
