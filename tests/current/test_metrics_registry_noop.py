from src.operational.metrics_registry import MetricsRegistry


def _force_no_prom(monkeypatch):
    def fake_import(self):
        # Simulate ImportError path
        self._enabled = False
        return False

    monkeypatch.setattr(MetricsRegistry, "_import_prometheus", fake_import, raising=True)


def test_noop_counter_inc_and_labels_no_raise(monkeypatch):
    _force_no_prom(monkeypatch)
    reg = MetricsRegistry()
    c = reg.get_counter("noop_counter", "desc", ["label"])
    # Should not raise:
    c.labels(label="x").inc()
    c.inc()


def test_noop_gauge_set_inc_dec_and_labels_no_raise(monkeypatch):
    _force_no_prom(monkeypatch)
    reg = MetricsRegistry()
    g = reg.get_gauge("noop_gauge", "desc", ["l"])
    # Should not raise:
    g.labels(l="v").set(1.23)
    g.set(0.0)
    g.inc()
    g.dec()


def test_noop_histogram_observe_and_labels_no_raise(monkeypatch):
    _force_no_prom(monkeypatch)
    reg = MetricsRegistry()
    h = reg.get_histogram("noop_hist", "desc", [0.1, 1.0], ["l"])
    # Should not raise:
    h.labels(l="v").observe(0.2)
    h.observe(0.3)
