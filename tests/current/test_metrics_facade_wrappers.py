
import src.operational.metrics as m


class DummyChildGauge:
    def __init__(self, calls):
        self.calls = calls

    def set(self, value: float) -> None:
        self.calls.append(("child_set", float(value)))


class DummyGauge:
    def __init__(self, calls):
        self.calls = calls

    def labels(self, **labels: str):
        self.calls.append(("labels", dict(labels)))
        return DummyChildGauge(self.calls)

    def set(self, value: float) -> None:
        # Not used by wrappers directly (they use child.set), but keep as safety.
        self.calls.append(("set", float(value)))


class DummyRegistry:
    def __init__(self):
        self.calls = []

    def get_gauge(self, name: str, description: str, labelnames=None):
        self.calls.append(
            (
                "get_gauge",
                name,
                tuple(labelnames) if labelnames is not None else None,
            )
        )
        return DummyGauge(self.calls)


def test_wrappers_call_registry_get_gauge_and_do_not_raise(monkeypatch):
    reg = DummyRegistry()
    monkeypatch.setattr(m, "get_registry", lambda: reg, raising=True)

    # No exceptions should be raised
    m.set_why_signal("EURUSD", 0.1)
    m.set_why_conf("EURUSD", 0.9)

    # Without extra labels
    m.set_why_feature("yields", True)

    # With extra labels (order should be deterministic: feature + sorted(keys))
    m.set_why_feature("yields", True, labels={"tenor": "10Y", "country": "US"})

    # Validate get_gauge invocations
    calls = [c for c in reg.calls if c and c[0] == "get_gauge"]
    assert len(calls) == 4

    # First two should be symbol-only gauges
    assert calls[0][1] == "why_composite_signal"
    assert calls[0][2] == ("symbol",)
    assert calls[1][1] == "why_confidence"
    assert calls[1][2] == ("symbol",)

    # Third: feature-only
    assert calls[2][1] == "why_feature_available"
    assert calls[2][2] == ("feature",)

    # Fourth: feature + sorted(extra label keys)
    assert calls[3][1] == "why_feature_available"
    assert calls[3][2] == ("feature", "country", "tenor")

    # Ensure labels then child.set were called at least once
    label_calls = [c for c in reg.calls if c and c[0] == "labels"]
    child_set_calls = [c for c in reg.calls if c and c[0] == "child_set"]
    assert label_calls, "Expected labels() to be invoked"
    assert child_set_calls, "Expected child.set() to be invoked"