import pytest

import src.operational.metrics as metrics


class _StubHistogram:
    def __init__(self):
        self.observations = []

    def observe(self, v: float):
        self.observations.append(float(v))


class _StubRegistry:
    def __init__(self):
        self.hist_calls = 0
        self.last_hist = None

    def get_histogram(self, *args, **kwargs):
        self.hist_calls += 1
        self.last_hist = _StubHistogram()
        return self.last_hist


def test_observe_exec_latency_negative_noop(monkeypatch: pytest.MonkeyPatch):
    stub = _StubRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: stub)

    metrics.observe_exec_latency(-0.1)
    # Guard should prevent get_histogram/observe
    assert stub.hist_calls == 0
    assert stub.last_hist is None

    metrics.observe_exec_latency(0.0)
    # Non-negative should hit histogram once and record observe(0.0)
    assert stub.hist_calls == 1
    assert stub.last_hist is not None
    assert stub.last_hist.observations == [0.0]


def test_observe_cancel_latency_negative_noop(monkeypatch: pytest.MonkeyPatch):
    stub = _StubRegistry()
    monkeypatch.setattr(metrics, "get_registry", lambda: stub)

    metrics.observe_cancel_latency(-5.0)
    # Guard should prevent get_histogram/observe
    assert stub.hist_calls == 0
    assert stub.last_hist is None

    metrics.observe_cancel_latency(2.5)
    # Non-negative should hit histogram once and record observe(2.5)
    assert stub.hist_calls == 1
    assert stub.last_hist is not None
    assert stub.last_hist.observations == [2.5]