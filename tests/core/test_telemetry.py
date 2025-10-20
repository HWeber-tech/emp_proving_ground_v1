import pytest

from src.core import telemetry


class _StubSink:
    def set_gauge(self, name: str, value: float, labels=None) -> None:
        pass

    def inc_counter(self, name: str, amount: float = 1.0, labels=None) -> None:
        pass

    def observe_histogram(self, name: str, value: float, buckets=None, labels=None) -> None:
        pass


def test_set_metrics_sink_rejects_invalid_object(monkeypatch):
    monkeypatch.setattr(telemetry, "_SINK", None)
    telemetry.set_metrics_sink(_StubSink())
    assert telemetry.has_metrics_sink()

    monkeypatch.setattr(telemetry, "_SINK", None)
    with pytest.raises(TypeError):
        telemetry.set_metrics_sink(object())  # type: ignore[arg-type]

    assert telemetry.has_metrics_sink() is False
    assert telemetry.get_metrics_sink() is not None
