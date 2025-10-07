from __future__ import annotations

import asyncio
from typing import Mapping

import sys
import types
import typing as _typing

import pytest

if not hasattr(_typing, "Unpack"):
    class _Placeholder:
        @classmethod
        def __class_getitem__(cls, item):  # pragma: no cover - compatibility shim
            return item

    _Placeholder.__name__ = "Unpack"
    _typing.Unpack = _Placeholder  # type: ignore[attr-defined]

if not hasattr(_typing, "Self"):
    _typing.Self = _typing.TypeVar("Self")  # type: ignore[attr-defined]

if "scipy" not in sys.modules:
    scipy_module = types.ModuleType("scipy")
    signal_module = types.ModuleType("scipy.signal")
    stats_module = types.ModuleType("scipy.stats")

    def _fake_find_peaks(*_args, **_kwargs):  # pragma: no cover - compatibility shim
        return [], {}

    def _fake_zscore(values, *_, **__):  # pragma: no cover - compatibility shim
        length = len(values) if hasattr(values, "__len__") else 0
        return [0.0] * length

    signal_module.find_peaks = _fake_find_peaks  # type: ignore[attr-defined]
    stats_module.zscore = _fake_zscore  # type: ignore[attr-defined]
    scipy_module.signal = signal_module  # type: ignore[attr-defined]
    scipy_module.stats = stats_module  # type: ignore[attr-defined]
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.signal"] = signal_module
    sys.modules["scipy.stats"] = stats_module

if "simplefix" not in sys.modules:
    simplefix_module = types.ModuleType("simplefix")

    class _FakeFixMessage:  # pragma: no cover - compatibility shim
        def __init__(self) -> None:
            self.pairs: list[tuple[int, str]] = []

        def append_pair(self, tag: int, value: str) -> None:
            self.pairs.append((tag, value))

        def encode(self) -> bytes:
            return b""

    simplefix_module.FixMessage = _FakeFixMessage  # type: ignore[attr-defined]
    sys.modules["simplefix"] = simplefix_module

if "src.runtime.fix_pilot" not in sys.modules:
    fix_pilot_module = types.ModuleType("src.runtime.fix_pilot")

    class _FakeFixIntegrationPilot:  # pragma: no cover - compatibility shim
        def __init__(self, *_, **__):
            pass

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        def snapshot(self) -> None:
            return None

    fix_pilot_module.FixIntegrationPilot = _FakeFixIntegrationPilot  # type: ignore[attr-defined]
    sys.modules["src.runtime.fix_pilot"] = fix_pilot_module

if "src.runtime.runtime_builder" not in sys.modules:
    runtime_builder_module = types.ModuleType("src.runtime.runtime_builder")

    class _StubRuntimeApplication:  # pragma: no cover - compatibility shim
        def __init__(self, *_, **__):
            pass

        def summary(self) -> Mapping[str, object]:
            return {}

    class _StubRuntimeWorkload:  # pragma: no cover - compatibility shim
        def __init__(self, *_, **__):
            pass

    def _stub_build_runtime_application(*_args, **_kwargs):  # pragma: no cover
        raise NotImplementedError

    runtime_builder_module.RuntimeApplication = _StubRuntimeApplication  # type: ignore[attr-defined]
    runtime_builder_module.RuntimeWorkload = _StubRuntimeWorkload  # type: ignore[attr-defined]
    runtime_builder_module.build_professional_runtime_application = (  # type: ignore[attr-defined]
        _stub_build_runtime_application
    )
    sys.modules["src.runtime.runtime_builder"] = runtime_builder_module

from src.core.event_bus import EventBus
from src.operations.sensory_drift import SensoryDriftSnapshot
from src.operations.sensory_metrics import SensoryMetrics
from src.operations.sensory_summary import SensorySummary
from src.runtime.bootstrap_runtime import BootstrapRuntime


@pytest.mark.asyncio()
async def test_bootstrap_runtime_exposes_sensory_status() -> None:
    event_bus = EventBus()
    runtime = BootstrapRuntime(event_bus=event_bus, tick_interval=0.0, max_ticks=4)

    await runtime.start()
    try:
        await asyncio.sleep(0.1)
        status = runtime.status()

        assert status.get("ticks_processed", 0) >= 1
        assert status.get("samples", 0) >= 1

        latest = status.get("latest")
        assert isinstance(latest, Mapping)
        dimensions = latest.get("dimensions")
        assert isinstance(dimensions, Mapping)
        assert {"HOW", "ANOMALY"}.issubset(set(dimensions.keys()))

        metrics = status.get("sensory_metrics")
        assert isinstance(metrics, Mapping)
        assert metrics.get("dimensions")

        audit_entries = status.get("sensor_audit") or status.get("legacy_sensor_audit")
        assert audit_entries
    finally:
        await runtime.stop()


@pytest.mark.asyncio()
async def test_bootstrap_runtime_publishes_sensory_telemetry(monkeypatch) -> None:
    import src.runtime.bootstrap_runtime as bootstrap_module

    summary_calls: list[SensorySummary] = []
    metrics_calls: list[SensoryMetrics] = []
    drift_calls: list[SensoryDriftSnapshot] = []

    def _capture_summary(
        summary: SensorySummary,
        *,
        event_bus: EventBus,
        event_type: str = "telemetry.sensory.summary",
        global_bus_factory=None,
    ) -> None:
        summary_calls.append(summary)

    def _capture_metrics(
        metrics: SensoryMetrics,
        *,
        event_bus: EventBus,
        event_type: str = "telemetry.sensory.metrics",
        global_bus_factory=None,
    ) -> None:
        metrics_calls.append(metrics)

    def _capture_drift(
        event_bus: EventBus,
        snapshot: SensoryDriftSnapshot,
        *,
        global_bus_factory=None,
    ) -> None:
        drift_calls.append(snapshot)

    monkeypatch.setattr(bootstrap_module, "publish_sensory_summary", _capture_summary)
    monkeypatch.setattr(bootstrap_module, "publish_sensory_metrics", _capture_metrics)
    monkeypatch.setattr(bootstrap_module, "publish_sensory_drift", _capture_drift)

    event_bus = EventBus()
    runtime = BootstrapRuntime(event_bus=event_bus, tick_interval=0.0, max_ticks=4)

    await runtime.start()
    try:
        await asyncio.sleep(0.1)
    finally:
        await runtime.stop()

    assert summary_calls
    assert metrics_calls
    assert drift_calls

    assert isinstance(summary_calls[-1], SensorySummary)
    assert isinstance(metrics_calls[-1], SensoryMetrics)
    assert isinstance(drift_calls[-1], SensoryDriftSnapshot)

    status = runtime.status()
    metrics_payload = status.get("sensory_metrics")
    assert metrics_payload == metrics_calls[-1].as_dict()
