from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Mapping

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

    async def _stub_execute_timescale_ingest(*_args, **_kwargs):  # pragma: no cover - stubbed runtime hook
        return True, None

    runtime_builder_module.RuntimeApplication = _StubRuntimeApplication  # type: ignore[attr-defined]
    runtime_builder_module.RuntimeWorkload = _StubRuntimeWorkload  # type: ignore[attr-defined]
    runtime_builder_module.build_professional_runtime_application = (  # type: ignore[attr-defined]
        _stub_build_runtime_application
    )
    runtime_builder_module._execute_timescale_ingest = _stub_execute_timescale_ingest  # type: ignore[attr-defined]
    runtime_builder_module.__all__ = [  # type: ignore[attr-defined]
        "RuntimeApplication",
        "RuntimeWorkload",
        "build_professional_runtime_application",
        "_execute_timescale_ingest",
    ]
    sys.modules["src.runtime.runtime_builder"] = runtime_builder_module

from src.core.event_bus import EventBus
from src.operations.sensory_drift import SensoryDriftSnapshot
from src.operations.sensory_metrics import SensoryMetrics
from src.operations.sensory_summary import SensorySummary
from src.runtime.bootstrap_runtime import BootstrapRuntime
from src.runtime.task_supervisor import TaskSupervisor


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

        lineage_history = status.get("sensory_lineage")
        assert isinstance(lineage_history, list)
        dimensions = {
            entry.get("dimension")
            for entry in lineage_history
            if isinstance(entry, Mapping)
        }
        assert {"HOW", "ANOMALY"}.issubset(dimensions)

        latest_lineage = status.get("sensory_lineage_latest")
        assert isinstance(latest_lineage, Mapping)
        assert latest_lineage.get("dimension") in {"WHY", "WHAT", "WHEN", "HOW", "ANOMALY"}
    finally:
        await runtime.stop()


class _StubEvolutionOrchestrator:
    def __init__(self) -> None:
        self.calls = 0
        self.telemetry: dict[str, Any] = {
            "adaptive_runs": {"enabled": False, "reason": "flag_missing"},
        }
        self.champion = None
        self.population_statistics: Mapping[str, Any] = {}

    async def run_cycle(self) -> None:
        self.calls += 1
        self.telemetry["calls"] = self.calls

    def build_readiness_snapshot(self) -> Mapping[str, Any]:
        return {
            "status": "review",
            "adaptive_runs_enabled": False,
            "seed_templates": (),
            "issues": ["Seed source missing from population statistics"],
            "generated_at": "2024-01-01T00:00:00+00:00",
        }


@pytest.mark.asyncio()
async def test_bootstrap_runtime_invokes_evolution_orchestrator() -> None:
    event_bus = EventBus()
    orchestrator = _StubEvolutionOrchestrator()
    runtime = BootstrapRuntime(
        event_bus=event_bus,
        tick_interval=0.0,
        max_ticks=3,
        evolution_orchestrator=orchestrator,
        evolution_cycle_interval=1,
    )

    await runtime.start()
    try:
        await asyncio.sleep(0.05)
    finally:
        await runtime.stop()

    assert orchestrator.calls >= 1
    assert orchestrator.telemetry.get("calls") == orchestrator.calls

    status = runtime.status()
    telemetry = status.get("telemetry", {})
    evolution_overview = telemetry.get("evolution", {}) if isinstance(telemetry, Mapping) else {}
    adaptive_runs = evolution_overview.get("adaptive_runs") if isinstance(evolution_overview, Mapping) else None
    readiness = evolution_overview.get("readiness") if isinstance(evolution_overview, Mapping) else None

    assert isinstance(evolution_overview, Mapping)
    assert isinstance(adaptive_runs, Mapping)
    assert adaptive_runs.get("enabled") is False
    assert isinstance(readiness, Mapping)
    assert readiness.get("status") in {"review", "blocked", "ready"}


def _make_record(symbol: str, volatility: float, close: float = 1.0) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "timestamp": now,
        "symbol": symbol,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1_000.0,
        "volatility": volatility,
        "spread": 0.0001,
        "depth": 50.0,
        "order_imbalance": 0.0,
        "data_quality": 0.95,
        "macro_bias": 0.0,
    }


def test_adaptive_sampling_detects_chaos_and_uses_high_resolution() -> None:
    event_bus = EventBus()
    runtime = BootstrapRuntime(event_bus=event_bus, tick_interval=0.2, max_ticks=0)

    symbol = "CHAOS"
    history = runtime._sensory_history[symbol]
    baseline = [
        0.011,
        0.012,
        0.013,
        0.012,
        0.014,
        0.013,
        0.012,
        0.015,
        0.014,
        0.013,
        0.012,
        0.016,
        0.014,
        0.013,
        0.012,
    ]
    for value in baseline:
        history.append(_make_record(symbol, value, close=1.0 + value))
    history.append(_make_record(symbol, 0.065, close=1.065))

    runtime._tick_counter = 42
    should_sample, sampling_state = runtime._should_capture_sensory_snapshot(symbol, history)
    assert should_sample is True
    assert sampling_state.current_state == "chaos"
    assert sampling_state.interval == 1

    runtime._tick_counter += 1
    history.append(_make_record(symbol, 0.07, close=1.07))
    follow_up, _ = runtime._should_capture_sensory_snapshot(symbol, history)
    assert follow_up is True, "chaos state must remain high resolution"


def test_adaptive_sampling_skips_ticks_when_calm() -> None:
    event_bus = EventBus()
    runtime = BootstrapRuntime(event_bus=event_bus, tick_interval=0.5, max_ticks=0)

    symbol = "CALM"
    history = runtime._sensory_history[symbol]
    baseline = [
        0.041,
        0.039,
        0.042,
        0.04,
        0.038,
        0.043,
        0.041,
        0.039,
        0.044,
        0.04,
        0.039,
        0.042,
        0.041,
        0.039,
        0.043,
    ]
    for value in baseline:
        history.append(_make_record(symbol, value, close=1.0 + value))
    history.append(_make_record(symbol, 0.004, close=0.99))

    runtime._tick_counter = 10
    should_sample, sampling_state = runtime._should_capture_sensory_snapshot(symbol, history)
    assert should_sample is True
    assert sampling_state.current_state == "calm"
    assert sampling_state.interval > 1

    runtime._tick_counter += 1
    history.append(_make_record(symbol, 0.0035, close=0.989))
    follow_up, follow_state = runtime._should_capture_sensory_snapshot(symbol, history)
    assert follow_state.current_state == "calm"
    assert follow_state.interval > 1
    assert follow_up is False, "calm regime should permit skipped ticks"


def test_status_reports_adaptive_sampling_snapshot() -> None:
    event_bus = EventBus()
    runtime = BootstrapRuntime(event_bus=event_bus, tick_interval=0.25, max_ticks=0)

    symbol = "STATUS"
    history = runtime._sensory_history[symbol]
    for idx in range(18):
        vol = 0.01 + 0.001 * idx
        history.append(_make_record(symbol, vol, close=1.0 + vol))

    runtime._tick_counter = 8
    runtime._should_capture_sensory_snapshot(symbol, history)

    status = runtime.status()
    adaptive = status.get("adaptive_sampling")
    assert isinstance(adaptive, Mapping)
    assert symbol in adaptive
    snapshot = adaptive[symbol]
    state = runtime._adaptive_sampling_states[symbol]
    assert snapshot["state"] == state.current_state
    assert snapshot["interval"] == state.interval
    assert snapshot["last_volatility"] == pytest.approx(state.last_volatility)


@pytest.mark.asyncio()
async def test_bootstrap_runtime_rebinds_task_supervisor() -> None:
    event_bus = EventBus()
    runtime = BootstrapRuntime(event_bus=event_bus, tick_interval=0.0, max_ticks=1)

    initial_supervisor = runtime.task_supervisor
    assert isinstance(initial_supervisor, TaskSupervisor)

    external_supervisor = TaskSupervisor(namespace="bootstrap-runtime-test")

    await runtime.start(task_supervisor=external_supervisor)
    try:
        assert runtime.task_supervisor is external_supervisor
        trading_supervisor = getattr(runtime.trading_manager, "_task_supervisor", None)
        assert trading_supervisor is external_supervisor

        snapshots = runtime.describe_background_tasks()
        assert any(
            snapshot.get("name") == "bootstrap-runtime-loop" for snapshot in snapshots
        )
    finally:
        await runtime.stop()
        await external_supervisor.cancel_all()


@pytest.mark.asyncio()
async def test_bootstrap_runtime_create_background_task_tracks_metadata() -> None:
    event_bus = EventBus()
    runtime = BootstrapRuntime(event_bus=event_bus, tick_interval=0.0, max_ticks=0)

    started = asyncio.Event()
    release = asyncio.Event()

    async def _job() -> None:
        started.set()
        await release.wait()

    task = runtime.create_background_task(
        _job(),
        name="unit-test-task",
        metadata={"component": "test", "role": "helper"},
    )

    await asyncio.wait_for(started.wait(), timeout=1.0)

    metadata = runtime.get_background_task_metadata(task)
    assert metadata is not None
    assert metadata["component"] == "test"

    snapshots = runtime.describe_background_tasks()
    assert any(
        snapshot.get("name") == "unit-test-task"
        and (snapshot.get("metadata") or {}).get("component") == "test"
        for snapshot in snapshots
    )

    release.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert runtime.get_background_task_metadata(task) is None


@pytest.mark.asyncio()
async def test_bootstrap_runtime_register_background_task_tracks_metadata() -> None:
    event_bus = EventBus()
    runtime = BootstrapRuntime(event_bus=event_bus, tick_interval=0.0, max_ticks=0)

    started = asyncio.Event()
    release = asyncio.Event()

    async def _job() -> None:
        started.set()
        await release.wait()

    external_task = asyncio.create_task(_job())
    runtime.register_background_task(
        external_task,
        metadata={"component": "external", "role": "listener"},
    )

    await asyncio.wait_for(started.wait(), timeout=1.0)

    metadata = runtime.get_background_task_metadata(external_task)
    assert metadata is not None
    assert metadata["component"] == "external"
    assert runtime.task_supervisor is not None
    assert runtime.task_supervisor.is_tracked(external_task)

    release.set()
    await asyncio.wait_for(external_task, timeout=1.0)

    assert runtime.get_background_task_metadata(external_task) is None


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


@pytest.mark.asyncio()
async def test_bootstrap_runtime_supervises_run_loop_restart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    event_bus = EventBus()
    runtime = BootstrapRuntime(event_bus=event_bus, tick_interval=0.05)

    supervisor = TaskSupervisor(namespace="test-bootstrap-restart", cancel_timeout=0.1)
    loop_counter = 0
    resumed = asyncio.Event()

    async def _failing_run_loop(self) -> None:
        nonlocal loop_counter
        loop_counter += 1
        if loop_counter == 1:
            raise RuntimeError("synthetic loop failure")
        resumed.set()
        while not self._stop_event.is_set():
            await asyncio.sleep(0.01)

    runtime._run_loop = types.MethodType(  # type: ignore[attr-defined]
        _failing_run_loop,
        runtime,
    )

    with caplog.at_level(logging.ERROR, supervisor._logger.name):
        await runtime.start(task_supervisor=supervisor)
        try:
            await asyncio.wait_for(resumed.wait(), timeout=2.0)
            assert loop_counter >= 2, "runtime loop did not resume after failure"

            snapshots = supervisor.describe()
            loop_snapshot = next(
                (snapshot for snapshot in snapshots if snapshot.get("name") == "bootstrap-runtime-loop"),
                None,
            )
            assert loop_snapshot is not None
            assert loop_snapshot.get("restarts", 0) >= 1
            assert any(
                "synthetic loop failure" in (record.exc_text or "") for record in caplog.records
            )
        finally:
            await runtime.stop()
            await supervisor.cancel_all()
