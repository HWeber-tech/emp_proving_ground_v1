import datetime as _datetime
import enum as _enum
import json
import typing as _typing
from pathlib import Path

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]

if not hasattr(_enum, "StrEnum"):
    class _StrEnum(str, _enum.Enum):
        pass

    _enum.StrEnum = _StrEnum


def _shim_class_getitem(name: str) -> type:
    class _Placeholder:
        @classmethod
        def __class_getitem__(cls, item):
            return item

    _Placeholder.__name__ = name
    return _Placeholder


if not hasattr(_typing, "Unpack"):
    _typing.Unpack = _shim_class_getitem("Unpack")  # type: ignore[attr-defined]

if not hasattr(_typing, "NotRequired"):
    _typing.NotRequired = _shim_class_getitem("NotRequired")  # type: ignore[attr-defined]

from datetime import UTC, datetime

import pytest

import src.operations.sensory_drift as sensory_drift_module

from collections.abc import Callable, Iterable
from typing import Any, Mapping

from src.core.event_bus import Event
from src.operations.alerts import AlertSeverity
from src.operations.observability_diary import ThrottleStateSnapshot
from src.operations.sensory_drift import (
    DriftSeverity,
    SensoryDriftSnapshot,
    derive_drift_alerts,
    evaluate_sensory_drift,
    export_drift_throttle_metrics,
    publish_sensory_drift,
)


class _StubEventBus:
    def __init__(self, *, running: bool = True) -> None:
        self.events: list[Event] = []
        self._running = running
        self.publish_from_sync: Callable[[Event], Any] | None = self._publish  # type: ignore[assignment]

    def _publish(self, event: Event) -> int:
        self.events.append(event)
        return 1

    def is_running(self) -> bool:
        return self._running


class _StubTopicBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any, str | None]] = []

    def publish_sync(self, event_type: str, payload: Any, *, source: str | None = None) -> None:
        self.events.append((event_type, payload, source))


def test_evaluate_sensory_drift_flags_alert_and_warn() -> None:
    audit_entries = [
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "unified_score": 0.62,
            "confidence": 0.78,
            "dimensions": {
                "why": {"signal": 0.55, "confidence": 0.74},
                "how": {"signal": -0.15, "confidence": 0.68},
            },
        },
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "unified_score": 0.10,
            "confidence": 0.45,
            "dimensions": {
                "why": {"signal": 0.05, "confidence": 0.40},
                "how": {"signal": 0.12, "confidence": 0.52},
            },
        },
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "unified_score": -0.05,
            "confidence": 0.48,
            "dimensions": {
                "why": {"signal": -0.02, "confidence": 0.42},
                "how": {"signal": 0.08, "confidence": 0.46},
            },
        },
    ]

    snapshot = evaluate_sensory_drift(audit_entries, metadata={"ingest_success": True})

    assert snapshot.status is DriftSeverity.alert
    assert snapshot.metadata["ingest_success"] is True
    assert snapshot.metadata["entries"] == len(audit_entries)

    why = snapshot.dimensions["why"]
    assert why.severity is DriftSeverity.alert
    assert why.baseline_signal is not None
    assert pytest.approx(why.baseline_signal, rel=1e-6) == 0.015
    assert why.delta is not None and why.delta > 0.5

    how = snapshot.dimensions["how"]
    assert how.severity is DriftSeverity.warn
    assert how.delta is not None and pytest.approx(how.delta, rel=1e-6) == -0.25
    markdown = snapshot.to_markdown()
    assert "why" in markdown and "how" in markdown


def test_evaluate_sensory_drift_handles_single_entry() -> None:
    audit_entries = [
        {
            "symbol": "GBPUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "dimensions": {
                "why": {"signal": 0.25, "confidence": 0.6},
            },
        }
    ]

    snapshot = evaluate_sensory_drift(audit_entries)

    assert snapshot.status is DriftSeverity.normal
    assert snapshot.sample_window == 1
    why = snapshot.dimensions["why"]
    assert why.baseline_signal is None
    assert why.delta is None
    assert why.severity is DriftSeverity.normal


def _entry(signal: float, confidence: float = 0.6) -> dict[str, object]:
    return {
        "symbol": "EURUSD",
        "generated_at": datetime.utcnow().isoformat(),
        "dimensions": {
            "why": {"signal": signal, "confidence": confidence},
        },
    }


def _psi_entry(signal_level: float, confidence_level: float) -> dict[str, object]:
    dimension_offsets = {
        "WHY": 0.00,
        "WHAT": 0.03,
        "WHEN": -0.02,
        "HOW": 0.05,
        "ANOMALY": -0.04,
    }

    dimensions: dict[str, dict[str, float]] = {}
    for name, offset in dimension_offsets.items():
        dimensions[name] = {
            "signal": signal_level + offset,
            "confidence": confidence_level - (offset / 2.0),
        }

    return {
        "symbol": "EURUSD",
        "generated_at": datetime.utcnow().isoformat(),
        "unified_score": signal_level,
        "confidence": confidence_level,
        "dimensions": dimensions,
    }


def _load_throttle_fixture() -> tuple[tuple[ThrottleStateSnapshot, ...], str | None, str | None]:
    fixture_path = (
        Path(__file__).resolve().parent.parent
        / "understanding"
        / "fixtures"
        / "throttle_prometheus_replay.json"
    )
    payload = json.loads(fixture_path.read_text())
    states: list[ThrottleStateSnapshot] = []
    for entry in payload.get("throttle_states", []):
        states.append(
            ThrottleStateSnapshot(
                name=str(entry.get("name", "unknown")),
                state=str(entry.get("state", "observing")),
                active=bool(entry.get("active", False)),
                multiplier=float(entry.get("multiplier", 0.0))
                if entry.get("multiplier") is not None
                else None,
                reason=entry.get("reason"),
                metadata={},
            )
        )
    return tuple(states), payload.get("regime"), payload.get("decision")


def test_page_hinkley_drift_escalates_without_delta_trigger() -> None:
    baseline = [0.02, 0.03, 0.015, 0.01, 0.025, 0.018]
    audit_entries = [_entry(1.2)] + [_entry(value) for value in baseline]

    snapshot = evaluate_sensory_drift(
        audit_entries,
        lookback=6,
        warn_threshold=1.5,
        alert_threshold=2.0,
        page_hinkley_delta=0.0,
        page_hinkley_warn=0.4,
        page_hinkley_alert=0.8,
        min_variance_samples=2,
    )

    why = snapshot.dimensions["why"]
    assert why.page_hinkley_stat is not None and why.page_hinkley_stat >= 0.8
    assert why.severity is DriftSeverity.alert
    assert "page_hinkley_alert" in why.detectors


def test_page_hinkley_replay_fixture_triggers_alert() -> None:
    fixture_path = Path(__file__).with_name("fixtures") / "page_hinkley_replay.json"
    with fixture_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    audit_entries = payload["audit_entries"]
    replay_metadata = dict(payload.get("metadata", {}))

    snapshot = evaluate_sensory_drift(
        audit_entries,
        lookback=11,
        warn_threshold=1.0,
        alert_threshold=1.5,
        page_hinkley_delta=0.0,
        page_hinkley_warn=0.6,
        page_hinkley_alert=0.9,
        min_variance_samples=50,
        metadata={"replay_id": replay_metadata.get("replay_id", "unknown")},
    )

    assert snapshot.status is DriftSeverity.alert
    assert snapshot.sample_window == len(audit_entries)

    why = snapshot.dimensions["why"]
    assert why.severity is DriftSeverity.alert
    assert why.detectors == ("page_hinkley_alert",)
    assert why.page_hinkley_stat is not None and why.page_hinkley_stat >= 0.9

    metadata = snapshot.metadata
    assert metadata.get("replay_id") == replay_metadata.get("replay_id")
    detector_catalog = metadata.get("detectors", {})
    why_metadata = detector_catalog.get("why")
    assert why_metadata is not None
    assert why_metadata.get("severity") == "alert"
    assert why_metadata.get("detectors") == ["page_hinkley_alert"]
    if why.page_hinkley_stat is not None:
        assert why_metadata.get("page_hinkley_stat") == pytest.approx(
            why.page_hinkley_stat, rel=1e-6
        )
    severity_counts = metadata.get("severity_counts")
    assert severity_counts == {"alert": 1}


def test_population_stability_index_triggers_alert() -> None:
    baseline_entries = [_psi_entry(0.12, 0.58) for _ in range(36)]
    evaluation_entries = [_psi_entry(0.40, 0.66)] + [_psi_entry(0.88, 0.92) for _ in range(11)]
    audit_entries = evaluation_entries + baseline_entries

    snapshot = evaluate_sensory_drift(
        audit_entries,
        warn_threshold=10.0,
        alert_threshold=10.0,
        page_hinkley_delta=10.0,
        page_hinkley_warn=1e6,
        page_hinkley_alert=1e6,
        variance_warn_ratio=1e6,
        variance_alert_ratio=1e6,
        metadata={"scenario": "psi_shift"},
    )

    assert snapshot.status is DriftSeverity.alert
    psi_dimension = snapshot.dimensions["psi:WHY.signal"]
    assert psi_dimension.severity is DriftSeverity.alert
    assert psi_dimension.current_signal >= 0.25

    psi_metadata = snapshot.metadata.get("psi")
    assert isinstance(psi_metadata, Mapping)
    assert psi_metadata.get("feature_count", 0) >= 8
    assert psi_metadata.get("max_psi", 0.0) >= 0.25
    alerts = psi_metadata.get("alerts", [])
    assert alerts
    alert_names = {entry["name"] for entry in alerts}
    assert "WHY.signal" in alert_names
def test_variance_ratio_flags_alert() -> None:
    baseline = [0.015, 0.018, 0.020, 0.017, 0.019, 0.016]
    evaluation = [0.5, 0.6, 0.8]
    history = baseline + evaluation[:-1]
    audit_entries = [_entry(evaluation[-1])] + [_entry(value) for value in reversed(history)]

    snapshot = evaluate_sensory_drift(
        audit_entries,
        lookback=10,
        warn_threshold=1.5,
        alert_threshold=2.0,
        variance_window=3,
        variance_warn_ratio=1.2,
        variance_alert_ratio=1.5,
        min_variance_samples=3,
    )

    why = snapshot.dimensions["why"]
    assert why.variance_ratio is not None and why.variance_ratio >= 1.5
    assert "variance_alert" in why.detectors
    assert why.severity is DriftSeverity.alert


def test_derive_drift_alerts_emits_dimension_events() -> None:
    entries = [_entry(0.5)] + [_entry(0.1 + i * 0.01) for i in range(10)]
    snapshot = evaluate_sensory_drift(
        entries,
        lookback=8,
        warn_threshold=0.05,
        alert_threshold=0.15,
        page_hinkley_delta=0.0,
        page_hinkley_warn=0.2,
        page_hinkley_alert=0.4,
        variance_window=4,
        variance_warn_ratio=1.1,
        variance_alert_ratio=1.4,
        min_variance_samples=3,
    )

    events = derive_drift_alerts(snapshot, threshold=DriftSeverity.warn)
    categories = {event.category: event for event in events}
    assert "sensory.drift" in categories
    assert any(category.startswith("sensory.drift.") for category in categories if category != "sensory.drift")
    dimension_event = next(
        event for name, event in categories.items() if name.startswith("sensory.drift.")
    )
    assert dimension_event.severity is not AlertSeverity.info
    assert "snapshot" in dimension_event.context


def _snapshot() -> SensoryDriftSnapshot:
    entries = [
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "dimensions": {"why": {"signal": 0.6, "confidence": 0.8}},
        },
        {
            "symbol": "EURUSD",
            "generated_at": datetime.utcnow().isoformat(),
            "dimensions": {"why": {"signal": 0.5, "confidence": 0.75}},
        },
    ]
    return evaluate_sensory_drift(entries)


def test_publish_sensory_drift_prefers_runtime_bus() -> None:
    bus = _StubEventBus()
    topic_bus = _StubTopicBus()

    snapshot = _snapshot()

    publish_sensory_drift(bus, snapshot, global_bus_factory=lambda: topic_bus)

    assert len(bus.events) == 1
    assert topic_bus.events == []
    event = bus.events[0]
    assert event.type == "telemetry.sensory.drift"
    assert event.source == "operations.sensory_drift"
    assert event.payload["status"] == snapshot.status.value


def test_publish_sensory_drift_falls_back_to_global_bus_on_none_result() -> None:
    bus = _StubEventBus()
    topic_bus = _StubTopicBus()

    def _none(event: Event) -> None:
        bus.events.append(event)
        return None

    bus.publish_from_sync = _none  # type: ignore[method-assign]

    snapshot = _snapshot()

    publish_sensory_drift(bus, snapshot, global_bus_factory=lambda: topic_bus)

    assert topic_bus.events
    event_type, payload, source = topic_bus.events[-1]
    assert event_type == "telemetry.sensory.drift"
    assert source == "operations.sensory_drift"
    assert payload["status"] == snapshot.status.value


def test_export_drift_throttle_metrics_emits_when_severity_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = Path(__file__).with_name("fixtures") / "page_hinkley_replay.json"
    payload = json.loads(fixture_path.read_text())
    snapshot = evaluate_sensory_drift(
        payload["audit_entries"],
        lookback=11,
        page_hinkley_delta=0.0,
        page_hinkley_warn=0.6,
        page_hinkley_alert=0.9,
        min_variance_samples=50,
    )

    throttle_states, regime, decision = _load_throttle_fixture()

    calls: list[tuple[tuple[ThrottleStateSnapshot, ...], str | None, str | None]] = []

    def _capture(
        throttle_payload: Iterable[ThrottleStateSnapshot],
        *,
        regime: str | None = None,
        decision_id: str | None = None,
    ) -> None:
        calls.append((tuple(throttle_payload), regime, decision_id))

    monkeypatch.setattr(
        sensory_drift_module,
        "export_throttle_metrics",
        _capture,
    )

    exported = export_drift_throttle_metrics(
        snapshot,
        throttle_states,
        regime=regime,
        decision_id=decision,
    )

    assert exported is True
    assert len(calls) == 1
    exported_states, exported_regime, exported_decision = calls[0]
    assert exported_regime == regime
    assert exported_decision == decision
    assert [state.name for state in exported_states] == [state.name for state in throttle_states]


def test_export_drift_throttle_metrics_respects_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    throttle_states, _, _ = _load_throttle_fixture()
    calls: list[object] = []

    monkeypatch.setattr(
        sensory_drift_module,
        "export_throttle_metrics",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    baseline_entries = [_entry(0.11), _entry(0.1), _entry(0.09), _entry(0.12)]
    snapshot = evaluate_sensory_drift(
        baseline_entries,
        warn_threshold=0.5,
        alert_threshold=0.75,
    )

    exported = export_drift_throttle_metrics(
        snapshot,
        throttle_states,
        threshold=DriftSeverity.alert,
    )

    assert exported is False
    assert calls == []


def test_export_drift_throttle_metrics_requires_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_path = Path(__file__).with_name("fixtures") / "page_hinkley_replay.json"
    payload = json.loads(fixture_path.read_text())
    snapshot = evaluate_sensory_drift(
        payload["audit_entries"],
        lookback=11,
        page_hinkley_delta=0.0,
        page_hinkley_warn=0.6,
        page_hinkley_alert=0.9,
        min_variance_samples=50,
    )

    monkeypatch.setattr(
        sensory_drift_module,
        "export_throttle_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not call")),
    )

    exported = export_drift_throttle_metrics(snapshot, ())

    assert exported is False
