from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping

import datetime as _datetime
import enum as _enum
import json
import typing as _typing

import numpy as np
import pytest

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

if not hasattr(_typing, "Required"):
    _typing.Required = _shim_class_getitem("Required")  # type: ignore[attr-defined]

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]

from datetime import UTC

from src.core.event_bus import Event
from src.sensory.lineage import build_lineage_record
from src.understanding.belief import (
    BeliefBuffer,
    BeliefEmitter,
    RegimeFSM,
    hebbian_step,
)


@dataclass
class _StubEventBus:
    running: bool = True
    raise_runtime: bool = False

    def __post_init__(self) -> None:
        self.events: list[Event] = []

    def is_running(self) -> bool:
        return self.running

    def publish_from_sync(self, event: Event) -> int:
        if self.raise_runtime:
            raise RuntimeError("runtime bus failure")
        self.events.append(event)
        return 1


def _build_dimensions(strength: float, confidence: float) -> Mapping[str, Mapping[str, object]]:
    anomaly_strength = abs(strength) * 0.3
    anomaly_flag = anomaly_strength > 0.15
    anomaly_z_score = anomaly_strength * 4.0
    abs_strength = abs(strength)
    how_value = {
        "liquidity": float(0.55 + 0.25 * strength),
        "participation": float(0.45 + 0.2 * strength),
        "imbalance": float(0.15 * strength),
        "volatility_drag": float(0.08 * abs_strength),
        "volatility": float(0.35 * abs_strength),
    }

    return {
        "WHY": {"signal": strength * 0.6, "confidence": confidence * 0.9},
        "WHAT": {"signal": strength * 0.4, "confidence": confidence * 0.85},
        "WHEN": {"signal": strength * 0.2, "confidence": confidence * 0.8},
        "HOW": {
            "signal": strength * 0.1,
            "confidence": confidence * 0.75,
            "value": how_value,
            "metadata": {
                "telemetry": how_value,
            },
        },
        "ANOMALY": {
            "signal": anomaly_strength,
            "confidence": confidence * 0.7,
            "value": {
                "is_anomaly": anomaly_flag,
                "z_score": anomaly_z_score,
            },
            "metadata": {
                "source": "tests.synthetic.anomaly",
                "is_anomaly": anomaly_flag,
                "audit": {
                    "z_score": anomaly_z_score,
                },
            },
        },
    }


def _build_snapshot(
    *,
    strength: float,
    confidence: float,
    timestamp: datetime,
    lineage_counter: int,
) -> Mapping[str, Any]:
    dimensions = _build_dimensions(strength, confidence)
    lineage = build_lineage_record(
        "SENSORY_FUSION",
        "tests.sensory.synthetic",
        inputs={"symbol": "EURUSD", "sequence": lineage_counter},
        outputs={"strength": strength, "confidence": confidence},
    )
    return {
        "symbol": "EURUSD",
        "generated_at": timestamp,
        "integrated_signal": {
            "strength": strength,
            "confidence": confidence,
            "direction": 1.0 if strength >= 0 else -1.0,
            "contributing": sorted(dimensions.keys()),
        },
        "dimensions": dimensions,
        "lineage": lineage,
    }


def _normalise_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    def _scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            cleansed: dict[str, Any] = {}
            for key, value in obj.items():
                if key in {"generated_at", "timestamp"}:
                    cleansed[key] = "<dynamic>"
                else:
                    cleansed[key] = _scrub(value)
            return cleansed
        if isinstance(obj, list):
            return [_scrub(item) for item in obj]
        return obj

    return _scrub(payload)


@pytest.mark.guardrail
@pytest.mark.parametrize(
    "strengths, confidences",
    (
        (np.linspace(0.05, 0.15, 6), np.linspace(0.55, 0.65, 6)),
        (np.linspace(-0.2, 0.25, 8), np.linspace(0.4, 0.9, 8)),
        (np.linspace(-0.5, 0.6, 10), np.linspace(0.3, 0.95, 10)),
    ),
    ids=("calm", "normal", "storm"),
)
def test_hebbian_low_rank_sigma(strengths: np.ndarray, confidences: np.ndarray) -> None:
    buffer = BeliefBuffer(belief_id="understanding-test", learning_rate=0.25, decay=0.08)
    base_time = datetime(2025, 1, 1, tzinfo=UTC)

    for index, (strength, confidence) in enumerate(zip(strengths, confidences)):
        snapshot = _build_snapshot(
            strength=float(strength),
            confidence=float(confidence),
            timestamp=base_time + timedelta(minutes=index),
            lineage_counter=index,
        )
        state = buffer.update(snapshot)
        sigma = np.array(state.posterior.covariance)
        eigenvalues = np.linalg.eigvalsh(sigma)
        assert eigenvalues.min() >= -1e-9


def test_hebbian_step_preserves_psd_matrix() -> None:
    prior_mean = np.zeros(4)
    prior_covariance = np.diag([0.05, 0.03, 0.02, 0.01])
    observation = np.array([0.12, -0.04, 0.08, 0.03])

    _, covariance = hebbian_step(
        prior_mean,
        prior_covariance,
        observation,
        learning_rate=0.3,
        decay=0.1,
    )

    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues.min() >= -1e-9


def test_belief_emitter_publishes_belief_state(tmp_path: Any) -> None:
    bus = _StubEventBus()
    buffer = BeliefBuffer(belief_id="understanding-belief", learning_rate=0.2, decay=0.05)
    emitter = BeliefEmitter(buffer=buffer, event_bus=bus)

    snapshot = _build_snapshot(
        strength=0.42,
        confidence=0.78,
        timestamp=datetime(2025, 1, 2, 9, 30, tzinfo=UTC),
        lineage_counter=1,
    )

    state = emitter.emit(snapshot, regime_hint="balanced")

    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.type == "telemetry.understanding.belief"
    payload = _normalise_payload(event.payload)
    assert payload["belief_id"] == "understanding-belief"
    assert payload["posterior"]["confidence"] == pytest.approx(0.78)

    golden_path = tmp_path / "belief_payload.json"
    golden_path.write_text(json.dumps({"belief": payload}, indent=2, sort_keys=True))

    with open(
        "tests/understanding/golden/belief_snapshot.json", "r", encoding="utf-8"
    ) as handle:
        expected = json.load(handle)

    assert payload == expected["belief"]


def test_regime_fsm_publishes_signal() -> None:
    bus = _StubEventBus()
    buffer = BeliefBuffer(belief_id="understanding-belief")
    emitter = BeliefEmitter(buffer=buffer, event_bus=bus)
    snapshot = _build_snapshot(
        strength=0.5,
        confidence=0.82,
        timestamp=datetime(2025, 1, 2, 9, 45, tzinfo=UTC),
        lineage_counter=2,
    )
    state = emitter.emit(snapshot)

    fsm = RegimeFSM(event_bus=bus, signal_id="understanding-regime")
    signal = fsm.publish(state)

    regime_event = bus.events[-1]
    assert regime_event.type == "telemetry.understanding.regime"
    payload = _normalise_payload(regime_event.payload)

    assert payload["confidence"] >= 0.35
    assert payload["timestamp"] == "<dynamic>"
    assert payload["lineage"]["dimension"] == "UNDERSTANDING_REGIME"
    assert payload["features"]
    assert signal.regime_state.regime in {"bullish", "balanced", "bearish", "uncertain"}
    assert signal.regime_state.volatility_state in {"calm", "normal", "storm"}
    assert signal.regime_state.volatility >= 0.0
    assert payload["metadata"]["volatility_state"] == signal.regime_state.volatility_state


def test_belief_buffer_requires_lineage_metadata() -> None:
    buffer = BeliefBuffer(belief_id="understanding-belief")
    snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime.now(tz=UTC),
        "integrated_signal": {"strength": 0.1, "confidence": 0.5},
        "dimensions": _build_dimensions(0.1, 0.5),
    }

    with pytest.raises(ValueError):
        buffer.update(snapshot)


def test_belief_buffer_handles_dynamic_feature_sets() -> None:
    buffer = BeliefBuffer(belief_id="dynamic-belief", learning_rate=0.15, decay=0.05, min_variance=1e-6)
    base_time = datetime(2025, 7, 1, tzinfo=UTC)

    def _snapshot(dimensions: Mapping[str, Mapping[str, float]], offset: int) -> Mapping[str, object]:
        return {
            "symbol": "EURUSD",
            "generated_at": base_time + timedelta(minutes=offset),
            "integrated_signal": {"strength": 0.25, "confidence": 0.8},
            "dimensions": dimensions,
            "lineage": {"source": "tests.dynamic", "sequence": offset},
        }

    initial_dimensions = {
        "WHAT": {"signal": 0.2, "confidence": 0.85},
        "HOW": {"signal": 0.1, "confidence": 0.75},
    }
    buffer.update(_snapshot(initial_dimensions, 0))
    original_order = buffer.feature_order
    assert original_order is not None

    expanded_dimensions = {
        "WHAT": {"signal": 0.22, "confidence": 0.82},
        "HOW": {"signal": 0.14, "confidence": 0.7},
        "EXTRA": {"signal": 0.55, "confidence": 0.5},
    }
    expanded_state = buffer.update(_snapshot(expanded_dimensions, 1))
    extended_order = buffer.feature_order
    assert extended_order is not None
    assert len(extended_order) > len(original_order)
    assert "EXTRA_signal" in extended_order
    assert "EXTRA_confidence" in extended_order
    assert len(expanded_state.posterior.mean) == len(extended_order)
    assert expanded_state.metadata["observation"]["EXTRA_signal"] == pytest.approx(0.55)

    contracted_dimensions = {
        "WHAT": {"signal": 0.18, "confidence": 0.88},
        "HOW": {"signal": 0.09, "confidence": 0.78},
    }
    contracted_state = buffer.update(_snapshot(contracted_dimensions, 2))
    observation = contracted_state.metadata["observation"]
    assert observation["EXTRA_signal"] == pytest.approx(0.0)
    assert observation["EXTRA_confidence"] == pytest.approx(0.0)

    covariance = np.array(contracted_state.posterior.covariance)
    assert covariance.shape == (len(extended_order), len(extended_order))


def test_belief_buffer_apply_hyperparameters_resets_state_history() -> None:
    bus = _StubEventBus()
    buffer = BeliefBuffer(
        belief_id="hyper-params",
        learning_rate=0.12,
        decay=0.04,
        max_variance=0.4,
        min_variance=1e-6,
        volatility_window=16,
    )
    emitter = BeliefEmitter(buffer=buffer, event_bus=bus)

    base_time = datetime(2025, 1, 1, tzinfo=UTC)
    for index in range(3):
        snapshot = _build_snapshot(
            strength=0.1 * (index + 1),
            confidence=0.6,
            timestamp=base_time + timedelta(minutes=index),
            lineage_counter=index,
        )
        emitter.emit(snapshot)

    assert len(buffer) == 3

    buffer.apply_hyperparameters(
        learning_rate=0.28,
        decay=0.12,
        max_variance=0.15,
        min_variance=1e-5,
        volatility_features=("HOW_signal",),
        volatility_window=24,
        reset_states=True,
        reset_volatility=True,
    )

    assert len(buffer) == 0
    assert buffer.learning_rate == pytest.approx(0.28)
    assert buffer.decay == pytest.approx(0.12)
    assert buffer.max_variance == pytest.approx(0.15)
    assert buffer.min_variance == pytest.approx(1e-5)
    assert buffer.volatility_features == ("HOW_signal",)
    assert buffer.volatility_window == 24
    assert buffer.latest() is None
