"""Belief and regime scaffolding for the understanding loop.

This module delivers the roadmap task that asks for BeliefState buffers,
`RegimeSignal` publication, and health-checked BeliefEmitter/RegimeFSM
contracts.  It converts sensory organ snapshots into structured belief updates,
applies a Hebbian-style low-rank covariance refresh with deterministic decay,
and publishes both belief and regime signals via the hardened event-bus
failover helper.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from src.core.event_bus import Event, EventBus, TopicBus
from src.operations.event_bus_failover import publish_event_with_failover
from src.sensory.lineage import SensorLineageRecord, build_lineage_record
from src.thinking.adaptation.policy_router import RegimeState

logger = logging.getLogger(__name__)

__all__ = [
    "BeliefDistribution",
    "BeliefState",
    "BeliefBuffer",
    "BeliefEmitter",
    "RegimeSignal",
    "RegimeFSM",
    "hebbian_step",
]


@dataclass(slots=True, frozen=True)
class BeliefDistribution:
    """Posterior/prior distribution tracked by the belief buffer.

    | Field | Description |
    | --- | --- |
    | ``mean`` | Ordered feature means after the most recent update. |
    | ``covariance`` | Symmetric positive semi-definite covariance matrix. |
    | ``strength`` | Aggregate strength derived from integrated sensory signal. |
    | ``confidence`` | Aggregate confidence derived from integrated sensory signal. |
    | ``support`` | Number of observations folded into the distribution. |
    | ``decay`` | Exponential decay factor applied during the latest update. |
    """

    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    strength: float
    confidence: float
    support: int
    decay: float

    def as_dict(self) -> Mapping[str, object]:
        return {
            "mean": list(self.mean),
            "covariance": [list(row) for row in self.covariance],
            "strength": self.strength,
            "confidence": self.confidence,
            "support": self.support,
            "decay": self.decay,
        }


@dataclass(slots=True, frozen=True)
class BeliefState:
    """Canonical belief snapshot published by the understanding loop.

    | Field | Description |
    | --- | --- |
    | ``belief_id`` | Stable identifier for the emitting belief buffer. |
    | ``version`` | Schema version string for downstream consumers. |
    | ``symbol`` | Market symbol associated with the sensory snapshot. |
    | ``generated_at`` | Timestamp (UTC) when the state was produced. |
    | ``features`` | Ordered feature names aligned with ``mean``/``covariance``. |
    | ``prior`` | Distribution prior to applying the latest Hebbian step. |
    | ``posterior`` | Distribution after applying the latest Hebbian step. |
    | ``lineage`` | Sanitised lineage metadata describing the transformation. |
    | ``metadata`` | Free-form annotations such as regime hints and source notes. |
    """

    belief_id: str
    version: str
    symbol: str | None
    generated_at: datetime
    features: tuple[str, ...]
    prior: BeliefDistribution
    posterior: BeliefDistribution
    lineage: Mapping[str, object]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "belief_id": self.belief_id,
            "version": self.version,
            "symbol": self.symbol,
            "generated_at": self.generated_at.isoformat(),
            "features": list(self.features),
            "prior": dict(self.prior.as_dict()),
            "posterior": dict(self.posterior.as_dict()),
            "lineage": dict(self.lineage),
            "metadata": dict(self.metadata),
        }


def hebbian_step(
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    observation: np.ndarray,
    *,
    learning_rate: float,
    decay: float,
    jitter: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a low-rank Hebbian update ensuring the covariance stays PSD."""

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")

    if observation.ndim != 1:
        raise ValueError("observation must be a 1D vector")
    if prior_mean.shape != observation.shape:
        raise ValueError("prior_mean and observation must share the same shape")
    if prior_covariance.shape != (observation.size, observation.size):
        raise ValueError("prior_covariance must be square and match observation size")

    mean = (1.0 - decay) * prior_mean + learning_rate * observation
    centred = observation - mean
    outer = np.outer(centred, centred)
    updated = (1.0 - decay) * prior_covariance + learning_rate * outer

    # Symmetrise the covariance to avoid numerical drift.
    updated = (updated + updated.T) / 2.0

    # Clamp eigenvalues to keep the matrix positive semi-definite.
    eigenvalues, eigenvectors = np.linalg.eigh(updated)
    clipped = np.maximum(eigenvalues, jitter)
    covariance = (eigenvectors * clipped) @ eigenvectors.T

    return mean, covariance


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_anomaly_flag(
    value_payload: Mapping[str, object] | None,
    metadata_payload: Mapping[str, object] | None,
) -> bool | None:
    def _coerce_bool(candidate: object | None) -> bool | None:
        if isinstance(candidate, bool):
            return candidate
        if isinstance(candidate, (int, float)):
            return bool(candidate)
        return None

    if value_payload is not None:
        flag = _coerce_bool(value_payload.get("is_anomaly"))
        if flag is not None:
            return flag
    if metadata_payload is not None:
        flag = _coerce_bool(metadata_payload.get("is_anomaly"))
        if flag is not None:
            return flag
        audit = metadata_payload.get("audit")
        if isinstance(audit, Mapping):
            flag = _coerce_bool(audit.get("is_anomaly"))
            if flag is not None:
                return flag
    return None


def _extract_z_score(
    value_payload: Mapping[str, object] | None,
    metadata_payload: Mapping[str, object] | None,
) -> float | None:
    def _from_mapping(mapping: Mapping[str, object] | None) -> float | None:
        if not mapping:
            return None
        candidate = mapping.get("z_score")
        if candidate is None:
            audit = mapping.get("audit")
            if isinstance(audit, Mapping):
                candidate = audit.get("z_score")
        if candidate is None:
            return None
        try:
            return float(candidate)
        except (TypeError, ValueError):
            return None

    value_score = _from_mapping(value_payload)
    if value_score is not None:
        return value_score
    meta_score = _from_mapping(metadata_payload)
    if meta_score is not None:
        return meta_score
    return None


def _lineage_to_mapping(lineage: SensorLineageRecord | Mapping[str, object] | None) -> Mapping[str, object]:
    if isinstance(lineage, SensorLineageRecord):
        return lineage.as_dict()
    if isinstance(lineage, Mapping):
        return {str(key): value for key, value in lineage.items()}
    return {}


class BeliefBuffer:
    """Maintain a rolling buffer of belief states with deterministic ordering."""

    def __init__(
        self,
        *,
        belief_id: str,
        window: int = 256,
        learning_rate: float = 0.2,
        decay: float = 0.05,
        version: str = "1.0",
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._belief_id = belief_id
        self._window = window
        self._learning_rate = learning_rate
        self._decay = decay
        self._version = version
        self._states: deque[BeliefState] = deque(maxlen=window)
        self._feature_order: tuple[str, ...] | None = None
        self._support: int = 0

    def __len__(self) -> int:
        return len(self._states)

    @property
    def feature_order(self) -> tuple[str, ...] | None:
        return self._feature_order

    def latest(self) -> BeliefState | None:
        return self._states[0] if self._states else None

    def update(
        self,
        sensory_snapshot: Mapping[str, object],
        *,
        regime_hint: str | None = None,
    ) -> BeliefState:
        lineage = sensory_snapshot.get("lineage")
        if lineage is None:
            raise ValueError("sensory snapshot must include lineage metadata")

        features = self._extract_features(sensory_snapshot)
        ordered_features = self._ensure_feature_order(features)

        observation = np.array([features[name] for name in ordered_features], dtype=float)
        symbol = sensory_snapshot.get("symbol") if isinstance(sensory_snapshot, Mapping) else None
        generated_at = sensory_snapshot.get("generated_at") if isinstance(sensory_snapshot, Mapping) else None
        if not isinstance(generated_at, datetime):
            generated_at = datetime.now(tz=UTC)

        prior_state = self.latest()
        if prior_state is None:
            prior_mean = np.zeros_like(observation)
            prior_covariance = np.eye(observation.size) * 1e-6
            prior_strength = features.get("integrated_strength", 0.0)
            prior_confidence = features.get("integrated_confidence", 0.0)
        else:
            prior_mean = np.array(prior_state.posterior.mean)
            prior_covariance = np.array(prior_state.posterior.covariance)
            prior_strength = prior_state.posterior.strength
            prior_confidence = prior_state.posterior.confidence

        posterior_mean, posterior_covariance = hebbian_step(
            prior_mean,
            prior_covariance,
            observation,
            learning_rate=self._learning_rate,
            decay=self._decay,
        )

        integrated_strength = features.get("integrated_strength", prior_strength)
        integrated_confidence = features.get("integrated_confidence", prior_confidence)

        self._support += 1

        prior = BeliefDistribution(
            mean=tuple(prior_mean.tolist()),
            covariance=tuple(tuple(float(value) for value in row) for row in prior_covariance.tolist()),
            strength=prior_strength,
            confidence=prior_confidence,
            support=max(self._support - 1, 0),
            decay=self._decay,
        )
        posterior = BeliefDistribution(
            mean=tuple(posterior_mean.tolist()),
            covariance=tuple(tuple(float(value) for value in row) for row in posterior_covariance.tolist()),
            strength=integrated_strength,
            confidence=integrated_confidence,
            support=self._support,
            decay=self._decay,
        )

        belief_lineage = build_lineage_record(
            "UNDERSTANDING_BELIEF",
            "understanding.belief_buffer",
            inputs={"sensory_lineage": _lineage_to_mapping(lineage)},
            outputs={
                "posterior_strength": integrated_strength,
                "posterior_confidence": integrated_confidence,
            },
            metadata={
                "symbol": symbol,
                "feature_order": ordered_features,
                "regime_hint": regime_hint,
            },
        )

        metadata: MutableMapping[str, object] = {
            "learning_rate": self._learning_rate,
            "decay": self._decay,
            "regime_hint": regime_hint,
        }

        state = BeliefState(
            belief_id=self._belief_id,
            version=self._version,
            symbol=symbol if isinstance(symbol, str) else None,
            generated_at=generated_at,
            features=ordered_features,
            prior=prior,
            posterior=posterior,
            lineage=belief_lineage.as_dict(),
            metadata=metadata,
        )

        self._states.appendleft(state)
        return state

    def _ensure_feature_order(self, features: Mapping[str, float]) -> tuple[str, ...]:
        if self._feature_order is None:
            self._feature_order = tuple(sorted(features))
        missing = set(self._feature_order) - set(features)
        if missing:
            raise ValueError(f"missing features in sensory snapshot: {sorted(missing)!r}")
        return self._feature_order

    def _extract_features(self, snapshot: Mapping[str, object]) -> Mapping[str, float]:
        payload: MutableMapping[str, float] = {}

        integrated = snapshot.get("integrated_signal") if isinstance(snapshot, Mapping) else None
        if integrated is not None:
            strength = getattr(integrated, "strength", None)
            confidence = getattr(integrated, "confidence", None)
            if isinstance(integrated, Mapping):
                strength = integrated.get("strength", strength)
                confidence = integrated.get("confidence", confidence)
            payload["integrated_strength"] = _coerce_float(strength)
            payload["integrated_confidence"] = _coerce_float(confidence)

        dimensions = snapshot.get("dimensions") if isinstance(snapshot, Mapping) else None
        if isinstance(dimensions, Mapping):
            for name, dimension in dimensions.items():
                key_base = str(name).upper()
                value_payload: Mapping[str, object] | None = None
                metadata_payload: Mapping[str, object] | None = None
                if isinstance(dimension, Mapping):
                    payload[f"{key_base}_signal"] = _coerce_float(dimension.get("signal"))
                    payload[f"{key_base}_confidence"] = _coerce_float(dimension.get("confidence"))
                    raw_value = dimension.get("value")
                    if isinstance(raw_value, Mapping):
                        value_payload = {str(k): v for k, v in raw_value.items()}
                    raw_metadata = dimension.get("metadata")
                    if isinstance(raw_metadata, Mapping):
                        metadata_payload = {str(k): v for k, v in raw_metadata.items()}
                else:
                    signal = getattr(dimension, "signal", None)
                    confidence = getattr(dimension, "confidence", None)
                    payload[f"{key_base}_signal"] = _coerce_float(signal)
                    payload[f"{key_base}_confidence"] = _coerce_float(confidence)
                    raw_value = getattr(dimension, "value", None)
                    if isinstance(raw_value, Mapping):
                        value_payload = {str(k): v for k, v in raw_value.items()}
                    raw_metadata = getattr(dimension, "metadata", None)
                    if isinstance(raw_metadata, Mapping):
                        metadata_payload = {str(k): v for k, v in raw_metadata.items()}

                if key_base == "ANOMALY":
                    anomaly_flag = _extract_anomaly_flag(value_payload, metadata_payload)
                    if anomaly_flag is not None:
                        payload["ANOMALY_flag"] = 1.0 if anomaly_flag else 0.0

                    z_score = _extract_z_score(value_payload, metadata_payload)
                    if z_score is not None:
                        payload["ANOMALY_z_score"] = float(z_score)

        if not payload:
            raise ValueError("sensory snapshot missing dimension features")
        return dict(payload)


@dataclass(slots=True, frozen=True)
class RegimeSignal:
    """Event-bus friendly regime classification bundle."""

    signal_id: str
    regime_state: RegimeState
    features: Mapping[str, float]
    lineage: Mapping[str, object]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "signal_id": self.signal_id,
            "regime": self.regime_state.regime,
            "confidence": self.regime_state.confidence,
            "timestamp": self.regime_state.timestamp.isoformat(),
            "features": dict(self.features),
            "lineage": dict(self.lineage),
            "metadata": dict(self.metadata),
        }


class BeliefEmitter:
    """Translate sensory snapshots into belief states and publish them."""

    def __init__(
        self,
        *,
        buffer: BeliefBuffer,
        event_bus: EventBus,
        event_type: str = "telemetry.understanding.belief",
        global_bus_factory: Callable[[], TopicBus] | None = None,
    ) -> None:
        self._buffer = buffer
        self._event_bus = event_bus
        self._event_type = event_type
        self._global_bus_factory = global_bus_factory

    def emit(
        self,
        sensory_snapshot: Mapping[str, object],
        *,
        regime_hint: str | None = None,
    ) -> BeliefState:
        state = self._buffer.update(sensory_snapshot, regime_hint=regime_hint)
        event = Event(
            type=self._event_type,
            payload=state.as_dict(),
            source="understanding.belief_emitter",
        )
        publish_event_with_failover(
            self._event_bus,
            event,
            logger=logger,
            runtime_fallback_message="Runtime bus rejected belief state; falling back to global bus",
            runtime_unexpected_message="Unexpected error publishing belief state via runtime bus",
            runtime_none_message="Runtime bus returned no result while publishing belief state",
            global_not_running_message="Global event bus not running while publishing belief state",
            global_unexpected_message="Unexpected error publishing belief state via global bus",
            global_bus_factory=self._global_bus_factory,  # type: ignore[arg-type]
        )
        return state

    def healthcheck(self) -> Mapping[str, object]:
        latest = self._buffer.latest()
        return {
            "belief_id": latest.belief_id if latest else self._buffer._belief_id,
            "feature_order": list(self._buffer.feature_order or []),
            "window": len(self._buffer),
        }


class RegimeFSM:
    """Classify belief states into regime signals and publish them."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        signal_id: str,
        bullish_threshold: float = 0.25,
        bearish_threshold: float = -0.25,
        confidence_floor: float = 0.35,
        event_type: str = "telemetry.understanding.regime",
        global_bus_factory: Callable[[], TopicBus] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._signal_id = signal_id
        self._bullish_threshold = bullish_threshold
        self._bearish_threshold = bearish_threshold
        self._confidence_floor = confidence_floor
        self._event_type = event_type
        self._global_bus_factory = global_bus_factory

    def classify(self, belief_state: BeliefState) -> RegimeSignal:
        posterior = belief_state.posterior
        strength = posterior.strength
        confidence = max(posterior.confidence, 0.0)

        if confidence < self._confidence_floor:
            regime = "uncertain"
            regime_confidence = self._confidence_floor
        elif strength >= self._bullish_threshold:
            regime = "bullish"
            regime_confidence = min(1.0, confidence + 0.1)
        elif strength <= self._bearish_threshold:
            regime = "bearish"
            regime_confidence = min(1.0, confidence + 0.1)
        else:
            regime = "balanced"
            regime_confidence = confidence

        feature_map: MutableMapping[str, float] = {}
        for name, value in zip(belief_state.features, belief_state.posterior.mean):
            feature_map[name] = float(value)

        regime_state = RegimeState(
            regime=regime,
            confidence=float(np.clip(regime_confidence, 0.0, 1.0)),
            features=feature_map,
            timestamp=belief_state.generated_at,
        )

        lineage = build_lineage_record(
            "UNDERSTANDING_REGIME",
            "understanding.regime_fsm",
            inputs={"belief": belief_state.as_dict()},
            outputs={"regime": regime_state.regime, "confidence": regime_state.confidence},
        )

        metadata = {
            "bullish_threshold": self._bullish_threshold,
            "bearish_threshold": self._bearish_threshold,
        }

        return RegimeSignal(
            signal_id=self._signal_id,
            regime_state=regime_state,
            features=feature_map,
            lineage=lineage.as_dict(),
            metadata=metadata,
        )

    def publish(self, belief_state: BeliefState) -> RegimeSignal:
        signal = self.classify(belief_state)
        event = Event(
            type=self._event_type,
            payload=signal.as_dict(),
            source="understanding.regime_fsm",
        )
        publish_event_with_failover(
            self._event_bus,
            event,
            logger=logger,
            runtime_fallback_message="Runtime bus rejected regime signal; falling back to global bus",
            runtime_unexpected_message="Unexpected error publishing regime signal via runtime bus",
            runtime_none_message="Runtime bus returned no result while publishing regime signal",
            global_not_running_message="Global event bus not running while publishing regime signal",
            global_unexpected_message="Unexpected error publishing regime signal via global bus",
            global_bus_factory=self._global_bus_factory,  # type: ignore[arg-type]
        )
        return signal

    def healthcheck(self) -> Mapping[str, object]:
        return {
            "signal_id": self._signal_id,
            "confidence_floor": self._confidence_floor,
            "bullish_threshold": self._bullish_threshold,
            "bearish_threshold": self._bearish_threshold,
        }
