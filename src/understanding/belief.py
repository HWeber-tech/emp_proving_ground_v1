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
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from src.core.event_bus import Event, EventBus, TopicBus
from src.sensory.lineage import SensorLineageRecord, build_lineage_record
from src.thinking.adaptation.policy_router import RegimeState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import used for type hints only
    from src.operations.event_bus_failover import publish_event_with_failover as _publish_event_with_failover

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
    max_variance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a low-rank Hebbian update ensuring the covariance stays PSD."""

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    if jitter <= 0.0:
        raise ValueError("jitter must be positive")
    if max_variance is not None and max_variance <= 0.0:
        raise ValueError("max_variance must be positive when provided")

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
    if max_variance is not None:
        clipped = np.minimum(clipped, max_variance)
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
        max_variance: float | None = 5.0,
        min_variance: float = 1e-9,
        volatility_features: Sequence[str] | None = None,
        volatility_window: int = 64,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if min_variance <= 0.0:
            raise ValueError("min_variance must be positive")
        if max_variance is not None and max_variance <= 0.0:
            raise ValueError("max_variance must be positive when provided")
        if volatility_window <= 1:
            raise ValueError("volatility_window must be greater than 1")
        self._belief_id = belief_id
        self._window = window
        self._learning_rate = learning_rate
        self._decay = decay
        self._version = version
        self._states: deque[BeliefState] = deque(maxlen=window)
        self._feature_order: tuple[str, ...] | None = None
        self._support: int = 0
        self._max_variance = max_variance
        self._min_variance = min_variance
        features = tuple(str(name) for name in (volatility_features or ("HOW_signal", "WHAT_signal")))
        self._volatility_features: tuple[str, ...] = features or ("HOW_signal",)
        self._volatility_history: deque[float] = deque(maxlen=volatility_window)

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
            jitter=self._min_variance,
            max_variance=self._max_variance,
        )

        integrated_strength = features.get("integrated_strength", prior_strength)
        integrated_confidence = features.get("integrated_confidence", prior_confidence)

        volatility_sample = self._compute_volatility_sample(features)
        volatility: float | None = None
        if volatility_sample is not None:
            self._volatility_history.append(float(volatility_sample))
            if len(self._volatility_history) >= 2:
                volatility = float(np.std(self._volatility_history, ddof=0))
            else:
                volatility = float(volatility_sample)

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

        lineage_outputs: MutableMapping[str, object] = {
            "posterior_strength": integrated_strength,
            "posterior_confidence": integrated_confidence,
        }
        if volatility_sample is not None:
            lineage_outputs["volatility_sample"] = float(volatility_sample)
        if volatility is not None:
            lineage_outputs["volatility"] = float(volatility)

        belief_lineage = build_lineage_record(
            "UNDERSTANDING_BELIEF",
            "understanding.belief_buffer",
            inputs={"sensory_lineage": _lineage_to_mapping(lineage)},
            outputs=lineage_outputs,
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
            "observation": {name: float(value) for name, value in features.items()},
        }
        if volatility_sample is not None:
            metadata["volatility_sample"] = float(volatility_sample)
        if volatility is not None:
            metadata["volatility"] = float(volatility)

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

    def _compute_volatility_sample(self, features: Mapping[str, float]) -> float | None:
        samples: list[float] = []
        for feature in self._volatility_features:
            value = features.get(feature)
            if value is None:
                continue
            try:
                samples.append(abs(float(value)))
            except (TypeError, ValueError):
                continue
        if not samples:
            return None
        return float(np.mean(samples))


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
        _publish_event(
            self._event_bus,
            event,
            logger=logger,
            runtime_fallback_message="Runtime bus rejected belief state; falling back to global bus",
            runtime_unexpected_message="Unexpected error publishing belief state via runtime bus",
            runtime_none_message="Runtime bus returned no result while publishing belief state",
            global_not_running_message="Global event bus not running while publishing belief state",
            global_unexpected_message="Unexpected error publishing belief state via global bus",
            global_bus_factory=self._global_bus_factory,
        )
        return state

    def healthcheck(self) -> Mapping[str, object]:
        latest = self._buffer.latest()
        return {
            "belief_id": latest.belief_id if latest else self._buffer._belief_id,
            "feature_order": list(self._buffer.feature_order or []),
            "window": len(self._buffer),
            "volatility_features": list(self._buffer._volatility_features),
            "volatility_window": self._buffer._volatility_history.maxlen,
            "volatility_samples": len(self._buffer._volatility_history),
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
        volatility_feature: str = "HOW_signal",
        volatility_window: int = 48,
        calm_threshold: float = 0.05,
        storm_threshold: float = 0.25,
        adaptive_thresholds: bool = True,
        calibration_min_samples: int = 6,
        calm_percentile: float = 0.35,
        storm_percentile: float = 0.85,
    ) -> None:
        self._event_bus = event_bus
        self._signal_id = signal_id
        self._bullish_threshold = bullish_threshold
        self._bearish_threshold = bearish_threshold
        self._confidence_floor = confidence_floor
        self._event_type = event_type
        self._global_bus_factory = global_bus_factory
        if volatility_window <= 1:
            raise ValueError("volatility_window must be greater than 1")
        if calm_threshold < 0.0:
            raise ValueError("calm_threshold must be non-negative")
        if storm_threshold <= calm_threshold:
            raise ValueError("storm_threshold must be greater than calm_threshold")
        if not 0.0 <= calm_percentile < storm_percentile <= 1.0:
            raise ValueError("percentiles must satisfy 0 <= calm < storm <= 1")
        if calibration_min_samples <= 1:
            raise ValueError("calibration_min_samples must be greater than 1")
        self._volatility_feature = str(volatility_feature)
        self._volatility_history: deque[float] = deque(maxlen=volatility_window)
        self._calm_threshold = calm_threshold
        self._storm_threshold = storm_threshold
        self._adaptive_thresholds = adaptive_thresholds
        self._calibration_min_samples = calibration_min_samples
        self._calm_percentile = calm_percentile
        self._storm_percentile = storm_percentile
        self._dynamic_calm_threshold: float | None = None
        self._dynamic_storm_threshold: float | None = None

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

        volatility, volatility_state, volatility_sample = self._resolve_volatility(
            belief_state,
            feature_map,
        )

        regime_state = RegimeState(
            regime=regime,
            confidence=float(np.clip(regime_confidence, 0.0, 1.0)),
            features=feature_map,
            timestamp=belief_state.generated_at,
            volatility=volatility,
            volatility_state=volatility_state,
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
            "volatility_feature": self._volatility_feature,
            "volatility": volatility,
            "volatility_state": volatility_state,
            "calm_threshold": self._current_thresholds()[0],
            "storm_threshold": self._current_thresholds()[1],
        }
        if volatility_sample is not None:
            metadata["volatility_sample"] = volatility_sample

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
        _publish_event(
            self._event_bus,
            event,
            logger=logger,
            runtime_fallback_message="Runtime bus rejected regime signal; falling back to global bus",
            runtime_unexpected_message="Unexpected error publishing regime signal via runtime bus",
            runtime_none_message="Runtime bus returned no result while publishing regime signal",
            global_not_running_message="Global event bus not running while publishing regime signal",
            global_unexpected_message="Unexpected error publishing regime signal via global bus",
            global_bus_factory=self._global_bus_factory,
        )
        return signal

    def healthcheck(self) -> Mapping[str, object]:
        return {
            "signal_id": self._signal_id,
            "confidence_floor": self._confidence_floor,
            "bullish_threshold": self._bullish_threshold,
            "bearish_threshold": self._bearish_threshold,
            "volatility_feature": self._volatility_feature,
            "volatility_window": self._volatility_history.maxlen,
            "volatility_samples": len(self._volatility_history),
            "calm_threshold": self._calm_threshold,
            "storm_threshold": self._storm_threshold,
            "dynamic_calm_threshold": self._dynamic_calm_threshold,
            "dynamic_storm_threshold": self._dynamic_storm_threshold,
        }

    def apply_threshold_scale(self, scale: float) -> None:
        if not np.isfinite(scale) or scale <= 0.0:
            return
        self._calm_threshold = float(self._calm_threshold * scale)
        self._storm_threshold = float(self._storm_threshold * scale)
        if self._dynamic_calm_threshold is not None:
            self._dynamic_calm_threshold = float(self._dynamic_calm_threshold * scale)
        if self._dynamic_storm_threshold is not None:
            self._dynamic_storm_threshold = float(self._dynamic_storm_threshold * scale)
        if self._storm_threshold <= self._calm_threshold:
            self._storm_threshold = self._calm_threshold + 1e-6
        if (
            self._dynamic_calm_threshold is not None
            and self._dynamic_storm_threshold is not None
            and self._dynamic_storm_threshold <= self._dynamic_calm_threshold
        ):
            self._dynamic_storm_threshold = self._dynamic_calm_threshold + 1e-6

    def _resolve_volatility(
        self,
        belief_state: BeliefState,
        feature_map: Mapping[str, float],
    ) -> tuple[float, str, float | None]:
        metadata = belief_state.metadata
        volatility_sample: float | None = None
        volatility_metric: float | None = None

        if isinstance(metadata, Mapping):
            sample_value = metadata.get("volatility_sample")
            if sample_value is not None:
                try:
                    volatility_sample = abs(float(sample_value))
                except (TypeError, ValueError):
                    volatility_sample = None
            metric_value = metadata.get("volatility")
            if metric_value is not None:
                try:
                    volatility_metric = abs(float(metric_value))
                except (TypeError, ValueError):
                    volatility_metric = None
            if volatility_sample is None:
                observation = metadata.get("observation")
                if isinstance(observation, Mapping):
                    candidate = observation.get(self._volatility_feature)
                    if candidate is not None:
                        try:
                            volatility_sample = abs(float(candidate))
                        except (TypeError, ValueError):
                            volatility_sample = None

        if volatility_sample is None:
            candidate = feature_map.get(self._volatility_feature)
            if candidate is not None:
                try:
                    volatility_sample = abs(float(candidate))
                except (TypeError, ValueError):
                    volatility_sample = None

        recorded_sample = volatility_sample
        history_value = volatility_metric if volatility_metric is not None else volatility_sample
        if history_value is not None:
            self._volatility_history.append(float(history_value))
            self._update_dynamic_thresholds()
            if len(self._volatility_history) >= 2:
                std_metric = float(np.std(self._volatility_history, ddof=0))
                volatility_metric = max(std_metric, float(history_value))
            else:
                volatility_metric = float(history_value)
        elif volatility_metric is None:
            volatility_metric = 0.0

        calm_threshold, storm_threshold = self._current_thresholds()

        if volatility_metric <= calm_threshold:
            volatility_state = "calm"
        elif volatility_metric >= storm_threshold:
            volatility_state = "storm"
        else:
            volatility_state = "normal"

        return float(volatility_metric), volatility_state, recorded_sample

    def _current_thresholds(self) -> tuple[float, float]:
        calm = self._dynamic_calm_threshold if self._dynamic_calm_threshold is not None else self._calm_threshold
        storm = self._dynamic_storm_threshold if self._dynamic_storm_threshold is not None else self._storm_threshold
        if storm <= calm:
            storm = calm + 1e-6
        return float(calm), float(storm)

    def _update_dynamic_thresholds(self) -> None:
        if not self._adaptive_thresholds:
            return
        if len(self._volatility_history) < self._calibration_min_samples:
            return
        history = np.array(self._volatility_history, dtype=float)
        if history.size == 0 or not np.any(np.isfinite(history)):
            return
        calm_candidate = float(np.quantile(history, self._calm_percentile))
        storm_candidate = float(np.quantile(history, self._storm_percentile))
        base_calm = max(self._calm_threshold, calm_candidate)
        base_storm = max(self._storm_threshold, storm_candidate)
        # Prevent the calm threshold from drifting above the storm baseline so storms remain detectable.
        base_calm = min(base_calm, self._storm_threshold * 0.75)
        # Cap the storm threshold growth to avoid suppressing legitimate spikes.
        base_storm = min(max(base_storm, self._storm_threshold), self._storm_threshold * 1.5)
        if base_storm <= base_calm:
            spread = float(np.std(history, ddof=0)) or 1e-6
            base_storm = base_calm + spread
        self._dynamic_calm_threshold = float(base_calm)
        self._dynamic_storm_threshold = float(base_storm)
def _publish_event(
    event_bus: EventBus,
    event: Event,
    *,
    logger: logging.Logger,
    runtime_fallback_message: str,
    runtime_unexpected_message: str,
    runtime_none_message: str,
    global_not_running_message: str,
    global_unexpected_message: str,
    global_bus_factory: Callable[[], TopicBus] | None,
) -> None:
    from src.operations.event_bus_failover import publish_event_with_failover

    publish_event_with_failover(
        event_bus,
        event,
        logger=logger,
        runtime_fallback_message=runtime_fallback_message,
        runtime_unexpected_message=runtime_unexpected_message,
        runtime_none_message=runtime_none_message,
        global_not_running_message=global_not_running_message,
        global_unexpected_message=global_unexpected_message,
        global_bus_factory=global_bus_factory,  # type: ignore[arg-type]
    )

