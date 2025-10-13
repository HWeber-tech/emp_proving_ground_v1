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
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import json
from pathlib import Path
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
    "BeliefSnapshotPersister",
    "RegimeSignal",
    "RegimeTransition",
    "RegimeFSM",
    "hebbian_step",
]


def _json_safe_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    if isinstance(value, tuple):
        return list(value)
    try:
        if isinstance(value, np.generic):
            return value.item()
    except Exception:  # pragma: no cover - numpy can raise for unsupported item()
        pass
    return value


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
    if value is None:
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(coerced):
        return default
    return coerced


_DIMENSION_EXTRA_VALUE_KEYS: dict[str, tuple[str, ...]] = {
    "WHAT": ("last_close",),
    "WHEN": ("session", "news", "gamma"),
    "HOW": ("liquidity", "participation", "imbalance", "volatility_drag", "volatility"),
    "WHY": ("narrative_sentiment",),
}


def _populate_dimension_features(
    accumulator: MutableMapping[str, float],
    dimension: str,
    value_payload: Mapping[str, object] | None,
) -> None:
    extras = _DIMENSION_EXTRA_VALUE_KEYS.get(dimension, ())
    for key in extras:
        raw_value = value_payload.get(key) if value_payload is not None else None
        accumulator[f"{dimension}_{key}"] = _coerce_float(raw_value)


def _populate_quality_features(
    accumulator: MutableMapping[str, float],
    dimension: str,
    metadata_payload: Mapping[str, object] | None,
) -> None:
    if not metadata_payload:
        return

    quality_payload = metadata_payload.get("quality")
    if not isinstance(quality_payload, Mapping):
        return

    confidence = quality_payload.get("confidence")
    if confidence is not None:
        accumulator[f"{dimension}_quality_confidence"] = _coerce_float(confidence)

    strength = quality_payload.get("strength")
    if strength is not None:
        accumulator[f"{dimension}_quality_strength"] = _coerce_float(strength)

    data_quality = quality_payload.get("data_quality")
    if data_quality is not None:
        accumulator[f"{dimension}_data_quality"] = _coerce_float(data_quality)


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
            result = float(candidate)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(result):
            return None
        return result

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
        self._latest_covariance_trace: float | None = None
        self._latest_covariance_condition: float | None = None
        self._latest_covariance_max: float | None = None
        self._latest_covariance_min: float | None = None

    def __len__(self) -> int:
        return len(self._states)

    @property
    def feature_order(self) -> tuple[str, ...] | None:
        return self._feature_order

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def max_variance(self) -> float | None:
        return self._max_variance

    @property
    def min_variance(self) -> float:
        return self._min_variance

    @property
    def volatility_features(self) -> tuple[str, ...]:
        return self._volatility_features

    @property
    def volatility_window(self) -> int:
        return self._volatility_history.maxlen

    def latest(self) -> BeliefState | None:
        return self._states[0] if self._states else None

    def replace_latest(self, state: BeliefState) -> None:
        """Replace the most recent belief state in the buffer."""

        if not self._states:
            raise ValueError("cannot replace latest belief state on an empty buffer")
        self._states[0] = state

    def apply_hyperparameters(
        self,
        *,
        learning_rate: float | None = None,
        decay: float | None = None,
        max_variance: float | None = None,
        min_variance: float | None = None,
        volatility_features: Sequence[str] | None = None,
        volatility_window: int | None = None,
        reset_states: bool = False,
        reset_volatility: bool = False,
    ) -> None:
        """Adjust buffer hyperparameters without reinstantiation."""

        if learning_rate is not None:
            if learning_rate <= 0.0:
                raise ValueError("learning_rate must be positive")
            self._learning_rate = float(learning_rate)
        if decay is not None:
            if not 0.0 < decay <= 1.0:
                raise ValueError("decay must be in (0, 1]")
            self._decay = float(decay)
        if min_variance is not None:
            if min_variance <= 0.0:
                raise ValueError("min_variance must be positive")
            self._min_variance = float(min_variance)
        if max_variance is not None:
            if max_variance <= 0.0:
                raise ValueError("max_variance must be positive when provided")
            self._max_variance = float(max_variance)
        if self._max_variance is not None and self._max_variance <= self._min_variance:
            raise ValueError("max_variance must exceed min_variance")

        if volatility_features is not None:
            features = tuple(str(name) for name in volatility_features if str(name))
            self._volatility_features = features or ("HOW_signal",)
        if volatility_window is not None:
            if volatility_window <= 1:
                raise ValueError("volatility_window must be greater than 1")
            self._volatility_history = deque(self._volatility_history, maxlen=volatility_window)

        if reset_volatility:
            self._volatility_history.clear()
        if reset_states:
            self._states = deque(maxlen=self._window)
            self._support = 0
            self._latest_covariance_trace = None
            self._latest_covariance_condition = None
            self._latest_covariance_max = None
            self._latest_covariance_min = None

    def update(
        self,
        sensory_snapshot: Mapping[str, object],
        *,
        regime_hint: str | None = None,
    ) -> BeliefState:
        lineage = sensory_snapshot.get("lineage")
        if lineage is None:
            raise ValueError("sensory snapshot must include lineage metadata")

        feature_values: MutableMapping[str, float] = dict(self._extract_features(sensory_snapshot))
        ordered_features = self._ensure_feature_order(feature_values)

        observation = np.array([feature_values[name] for name in ordered_features], dtype=float)
        symbol = sensory_snapshot.get("symbol") if isinstance(sensory_snapshot, Mapping) else None
        generated_at = sensory_snapshot.get("generated_at") if isinstance(sensory_snapshot, Mapping) else None
        if not isinstance(generated_at, datetime):
            generated_at = datetime.now(tz=UTC)

        prior_state = self.latest()
        if prior_state is None:
            prior_mean = np.zeros_like(observation)
            prior_covariance = np.eye(observation.size) * 1e-6
            prior_strength = feature_values.get("integrated_strength", 0.0)
            prior_confidence = feature_values.get("integrated_confidence", 0.0)
        else:
            prior_mean = np.array(prior_state.posterior.mean)
            prior_covariance = np.array(prior_state.posterior.covariance)
            prior_mean, prior_covariance = self._align_prior_dimensions(
                prior_mean,
                prior_covariance,
                observation.size,
            )
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

        posterior_eigenvalues = np.linalg.eigvalsh(posterior_covariance)
        posterior_max = float(np.max(posterior_eigenvalues)) if posterior_eigenvalues.size else 0.0
        posterior_min = float(np.min(posterior_eigenvalues)) if posterior_eigenvalues.size else 0.0
        posterior_trace = float(np.trace(posterior_covariance))
        condition_denominator = max(posterior_min, self._min_variance, 1e-12)
        posterior_condition = float(posterior_max / condition_denominator) if posterior_max > 0.0 else 1.0

        self._latest_covariance_trace = posterior_trace
        self._latest_covariance_condition = posterior_condition
        self._latest_covariance_max = posterior_max
        self._latest_covariance_min = posterior_min

        integrated_strength = feature_values.get("integrated_strength", prior_strength)
        integrated_confidence = feature_values.get("integrated_confidence", prior_confidence)

        volatility_sample = self._compute_volatility_sample(feature_values)
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
        lineage_outputs["covariance_condition"] = posterior_condition
        lineage_outputs["covariance_trace"] = posterior_trace

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
            "observation": {name: float(value) for name, value in feature_values.items()},
        }
        if volatility_sample is not None:
            metadata["volatility_sample"] = float(volatility_sample)
        if volatility is not None:
            metadata["volatility"] = float(volatility)
        metadata["covariance_trace"] = posterior_trace
        metadata["covariance_max_eigenvalue"] = posterior_max
        metadata["covariance_min_eigenvalue"] = posterior_min
        metadata["covariance_condition"] = posterior_condition

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

    def _ensure_feature_order(self, features: MutableMapping[str, float]) -> tuple[str, ...]:
        if self._feature_order is None:
            self._feature_order = tuple(sorted(features))
            return self._feature_order

        existing = set(self._feature_order)
        missing = [name for name in self._feature_order if name not in features]
        if missing:
            for name in missing:
                features[name] = 0.0
            logger.debug(
                "BeliefBuffer %s backfilled missing features: %s",
                self._belief_id,
                missing,
            )

        extras = sorted(name for name in features if name not in existing)
        if extras:
            self._feature_order = self._feature_order + tuple(extras)
            logger.debug(
                "BeliefBuffer %s extended feature order with: %s",
                self._belief_id,
                extras,
            )

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

                _populate_dimension_features(payload, key_base, value_payload)
                _populate_quality_features(payload, key_base, metadata_payload)

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
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                continue
            samples.append(abs(numeric))
        if not samples:
            return None
        return float(np.mean(samples))

    def _align_prior_dimensions(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        target_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        current_size = int(mean.shape[0])
        if current_size == target_size:
            return mean, covariance

        if current_size > target_size:
            trimmed_mean = mean[:target_size]
            trimmed_covariance = covariance[:target_size, :target_size]
            return trimmed_mean, trimmed_covariance

        expanded_mean = np.zeros(target_size, dtype=float)
        expanded_mean[:current_size] = mean

        expanded_covariance = np.eye(target_size, dtype=float) * self._min_variance
        expanded_covariance[:current_size, :current_size] = covariance
        return expanded_mean, expanded_covariance


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


@dataclass(slots=True, frozen=True)
class RegimeTransition:
    """Structured transition emitted when the regime or volatility state changes."""

    signal_id: str
    timestamp: datetime
    previous_regime: str | None
    current_regime: str
    previous_volatility_state: str | None
    current_volatility_state: str
    confidence: float
    volatility: float
    latency_ms: float | None
    reason: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    features: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "previous_regime": self.previous_regime,
            "current_regime": self.current_regime,
            "previous_volatility_state": self.previous_volatility_state,
            "current_volatility_state": self.current_volatility_state,
            "confidence": self.confidence,
            "volatility": self.volatility,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "features": dict(self.features),
        }
        return payload


@dataclass(slots=True, frozen=True)
class BeliefSnapshotRecord:
    """Row persisted for each emitted belief snapshot."""

    belief_id: str
    generated_at: datetime
    version: str
    symbol: str | None
    regime_hint: str | None
    feature_count: int
    active_count: int
    sparsity: float
    activation_method: str
    top_features: tuple[Mapping[str, object], ...]
    payload: Mapping[str, object]

    def to_row(self) -> Mapping[str, object]:
        top_features_serialised = [
            {
                "feature": str(entry.get("feature")),
                "value": float(entry.get("value", 0.0)),
                "rank": int(entry.get("rank", index + 1)),
            }
            for index, entry in enumerate(self.top_features)
        ]
        payload_serialised = json.dumps(dict(self.payload), default=_json_safe_default)
        return {
            "belief_id": self.belief_id,
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "version": self.version,
            "symbol": self.symbol,
            "regime_hint": self.regime_hint,
            "feature_count": self.feature_count,
            "active_count": self.active_count,
            "sparsity": float(self.sparsity),
            "activation_method": self.activation_method,
            "top_features": json.dumps(top_features_serialised, default=_json_safe_default),
            "payload": payload_serialised,
        }


class BeliefSnapshotPersister:
    """Persist belief snapshots to DuckDB and/or Parquet for retrospective analysis."""

    def __init__(
        self,
        *,
        duckdb_path: str | Path | None = None,
        parquet_path: str | Path | None = None,
        duckdb_table: str = "belief_snapshots",
    ) -> None:
        if duckdb_path is None and parquet_path is None:
            raise ValueError("BeliefSnapshotPersister requires a DuckDB or Parquet destination")

        self._duckdb_path = Path(duckdb_path).expanduser() if duckdb_path else None
        if self._duckdb_path is not None:
            self._duckdb_path.parent.mkdir(parents=True, exist_ok=True)

        parquet_destination: Path | None = None
        if parquet_path is not None:
            parquet_destination = Path(parquet_path).expanduser()
            if parquet_destination.suffix.lower() == ".parquet":
                parquet_destination = parquet_destination.parent / parquet_destination.stem
            parquet_destination.mkdir(parents=True, exist_ok=True)
        self._parquet_dir = parquet_destination

        self._duckdb_table = duckdb_table
        self._duckdb_warning_logged = False
        self._parquet_warning_logged = False

    def persist(self, state: BeliefState) -> None:
        record = self._build_record(state)
        if self._duckdb_path is not None:
            self._persist_duckdb(record)
        if self._parquet_dir is not None:
            self._persist_parquet(record)

    def _build_record(self, state: BeliefState) -> BeliefSnapshotRecord:
        payload = state.as_dict()
        metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
        activation_summary = metadata.get("activation_summary")

        top_features: list[Mapping[str, object]] = []
        activation_method = "unknown"
        active_count = 0
        sparsity = 1.0
        feature_count = len(state.features)
        if isinstance(activation_summary, Mapping):
            activation_method = str(activation_summary.get("method", "unknown"))
            try:
                active_count = int(activation_summary.get("active_count", 0))
            except (TypeError, ValueError):
                active_count = 0
            try:
                sparsity = float(activation_summary.get("sparsity", 1.0))
            except (TypeError, ValueError):
                sparsity = 1.0
            top_features_payload = activation_summary.get("top_features", ())
            if isinstance(top_features_payload, Mapping):
                top_features_payload = top_features_payload.values()
            if isinstance(top_features_payload, Iterable):
                for entry in top_features_payload:
                    if not isinstance(entry, Mapping):
                        continue
                    feature_name = str(entry.get("feature"))
                    try:
                        value = float(entry.get("value", 0.0))
                    except (TypeError, ValueError):
                        value = 0.0
                    try:
                        rank = int(entry.get("rank", len(top_features) + 1))
                    except (TypeError, ValueError):
                        rank = len(top_features) + 1
                    top_features.append({"feature": feature_name, "value": value, "rank": rank})

        regime_hint = metadata.get("regime_hint") if isinstance(metadata, Mapping) else None
        regime_hint_str = str(regime_hint) if regime_hint is not None else None

        return BeliefSnapshotRecord(
            belief_id=state.belief_id,
            generated_at=state.generated_at,
            version=state.version,
            symbol=state.symbol,
            regime_hint=regime_hint_str,
            feature_count=feature_count,
            active_count=active_count,
            sparsity=sparsity,
            activation_method=activation_method,
            top_features=tuple(top_features),
            payload=payload,
        )

    def _persist_duckdb(self, record: BeliefSnapshotRecord) -> None:
        try:
            import duckdb  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            if not self._duckdb_warning_logged:
                logger.warning(
                    "duckdb not available; skipping belief snapshot persistence to %s",
                    self._duckdb_path,
                )
                self._duckdb_warning_logged = True
            return
        except ImportError as exc:
            if not self._duckdb_warning_logged:
                logger.warning(
                    "duckdb import failed (%s); skipping persistence to %s",
                    exc,
                    self._duckdb_path,
                    exc_info=exc,
                )
                self._duckdb_warning_logged = True
            return

        row = record.to_row()
        columns = (
            "belief_id",
            "generated_at",
            "version",
            "symbol",
            "regime_hint",
            "feature_count",
            "active_count",
            "sparsity",
            "activation_method",
            "top_features",
            "payload",
        )
        sql = f"INSERT INTO {self._duckdb_table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {self._duckdb_table} (
                belief_id TEXT,
                generated_at TIMESTAMP,
                version TEXT,
                symbol TEXT,
                regime_hint TEXT,
                feature_count INTEGER,
                active_count INTEGER,
                sparsity DOUBLE,
                activation_method TEXT,
                top_features JSON,
                payload JSON
            )
        """

        with duckdb.connect(str(self._duckdb_path)) as connection:  # type: ignore[attr-defined]
            connection.execute(create_sql)
            connection.execute(sql, [row[name] for name in columns])

    def _persist_parquet(self, record: BeliefSnapshotRecord) -> None:
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            if not self._parquet_warning_logged:
                logger.warning(
                    "pandas/pyarrow not available; skipping belief snapshot parquet export to %s",
                    self._parquet_dir,
                )
                self._parquet_warning_logged = True
            return
        except ImportError as exc:
            if not self._parquet_warning_logged:
                logger.warning(
                    "failed to import pandas for parquet export (%s); destination=%s",
                    exc,
                    self._parquet_dir,
                    exc_info=exc,
                )
                self._parquet_warning_logged = True
            return

        row = record.to_row()
        frame = pd.DataFrame([row])
        filename = self._build_parquet_filename(record)
        frame.to_parquet(filename, index=False)

    def _build_parquet_filename(self, record: BeliefSnapshotRecord) -> Path:
        assert self._parquet_dir is not None
        timestamp = record.generated_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%f")
        belief_slug = self._slugify(record.belief_id)
        return self._parquet_dir / f"{timestamp}_{belief_slug}.parquet"

    @staticmethod
    def _slugify(value: str) -> str:
        return value.replace("/", "_").replace(":", "-")


class BeliefEmitter:
    """Translate sensory snapshots into belief states and publish them."""

    def __init__(
        self,
        *,
        buffer: BeliefBuffer,
        event_bus: EventBus,
        event_type: str = "telemetry.understanding.belief",
        global_bus_factory: Callable[[], TopicBus] | None = None,
        top_k: int = 5,
        activation_floor: float = 0.0,
        snapshot_persister: BeliefSnapshotPersister | None = None,
    ) -> None:
        self._buffer = buffer
        self._event_bus = event_bus
        self._event_type = event_type
        self._global_bus_factory = global_bus_factory
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if activation_floor < 0.0:
            raise ValueError("activation_floor must be non-negative")
        self._top_k = int(top_k)
        self._activation_floor = float(activation_floor)
        self._snapshot_persister = snapshot_persister

    @property
    def buffer(self) -> BeliefBuffer:
        return self._buffer

    def emit(
        self,
        sensory_snapshot: Mapping[str, object],
        *,
        regime_hint: str | None = None,
    ) -> BeliefState:
        state = self._buffer.update(sensory_snapshot, regime_hint=regime_hint)
        state = self._apply_activation(state)
        try:
            self._buffer.replace_latest(state)
        except ValueError:
            logger.debug("Belief buffer empty while attempting to replace latest state")
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
        if self._snapshot_persister is not None:
            try:
                self._snapshot_persister.persist(state)
            except Exception:  # pragma: no cover - defensive against IO issues
                logger.exception("Failed to persist belief snapshot for %s", state.belief_id)
        return state

    def _apply_activation(self, state: BeliefState) -> BeliefState:
        posterior_mean = np.asarray(state.posterior.mean, dtype=float)
        if posterior_mean.size == 0:
            return state

        activated = np.maximum(posterior_mean, 0.0)
        active_indices = np.where(activated > self._activation_floor)[0]
        active_count = int(active_indices.size)
        feature_count = int(activated.size)

        if active_indices.size:
            ranked_indices = active_indices[np.argsort(activated[active_indices])[::-1]]
        else:
            ranked_indices = np.array([], dtype=int)

        top_indices = ranked_indices[: self._top_k]
        top_features: list[dict[str, object]] = []
        for rank, index in enumerate(top_indices, start=1):
            feature_name = state.features[index] if index < len(state.features) else f"feature_{index}"
            top_features.append(
                {
                    "feature": feature_name,
                    "value": float(activated[index]),
                    "rank": rank,
                }
            )

        sparsity = 1.0
        if feature_count > 0:
            sparsity = 1.0 - (active_count / feature_count)
            if sparsity < 0.0:
                sparsity = 0.0
            elif sparsity > 1.0:
                sparsity = 1.0

        activation_summary = {
            "method": "relu_topk",
            "top_k": self._top_k,
            "activation_floor": self._activation_floor,
            "feature_count": feature_count,
            "active_count": active_count,
            "sparsity": float(sparsity),
            "top_features": top_features,
        }
        top_feature_map = {
            entry["feature"]: entry["value"]
            for entry in top_features
            if isinstance(entry.get("feature"), str)
        }

        metadata = dict(state.metadata)
        metadata["activation_summary"] = activation_summary
        metadata["activation_top_features"] = top_feature_map
        return replace(state, metadata=metadata)

    def healthcheck(self) -> Mapping[str, object]:
        latest = self._buffer.latest()
        return {
            "belief_id": latest.belief_id if latest else self._buffer._belief_id,
            "feature_order": list(self._buffer.feature_order or []),
            "window": len(self._buffer),
            "volatility_features": list(self._buffer._volatility_features),
            "volatility_window": self._buffer._volatility_history.maxlen,
            "volatility_samples": len(self._buffer._volatility_history),
            "covariance_trace": self._buffer._latest_covariance_trace,
            "covariance_condition": self._buffer._latest_covariance_condition,
            "covariance_max": self._buffer._latest_covariance_max,
            "covariance_min": self._buffer._latest_covariance_min,
        }


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Stable log-sum-exp helper that guards against overflow."""

    if axis is None:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.array(float("-inf"))
        max_value = float(np.max(values))
        if not np.isfinite(max_value):
            return np.array(max_value)
        shifted = values - max_value
        total = float(np.sum(np.exp(shifted)))
        return np.array(max_value + np.log(total + 1e-12))

    max_value = np.max(values, axis=axis, keepdims=True)
    shifted = values - max_value
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    return (max_value + np.log(total + 1e-12)).squeeze(axis)


def _softmax(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable softmax."""

    shifted = values - np.max(values, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
    return numerator / np.clip(denominator, 1e-12, None)


class _OnlineRegimeHMM:
    """Minimal online Gaussian HMM tailored for regime detection."""

    def __init__(
        self,
        *,
        state_names: Sequence[str],
        regime_labels: Sequence[str],
        transition_concentration: float,
        transition_forgetting: float,
        emission_learning_rate: float,
        initial_means: Sequence[float],
        initial_variance: float,
        min_variance: float,
        max_variance: float | None,
    ) -> None:
        if len(state_names) != len(regime_labels):
            raise ValueError("state_names and regime_labels must align")
        if len(initial_means) != len(state_names):
            raise ValueError("initial_means must align with states")
        if transition_concentration <= 0.0:
            raise ValueError("transition_concentration must be positive")
        if not 0.0 < transition_forgetting <= 1.0:
            raise ValueError("transition_forgetting must be in (0, 1]")
        if not 0.0 < emission_learning_rate <= 1.0:
            raise ValueError("emission_learning_rate must be in (0, 1]")
        if initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive")
        if min_variance <= 0.0:
            raise ValueError("min_variance must be positive")
        if max_variance is not None and max_variance <= min_variance:
            raise ValueError("max_variance must exceed min_variance")

        self._state_names = tuple(state_names)
        self._regime_labels = tuple(regime_labels)
        self._state_index = {name: idx for idx, name in enumerate(self._state_names)}
        self._count = len(self._state_names)
        self._transition_forgetting = transition_forgetting
        self._emission_learning_rate = emission_learning_rate
        self._min_variance = float(min_variance)
        self._max_variance = float(max_variance) if max_variance is not None else None

        base = np.full((self._count, self._count), float(transition_concentration))
        np.fill_diagonal(base, float(transition_concentration) * 2.5)
        self._base_transition_counts = base
        self._transition_counts = base.copy()

        self._state_probabilities = np.full(self._count, 1.0 / self._count)
        self._emission_means = np.asarray(initial_means, dtype=float)
        self._emission_variances = np.full(self._count, float(initial_variance))

    @property
    def regime_labels(self) -> tuple[str, ...]:
        return self._regime_labels

    @property
    def state_probabilities(self) -> np.ndarray:
        return self._state_probabilities.copy()

    @property
    def emission_means(self) -> np.ndarray:
        return self._emission_means.copy()

    @property
    def emission_variances(self) -> np.ndarray:
        return self._emission_variances.copy()

    @property
    def transition_matrix(self) -> np.ndarray:
        counts = np.clip(self._transition_counts, 1e-12, None)
        normalised = counts / np.sum(counts, axis=1, keepdims=True)
        return normalised

    def reset(self) -> None:
        self._transition_counts = self._base_transition_counts.copy()
        self._state_probabilities = np.full(self._count, 1.0 / self._count)

    def _gaussian_log_pdf(self, observation: float) -> np.ndarray:
        variance = np.clip(self._emission_variances, self._min_variance, None)
        if self._max_variance is not None:
            variance = np.clip(variance, None, self._max_variance)
        diff = observation - self._emission_means
        return -0.5 * (
            np.log(2.0 * np.pi * variance)
            + (diff**2) / np.clip(variance, self._min_variance, None)
        )

    def _update_emissions(self, observation: float, weights: np.ndarray, confidence: float) -> None:
        belief = float(np.clip(confidence, 0.0, 1.0))
        if belief <= 0.0:
            return
        scaled = float(self._emission_learning_rate) * belief
        for idx in range(self._count):
            weight = float(np.clip(weights[idx], 0.0, 1.0))
            if weight <= 0.0:
                continue
            learning_rate = scaled * weight
            if learning_rate <= 0.0:
                continue
            mean = self._emission_means[idx]
            variance = self._emission_variances[idx]
            delta = observation - mean
            mean = mean + learning_rate * delta
            centred = observation - mean
            variance = (1.0 - learning_rate) * variance + learning_rate * (centred**2)
            variance = float(np.clip(variance, self._min_variance, self._max_variance))
            self._emission_means[idx] = float(mean)
            self._emission_variances[idx] = variance

    def step(
        self,
        observation: float,
        *,
        confidence: float,
        prior_hint: str | None = None,
    ) -> Mapping[str, float]:
        obs = float(np.clip(observation, -10.0, 10.0))
        log_emission = self._gaussian_log_pdf(obs)

        transition_matrix = np.clip(self.transition_matrix, 1e-12, None)
        log_transition = np.log(transition_matrix)
        log_state = np.log(np.clip(self._state_probabilities, 1e-12, None))

        joint = log_state[:, None] + log_transition + log_emission[None, :]
        log_marginal = _logsumexp(joint, axis=0)
        posterior = _softmax(log_marginal)

        if prior_hint is not None and prior_hint in self._state_index:
            idx = self._state_index[prior_hint]
            posterior[idx] = posterior[idx] + 0.1
            posterior = posterior / np.sum(posterior)

        log_normaliser = float(_logsumexp(joint))
        xi = np.exp(joint - log_normaliser)

        forgetting = float(self._transition_forgetting)
        refreshed = (1.0 - forgetting) * self._transition_counts + forgetting * (
            self._base_transition_counts + xi
        )
        self._transition_counts = np.clip(refreshed, 1e-8, None)

        self._update_emissions(obs, posterior, confidence)
        self._state_probabilities = posterior

        probabilities = {
            regime: float(prob)
            for regime, prob in zip(self._regime_labels, posterior)
        }
        return probabilities


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
        transition_event_type: str = "telemetry.understanding.regime_transition",
        transition_history_size: int = 256,
        global_bus_factory: Callable[[], TopicBus] | None = None,
        volatility_feature: str = "HOW_signal",
        volatility_window: int = 48,
        calm_threshold: float = 0.05,
        storm_threshold: float = 0.25,
        adaptive_thresholds: bool = True,
        calibration_min_samples: int = 6,
        calm_percentile: float = 0.35,
        storm_percentile: float = 0.85,
        hmm_state_names: Sequence[str] | None = None,
        hmm_regime_labels: Sequence[str] | None = None,
        hmm_initial_means: Sequence[float] | None = None,
        hmm_initial_variance: float = 0.25,
        hmm_min_variance: float = 0.02,
        hmm_max_variance: float | None = 1.5,
        hmm_transition_concentration: float = 3.0,
        hmm_transition_forgetting: float = 0.08,
        hmm_emission_learning_rate: float = 0.3,
    ) -> None:
        self._event_bus = event_bus
        self._signal_id = signal_id
        self._bullish_threshold = bullish_threshold
        self._bearish_threshold = bearish_threshold
        self._confidence_floor = confidence_floor
        self._event_type = event_type
        self._transition_event_type = transition_event_type
        self._global_bus_factory = global_bus_factory
        if transition_history_size <= 0:
            raise ValueError("transition_history_size must be positive")
        self._transition_history_limit = int(transition_history_size)
        self._transition_history: deque[RegimeTransition] = deque(
            maxlen=self._transition_history_limit
        )
        self._last_signal: RegimeSignal | None = None
        self._last_transition: RegimeTransition | None = None
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
        default_states = ("bear", "neutral", "bull")
        state_names = tuple(hmm_state_names) if hmm_state_names is not None else default_states
        if len(state_names) < 2:
            raise ValueError("hmm_state_names must include at least two states")
        default_labels = ("bearish", "balanced", "bullish")[: len(state_names)]
        regime_labels = (
            tuple(hmm_regime_labels)
            if hmm_regime_labels is not None
            else default_labels
        )
        if len(regime_labels) != len(state_names):
            raise ValueError("hmm_regime_labels must align with hmm_state_names")
        default_means = (-0.45, 0.0, 0.45)
        if len(state_names) != len(default_means):
            if hmm_initial_means is None:
                raise ValueError(
                    "hmm_initial_means must be provided when using a custom number of states"
                )
        initial_means = (
            tuple(float(value) for value in hmm_initial_means)
            if hmm_initial_means is not None
            else default_means[: len(state_names)]
        )
        self._hmm = _OnlineRegimeHMM(
            state_names=state_names,
            regime_labels=regime_labels,
            transition_concentration=float(hmm_transition_concentration),
            transition_forgetting=float(hmm_transition_forgetting),
            emission_learning_rate=float(hmm_emission_learning_rate),
            initial_means=initial_means,
            initial_variance=float(hmm_initial_variance),
            min_variance=float(hmm_min_variance),
            max_variance=float(hmm_max_variance) if hmm_max_variance is not None else None,
        )
        self._hmm_state_names = state_names
        self._hmm_regime_labels = regime_labels
        self._hint_state_map: dict[str, str] = {}
        for state, regime in zip(state_names, regime_labels):
            self._hint_state_map[state.lower()] = state
            self._hint_state_map[regime.lower()] = state

    @property
    def calm_threshold(self) -> float:
        return self._calm_threshold

    @property
    def storm_threshold(self) -> float:
        return self._storm_threshold

    @property
    def volatility_feature(self) -> str:
        return self._volatility_feature

    @property
    def volatility_window(self) -> int:
        return self._volatility_history.maxlen

    @property
    def adaptive_thresholds(self) -> bool:
        return self._adaptive_thresholds

    def transition_history(self) -> Sequence[RegimeTransition]:
        """Return recorded regime transitions in chronological order."""

        return tuple(self._transition_history)

    @property
    def last_transition(self) -> RegimeTransition | None:
        return self._last_transition

    @property
    def last_signal(self) -> RegimeSignal | None:
        return self._last_signal

    def reconfigure(
        self,
        *,
        calm_threshold: float | None = None,
        storm_threshold: float | None = None,
        volatility_feature: str | None = None,
        volatility_window: int | None = None,
        adaptive_thresholds: bool | None = None,
        calibration_min_samples: int | None = None,
        calm_percentile: float | None = None,
        storm_percentile: float | None = None,
        reset_history: bool = False,
        reset_dynamic_thresholds: bool = True,
        reset_hmm: bool = False,
    ) -> None:
        """Update volatility thresholds and history to match new calibration."""

        if calm_threshold is not None:
            if calm_threshold < 0.0:
                raise ValueError("calm_threshold must be non-negative")
            self._calm_threshold = float(calm_threshold)
        if storm_threshold is not None:
            if storm_threshold <= 0.0:
                raise ValueError("storm_threshold must be positive")
            self._storm_threshold = float(storm_threshold)
        if self._storm_threshold <= self._calm_threshold:
            self._storm_threshold = self._calm_threshold + 1e-6

        if volatility_feature is not None:
            self._volatility_feature = str(volatility_feature)
        if volatility_window is not None:
            if volatility_window <= 1:
                raise ValueError("volatility_window must be greater than 1")
            self._volatility_history = deque(self._volatility_history, maxlen=volatility_window)
        if adaptive_thresholds is not None:
            self._adaptive_thresholds = bool(adaptive_thresholds)
        if calibration_min_samples is not None:
            if calibration_min_samples <= 1:
                raise ValueError("calibration_min_samples must be greater than 1")
            self._calibration_min_samples = int(calibration_min_samples)
        if calm_percentile is not None or storm_percentile is not None:
            calm_value = calm_percentile if calm_percentile is not None else self._calm_percentile
            storm_value = storm_percentile if storm_percentile is not None else self._storm_percentile
            if not 0.0 <= calm_value < storm_value <= 1.0:
                raise ValueError("percentiles must satisfy 0 <= calm < storm <= 1")
            self._calm_percentile = float(calm_value)
            self._storm_percentile = float(storm_value)

        if reset_history:
            self._volatility_history.clear()
        if reset_dynamic_thresholds:
            self._dynamic_calm_threshold = None
            self._dynamic_storm_threshold = None
        if reset_hmm:
            self._hmm.reset()

    def classify(self, belief_state: BeliefState) -> RegimeSignal:
        posterior = belief_state.posterior
        strength = posterior.strength
        confidence = max(posterior.confidence, 0.0)
        regime_hint: str | None = None
        metadata_payload = belief_state.metadata
        if isinstance(metadata_payload, Mapping):
            raw_hint = metadata_payload.get("regime_hint")
            if isinstance(raw_hint, str):
                regime_hint = raw_hint.strip().lower()

        prior_state = self._hint_state_map.get(regime_hint) if regime_hint else None
        regime_probabilities = self._hmm.step(
            strength,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            prior_hint=prior_state,
        )

        regime_name = max(regime_probabilities, key=regime_probabilities.get)
        hmm_confidence = float(np.clip(regime_probabilities[regime_name], 0.0, 1.0))
        belief_confidence = float(np.clip(confidence, 0.0, 1.0))
        combined_confidence = float(
            np.clip(0.5 * hmm_confidence + 0.5 * belief_confidence, 0.0, 1.0)
        )

        if hmm_confidence < self._confidence_floor or belief_confidence < self._confidence_floor:
            regime = "uncertain"
            regime_confidence = max(self._confidence_floor, combined_confidence)
        else:
            regime = regime_name
            regime_confidence = max(combined_confidence, self._confidence_floor)

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
            "posterior_strength": float(strength),
            "posterior_confidence": float(confidence),
            "regime_probabilities": dict(regime_probabilities),
            "hmm_state_means": {
                regime_label: float(mean)
                for regime_label, mean in zip(
                    self._hmm_regime_labels, self._hmm.emission_means
                )
            },
            "hmm_state_variances": {
                regime_label: float(variance)
                for regime_label, variance in zip(
                    self._hmm_regime_labels,
                    self._hmm.emission_variances,
                )
            },
            "hmm_transition_matrix": [
                [float(value) for value in row]
                for row in self._hmm.transition_matrix.tolist()
            ],
        }
        if volatility_sample is not None:
            metadata["volatility_sample"] = volatility_sample
        if regime_hint is not None:
            metadata["regime_hint"] = regime_hint

        return RegimeSignal(
            signal_id=self._signal_id,
            regime_state=regime_state,
            features=feature_map,
            lineage=lineage.as_dict(),
            metadata=metadata,
        )

    def publish(self, belief_state: BeliefState) -> RegimeSignal:
        signal = self.classify(belief_state)
        self._record_transition(signal)
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
            "transition_history_size": self._transition_history_limit,
            "transition_count": len(self._transition_history),
            "last_regime": (
                self._last_signal.regime_state.regime if self._last_signal else None
            ),
            "last_transition_reason": (
                self._last_transition.reason if self._last_transition else None
            ),
            "last_transition_timestamp": (
                self._last_transition.timestamp.isoformat()
                if self._last_transition
                else None
            ),
            "hmm_state_probabilities": [
                float(value) for value in self._hmm.state_probabilities.tolist()
            ],
            "hmm_emission_means": [
                float(value) for value in self._hmm.emission_means.tolist()
            ],
            "hmm_emission_variances": [
                float(value) for value in self._hmm.emission_variances.tolist()
            ],
            "hmm_transition_matrix": [
                [float(value) for value in row]
                for row in self._hmm.transition_matrix.tolist()
            ],
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

    def _record_transition(self, signal: RegimeSignal) -> None:
        previous_signal = self._last_signal
        current_state = signal.regime_state

        def _normalise(timestamp: datetime) -> datetime:
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=UTC)
            return timestamp.astimezone(UTC)

        if previous_signal is None:
            reason = "initial"
            previous_state: RegimeState | None = None
        else:
            previous_state = previous_signal.regime_state
            regime_changed = current_state.regime != previous_state.regime
            volatility_changed = (
                current_state.volatility_state != previous_state.volatility_state
            )
            if not regime_changed and not volatility_changed:
                self._last_signal = signal
                return
            if regime_changed and volatility_changed:
                reason = "regime_and_volatility"
            elif regime_changed:
                reason = "regime"
            else:
                reason = "volatility"

        current_timestamp = _normalise(current_state.timestamp)
        previous_timestamp = (
            _normalise(previous_state.timestamp) if previous_signal is not None else None
        )
        latency_ms = (
            max(0.0, (current_timestamp - previous_timestamp).total_seconds() * 1000.0)
            if previous_timestamp is not None
            else None
        )

        calm_threshold, storm_threshold = self._current_thresholds()
        metadata = dict(signal.metadata)
        metadata.setdefault("calm_threshold", calm_threshold)
        metadata.setdefault("storm_threshold", storm_threshold)

        transition = RegimeTransition(
            signal_id=signal.signal_id,
            timestamp=current_timestamp,
            previous_regime=previous_state.regime if previous_signal else None,
            current_regime=current_state.regime,
            previous_volatility_state=(
                previous_state.volatility_state if previous_signal else None
            ),
            current_volatility_state=current_state.volatility_state,
            confidence=current_state.confidence,
            volatility=current_state.volatility,
            latency_ms=latency_ms,
            reason=reason,
            metadata=metadata,
            features=dict(signal.features),
        )

        self._transition_history.append(transition)
        self._last_transition = transition
        self._last_signal = signal

        logger.info(
            "Regime transition %s -> %s (reason=%s, volatility_state=%s, confidence=%.3f)",
            transition.previous_regime or "<init>",
            transition.current_regime,
            transition.reason,
            transition.current_volatility_state,
            transition.confidence,
        )

        event = Event(
            type=self._transition_event_type,
            payload=transition.as_dict(),
            source="understanding.regime_fsm",
        )
        _publish_event(
            self._event_bus,
            event,
            logger=logger,
            runtime_fallback_message="Runtime bus rejected regime transition; falling back to global bus",
            runtime_unexpected_message="Unexpected error publishing regime transition via runtime bus",
            runtime_none_message="Runtime bus returned no result while publishing regime transition",
            global_not_running_message="Global event bus not running while publishing regime transition",
            global_unexpected_message="Unexpected error publishing regime transition via global bus",
            global_bus_factory=self._global_bus_factory,
        )

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
                else:
                    if not np.isfinite(volatility_sample):
                        volatility_sample = None
            metric_value = metadata.get("volatility")
            if metric_value is not None:
                try:
                    volatility_metric = abs(float(metric_value))
                except (TypeError, ValueError):
                    volatility_metric = None
                else:
                    if not np.isfinite(volatility_metric):
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
                        else:
                            if not np.isfinite(volatility_sample):
                                volatility_sample = None

        if volatility_sample is None:
            candidate = feature_map.get(self._volatility_feature)
            if candidate is not None:
                try:
                    volatility_sample = abs(float(candidate))
                except (TypeError, ValueError):
                    volatility_sample = None
                else:
                    if not np.isfinite(volatility_sample):
                        volatility_sample = None

        recorded_sample = volatility_sample
        history_value = volatility_metric if volatility_metric is not None else volatility_sample
        if history_value is not None and np.isfinite(history_value):
            numeric_history = float(history_value)
            if np.isfinite(numeric_history):
                self._volatility_history.append(numeric_history)
            history_value = numeric_history
        if history_value is not None and np.isfinite(history_value):
            self._update_dynamic_thresholds()
            if len(self._volatility_history) >= 2:
                std_metric = float(np.std(self._volatility_history, ddof=0))
                if not np.isfinite(std_metric):
                    std_metric = abs(history_value)
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
