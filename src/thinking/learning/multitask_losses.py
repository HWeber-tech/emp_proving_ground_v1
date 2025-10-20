"""Multi-task loss utilities for the LOBSTER pre-training roadmap slice (E.1.1).

The roadmap calls for combining a collection of heterogeneous objectives: a
robust regression target (Huber + quantile pinball), directional classifiers,
"big move" detection, and a trio of auxiliary heads covering next-event
prediction, masked depth reconstruction, and queue depletion alerts.  This
module implements light-weight, dependency free helpers that ingest scalar or
vector predictions/targets and return a structured loss breakdown that hidden
training harnesses can consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

_EPSILON = 1e-9


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            return default
    return default


def _as_sequence(payload: object) -> list[float]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        return [_as_float(value) for _, value in payload.items()]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [_as_float(item) for item in payload]
    return [_as_float(payload)]


def _mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    if count == 0:
        return 0.0
    return total / count


def _binary_target(value: object) -> float:
    numeric = _as_float(value)
    if numeric <= 0.0:
        return 0.0
    if numeric >= 1.0:
        return 1.0
    # Map -1/1 or other centred encodings into {0, 1}
    if numeric <= 0.5:
        return 0.0
    return 1.0


def _binary_cross_entropy(probability: float, target: float) -> float:
    prob = min(max(probability, _EPSILON), 1.0 - _EPSILON)
    return -(target * math.log(prob) + (1.0 - target) * math.log(1.0 - prob))


def _normalise_distribution(values: Sequence[float]) -> list[float]:
    cleaned = [max(float(v), 0.0) for v in values]
    total = sum(cleaned)
    if total <= 0.0:
        length = len(cleaned)
        if length == 0:
            return []
        uniform = 1.0 / length
        return [uniform for _ in cleaned]
    return [v / total for v in cleaned]


def _negative_log(probability: float) -> float:
    prob = min(max(probability, _EPSILON), 1.0 - _EPSILON)
    return -math.log(prob)


def _quantile_keys(level: float) -> tuple[str, ...]:
    q = float(level)
    percent = int(round(q * 100))
    return (
        f"{q}",
        f"{q:.2f}",
        f"q{percent}",
        f"tau_{percent}",
        f"quantile_{percent}",
    )


def _resolve_quantile_predictions(
    predictions: object,
    levels: Sequence[float],
) -> list[tuple[float, float]]:
    resolved: list[tuple[float, float]] = []
    if isinstance(predictions, Mapping):
        lower_mapping = {str(key).strip(): _as_float(value) for key, value in predictions.items()}
        for idx, level in enumerate(levels):
            value = None
            for key in _quantile_keys(level):
                if key in lower_mapping:
                    value = lower_mapping[key]
                    break
            if value is None:
                # Fall back to positional access if available
                fallback_key = str(idx)
                if fallback_key in lower_mapping:
                    value = lower_mapping[fallback_key]
            if value is not None:
                resolved.append((float(level), float(value)))
    elif isinstance(predictions, Sequence) and not isinstance(predictions, (str, bytes, bytearray)):
        numeric = [_as_float(entry) for entry in predictions]
        for idx, level in enumerate(levels):
            if idx < len(numeric):
                resolved.append((float(level), numeric[idx]))
    else:
        value = _as_float(predictions)
        if levels:
            resolved.append((float(levels[0]), value))
    return resolved


def _huber_loss(predictions: object, targets: object, delta: float) -> float:
    pred_seq = _as_sequence(predictions)
    target_seq = _as_sequence(targets)
    length = max(len(pred_seq), len(target_seq))
    if length == 0:
        return 0.0
    # Extend sequences to matching length
    if len(pred_seq) < length:
        pred_seq.extend([pred_seq[-1] if pred_seq else 0.0] * (length - len(pred_seq)))
    if len(target_seq) < length:
        target_seq.extend([target_seq[-1] if target_seq else 0.0] * (length - len(target_seq)))

    losses = []
    for pred, target in zip(pred_seq, target_seq):
        diff = pred - target
        abs_diff = abs(diff)
        if abs_diff <= delta:
            losses.append(0.5 * diff * diff)
        else:
            losses.append(delta * (abs_diff - 0.5 * delta))
    return _mean(losses)


def _quantile_loss(predictions: object, target: object, levels: Sequence[float]) -> float:
    resolved = _resolve_quantile_predictions(predictions, levels)
    if not resolved:
        return 0.0
    target_value = _as_float(target)
    losses = []
    for level, prediction in resolved:
        diff = target_value - prediction
        if diff >= 0.0:
            losses.append(level * diff)
        else:
            losses.append((level - 1.0) * diff)
    return _mean(losses)


def _categorical_cross_entropy(
    predictions: object,
    target: object,
) -> float:
    if predictions is None:
        return 0.0

    # Case 1: target provided as full distribution
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes, bytearray)):
        target_distribution = _normalise_distribution(_as_sequence(target))
        pred_distribution = _normalise_distribution(_as_sequence(predictions))
        pairs = zip(target_distribution, pred_distribution)
        return -sum(t * math.log(max(p, _EPSILON)) for t, p in pairs)

    # Case 2: mapping predictions with label/index target
    if isinstance(predictions, Mapping):
        mapping = {str(key): _as_float(value) for key, value in predictions.items()}
        if isinstance(target, Mapping):
            if "label" in target:
                label = str(target["label"])
                if label in mapping:
                    return _negative_log(mapping[label])
            if "index" in target:
                index = int(_as_float(target["index"]))
                keys = list(mapping)
                if 0 <= index < len(keys):
                    return _negative_log(mapping[keys[index]])
        elif isinstance(target, str):
            label = target.strip()
            if label in mapping:
                return _negative_log(mapping[label])
        elif isinstance(target, (int, float)):
            index = int(_as_float(target))
            keys = list(mapping)
            if 0 <= index < len(keys):
                return _negative_log(mapping[keys[index]])
        return 0.0

    # Case 3: sequence predictions with index label
    sequence = _as_sequence(predictions)
    if isinstance(target, Mapping) and "index" in target:
        index = int(_as_float(target["index"]))
        if 0 <= index < len(sequence):
            return _negative_log(sequence[index])
    if isinstance(target, (int, float)):
        index = int(_as_float(target))
        if 0 <= index < len(sequence):
            return _negative_log(sequence[index])
    if isinstance(target, str):
        # String targets without mapping context cannot be resolved meaningfully
        return 0.0
    return 0.0


def _masked_mse(
    predictions: object,
    targets: object,
    mask: object | None,
) -> float:
    pred_seq = _as_sequence(predictions)
    target_seq = _as_sequence(targets)
    length = max(len(pred_seq), len(target_seq))
    if length == 0:
        return 0.0
    if len(pred_seq) < length:
        pred_seq.extend([pred_seq[-1] if pred_seq else 0.0] * (length - len(pred_seq)))
    if len(target_seq) < length:
        target_seq.extend([target_seq[-1] if target_seq else 0.0] * (length - len(target_seq)))

    mask_seq = []
    if mask is None:
        mask_seq = [1.0] * length
    else:
        mask_seq = _as_sequence(mask)
        if len(mask_seq) < length:
            mask_seq.extend([mask_seq[-1] if mask_seq else 0.0] * (length - len(mask_seq)))
    weighted_losses = []
    weights = []
    for pred, target, weight in zip(pred_seq, target_seq, mask_seq):
        w = max(_as_float(weight), 0.0)
        if w <= 0.0:
            continue
        diff = pred - target
        weighted_losses.append(w * diff * diff)
        weights.append(w)
    if not weighted_losses:
        return 0.0
    return sum(weighted_losses) / sum(weights)


@dataclass(frozen=True)
class MultiTaskLossResult:
    """Structured view of multi-task losses with total aggregation."""

    total: float
    primary: Mapping[str, float]
    auxiliary: Mapping[str, float]
    weights: Mapping[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "primary": dict(self.primary),
            "auxiliary": dict(self.auxiliary),
            "weights": dict(self.weights),
        }

    def component(self, name: str) -> float:
        if name in self.primary:
            return self.primary[name]
        if name in self.auxiliary:
            return self.auxiliary[name]
        raise KeyError(name)


_DEFAULT_LEVELS: tuple[float, ...] = (0.25, 0.5, 0.75)
_DEFAULT_WEIGHTS: Mapping[str, float] = {
    "huber": 1.0,
    "quantile": 1.0,
    "direction": 1.0,
    "big_move": 1.0,
    "next_event": 0.5,
    "masked_depth": 0.5,
    "queue_depletion": 0.5,
}


def compute_multitask_losses(
    predictions: Mapping[str, object],
    targets: Mapping[str, object],
    *,
    weights: Mapping[str, float] | None = None,
    quantile_levels: Sequence[float] | None = None,
    huber_delta: float = 1.0,
) -> MultiTaskLossResult:
    """Compute the roadmap-specified multi-task loss breakdown.

    Parameters
    ----------
    predictions:
        Mapping containing model head outputs. Recognised keys include
        ``delta`` (regression), ``quantiles``, ``direction``, ``big_move``,
        ``next_event``, ``masked_depth``, ``queue_depletion``.
    targets:
        Mapping containing matching supervision targets. Recognised keys mirror
        ``predictions`` and additionally accept ``depth_mask`` for the masked
        depth reconstruction task.
    weights:
        Optional per-component weights. Missing entries fall back to
        :data:`_DEFAULT_WEIGHTS`.
    quantile_levels:
        Sequence of quantile levels (between 0 and 1) used for the pinball loss.
        Defaults to ``(0.25, 0.5, 0.75)``.
    huber_delta:
        Delta parameter for the Huber loss.
    """

    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping")
    if not isinstance(targets, Mapping):
        raise TypeError("targets must be a mapping")

    levels = tuple(float(level) for level in (quantile_levels or _DEFAULT_LEVELS) if 0.0 < float(level) < 1.0)
    if not levels:
        levels = _DEFAULT_LEVELS

    combined_weights: MutableMapping[str, float] = dict(_DEFAULT_WEIGHTS)
    if weights:
        for key, value in weights.items():
            combined_weights[str(key)] = float(value)

    primary: MutableMapping[str, float] = {}
    auxiliary: MutableMapping[str, float] = {}

    if "delta" in predictions and "delta" in targets:
        primary["huber"] = _huber_loss(predictions["delta"], targets["delta"], float(max(huber_delta, _EPSILON)))

    if "quantiles" in predictions and "delta" in targets:
        primary["quantile"] = _quantile_loss(predictions["quantiles"], targets["delta"], levels)

    if "direction" in predictions and "direction" in targets:
        target_value = _binary_target(targets["direction"])
        probability = _as_float(predictions["direction"], 0.5)
        primary["direction"] = _binary_cross_entropy(probability, target_value)

    if "big_move" in predictions and "big_move" in targets:
        target_value = _binary_target(targets["big_move"])
        probability = _as_float(predictions["big_move"], 0.5)
        primary["big_move"] = _binary_cross_entropy(probability, target_value)

    if "next_event" in predictions and "next_event" in targets:
        auxiliary["next_event"] = _categorical_cross_entropy(
            predictions["next_event"], targets["next_event"],
        )

    masked_depth_pred = None
    if "masked_depth" in predictions:
        masked_depth_pred = predictions["masked_depth"]
    if masked_depth_pred is not None and "masked_depth" in targets:
        mask = targets.get("depth_mask")
        if mask is None:
            mask = predictions.get("depth_mask")
        auxiliary["masked_depth"] = _masked_mse(masked_depth_pred, targets["masked_depth"], mask)

    if "queue_depletion" in predictions and "queue_depletion" in targets:
        target_value = _binary_target(targets["queue_depletion"])
        probability = _as_float(predictions["queue_depletion"], 0.5)
        auxiliary["queue_depletion"] = _binary_cross_entropy(probability, target_value)

    active_weights: MutableMapping[str, float] = {}
    for name in (*primary.keys(), *auxiliary.keys()):
        active_weights[name] = combined_weights.get(name, 1.0)

    total = 0.0
    for name, loss in primary.items():
        total += loss * active_weights.get(name, 1.0)
    for name, loss in auxiliary.items():
        total += loss * active_weights.get(name, 1.0)

    return MultiTaskLossResult(
        total=total,
        primary=dict(primary),
        auxiliary=dict(auxiliary),
        weights=dict(active_weights),
    )


__all__ = ["MultiTaskLossResult", "compute_multitask_losses"]
