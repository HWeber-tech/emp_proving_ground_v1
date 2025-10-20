"""Layer freezing and LoRA allocation planner for FX adaptation (E.2.1).

This module implements the roadmap requirement to freeze the bottom 60–80% of
layers while enabling LoRA adapters with rank 8–16 across the top 30–40%.  The
implementation is intentionally framework-agnostic; it works on ordered layer
names (strings or objects with a ``name`` attribute) and produces a deterministic
plan describing which layers should be frozen and how LoRA adapters should be
configured.  Downstream training harnesses can consume the resulting plan to
apply the actual parameter updates in their preferred deep-learning library.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LoRALayerConfig:
    """Configuration describing a single LoRA adapter placement."""

    layer: str
    rank: int
    alpha: float
    dropout: float

    def as_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
        }


@dataclass(frozen=True)
class LoRAFreezePlan:
    """Plan capturing frozen layers and LoRA adapter assignments."""

    total_layers: int
    frozen_layers: tuple[str, ...]
    lora_layers: tuple[LoRALayerConfig, ...]
    trainable_layers: tuple[str, ...]
    freeze_fraction: float
    lora_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "total_layers": self.total_layers,
            "freeze_fraction": self.freeze_fraction,
            "lora_fraction": self.lora_fraction,
            "frozen_layers": list(self.frozen_layers),
            "trainable_layers": list(self.trainable_layers),
            "lora_layers": [config.as_dict() for config in self.lora_layers],
        }

    @property
    def non_lora_trainable_layers(self) -> tuple[str, ...]:
        lora_layer_names = {config.layer for config in self.lora_layers}
        return tuple(layer for layer in self.trainable_layers if layer not in lora_layer_names)


def plan_lora_freeze(
    layers: Sequence[object],
    *,
    freeze_fraction: float = 0.7,
    freeze_range: tuple[float, float] = (0.6, 0.8),
    lora_fraction: float = 0.35,
    lora_range: tuple[float, float] = (0.3, 0.4),
    rank_range: tuple[int, int] = (8, 16),
    lora_alpha_multiplier: float = 2.0,
    lora_dropout: float = 0.05,
) -> LoRAFreezePlan:
    """Return a :class:`LoRAFreezePlan` for the supplied ``layers``.

    Parameters
    ----------
    layers:
        Ordered sequence of layer descriptors.  Each entry may be a string, an
        object exposing ``name``, or any object whose string representation is
        stable for identification purposes.  The sequence is expected to be
        ordered from "bottom" (indices near zero) to "top" (last indices).
    freeze_fraction:
        Target fraction of layers to freeze.  Values are clamped into
        ``freeze_range`` when necessary.
    freeze_range:
        Inclusive range describing the allowable fraction of frozen layers.
    lora_fraction:
        Target fraction of layers that should receive LoRA adapters.  Values are
        clamped into ``lora_range`` when possible.
    lora_range:
        Inclusive range for the fraction of LoRA-enabled layers.
    rank_range:
        Inclusive integer range for the LoRA rank allocation per layer.
    lora_alpha_multiplier:
        Multiplier applied to each LoRA rank to derive the LoRA ``alpha``.
    lora_dropout:
        Dropout probability assigned to each LoRA adapter.
    """

    layer_names = _normalise_layer_names(layers)
    total = len(layer_names)
    if total == 0:
        raise ValueError("layers must contain at least one entry")

    freeze_min, freeze_max = _validate_fraction_range(freeze_range, "freeze_range")
    lora_min, lora_max = _validate_fraction_range(lora_range, "lora_range")
    target_freeze_fraction = _clamp_fraction(freeze_fraction, freeze_min, freeze_max)
    target_lora_fraction = _clamp_fraction(lora_fraction, lora_min, lora_max)

    min_rank, max_rank = _validate_rank_range(rank_range)
    if lora_alpha_multiplier <= 0:
        raise ValueError("lora_alpha_multiplier must be positive")
    if not (0.0 <= lora_dropout < 1.0):
        raise ValueError("lora_dropout must be within [0, 1)")

    freeze_count = _resolve_freeze_count(
        total,
        target_freeze_fraction,
        freeze_min,
        freeze_max,
        lora_min,
    )
    available_for_lora = total - freeze_count

    lora_count = _resolve_lora_count(
        total,
        available_for_lora,
        target_lora_fraction,
        lora_min,
        lora_max,
    )

    frozen_layers = tuple(layer_names[:freeze_count])
    trainable_pool = layer_names[freeze_count:]
    lora_layer_names = tuple(trainable_pool[-lora_count:]) if lora_count else tuple()
    trainable_layers = tuple(layer_names[freeze_count:])

    lora_layers = _build_lora_configs(
        lora_layer_names,
        min_rank,
        max_rank,
        alpha_multiplier=lora_alpha_multiplier,
        dropout=lora_dropout,
    )

    freeze_fraction_actual = freeze_count / total
    lora_fraction_actual = (len(lora_layers) / total) if total else 0.0

    return LoRAFreezePlan(
        total_layers=total,
        frozen_layers=frozen_layers,
        lora_layers=lora_layers,
        trainable_layers=trainable_layers,
        freeze_fraction=freeze_fraction_actual,
        lora_fraction=lora_fraction_actual,
    )


def _normalise_layer_names(layers: Sequence[object]) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, layer in enumerate(layers):
        name: str
        if isinstance(layer, str):
            name = layer.strip() or f"layer_{index}"
        elif hasattr(layer, "name"):
            name = str(getattr(layer, "name"))
            name = name.strip() or f"layer_{index}"
        else:
            name = str(layer).strip() or f"layer_{index}"
        count = seen.get(name, 0)
        if count:
            unique_name = f"{name}#{count}"
            seen[name] = count + 1
            names.append(unique_name)
        else:
            seen[name] = 1
            names.append(name)
    return names


def _validate_fraction_range(value: tuple[float, float], label: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{label} must contain exactly two entries")
    lower, upper = float(value[0]), float(value[1])
    if not (0.0 <= lower <= 1.0):
        raise ValueError(f"{label} lower bound must be within [0, 1]")
    if not (0.0 <= upper <= 1.0):
        raise ValueError(f"{label} upper bound must be within [0, 1]")
    if lower > upper:
        raise ValueError(f"{label} lower bound cannot exceed upper bound")
    return lower, upper


def _clamp_fraction(value: float, lower: float, upper: float) -> float:
    fraction = float(value)
    if fraction < lower:
        return lower
    if fraction > upper:
        return upper
    return fraction


def _validate_rank_range(rank_range: tuple[int, int]) -> tuple[int, int]:
    if len(rank_range) != 2:
        raise ValueError("rank_range must contain exactly two integers")
    lower, upper = int(rank_range[0]), int(rank_range[1])
    if lower <= 0 or upper <= 0:
        raise ValueError("LoRA rank bounds must be positive integers")
    if lower > upper:
        raise ValueError("LoRA rank lower bound cannot exceed upper bound")
    return lower, upper


def _resolve_freeze_count(
    total: int,
    target_fraction: float,
    min_fraction: float,
    max_fraction: float,
    lora_min_fraction: float,
) -> int:
    target = int(round(total * target_fraction))
    min_required_for_lora = _minimum_lora_layers(total, lora_min_fraction)

    candidates: list[int] = []
    for count in range(total + 1):
        fraction = count / total
        if min_fraction <= fraction <= max_fraction and total - count >= min_required_for_lora:
            candidates.append(count)

    if candidates:
        return min(candidates, key=lambda c: (abs(c - target), c))

    fallback = max(0, total - min_required_for_lora)
    return min(fallback, total)


def _resolve_lora_count(
    total: int,
    available: int,
    target_fraction: float,
    min_fraction: float,
    max_fraction: float,
) -> int:
    if available <= 0:
        return 0

    target = int(round(total * target_fraction))
    min_count = min(available, _minimum_lora_layers(total, min_fraction))
    max_count = min(available, max(min_count, int(math.ceil(total * max_fraction))))

    candidates: list[int] = []
    for count in range(min_count, max_count + 1):
        fraction = count / total
        if min_fraction <= fraction <= max_fraction and count <= available:
            candidates.append(count)

    if candidates:
        return min(candidates, key=lambda c: (abs(c - target), -c))

    return min_count


def _minimum_lora_layers(total: int, fraction: float) -> int:
    if total == 0 or fraction <= 0.0:
        return 1 if total > 0 else 0
    return max(1, int(math.ceil(total * fraction)))


def _build_lora_configs(
    layer_names: Iterable[str],
    min_rank: int,
    max_rank: int,
    *,
    alpha_multiplier: float,
    dropout: float,
) -> tuple[LoRALayerConfig, ...]:
    names = list(layer_names)
    count = len(names)
    if count == 0:
        return tuple()
    if count == 1:
        rank = int(round((min_rank + max_rank) / 2))
        return (
            LoRALayerConfig(
                layer=names[0],
                rank=rank,
                alpha=rank * alpha_multiplier,
                dropout=dropout,
            ),
        )

    ranks: list[int] = []
    span = max(count - 1, 1)
    for index in range(count):
        blend = index / span
        interpolated = min_rank + (max_rank - min_rank) * blend
        ranks.append(int(round(interpolated)))

    configs = []
    for layer, rank in zip(names, ranks):
        clamped_rank = max(min_rank, min(max_rank, rank))
        configs.append(
            LoRALayerConfig(
                layer=layer,
                rank=clamped_rank,
                alpha=clamped_rank * alpha_multiplier,
                dropout=dropout,
            )
        )
    return tuple(configs)


__all__ = [
    "LoRALayerConfig",
    "LoRAFreezePlan",
    "plan_lora_freeze",
]
