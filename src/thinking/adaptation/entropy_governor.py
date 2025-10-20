"""Entropy Governor for tactic rotation and microtiming jitter.

This module implements the roadmap item labelled **Entropy Governor**.  The
governor injects controlled unpredictability into the policy router by
rotating tactic selections and jittering execution microtiming.  Rotation keeps
recently favoured tactics on a short cooldown so adversaries cannot lock onto
deterministic patterns, while microtiming jitter applies small, bounded delays
to decision timestamps so downstream schedulers avoid perfectly regular
cadence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import random
from typing import Deque, Iterable, Mapping, MutableMapping, Sequence


UTC = timezone.utc


def _coerce_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(slots=True)
class EntropyGovernorConfig:
    """Configuration for :class:`EntropyGovernor`."""

    history_window: int = 64
    rotation_cooldown: int = 1
    rotation_probability: float = 0.2
    minimum_jitter_ms: float = 0.75
    maximum_jitter_ms: float = 3.5
    allow_negative_jitter: bool = False
    max_rotation_depth: int = 5

    def __post_init__(self) -> None:
        if self.history_window <= 0:
            raise ValueError("history_window must be positive")
        if self.rotation_cooldown < 0:
            raise ValueError("rotation_cooldown cannot be negative")
        if not 0.0 <= self.rotation_probability <= 1.0:
            raise ValueError("rotation_probability must be within [0, 1]")
        if self.maximum_jitter_ms < 0:
            raise ValueError("maximum_jitter_ms cannot be negative")
        if self.minimum_jitter_ms < 0:
            raise ValueError("minimum_jitter_ms cannot be negative")
        if self.minimum_jitter_ms > self.maximum_jitter_ms and self.maximum_jitter_ms > 0.0:
            raise ValueError("minimum_jitter_ms cannot exceed maximum_jitter_ms")
        if self.max_rotation_depth <= 0:
            raise ValueError("max_rotation_depth must be positive")


class EntropyGovernor:
    """Rotate tactics and jitter microtiming to stay unpredictable."""

    def __init__(
        self,
        config: EntropyGovernorConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config or EntropyGovernorConfig()
        self._rng: random.Random = rng or random.SystemRandom()
        self._history: Deque[str] = deque(maxlen=self._config.history_window)

    @staticmethod
    def _extract_tactic_id(entry: Mapping[str, object]) -> str | None:
        tactic = entry.get("tactic")
        if tactic is not None:
            tactic_id = getattr(tactic, "tactic_id", None)
            if isinstance(tactic_id, str) and tactic_id.strip():
                return tactic_id.strip()
        value = entry.get("tactic_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _remember(self, tactic_id: str | None) -> None:
        if not tactic_id:
            return
        self._history.append(tactic_id)

    def history(self) -> Sequence[str]:
        return tuple(self._history)

    def apply_rotation(
        self,
        *,
        ranked: Sequence[Mapping[str, object]],
        current: Mapping[str, object],
        forced: bool = False,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        if not ranked:
            raise ValueError("ranked must not be empty")

        context: MutableMapping[str, object] = {
            "forced": bool(forced),
            "rotation_applied": False,
            "history_before": list(self._history),
        }

        selected = current
        tactic_id = self._extract_tactic_id(current)

        if forced or tactic_id is None:
            self._remember(tactic_id)
            context["history_after"] = list(self._history)
            context["reason"] = "forced" if forced else "unknown_tactic"
            context["selected_tactic"] = tactic_id
            return selected, context

        recent = list(self._history)[-self._config.rotation_cooldown :] if self._config.rotation_cooldown else []
        context["cooldown_ids"] = list(recent)
        skip_ids = set(recent)

        def find_alternative(additional_skip: Iterable[str] | None = None) -> Mapping[str, object] | None:
            ignored = set(skip_ids)
            if additional_skip:
                ignored.update(additional_skip)
            depth = 0
            for entry in ranked:
                if depth >= self._config.max_rotation_depth:
                    break
                candidate_id = self._extract_tactic_id(entry)
                if candidate_id is None:
                    depth += 1
                    continue
                if candidate_id in ignored:
                    depth += 1
                    continue
                return entry
            return None

        rotated = False
        rotation_reason: str | None = None
        skipped: list[str] = []

        if tactic_id in skip_ids:
            alternative = find_alternative(additional_skip=None)
            if alternative is not None:
                selected = alternative
                rotated = True
                rotation_reason = "cooldown_repeat"
                skipped.append(tactic_id)
        elif self._config.rotation_probability > 0.0:
            if self._rng.random() < self._config.rotation_probability:
                alternative = find_alternative(additional_skip={tactic_id})
                if alternative is not None:
                    selected = alternative
                    rotated = True
                    rotation_reason = "stochastic_rotation"
                    skipped.append(tactic_id)

        selected_id = self._extract_tactic_id(selected)
        self._remember(selected_id)

        context.update(
            {
                "rotation_applied": rotated,
                "rotation_reason": rotation_reason,
                "skipped_tactics": skipped,
                "selected_tactic": selected_id,
                "history_after": list(self._history),
            }
        )
        return selected, context

    def jitter_timestamp(
        self,
        timestamp: datetime,
    ) -> tuple[datetime, Mapping[str, object]]:
        reference = _coerce_datetime(timestamp)
        context: MutableMapping[str, object] = {
            "jitter_applied": False,
            "base_timestamp": reference.isoformat(),
        }

        maximum = max(self._config.maximum_jitter_ms, 0.0)
        minimum = max(self._config.minimum_jitter_ms, 0.0)

        if maximum <= 0.0:
            context["reason"] = "disabled"
            return reference, context

        if minimum > maximum:
            minimum = maximum

        magnitude = self._rng.uniform(minimum, maximum) if maximum > minimum else maximum
        direction = 1.0
        if self._config.allow_negative_jitter and self._rng.random() < 0.5:
            direction = -1.0

        offset_ms = direction * magnitude
        jittered = reference + timedelta(milliseconds=offset_ms)

        context.update(
            {
                "jitter_applied": bool(magnitude > 0.0),
                "offset_ms": offset_ms,
                "offset_seconds": offset_ms / 1000.0,
                "minimum_ms": minimum,
                "maximum_ms": maximum,
                "direction": "advance" if offset_ms < 0 else "delay",
                "timestamp": jittered.isoformat(),
            }
        )
        return jittered, context


__all__ = [
    "EntropyGovernor",
    "EntropyGovernorConfig",
]

