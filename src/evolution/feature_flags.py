"""Feature flag helpers for evolution experiments and adaptive runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

ADAPTIVE_RUNS_FLAG = "EVOLUTION_ENABLE_ADAPTIVE_RUNS"
_TRUTHY = {"1", "true", "yes", "on", "enable", "enabled"}


def _coerce_bool(value: object) -> bool:
    """Normalise loose flag values into a strict boolean."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return False


@dataclass(slots=True)
class EvolutionFeatureFlags:
    """Resolve evolution feature flags with optional overrides for tests."""

    env: Mapping[str, str] | None = None

    def adaptive_runs_enabled(self, override: bool | None = None) -> bool:
        """Return whether adaptive evolution runs are enabled."""

        if override is not None:
            return bool(override)

        source = self.env if self.env is not None else os.environ
        raw = source.get(ADAPTIVE_RUNS_FLAG)
        return _coerce_bool(raw)


__all__ = ["EvolutionFeatureFlags", "ADAPTIVE_RUNS_FLAG"]
