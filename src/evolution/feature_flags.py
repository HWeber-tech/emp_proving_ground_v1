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


@dataclass(frozen=True, slots=True)
class AdaptiveRunDecision:
    """Decision metadata describing whether adaptive runs are enabled."""

    enabled: bool
    source: str
    raw_value: object | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"enabled": self.enabled, "source": self.source}
        if self.reason:
            payload["reason"] = self.reason
        if self.raw_value is not None:
            payload["raw_value"] = self.raw_value
        return payload


@dataclass(slots=True)
class EvolutionFeatureFlags:
    """Resolve evolution feature flags with optional overrides for tests."""

    env: Mapping[str, str] | None = None

    def adaptive_runs_enabled(self, override: bool | None = None) -> bool:
        """Return whether adaptive evolution runs are enabled."""

        return self.adaptive_runs_decision(override=override).enabled

    def adaptive_runs_decision(
        self, override: bool | None = None
    ) -> AdaptiveRunDecision:
        """Return the gating decision for adaptive evolution runs."""

        if override is not None:
            enabled = bool(override)
            reason = "override_enabled" if enabled else "override_disabled"
            return AdaptiveRunDecision(
                enabled=enabled,
                source="override",
                raw_value=override,
                reason=reason,
            )

        source = self.env if self.env is not None else os.environ
        raw = source.get(ADAPTIVE_RUNS_FLAG)
        if raw is None:
            return AdaptiveRunDecision(
                enabled=False,
                source="environment",
                raw_value=None,
                reason="flag_missing",
            )

        enabled = _coerce_bool(raw)
        reason = "flag_enabled" if enabled else "flag_disabled"
        return AdaptiveRunDecision(
            enabled=enabled,
            source="environment",
            raw_value=raw,
            reason=reason,
        )


__all__ = ["AdaptiveRunDecision", "EvolutionFeatureFlags", "ADAPTIVE_RUNS_FLAG"]
