"""Environment-aware feature toggles for adaptation components.

This module wires the roadmap requirement for environment-scoped feature flags
covering fast-weights, linear attention, and exploration knobs.  It resolves a
`SystemConfig` snapshot into explicit booleans that downstream orchestrators can
consult when constructing belief snapshots, routing decisions, or enabling
evolution trials.  Operators can override the defaults via `SystemConfig.extras`
so CI, staging, and production environments expose deterministic behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from src.governance.system_config import EmpEnvironment, EmpTier, RunMode, SystemConfig


_TRUTHY = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSY = {"0", "false", "no", "n", "off", "disable", "disabled"}

FAST_WEIGHTS_FLAG = "fast_weights_live"
LINEAR_ATTENTION_FLAG = "linear_attention_live"
EXPLORATION_FLAG = "exploration_live"


def _coerce_optional_bool(value: object | None) -> bool | None:
    """Convert loose flag values into ``bool`` while allowing ``None``."""

    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return None


@dataclass(frozen=True)
class AdaptationFeatureToggles:
    """Resolved booleans guarding adaptation features for a runtime deployment."""

    fast_weights: bool | None = None
    linear_attention: bool | None = None
    exploration: bool | None = None

    def as_feature_flags(self) -> dict[str, bool]:
        """Return a mapping suitable for propagation on belief snapshots."""

        flags: dict[str, bool] = {}
        if self.fast_weights is not None:
            flags[FAST_WEIGHTS_FLAG] = self.fast_weights
        if self.linear_attention is not None:
            flags[LINEAR_ATTENTION_FLAG] = self.linear_attention
        if self.exploration is not None:
            flags[EXPLORATION_FLAG] = self.exploration
        return flags

    def merge_flags(self, snapshot_flags: Mapping[str, bool] | None) -> dict[str, bool]:
        """Merge environment defaults with snapshot-provided feature flags."""

        merged = dict(self.as_feature_flags())
        if snapshot_flags:
            for key, value in snapshot_flags.items():
                merged[str(key)] = bool(value)
        return merged

    def resolve_fast_weights_enabled(self, explicit: bool | None) -> bool:
        """Derive the fast-weight enabled flag for a belief snapshot."""

        if explicit is not None:
            return explicit
        if self.fast_weights is not None:
            return self.fast_weights
        return True

    def with_overrides(
        self,
        *,
        fast_weights: bool | None = None,
        linear_attention: bool | None = None,
        exploration: bool | None = None,
    ) -> "AdaptationFeatureToggles":
        """Return a new instance with selective overrides applied."""

        return replace(
            self,
            fast_weights=self.fast_weights if fast_weights is None else fast_weights,
            linear_attention=
            self.linear_attention if linear_attention is None else linear_attention,
            exploration=self.exploration if exploration is None else exploration,
        )

    @classmethod
    def from_system_config(
        cls,
        config: SystemConfig,
        *,
        extras: Mapping[str, object] | None = None,
    ) -> "AdaptationFeatureToggles":
        """Resolve feature toggles from ``SystemConfig`` plus optional overrides."""

        extras_mapping: Mapping[str, object] = extras or config.extras or {}

        defaults = _environment_defaults(config)

        if config.run_mode is RunMode.live:
            # Live trading enforces the conservative posture regardless of overrides.
            return defaults

        fast_override = _coerce_optional_bool(extras_mapping.get("FEATURE_FAST_WEIGHTS"))
        linear_override = _coerce_optional_bool(extras_mapping.get("FEATURE_LINEAR_ATTENTION"))
        exploration_override = _coerce_optional_bool(
            extras_mapping.get("FEATURE_EXPLORATION")
        )

        toggles = defaults.with_overrides(
            fast_weights=fast_override,
            linear_attention=linear_override,
            exploration=exploration_override,
        )

        return toggles


def _environment_defaults(config: SystemConfig) -> AdaptationFeatureToggles:
    """Map runtime posture (environment, run mode, tier) into feature defaults."""

    env_defaults: dict[EmpEnvironment, AdaptationFeatureToggles] = {
        EmpEnvironment.demo: AdaptationFeatureToggles(
            fast_weights=False,
            linear_attention=False,
            exploration=False,
        ),
        EmpEnvironment.staging: AdaptationFeatureToggles(
            fast_weights=True,
            linear_attention=True,
            exploration=False,
        ),
        EmpEnvironment.production: AdaptationFeatureToggles(
            fast_weights=False,
            linear_attention=False,
            exploration=False,
        ),
    }

    toggles = env_defaults.get(config.environment, AdaptationFeatureToggles())

    if config.run_mode is RunMode.live:
        return AdaptationFeatureToggles(fast_weights=False, linear_attention=False, exploration=False)

    if config.tier is EmpTier.tier_0:
        toggles = toggles.with_overrides(fast_weights=False)

    return toggles


__all__ = [
    "AdaptationFeatureToggles",
    "FAST_WEIGHTS_FLAG",
    "LINEAR_ATTENTION_FLAG",
    "EXPLORATION_FLAG",
]
