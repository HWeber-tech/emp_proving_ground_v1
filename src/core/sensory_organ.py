"""Canonical sensory organ exports for the public ``src.core`` API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.sensory.real_sensory_organ import RealSensoryOrgan, SensoryDriftConfig

SensoryOrgan = RealSensoryOrgan

__all__ = [
    "SensoryOrgan",
    "RealSensoryOrgan",
    "SensoryDriftConfig",
    "create_sensory_organ",
]


def _coerce_drift_config(candidate: object | None) -> SensoryDriftConfig | None:
    """Normalise drift configuration inputs used by legacy callers."""

    if candidate is None or isinstance(candidate, SensoryDriftConfig):
        return candidate
    if isinstance(candidate, Mapping):
        return SensoryDriftConfig(**dict(candidate))
    raise TypeError(
        "drift_config must be None, a SensoryDriftConfig, or a mapping of overrides"
    )


def create_sensory_organ(*, drift_config: object | None = None, **kwargs: Any) -> RealSensoryOrgan:
    """Factory that mirrors the historical shim while delegating to the real organ."""

    normalised_config = _coerce_drift_config(drift_config)
    return RealSensoryOrgan(drift_config=normalised_config, **kwargs)

