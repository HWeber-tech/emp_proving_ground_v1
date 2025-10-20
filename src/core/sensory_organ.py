"""Canonical sensory organ exports for the public ``src.core`` API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        try:
            return SensoryDriftConfig(**dict(candidate))
        except (TypeError, ValueError) as error:
            raise TypeError(
                "drift_config sequence must contain key/value pairs convertible to SensoryDriftConfig"
            ) from error
    raise TypeError(
        "drift_config must be None, a SensoryDriftConfig, a mapping, or a sequence of overrides"
    )


def create_sensory_organ(*, drift_config: object | None = None, **kwargs: Any) -> RealSensoryOrgan:
    """Factory that mirrors the historical shim while delegating to the real organ."""

    normalised_config = _coerce_drift_config(drift_config)
    return RealSensoryOrgan(drift_config=normalised_config, **kwargs)
