"""Monitoring helpers for sensory output quality."""

from __future__ import annotations

from .sensor_drift import (
    SensorDriftBaseline,
    SensorDriftParameters,
    SensorDriftResult,
    SensorDriftSummary,
    evaluate_sensor_drift,
)

__all__ = [
    "LiveSensoryDiagnostics",
    "build_live_sensory_diagnostics",
    "build_live_sensory_diagnostics_from_manager",
    "SensorDriftBaseline",
    "SensorDriftParameters",
    "SensorDriftResult",
    "SensorDriftSummary",
    "evaluate_sensor_drift",
]


_LAZY_EXPORTS = {
    "LiveSensoryDiagnostics",
    "build_live_sensory_diagnostics",
    "build_live_sensory_diagnostics_from_manager",
}


def __getattr__(name: str):  # pragma: no cover - exercised via import machinery
    if name in _LAZY_EXPORTS:
        from . import live_diagnostics as _live

        return getattr(_live, name)
    raise AttributeError(f"module 'src.sensory.monitoring' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - import tooling helper
    return sorted(set(__all__))
