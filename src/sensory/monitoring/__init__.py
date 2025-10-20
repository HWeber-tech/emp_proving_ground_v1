"""Monitoring helpers for sensory output quality."""

from __future__ import annotations

from .sensor_drift import (
    SensorDriftBaseline,
    SensorDriftParameters,
    SensorDriftResult,
    SensorDriftSummary,
    evaluate_sensor_drift,
)
from .signal_roi import (
    SignalRoiContribution,
    SignalRoiMonitor,
    SignalRoiSummary,
    evaluate_signal_roi,
)

__all__ = [
    "LiveSensoryDiagnostics",
    "build_live_sensory_diagnostics",
    "build_live_sensory_diagnostics_from_manager",
    "OptionsSurfaceMonitor",
    "OptionsSurfaceMonitorConfig",
    "OptionsSurfaceSummary",
    "OpenInterestWall",
    "ImpliedVolSkewSnapshot",
    "DeltaImbalanceSnapshot",
    "SensorDriftBaseline",
    "SensorDriftParameters",
    "SensorDriftResult",
    "SensorDriftSummary",
    "evaluate_sensor_drift",
    "SignalRoiContribution",
    "SignalRoiMonitor",
    "SignalRoiSummary",
    "evaluate_signal_roi",
]


_LAZY_EXPORTS = {
    "LiveSensoryDiagnostics",
    "build_live_sensory_diagnostics",
    "build_live_sensory_diagnostics_from_manager",
    "OptionsSurfaceMonitor",
    "OptionsSurfaceMonitorConfig",
    "OptionsSurfaceSummary",
    "OpenInterestWall",
    "ImpliedVolSkewSnapshot",
    "DeltaImbalanceSnapshot",
}


def __getattr__(name: str):  # pragma: no cover - exercised via import machinery
    if name in _LAZY_EXPORTS:
        if name in {"LiveSensoryDiagnostics", "build_live_sensory_diagnostics", "build_live_sensory_diagnostics_from_manager"}:
            from . import live_diagnostics as _live

            return getattr(_live, name)
        from . import options_surface as _options_surface

        return getattr(_options_surface, name)
    raise AttributeError(f"module 'src.sensory.monitoring' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - import tooling helper
    return sorted(set(__all__))
