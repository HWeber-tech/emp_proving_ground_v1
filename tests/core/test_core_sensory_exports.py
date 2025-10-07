"""Validate core sensory exports align with the canonical sensory organ."""

from __future__ import annotations

import importlib


def test_core_exposes_real_sensory_organ():
    core = importlib.import_module("src.core")
    reloaded = importlib.reload(core)

    from src.sensory.real_sensory_organ import RealSensoryOrgan, SensoryDriftConfig

    assert reloaded.SensoryOrgan is RealSensoryOrgan

    organ = reloaded.create_sensory_organ()
    assert isinstance(organ, RealSensoryOrgan)

    tuned = reloaded.create_sensory_organ(drift_config={"baseline_window": 42})
    assert isinstance(tuned, RealSensoryOrgan)
    assert tuned._drift_config.baseline_window == 42  # type: ignore[attr-defined]
    assert isinstance(tuned._drift_config, SensoryDriftConfig)  # type: ignore[attr-defined]

