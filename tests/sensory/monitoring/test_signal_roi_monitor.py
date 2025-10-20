from __future__ import annotations

import numpy as np
import pytest

from src.sensory.monitoring import SignalRoiMonitor, evaluate_signal_roi


def test_signal_roi_monitor_ranks_high_value_streams() -> None:
    rng = np.random.default_rng(123)
    primary = rng.normal(size=512)
    secondary = rng.normal(size=512)
    noise = rng.normal(size=512)
    target = 1.8 * primary + 0.6 * secondary + 0.1 * rng.normal(size=512)

    streams = {
        "primary": primary,
        "secondary": secondary,
        "noise": noise,
    }

    monitor = SignalRoiMonitor(regularisation=1e-6)
    summary = monitor.evaluate(target, streams)

    assert summary.samples == len(target)
    assert summary.r_squared > 0.85
    ranked_streams = [contrib.stream for contrib in summary.contributions]
    assert ranked_streams[0] == "primary"
    assert summary.contributions[0].marginal_r2 > summary.contributions[1].marginal_r2

    positive_shares = [
        share for share in (contrib.share_of_gain for contrib in summary.contributions) if share is not None
    ]
    assert positive_shares
    assert sum(positive_shares) == pytest.approx(1.0, abs=1e-6)


def test_signal_roi_drops_nan_rows() -> None:
    target = np.array([0.5, 1.0, np.nan, 1.5, 2.0])
    streams = {
        "alpha": np.array([1.0, 2.0, 3.0, np.nan, 5.0]),
        "beta": np.array([0.1, 0.2, 0.3, 0.4, np.nan]),
    }

    summary = evaluate_signal_roi(target, streams)

    assert summary.samples == 2
    assert len(summary.contributions) == 2
    assert summary.r_squared >= 0.0


def test_signal_roi_validation() -> None:
    with pytest.raises(ValueError):
        SignalRoiMonitor(regularisation=-0.1)

    monitor = SignalRoiMonitor()
    with pytest.raises(ValueError):
        monitor.evaluate([0.1, 0.2, 0.3], {})

    with pytest.raises(ValueError):
        evaluate_signal_roi([1.0, 2.0], {"x": [1.0, 2.0, 3.0]})
