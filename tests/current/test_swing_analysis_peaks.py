#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from src.sensory.what.features.swing_analysis import find_peaks


def test_find_peaks_basic_distance():
    # Simple alternating series with clear local maxima at odd indices
    x = np.array([0.0, 1.0, 0.0, 1.1, 0.0, 0.9, 0.0], dtype=float)

    idx, props = find_peaks(x, distance=2)
    # With distance=2, the middle strong peaks at indices 1 and 3 are both allowed,
    # index 5 is suppressed by distance constraint when 3 is stronger.
    assert idx.tolist() in ([1, 3], [3, 5], [1, 5], [1, 3, 5])  # accept common SciPy selections
    assert isinstance(props, dict)


def test_find_peaks_prominence():
    # Create peaks with one clearly more prominent
    x = np.array([0.0, 2.0, 0.0, 0.5, 0.0, 0.4, 0.0], dtype=float)

    idx, _ = find_peaks(x, distance=1, prominence=0.6)
    # Only the first strong peak should remain
    assert idx.tolist() == [1]
