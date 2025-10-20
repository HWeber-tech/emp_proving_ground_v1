"""
Swing analysis feature functions for the WHAT dimension.

Pure, reusable utilities extracted from the monolithic pattern engine to
reduce coupling and aid unit testing.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import Timestamp as PandasTimestamp

try:  # SciPy is optional in runtime environments
    from scipy.signal import find_peaks as _scipy_find_peaks
except Exception:  # pragma: no cover - exercised via unit tests
    _scipy_find_peaks = None  # type: ignore[assignment]


def _fallback_find_peaks(
    values: NDArray[np.float64], distance: int = 1, prominence: float | None = None
) -> tuple[NDArray[np.intp], dict[str, NDArray[np.float64] | NDArray[np.intp]]]:
    array = np.asarray(values, dtype=float)
    length = array.size
    if length < 3:
        empty_peaks = np.empty(0, dtype=np.intp)
        props: dict[str, NDArray[np.float64] | NDArray[np.intp]] = {}
        if prominence is not None:
            props["prominences"] = np.empty(0, dtype=np.float64)
        return empty_peaks, props

    candidate_indices: list[int] = []
    for idx in range(1, length - 1):
        if array[idx] > array[idx - 1] and array[idx] > array[idx + 1]:
            candidate_indices.append(idx)

    if not candidate_indices:
        empty_peaks = np.empty(0, dtype=np.intp)
        props: dict[str, NDArray[np.float64] | NDArray[np.intp]] = {}
        if prominence is not None:
            props["prominences"] = np.empty(0, dtype=np.float64)
        return empty_peaks, props

    min_distance = max(int(distance), 1)
    if min_distance > 1 and len(candidate_indices) > 1:
        ordered = sorted(candidate_indices, key=lambda i: array[i], reverse=True)
        selected: list[int] = []
        for candidate in ordered:
            if all(abs(candidate - chosen) >= min_distance for chosen in selected):
                selected.append(candidate)
        candidate_indices = sorted(selected)
    else:
        candidate_indices.sort()

    props: dict[str, NDArray[np.float64] | NDArray[np.intp]] = {}
    if prominence is not None:
        filtered_indices: list[int] = []
        prominences: list[float] = []
        threshold = float(prominence)
        for idx in candidate_indices:
            left_min = array[: idx + 1].min() if idx > 0 else array[idx]
            right_min = array[idx:].min() if idx < length - 1 else array[idx]
            prominence_value = float(array[idx] - max(left_min, right_min))
            if prominence_value >= threshold:
                filtered_indices.append(idx)
                prominences.append(prominence_value)
        candidate_indices = filtered_indices
        props["prominences"] = np.asarray(prominences, dtype=np.float64)

    peaks_array = np.asarray(candidate_indices, dtype=np.intp)
    return peaks_array, props


def find_peaks(
    values: NDArray[np.float64], distance: int = 1, prominence: float | None = None
) -> tuple[NDArray[np.intp], dict[str, NDArray[np.float64] | NDArray[np.intp]]]:
    """
    SciPy-backed peak finder with a lightweight NumPy fallback for portability.
    """
    if _scipy_find_peaks is not None and "scipy" in getattr(_scipy_find_peaks, "__module__", ""):
        peaks, props = _scipy_find_peaks(values, distance=distance, prominence=prominence)

        normalised_peaks = np.asarray(peaks, dtype=np.intp)
        normalised_props: dict[str, NDArray[np.float64] | NDArray[np.intp]] = {}
        for key, value in props.items():
            arr = np.asarray(value)
            if np.issubdtype(arr.dtype, np.integer):
                normalised_props[key] = arr.astype(np.intp, copy=False)
            else:
                normalised_props[key] = arr.astype(np.float64, copy=False)

        return cast(NDArray[np.intp], normalised_peaks), cast(
            dict[str, NDArray[np.float64] | NDArray[np.intp]], normalised_props
        )

    return _fallback_find_peaks(values, distance=distance, prominence=prominence)


def identify_significant_swings(data: pd.DataFrame) -> list[dict[str, object]]:
    """Identify significant price swings from OHLCV data."""
    swings: list[dict[str, object]] = []

    # Find local maxima and minima
    highs = find_peaks(data["high"].to_numpy(dtype=float), distance=10, prominence=0.02)[0]
    lows = find_peaks(-data["low"].to_numpy(dtype=float), distance=10, prominence=0.02)[0]

    # Combine and sort points
    points: list[dict[str, object]] = []
    for idx in highs:
        points.append(
            {"time": data.index[idx], "price": float(data["high"].iloc[idx]), "type": "high"}
        )
    for idx in lows:
        points.append(
            {"time": data.index[idx], "price": float(data["low"].iloc[idx]), "type": "low"}
        )

    points.sort(key=lambda x: cast(PandasTimestamp, x["time"]))

    # Identify significant swings
    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]

        if prev["type"] != curr["type"]:
            # Protect against division by zero with explicit float casts
            prev_price = cast(float, prev["price"])
            curr_price = cast(float, curr["price"])
            denom = prev_price if prev_price != 0.0 else 1e-12
            price_change = abs(curr_price - prev_price) / denom
            if price_change > 0.05:  # 5% minimum swing
                swings.append(
                    {
                        "start_time": prev["time"],
                        "end_time": curr["time"],
                        "start_price": prev_price,
                        "end_price": curr_price,
                        "confidence": float(min(1.0, price_change * 10)),
                        "strength": float(price_change),
                    }
                )

    return swings


def calculate_fibonacci_levels(swing: dict[str, object]) -> list[float]:
    """Calculate Fibonacci retracement levels for a swing."""
    start_price = cast(float, swing["start_price"])
    end_price = cast(float, swing["end_price"])
    _ = abs(end_price - start_price)  # price_range (kept for parity, not used directly)
    levels: list[float] = []
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

    for ratio in fib_ratios:
        level = start_price + (end_price - start_price) * ratio
        levels.append(float(level))

    return levels


def identify_significant_moves(data: pd.DataFrame) -> list[dict[str, object]]:
    """Identify significant price moves for extension patterns.

    In this first extraction, significant moves are the same as significant swings.
    """
    return identify_significant_swings(data)


def calculate_extension_levels(move: dict[str, object]) -> list[float]:
    """Calculate Fibonacci extension levels for a move."""
    start_price = cast(float, move["start_price"])
    end_price = cast(float, move["end_price"])
    move_distance = abs(end_price - start_price)
    extensions: list[float] = []
    extension_ratios = [1.618, 2.618, 4.236, 6.854]

    direction = 1 if end_price > start_price else -1

    for ratio in extension_ratios:
        extension = end_price + (direction * move_distance * ratio)
        extensions.append(float(extension))

    return extensions
