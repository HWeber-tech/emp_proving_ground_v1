"""
Swing analysis feature functions for the WHAT dimension.

Pure, reusable utilities extracted from the monolithic pattern engine to
reduce coupling and aid unit testing.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Strict SciPy-backed implementation (no functional fallback).
from scipy.signal import find_peaks as _scipy_find_peaks  # type: ignore


def find_peaks(values: np.ndarray, distance: int = 1, prominence: float | None = None) -> Tuple[np.ndarray, Dict]:
    """
    Strict SciPy-backed peak finder. Raises ImportError if SciPy is unavailable.
    """
    return _scipy_find_peaks(values, distance=distance, prominence=prominence)


def identify_significant_swings(data: pd.DataFrame) -> List[Dict]:
    """Identify significant price swings from OHLCV data."""
    swings: List[Dict] = []

    # Find local maxima and minima
    highs = find_peaks(data["high"].values, distance=10, prominence=0.02)[0]
    lows = find_peaks(-data["low"].values, distance=10, prominence=0.02)[0]

    # Combine and sort points
    points: List[Dict] = []
    for idx in highs:
        points.append(
            {"time": data.index[idx], "price": float(data["high"].iloc[idx]), "type": "high"}
        )
    for idx in lows:
        points.append(
            {"time": data.index[idx], "price": float(data["low"].iloc[idx]), "type": "low"}
        )

    points.sort(key=lambda x: x["time"])

    # Identify significant swings
    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]

        if prev["type"] != curr["type"]:
            # Protect against division by zero
            denom = prev["price"] if prev["price"] else 1e-12
            price_change = abs(curr["price"] - prev["price"]) / denom
            if price_change > 0.05:  # 5% minimum swing
                swings.append(
                    {
                        "start_time": prev["time"],
                        "end_time": curr["time"],
                        "start_price": prev["price"],
                        "end_price": curr["price"],
                        "confidence": float(min(1.0, price_change * 10)),
                        "strength": float(price_change),
                    }
                )

    return swings


def calculate_fibonacci_levels(swing: Dict) -> List[float]:
    """Calculate Fibonacci retracement levels for a swing."""
    start_price = float(swing["start_price"])
    end_price = float(swing["end_price"])
    _ = abs(end_price - start_price)  # price_range (kept for parity, not used directly)

    levels: List[float] = []
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

    for ratio in fib_ratios:
        level = start_price + (end_price - start_price) * ratio
        levels.append(float(level))

    return levels


def identify_significant_moves(data: pd.DataFrame) -> List[Dict]:
    """Identify significant price moves for extension patterns.

    In this first extraction, significant moves are the same as significant swings.
    """
    return identify_significant_swings(data)


def calculate_extension_levels(move: Dict) -> List[float]:
    """Calculate Fibonacci extension levels for a move."""
    start_price = float(move["start_price"])
    end_price = float(move["end_price"])
    move_distance = abs(end_price - start_price)

    extensions: List[float] = []
    extension_ratios = [1.618, 2.618, 4.236, 6.854]

    direction = 1 if end_price > start_price else -1

    for ratio in extension_ratios:
        extension = end_price + (direction * move_distance * ratio)
        extensions.append(float(extension))

    return extensions