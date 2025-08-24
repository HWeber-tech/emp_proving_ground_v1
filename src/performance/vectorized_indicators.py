"""
Vectorized technical indicators for performance.
Moved out of performance package __init__ to avoid heavy imports at package import time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


class VectorizedIndicators:
    """Vectorized technical indicators for performance."""

    @staticmethod
    def sma(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
        """Simple Moving Average."""
        if data.shape[0] < period:
            return np.asarray([], dtype=float)
        return np.convolve(data, np.ones(period, dtype=float) / float(period), mode="valid").astype(
            float
        )

    @staticmethod
    def rsi(data: NDArray[np.float64], period: int = 14) -> NDArray[np.float64]:
        """Relative Strength Index."""
        if data.shape[0] < period + 1:
            return np.asarray([], dtype=float)

        deltas = np.diff(data)
        seed = deltas[: period + 1]
        up = float(seed[seed >= 0].sum()) / float(period) if period > 0 else 0.0
        down = -float(seed[seed < 0].sum()) / float(period) if period > 0 else 0.0
        rs = up / down if down != 0 else 100.0

        rsi_values: list[float] = [100.0 - 100.0 / (1.0 + rs)]

        for i in range(period + 1, int(data.shape[0])):
            delta = float(deltas[i - 1])
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta

            up = (up * (period - 1) + upval) / float(period)
            down = (down * (period - 1) + downval) / float(period)
            rs = up / down if down != 0 else 100.0
            rsi_values.append(100.0 - 100.0 / (1.0 + rs))

        return np.asarray(rsi_values, dtype=float)

    @staticmethod
    def bollinger_bands(
        data: NDArray[np.float64],
        period: int = 20,
        std_dev: float = 2.0,
    ) -> dict[str, NDArray[np.float64]]:
        """Bollinger Bands."""
        if data.shape[0] < period:
            empty = np.asarray([], dtype=float)
            return {"upper": empty, "middle": empty, "lower": empty}

        sma = VectorizedIndicators.sma(data, period)
        rolling_std = np.asarray(
            [float(np.std(data[i : i + period])) for i in range(data.shape[0] - period + 1)],
            dtype=float,
        )

        upper = sma + (rolling_std * std_dev)
        lower = sma - (rolling_std * std_dev)

        return {
            "upper": upper.astype(float),
            "middle": sma.astype(float),
            "lower": lower.astype(float),
        }

    @staticmethod
    def calculate_all_indicators(
        market_data: dict[str, object],
        indicators: Optional[list[str]] = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Calculate all requested indicators."""
        if indicators is None:
            indicators = ["sma", "rsi", "bb"]

        results: dict[str, NDArray[np.float64]] = {}
        close = np.asarray(market_data.get("close", []), dtype=float)

        if "sma" in indicators:
            results["sma_20"] = VectorizedIndicators.sma(close, 20)
            results["sma_50"] = VectorizedIndicators.sma(close, 50)

        if "rsi" in indicators:
            results["rsi_14"] = VectorizedIndicators.rsi(close, 14)

        if "bb" in indicators:
            bb = VectorizedIndicators.bollinger_bands(close)
            results["bb_upper"] = bb["upper"]
            results["bb_middle"] = bb["middle"]
            results["bb_lower"] = bb["lower"]

        return results
