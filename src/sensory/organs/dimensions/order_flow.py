from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


_REQUIRED_COLUMNS = ("high", "low", "close")


def _coerce_dataframe(data: object) -> pd.DataFrame:
    """Return a sanitized DataFrame with the expected OHLC columns."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, Iterable):
        frame = pd.DataFrame(data)
    else:  # pragma: no cover - defensive guardrail
        raise TypeError("Order flow analysis expects a DataFrame or iterable input")

    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for order flow analysis: {missing}")

    if "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp")
    frame = frame.reset_index(drop=True)

    numeric_columns = [column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


@dataclass(frozen=True)
class _MicrostructureResult:
    microstructure_score: float
    bullish_fvg: int
    bearish_fvg: int
    average_gap_size: float
    buy_sweeps: int
    sell_sweeps: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "microstructure_score": float(self.microstructure_score),
            "fair_value_gaps": {
                "bullish_count": int(self.bullish_fvg),
                "bearish_count": int(self.bearish_fvg),
                "average_gap_size": float(self.average_gap_size),
            },
            "liquidity_sweeps": {
                "buy_side": int(self.buy_sweeps),
                "sell_side": int(self.sell_sweeps),
            },
        }


class MarketMicrostructureAnalyzer:
    """Detect ICT-style market microstructure patterns from OHLC data."""

    def __init__(self, *, minimum_candles: int = 3) -> None:
        self._minimum_candles = max(minimum_candles, 3)

    # ------------------------------------------------------------------
    def analyze_microstructure(self, df: object) -> dict[str, Any]:
        """Surface fair value gaps and liquidity sweeps from price history."""

        frame = _coerce_dataframe(df)
        if len(frame) < self._minimum_candles:
            return _MicrostructureResult(0.0, 0, 0, 0.0, 0, 0).as_dict()

        bullish_gaps: list[float] = []
        bearish_gaps: list[float] = []
        buy_sweeps = 0
        sell_sweeps = 0

        highs = frame["high"].tolist()
        lows = frame["low"].tolist()
        closes = frame["close"].tolist()

        for idx in range(2, len(frame)):
            current_low = lows[idx]
            current_high = highs[idx]
            prior_high = highs[idx - 2]
            prior_low = lows[idx - 2]

            if pd.notna(current_low) and pd.notna(prior_high) and float(current_low) > float(prior_high):
                bullish_gaps.append(float(current_low) - float(prior_high))

            if pd.notna(current_high) and pd.notna(prior_low) and float(current_high) < float(prior_low):
                bearish_gaps.append(float(prior_low) - float(current_high))

        for idx in range(1, len(frame)):
            if pd.notna(highs[idx]) and pd.notna(highs[idx - 1]) and pd.notna(closes[idx]) and pd.notna(closes[idx - 1]):
                if float(highs[idx]) > float(highs[idx - 1]) and float(closes[idx]) < float(closes[idx - 1]):
                    buy_sweeps += 1

            if pd.notna(lows[idx]) and pd.notna(lows[idx - 1]) and pd.notna(closes[idx]) and pd.notna(closes[idx - 1]):
                if float(lows[idx]) < float(lows[idx - 1]) and float(closes[idx]) > float(closes[idx - 1]):
                    sell_sweeps += 1

        total_events = len(bullish_gaps) + len(bearish_gaps) + buy_sweeps + sell_sweeps
        if total_events == 0:
            score = 0.0
        else:
            directional = (len(bullish_gaps) + buy_sweeps) - (len(bearish_gaps) + sell_sweeps)
            score = directional / float(total_events)

        all_gaps = bullish_gaps + bearish_gaps
        average_gap = sum(all_gaps) / len(all_gaps) if all_gaps else 0.0

        return _MicrostructureResult(score, len(bullish_gaps), len(bearish_gaps), average_gap, buy_sweeps, sell_sweeps).as_dict()


class OrderFlowAnalyzer:
    """Aggregate institutional order flow bias from microstructure signals."""

    def __init__(self, *, microstructure: MarketMicrostructureAnalyzer | None = None) -> None:
        self._microstructure = microstructure or MarketMicrostructureAnalyzer()

    # ------------------------------------------------------------------
    def analyze_institutional_flow(self, df: object) -> dict[str, Any]:
        """Summarise directional liquidity sweeps and fair value gap bias."""

        frame = _coerce_dataframe(df)
        if len(frame) < 2:
            return {
                "flow_strength": 0.0,
                "institutional_pressure": {"buying_pressure": 0.0, "selling_pressure": 0.0},
                "microstructure": _MicrostructureResult(0.0, 0, 0, 0.0, 0, 0).as_dict(),
            }

        microstructure = self._microstructure.analyze_microstructure(frame)
        sweeps = microstructure.get("liquidity_sweeps", {})
        fvg = microstructure.get("fair_value_gaps", {})

        buy_sweeps = int(sweeps.get("buy_side", 0))
        sell_sweeps = int(sweeps.get("sell_side", 0))
        total_sweeps = buy_sweeps + sell_sweeps

        bullish_fvg = int(fvg.get("bullish_count", 0))
        bearish_fvg = int(fvg.get("bearish_count", 0))
        total_gaps = bullish_fvg + bearish_fvg

        sweep_bias = (buy_sweeps - sell_sweeps) / float(total_sweeps) if total_sweeps else 0.0
        gap_bias = (bullish_fvg - bearish_fvg) / float(total_gaps) if total_gaps else 0.0

        flow_bias = 0.5 * sweep_bias + 0.5 * gap_bias
        flow_strength = abs(flow_bias)

        institutional_pressure = {
            "buying_pressure": float(max(flow_bias, 0.0)),
            "selling_pressure": float(max(-flow_bias, 0.0)),
        }

        return {
            "flow_strength": float(flow_strength),
            "institutional_pressure": institutional_pressure,
            "microstructure": microstructure,
        }


__all__ = ["MarketMicrostructureAnalyzer", "OrderFlowAnalyzer"]
