"""ICT-style microstructure pattern detection utilities.

This module implements a light-weight detector for two staples from the
Inner Circle Trader (ICT) playbook:

* **Fair Value Gaps (FVGs)** – three-candle displacement structures where
  the middle candle creates an inefficiency between the first and third
  candles.
* **Liquidity Sweeps** – stop runs where price wicks beyond a recent swing
  high/low but fails to close through it.

The goal is to surface the patterns as structured telemetry that can be
consumed by the HOW sensory organ, execution modules, or strategies without
requiring heavy numeric dependencies. The implementation intentionally
follows pragmatic heuristics that are robust to noisy retail data feeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "ICTPatternAnalyzerConfig",
    "ICTPatternSnapshot",
    "ICTPatternAnalyzer",
]


@dataclass(slots=True)
class ICTPatternAnalyzerConfig:
    """Configuration surface for ICT pattern detection heuristics."""

    min_gap_fraction: float = 0.0005
    """Minimum fractional gap (relative to price) to flag a fair value gap."""

    sweep_lookback: int = 8
    """How many prior candles to inspect when checking for liquidity sweeps."""

    wick_ratio_threshold: float = 0.6
    """Minimum wick/total range ratio to classify a sweep displacement."""


@dataclass(slots=True)
class ICTPatternSnapshot:
    """Structured representation of ICT pattern detections."""

    bullish_fvg: bool
    bearish_fvg: bool
    bullish_gap_size: float
    bearish_gap_size: float
    liquidity_sweep_up: bool
    liquidity_sweep_down: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "bullish_fvg": bool(self.bullish_fvg),
            "bearish_fvg": bool(self.bearish_fvg),
            "bullish_gap_size": float(self.bullish_gap_size),
            "bearish_gap_size": float(self.bearish_gap_size),
            "liquidity_sweep_up": bool(self.liquidity_sweep_up),
            "liquidity_sweep_down": bool(self.liquidity_sweep_down),
        }


class ICTPatternAnalyzer:
    """Detect ICT-inspired price dislocations on OHLC candles."""

    def __init__(self, config: ICTPatternAnalyzerConfig | None = None) -> None:
        self._config = config or ICTPatternAnalyzerConfig()

    def evaluate(self, candles: pd.DataFrame | None) -> ICTPatternSnapshot | None:
        """Return an :class:`ICTPatternSnapshot` from the supplied candles."""

        if candles is None or candles.empty:
            return None

        required_columns = {"open", "high", "low", "close"}
        frame = candles.dropna(subset=required_columns, how="any")
        if len(frame) < 3:
            return None

        bullish_fvg, bullish_gap = self._detect_bullish_fvg(frame)
        bearish_fvg, bearish_gap = self._detect_bearish_fvg(frame)
        sweep_up, sweep_down = self._detect_liquidity_sweeps(frame)

        return ICTPatternSnapshot(
            bullish_fvg=bullish_fvg,
            bearish_fvg=bearish_fvg,
            bullish_gap_size=bullish_gap,
            bearish_gap_size=bearish_gap,
            liquidity_sweep_up=sweep_up,
            liquidity_sweep_down=sweep_down,
        )

    # Fair value gap detection -------------------------------------------------
    def _detect_bullish_fvg(self, frame: pd.DataFrame) -> tuple[bool, float]:
        last_three = frame.iloc[-3:]
        first = last_three.iloc[0]
        middle = last_three.iloc[1]
        last = last_three.iloc[2]

        min_gap = self._minimum_gap_threshold(middle)
        gap = float(last["low"] - first["high"])
        body_bias = float(middle["close"] - middle["open"])
        bullish = gap > min_gap and body_bias > 0
        return bullish, gap if gap > 0 else 0.0

    def _detect_bearish_fvg(self, frame: pd.DataFrame) -> tuple[bool, float]:
        last_three = frame.iloc[-3:]
        first = last_three.iloc[0]
        middle = last_three.iloc[1]
        last = last_three.iloc[2]

        min_gap = self._minimum_gap_threshold(middle)
        gap = float(first["low"] - last["high"])
        body_bias = float(middle["close"] - middle["open"])
        bearish = gap > min_gap and body_bias < 0
        return bearish, gap if gap > 0 else 0.0

    def _minimum_gap_threshold(self, candle: pd.Series) -> float:
        reference_price = float((candle["open"] + candle["close"]) / 2)
        return abs(reference_price) * max(self._config.min_gap_fraction, 0.0)

    # Liquidity sweeps --------------------------------------------------------
    def _detect_liquidity_sweeps(self, frame: pd.DataFrame) -> tuple[bool, bool]:
        lookback = max(2, int(self._config.sweep_lookback))
        window = frame.iloc[-lookback:]
        if len(window) < 2:
            return (False, False)

        recent = window.iloc[-1]
        history = window.iloc[:-1]

        prior_high = float(history["high"].max())
        prior_low = float(history["low"].min())

        sweep_up = self._is_liquidity_sweep_up(recent, prior_high)
        sweep_down = self._is_liquidity_sweep_down(recent, prior_low)
        return sweep_up, sweep_down

    def _is_liquidity_sweep_up(self, candle: pd.Series, prior_high: float) -> bool:
        if candle["high"] <= prior_high:
            return False
        close_below_prior = candle["close"] < prior_high
        wick_length = float(candle["high"] - max(candle["open"], candle["close"]))
        total_range = float(candle["high"] - candle["low"])
        wick_ratio = wick_length / total_range if total_range > 0 else 0.0
        return close_below_prior and wick_ratio >= self._config.wick_ratio_threshold

    def _is_liquidity_sweep_down(self, candle: pd.Series, prior_low: float) -> bool:
        if candle["low"] >= prior_low:
            return False
        close_above_prior = candle["close"] > prior_low
        wick_length = float(min(candle["open"], candle["close"]) - candle["low"])
        total_range = float(candle["high"] - candle["low"])
        wick_ratio = wick_length / total_range if total_range > 0 else 0.0
        return close_above_prior and wick_ratio >= self._config.wick_ratio_threshold
