"""Helpers for calibrating belief/regime components from real market data."""

from __future__ import annotations

import logging
import math
from typing import Mapping, Sequence

import pandas as pd

from src.understanding.belief_regime_calibrator import (
    BeliefRegimeCalibration,
    calibrate_belief_and_regime,
)

logger = logging.getLogger(__name__)


def price_series_from_frame(frame: pd.DataFrame) -> Sequence[float]:
    """Extract a numeric close/price series suitable for calibration."""

    if frame.empty:
        raise ValueError("market data frame is empty")

    for column in ("close", "price", "adj_close"):
        if column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            values = series.to_list()
            if len(values) >= 3:
                return values
    raise ValueError("market data must contain at least three valid close/price values")


def calibrate_from_market_data(frame: pd.DataFrame) -> BeliefRegimeCalibration | None:
    """Derive calibration parameters from a market data frame if possible."""

    try:
        prices = price_series_from_frame(frame)
    except ValueError as exc:  # pragma: no cover - defensive guard
        logger.warning("Skipping belief/regime calibration: %s", exc)
        return None

    try:
        return calibrate_belief_and_regime(prices)
    except ValueError as exc:  # pragma: no cover - calibration fallback path
        logger.warning("Belief/regime calibration failed: %s", exc)
        return None


def extract_snapshot_volatility(
    snapshot: Mapping[str, object],
    feature_name: str,
) -> float | None:
    """Pull a volatility sample for the configured feature from a sensory snapshot."""

    dimensions = snapshot.get("dimensions") if isinstance(snapshot, Mapping) else None
    if isinstance(dimensions, Mapping):
        dimension_key = feature_name.split("_", 1)[0].upper()
        payload = dimensions.get(dimension_key)
        if isinstance(payload, Mapping):
            candidate = payload.get("signal")
        else:
            candidate = getattr(payload, "signal", None)
        if candidate is not None:
            try:
                return abs(float(candidate))
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                return None
    return None


def resolve_threshold_scale(
    calibration: BeliefRegimeCalibration | None,
    sample: float | None,
    *,
    trigger: float = 10.0,
    max_scale: float = 1e4,
) -> float | None:
    """Scale regime thresholds when live volatility dwarfs the calibration baseline."""

    if calibration is None or sample is None:
        return None
    baseline = max(calibration.calm_threshold, 1e-8)
    try:
        sample_abs = abs(float(sample))
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None
    if not math.isfinite(sample_abs) or sample_abs == 0.0:
        return None
    ratio = sample_abs / baseline
    if ratio <= trigger:
        return None
    return float(min(ratio, max_scale))
