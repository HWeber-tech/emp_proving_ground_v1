"""Daily class prior estimation utilities for imbalance-aware training.

This module exposes helpers for computing binary class priors on a daily cadence
without leaking future information.  The resulting ``pos_weight`` mirrors the
``BCEWithLogitsLoss`` expectation: ``pos_weight = negatives / positives`` based
solely on observations strictly before the day being weighted.  A light amount
of additive smoothing plus a configurable default keeps early days numerically
stable while still preventing future peeking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

__all__ = [
    "DailyClassPrior",
    "compute_daily_class_priors",
    "assign_daily_pos_weight",
]

_EPSILON = 1e-12


@dataclass(slots=True, frozen=True)
class DailyClassPrior:
    """Binary class prior derived from historical data before ``date``.

    Attributes
    ----------
    date:
        Normalised trading day associated with the prior.
    history_positive:
        Count of positive labels observed strictly before ``date``.
    history_negative:
        Count of negative labels observed strictly before ``date``.
    pos_weight:
        Weight applied to positive samples for that day, computed as
        ``(history_negative + smoothing) / (history_positive + smoothing)``
        using the smoothing provided to :func:`compute_daily_class_priors`.
    """

    date: pd.Timestamp
    history_positive: int
    history_negative: int
    pos_weight: float

    def as_dict(self) -> dict[str, object]:
        """Serialise the prior for downstream telemetry or testing."""

        return {
            "date": self.date.isoformat(),
            "history_positive": self.history_positive,
            "history_negative": self.history_negative,
            "pos_weight": self.pos_weight,
        }


def _coerce_binary(series: pd.Series) -> pd.Series:
    """Convert an arbitrary label series into {0, 1} without mutating input."""

    if series.empty:
        return pd.Series(dtype="int64")
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    # Treat strictly positive values as the positive class; everything else is 0.
    binary = (numeric > 0.0).astype("int64")
    return binary


def _validated_timestamps(frame: pd.DataFrame, column: str) -> pd.Series:
    timestamps = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if timestamps.isna().any():  # pragma: no cover - defensive guardrail
        raise ValueError(f"unable to coerce null/invalid timestamps from column '{column}'")
    return timestamps


def compute_daily_class_priors(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    label_col: str = "label",
    smoothing: float = 1.0,
    default_weight: float = 1.0,
) -> list[DailyClassPrior]:
    """Return chronological class priors with no access to future data.

    The caller supplies a frame containing at least a timestamp column and a
    binary label column.  For each trading day we aggregate label counts and
    derive a ``pos_weight`` using only the history strictly prior to that day.

    Parameters
    ----------
    frame:
        Source dataframe containing ``timestamp_col`` and ``label_col``.
    timestamp_col:
        Column holding event timestamps; coerced to UTC ``datetime64``.
    label_col:
        Column holding binary labels (any positive value is treated as ``1``).
    smoothing:
        Additive smoothing applied to both numerator and denominator.  Values
        must be non-negative.
    default_weight:
        Weight returned when no historical observations exist (e.g., the very
        first day with ``smoothing == 0``).  Must be positive.
    """

    if timestamp_col not in frame:
        raise KeyError(f"timestamp column '{timestamp_col}' missing from frame")
    if label_col not in frame:
        raise KeyError(f"label column '{label_col}' missing from frame")
    if smoothing < 0.0:
        raise ValueError("smoothing must be non-negative")
    if default_weight <= 0.0:
        raise ValueError("default_weight must be positive")

    if frame.empty:
        return []

    timestamps = _validated_timestamps(frame, timestamp_col)
    labels = _coerce_binary(frame[label_col])
    if len(labels) != len(frame):  # pragma: no cover - defensive guardrail
        raise ValueError("label column could not be aligned with frame index")

    dates = timestamps.dt.floor("D")
    positives = labels
    negatives = 1 - positives

    grouped = (
        pd.DataFrame({"date": dates, "positive": positives, "negative": negatives})
        .groupby("date", sort=True)[["positive", "negative"]]
        .sum()
    )

    grouped = grouped.astype("int64", copy=False)
    if grouped.empty:
        return []

    cumulative_positive = grouped["positive"].cumsum().shift(fill_value=0).astype("int64")
    cumulative_negative = grouped["negative"].cumsum().shift(fill_value=0).astype("int64")

    priors: list[DailyClassPrior] = []
    for date, row in grouped.iterrows():
        history_positive = int(cumulative_positive.loc[date])
        history_negative = int(cumulative_negative.loc[date])

        if history_positive == 0 and history_negative == 0 and smoothing == 0.0:
            weight = float(default_weight)
        else:
            denominator = history_positive + smoothing
            if denominator <= 0.0:
                denominator = _EPSILON
            weight = float((history_negative + smoothing) / denominator)

        priors.append(
            DailyClassPrior(
                date=pd.Timestamp(date),
                history_positive=history_positive,
                history_negative=history_negative,
                pos_weight=weight,
            )
        )

    return priors


def assign_daily_pos_weight(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    label_col: str = "label",
    smoothing: float = 1.0,
    default_weight: float = 1.0,
) -> pd.Series:
    """Map the daily class priors onto each row of ``frame``.

    The returned series is aligned with ``frame.index`` and carries the
    per-sample ``pos_weight`` appropriate for the sample's trading day.
    """

    priors = compute_daily_class_priors(
        frame,
        timestamp_col=timestamp_col,
        label_col=label_col,
        smoothing=smoothing,
        default_weight=default_weight,
    )

    if not priors:
        return pd.Series(index=frame.index, dtype="float64", name="pos_weight")

    weight_by_date = {prior.date: prior.pos_weight for prior in priors}
    timestamps = _validated_timestamps(frame, timestamp_col)
    dates = timestamps.dt.floor("D")

    weights: list[float] = []
    for date in dates:
        weight = weight_by_date.get(date)
        if weight is None:
            weight = float(default_weight)
        weights.append(weight)

    return pd.Series(weights, index=frame.index, dtype="float64", name="pos_weight")
