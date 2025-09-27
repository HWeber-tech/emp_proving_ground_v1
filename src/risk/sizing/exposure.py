"""Utilities for sector and asset-class exposure limits."""

from __future__ import annotations

from typing import Mapping

__all__ = ["compute_classified_exposure", "check_classification_limits"]


def compute_classified_exposure(
    exposures: Mapping[str, float],
    classifications: Mapping[str, Mapping[str, str]],
    *,
    classification_key: str,
) -> dict[str, float]:
    """Aggregate absolute exposure by the requested classification key."""

    totals: dict[str, float] = {}
    for symbol, value in exposures.items():
        if symbol not in classifications:
            continue
        classification = classifications[symbol]
        bucket = classification.get(classification_key)
        if not bucket:
            continue
        amount = abs(float(value))
        totals[bucket] = totals.get(bucket, 0.0) + amount
    return totals


def check_classification_limits(
    exposures: Mapping[str, float],
    classifications: Mapping[str, Mapping[str, str]],
    limits: Mapping[str, float],
    *,
    classification_key: str,
) -> dict[str, dict[str, float]]:
    """Return breaches for the supplied classification limits."""

    if not limits:
        return {}

    totals = compute_classified_exposure(
        exposures, classifications, classification_key=classification_key
    )

    breaches: dict[str, dict[str, float]] = {}
    for bucket, limit in limits.items():
        if limit is None:
            continue
        exposure = totals.get(bucket, 0.0)
        limit_value = float(limit)
        if limit_value <= 0:
            continue
        if exposure > limit_value:
            breaches[bucket] = {"exposure": exposure, "limit": limit_value}
    return breaches

