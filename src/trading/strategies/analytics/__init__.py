"""Analytics utilities for evaluating trading strategy performance."""

from __future__ import annotations

from .performance_attribution import (
    AttributionResult,
    FeatureContribution,
    compute_performance_attribution,
    result_to_dataframe,
)

__all__ = [
    "AttributionResult",
    "FeatureContribution",
    "compute_performance_attribution",
    "result_to_dataframe",
]
