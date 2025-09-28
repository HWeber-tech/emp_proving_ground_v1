from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import pandas as pd
import pytest

from src.data_foundation.ingest.multi_source import (
    CoverageValidator,
    CrossSourceDriftValidator,
    DataQualitySeverity,
    MultiSourceAggregator,
    ProviderSpec,
    StalenessValidator,
)


def _frame(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_fetch(data: pd.DataFrame):
    def _fetch(symbols: Sequence[str], start: datetime, end: datetime) -> pd.DataFrame:
        _ = symbols, start, end  # unused in deterministic fixture
        return data.copy()

    return _fetch


def test_multi_source_aggregator_merges_providers() -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)

    yahoo = _frame(
        [
            {
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "symbol": "eurusd",
                "open": 1.05,
                "high": 1.06,
                "low": 1.04,
                "close": 1.055,
                "volume": 1_000,
            },
            {
                "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "symbol": "eurusd",
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
            },
        ]
    )

    alpha = _frame(
        [
            {
                "timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "symbol": "EURUSD",
                "open": 1.056,
                "high": 1.07,
                "low": 1.05,
                "close": 1.065,
                "volume": 1_500,
            }
        ]
    )

    fred = _frame(
        [
            {
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "symbol": "EURUSD",
                "open": 1.051,
                "high": 1.062,
                "low": 1.043,
                "close": 1.054,
                "volume": 800,
            }
        ]
    )

    aggregator = MultiSourceAggregator(
        providers=[
            ProviderSpec(name="yahoo", fetch=_make_fetch(yahoo)),
            ProviderSpec(name="alpha_vantage", fetch=_make_fetch(alpha)),
            ProviderSpec(name="fred", fetch=_make_fetch(fred)),
        ],
        validators=[
            CoverageValidator(frequency="1D", warn_ratio=0.8, error_ratio=0.5),
            StalenessValidator(max_staleness=timedelta(days=2)),
        ],
    )

    result = aggregator.aggregate(["EURUSD"], start=start, end=end)

    assert list(result.data.columns) == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "primary_source",
        "sources",
    ]

    assert len(result.data) == 2
    first_row = result.data.iloc[0]
    assert first_row["primary_source"] == "yahoo"
    assert first_row["sources"] == ("yahoo", "fred")
    second_row = result.data.iloc[1]
    assert second_row["primary_source"] == "alpha_vantage"
    assert pytest.approx(second_row["close"], rel=1e-6) == 1.065
    assert second_row["sources"] == ("alpha_vantage",)

    severities = {finding.name: finding.severity for finding in result.quality_findings}
    assert severities["coverage"] == DataQualitySeverity.warning
    assert severities["staleness"] == DataQualitySeverity.ok


def test_cross_source_drift_validator_flags_large_divergence() -> None:
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = datetime(2024, 2, 2, tzinfo=timezone.utc)

    yahoo = _frame(
        [
            {
                "timestamp": datetime(2024, 2, 1, tzinfo=timezone.utc),
                "symbol": "GBPUSD",
                "open": 1.20,
                "high": 1.21,
                "low": 1.19,
                "close": 1.205,
                "volume": 2_000,
            }
        ]
    )

    alpha = _frame(
        [
            {
                "timestamp": datetime(2024, 2, 1, tzinfo=timezone.utc),
                "symbol": "GBPUSD",
                "open": 1.24,
                "high": 1.25,
                "low": 1.23,
                "close": 1.245,
                "volume": 1_800,
            }
        ]
    )

    aggregator = MultiSourceAggregator(
        providers=[
            ProviderSpec(name="yahoo", fetch=_make_fetch(yahoo)),
            ProviderSpec(name="alpha_vantage", fetch=_make_fetch(alpha)),
        ],
        validators=[
            CoverageValidator(frequency="1D"),
            CrossSourceDriftValidator(tolerance=0.02, warn_tolerance=0.01),
        ],
    )

    result = aggregator.aggregate(["GBPUSD"], start=start, end=end)

    drift_finding = next(f for f in result.quality_findings if f.name == "cross_source_drift")
    assert drift_finding.severity == DataQualitySeverity.error
    assert drift_finding.metrics["max_relative_drift"] > 0.02
    assert "GBPUSD@" in next(iter(drift_finding.details["breaches"]))

