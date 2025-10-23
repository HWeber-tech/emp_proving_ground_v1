from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.data_foundation.monitoring.fundamental_quality import (
    FundamentalQualityStatus,
    evaluate_fundamental_quality,
)


def test_fundamental_quality_reports_ok_for_complete_fresh_data() -> None:
    now = datetime(2024, 3, 10, 15, 30, tzinfo=UTC)
    records = [
        {
            "symbol": "AAPL",
            "as_of": now.isoformat(),
            "metrics": {
                "price": 175.4,
                "eps": 6.2,
                "book_value_per_share": 25.8,
                "free_cash_flow_per_share": 4.1,
                "growth_rate": 0.12,
                "discount_rate": 0.09,
                "revenue": 4.2e5,
                "net_income": 1.1e5,
                "dividend_yield": 0.012,
            },
            "source": "fmp",
        },
        {
            "symbol": "MSFT",
            "as_of": (now - timedelta(hours=4)).isoformat(),
            "metrics": {
                "price": 320.1,
                "eps": 9.4,
                "book_value_per_share": 22.5,
                "free_cash_flow_per_share": 6.2,
                "growth_rate": 0.15,
                "discount_rate": 0.1,
                "revenue": 3.8e5,
                "net_income": 9.2e4,
                "dividend_yield": 0.009,
            },
            "source": "polygon",
        },
    ]

    report = evaluate_fundamental_quality(
        records,
        expected_symbols=["AAPL", "MSFT"],
        now=now,
    )

    assert report.status is FundamentalQualityStatus.ok
    assert report.score >= 0.85
    assert len(report.checks) == 2

    mapping = {check.symbol: check for check in report.checks}
    aapl = mapping["AAPL"]
    assert aapl.status is FundamentalQualityStatus.ok
    assert aapl.coverage_ratio == pytest.approx(1.0)
    assert aapl.staleness_hours == pytest.approx(0.0)
    assert not aapl.missing_fields
    assert aapl.metadata["source"] == "fmp"


def test_fundamental_quality_flags_missing_symbols_and_stale_records() -> None:
    now = datetime(2024, 3, 10, 15, 30, tzinfo=UTC)
    stale = now - timedelta(days=21)
    records = [
        {
            "symbol": "AAPL",
            "as_of": stale,
            "metrics": {
                "price": 120.0,
                "book_value_per_share": 18.0,
            },
            "source": "fmp",
        }
    ]

    report = evaluate_fundamental_quality(
        records,
        expected_symbols=["AAPL", "GOOG"],
        now=now,
        freshness_warn_hours=72.0,
        freshness_error_hours=240.0,
    )

    assert report.status is FundamentalQualityStatus.error
    mapping = {check.symbol: check for check in report.checks}

    aapl = mapping["AAPL"]
    assert aapl.status is FundamentalQualityStatus.error
    assert "missing_fields" in aapl.issues
    assert any("Missing fundamental fields" in msg for msg in aapl.messages)
    assert any("stale" in msg for msg in aapl.messages)

    goog = mapping["GOOG"]
    assert goog.status is FundamentalQualityStatus.error
    assert "no_fundamentals" in goog.issues
    assert any("No fundamental data" in msg for msg in goog.messages)
    assert "GOOG" in report.metadata.get("missing_symbols", [])


def test_fundamental_quality_prefers_latest_snapshot() -> None:
    now = datetime(2024, 3, 10, 15, 30, tzinfo=UTC)
    older = now - timedelta(days=3)
    newer = now - timedelta(hours=6)
    records = [
        {
            "symbol": "MSFT",
            "as_of": older,
            "metrics": {
                "price": 315.0,
                "eps": 8.8,
                "book_value_per_share": 20.0,
                "free_cash_flow_per_share": 5.5,
                "growth_rate": 0.14,
                "discount_rate": 0.1,
                "revenue": 3.6e5,
                "net_income": 8.8e4,
            },
            "source": "fmp",
        },
        {
            "symbol": "MSFT",
            "as_of": newer,
            "metrics": {
                "price": 322.0,
                "eps": 9.2,
                "book_value_per_share": 21.0,
                "free_cash_flow_per_share": 5.9,
                "growth_rate": 0.16,
                "discount_rate": 0.1,
                "revenue": 3.7e5,
                "net_income": 9.0e4,
            },
            "source": "polygon",
        },
    ]

    report = evaluate_fundamental_quality(records, now=now)

    assert len(report.checks) == 1
    check = report.checks[0]
    assert check.metadata["source"] == "polygon"
    assert check.metadata["as_of"] == newer.isoformat()
    assert check.staleness_hours == pytest.approx(6.0)
