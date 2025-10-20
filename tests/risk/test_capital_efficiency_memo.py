from __future__ import annotations

from datetime import datetime, timezone

from src.risk import (
    ExposureBreakdown,
    PortfolioRiskLimits,
    RiskReport,
    generate_capital_efficiency_memo,
)


def build_sample_report() -> RiskReport:
    exposures = (
        ExposureBreakdown(symbol="regime:carry-balanced", notional=0.6, percentage=60.0),
        ExposureBreakdown(symbol="regime:flight-to-safety", notional=0.4, percentage=40.0),
    )
    limits = PortfolioRiskLimits(
        per_asset_cap=0.5,
        aggregate_cap=1.0,
        usd_beta_cap=None,
        var95_cap=0.02,
    )
    return RiskReport(
        generated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        confidence=0.95,
        sample_size=128,
        historical_var=0.015,
        parametric_var=0.018,
        monte_carlo_var=0.02,
        monte_carlo_simulations=5000,
        historical_expected_shortfall=0.02,
        parametric_expected_shortfall=0.022,
        total_exposure=1.0,
        exposures=exposures,
        breaches={"per_asset": {"symbols": ["regime:carry-balanced"], "limit": 0.5}},
        limits=limits,
    )


def test_generate_capital_efficiency_memo_includes_utilisation() -> None:
    report = build_sample_report()

    memo = generate_capital_efficiency_memo(report)

    assert "Aggregate exposure" in memo
    assert "100.0%" in memo
    assert "⚠️" in memo
    assert memo.endswith("\n")


def test_risk_report_from_mapping_round_trip() -> None:
    report = build_sample_report()
    payload = report.to_dict()

    hydrated = RiskReport.from_mapping(payload)

    assert hydrated.generated_at == report.generated_at
    assert hydrated.sample_size == report.sample_size
    assert hydrated.breaches == report.breaches
    assert hydrated.limits and hydrated.limits.per_asset_cap == report.limits.per_asset_cap
