from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.risk import (
    generate_risk_report,
    load_portfolio_limits,
    render_risk_report_json,
    render_risk_report_markdown,
)
from src.risk.reporting.report_generator import parse_returns_file


@pytest.fixture()
def sample_returns() -> list[float]:
    return [-0.045, -0.02, -0.015, 0.005, 0.012, 0.02, -0.03, 0.018]


def test_generate_risk_report_includes_expected_metrics(sample_returns: list[float]) -> None:
    limits = load_portfolio_limits()
    exposures = {"EURUSD": 2.5, "GBPUSD": 1.2}

    report = generate_risk_report(
        sample_returns,
        confidence=0.95,
        simulations=5000,
        exposures=exposures,
        limits=limits,
    )

    assert report.sample_size == len(sample_returns)
    assert report.monte_carlo_simulations == 5000
    # VaR/ES values should align with direct calculations
    assert report.historical_var > 0
    assert report.parametric_expected_shortfall >= report.historical_expected_shortfall
    # Exposure totals
    assert pytest.approx(report.total_exposure, rel=1e-12) == sum(abs(v) for v in exposures.values())
    # Breach detection (aggregate + per asset + VaR)
    assert "aggregate_exposure" in report.breaches
    assert "per_asset" in report.breaches
    assert "var_limit" in report.breaches


def test_render_risk_report_markdown_contains_tables(sample_returns: list[float]) -> None:
    report = generate_risk_report(sample_returns, exposures=None, limits=None)
    markdown = render_risk_report_markdown(report)

    assert "# Portfolio Risk Report" in markdown
    assert "| Metric | Value |" in markdown
    assert "_No exposures supplied._" in markdown
    assert markdown.endswith("\n")


def test_render_risk_report_json_round_trip(sample_returns: list[float]) -> None:
    report = generate_risk_report(sample_returns, exposures=None, limits=None)
    payload = render_risk_report_json(report)
    decoded = json.loads(payload)

    assert decoded["sample_size"] == len(sample_returns)
    assert decoded["historical_var"] == report.historical_var


def test_parse_returns_file(tmp_path: Path) -> None:
    data = "-0.01, 0.02\n-0.03 0.04"
    returns_file = tmp_path / "returns.txt"
    returns_file.write_text(data, encoding="utf-8")

    parsed = parse_returns_file(returns_file)

    assert parsed == [-0.01, 0.02, -0.03, 0.04]


def test_load_portfolio_limits_allows_override(tmp_path: Path) -> None:
    custom_yaml = tmp_path / "limits.yaml"
    custom_yaml.write_text(
        """
portfolio_risk:
  per_asset_cap: 0.5
  aggregate_cap: 1.0
  var95_cap: 0.01
        """.strip(),
        encoding="utf-8",
    )

    limits = load_portfolio_limits(custom_yaml)

    assert limits.per_asset_cap == 0.5
    assert limits.aggregate_cap == 1.0
    assert limits.var95_cap == 0.01
