"""Generate portfolio risk reports aligned with the high-impact roadmap."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

from ..analytics import (
    compute_historical_expected_shortfall,
    compute_historical_var,
    compute_monte_carlo_var,
    compute_parametric_expected_shortfall,
    compute_parametric_var,
)

__all__ = [
    "ExposureBreakdown",
    "PortfolioRiskLimits",
    "RiskReport",
    "generate_risk_report",
    "load_portfolio_limits",
    "render_risk_report_json",
    "render_risk_report_markdown",
]


@dataclass(slots=True)
class ExposureBreakdown:
    """Summary of exposure for a single instrument."""

    symbol: str
    notional: float
    percentage: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "symbol": self.symbol,
            "notional": self.notional,
            "percentage": self.percentage,
        }


@dataclass(slots=True)
class PortfolioRiskLimits:
    """Structured representation of portfolio risk limits."""

    per_asset_cap: float | None = None
    aggregate_cap: float | None = None
    usd_beta_cap: float | None = None
    var95_cap: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "PortfolioRiskLimits":
        def _maybe_float(value: object | None) -> float | None:
            if value is None:
                return None
            try:
                return float(value)  # type: ignore[return-value]
            except (TypeError, ValueError):
                return None

        return cls(
            per_asset_cap=_maybe_float(mapping.get("per_asset_cap")),
            aggregate_cap=_maybe_float(mapping.get("aggregate_cap")),
            usd_beta_cap=_maybe_float(mapping.get("usd_beta_cap")),
            var95_cap=_maybe_float(mapping.get("var95_cap")),
        )


@dataclass(slots=True)
class RiskReport:
    """Computed risk report for a portfolio snapshot."""

    generated_at: datetime
    confidence: float
    sample_size: int
    historical_var: float
    parametric_var: float
    monte_carlo_var: float
    monte_carlo_simulations: int
    historical_expected_shortfall: float
    parametric_expected_shortfall: float
    total_exposure: float
    exposures: tuple[ExposureBreakdown, ...]
    breaches: dict[str, object]
    limits: PortfolioRiskLimits | None

    def to_dict(self) -> dict[str, object]:
        data = {
            "generated_at": self.generated_at.isoformat(),
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "historical_var": self.historical_var,
            "parametric_var": self.parametric_var,
            "monte_carlo_var": self.monte_carlo_var,
            "monte_carlo_simulations": self.monte_carlo_simulations,
            "historical_expected_shortfall": self.historical_expected_shortfall,
            "parametric_expected_shortfall": self.parametric_expected_shortfall,
            "total_exposure": self.total_exposure,
            "exposures": [exposure.to_dict() for exposure in self.exposures],
            "breaches": self.breaches,
        }
        if self.limits is not None:
            data["limits"] = self.limits.to_dict()
        return data


def _normalise_returns(returns: Sequence[float] | Iterable[float]) -> np.ndarray:
    array = np.asarray(list(returns), dtype=float)
    if array.size == 0:
        raise ValueError("returns must contain at least one observation")
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("returns must contain at least one finite observation")
    return array


def _prepare_exposures(
    exposures: Mapping[str, float] | None,
) -> tuple[tuple[ExposureBreakdown, ...], float]:
    if not exposures:
        return tuple(), 0.0

    resolved: list[ExposureBreakdown] = []
    total_abs = sum(abs(float(value)) for value in exposures.values())
    total_abs = float(total_abs)

    for symbol, value in sorted(
        exposures.items(), key=lambda item: abs(float(item[1])), reverse=True
    ):
        notional = float(value)
        if total_abs == 0.0:
            pct = 0.0
        else:
            pct = abs(notional) / total_abs * 100.0
        resolved.append(
            ExposureBreakdown(symbol=symbol, notional=notional, percentage=pct)
        )

    return tuple(resolved), total_abs


def _evaluate_breaches(
    *,
    exposures: tuple[ExposureBreakdown, ...],
    total_exposure: float,
    historical_var: float,
    limits: PortfolioRiskLimits | None,
) -> dict[str, object]:
    if limits is None:
        return {}

    breaches: dict[str, object] = {}

    if limits.aggregate_cap is not None and total_exposure > limits.aggregate_cap:
        breaches["aggregate_exposure"] = {
            "current": total_exposure,
            "limit": limits.aggregate_cap,
        }

    if limits.per_asset_cap is not None:
        violators = [
            exposure.symbol
            for exposure in exposures
            if abs(exposure.notional) > limits.per_asset_cap
        ]
        if violators:
            breaches["per_asset"] = {
                "symbols": violators,
                "limit": limits.per_asset_cap,
            }

    if limits.var95_cap is not None and historical_var > limits.var95_cap:
        breaches["var_limit"] = {
            "historical_var": historical_var,
            "limit": limits.var95_cap,
        }

    return breaches


def generate_risk_report(
    returns: Sequence[float] | Iterable[float],
    *,
    confidence: float = 0.99,
    simulations: int = 10_000,
    exposures: Mapping[str, float] | None = None,
    limits: PortfolioRiskLimits | None = None,
) -> RiskReport:
    """Compute a risk report from historical returns and exposures."""

    sample = _normalise_returns(returns)

    historical_var = compute_historical_var(sample, confidence=confidence)
    parametric_var = compute_parametric_var(sample, confidence=confidence)
    monte_carlo_var = compute_monte_carlo_var(
        sample, confidence=confidence, simulations=simulations
    )
    historical_es = compute_historical_expected_shortfall(sample, confidence=confidence)
    parametric_es = compute_parametric_expected_shortfall(sample, confidence=confidence)

    exposures_breakdown, total_exposure = _prepare_exposures(exposures)
    breaches = _evaluate_breaches(
        exposures=exposures_breakdown,
        total_exposure=total_exposure,
        historical_var=historical_var.value,
        limits=limits,
    )

    return RiskReport(
        generated_at=datetime.now(timezone.utc),
        confidence=confidence,
        sample_size=historical_var.sample_size,
        historical_var=historical_var.value,
        parametric_var=parametric_var.value,
        monte_carlo_var=monte_carlo_var.value,
        monte_carlo_simulations=simulations,
        historical_expected_shortfall=historical_es.value,
        parametric_expected_shortfall=parametric_es.value,
        total_exposure=total_exposure,
        exposures=exposures_breakdown,
        breaches=breaches,
        limits=limits,
    )


def render_risk_report_markdown(report: RiskReport) -> str:
    """Render a risk report to Markdown."""

    def _format_pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    lines = [
        "# Portfolio Risk Report",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        "",
        "## Risk Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Historical VaR ({report.confidence:.0%}) | {_format_pct(report.historical_var)} |",
        f"| Parametric VaR ({report.confidence:.0%}) | {_format_pct(report.parametric_var)} |",
        f"| Monte Carlo VaR ({report.confidence:.0%}) | {_format_pct(report.monte_carlo_var)} |",
        f"| Historical Expected Shortfall | {_format_pct(report.historical_expected_shortfall)} |",
        f"| Parametric Expected Shortfall | {_format_pct(report.parametric_expected_shortfall)} |",
        "",
        "## Exposure Breakdown",
        "",
        f"Total absolute exposure: {report.total_exposure:.4f}",
    ]

    if report.exposures:
        lines.extend(
            [
                "",
                "| Symbol | Notional | Share |",
                "| --- | ---: | ---: |",
            ]
        )
        for exposure in report.exposures:
            lines.append(
                f"| {exposure.symbol} | {exposure.notional:.4f} | {exposure.percentage:.2f}% |"
            )
    else:
        lines.append("\n_No exposures supplied._")

    if report.breaches:
        lines.extend(["", "## Breach Summary", ""])
        for name, payload in report.breaches.items():
            lines.append(f"- **{name.replace('_', ' ').title()}**: {json.dumps(payload)}")

    return "\n".join(lines).strip() + "\n"


def render_risk_report_json(report: RiskReport) -> str:
    """Render a risk report to JSON."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def load_portfolio_limits(path: str | Path | None = None) -> PortfolioRiskLimits:
    """Load portfolio risk limits from a YAML document."""

    candidate = Path(path) if path is not None else _default_limits_path()
    if not candidate.exists():
        raise FileNotFoundError(f"risk limits file not found: {candidate}")

    with candidate.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    section = data.get("portfolio_risk", data)
    if not isinstance(section, Mapping):
        raise ValueError("portfolio risk configuration must be a mapping")
    return PortfolioRiskLimits.from_mapping(section)


def _default_limits_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "risk" / "portfolio.yaml"


def parse_returns_file(path: Path) -> list[float]:
    """Parse a text file containing delimited return series."""

    raw = path.read_text(encoding="utf-8")
    tokens = re.split(r"[\s,]+", raw.strip())
    returns = [float(token) for token in tokens if token]
    if not returns:
        raise ValueError("returns file did not contain any numeric values")
    return returns
