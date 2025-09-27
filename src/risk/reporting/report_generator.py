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
from ..sizing import check_classification_limits

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
    sector: str | None = None
    asset_class: str | None = None

    def to_dict(self) -> dict[str, float | str]:
        return {
            "symbol": self.symbol,
            "notional": self.notional,
            "percentage": self.percentage,
            "sector": self.sector,
            "asset_class": self.asset_class,
        }


@dataclass(slots=True)
class PortfolioRiskLimits:
    """Structured representation of portfolio risk limits."""

    per_asset_cap: float | None = None
    aggregate_cap: float | None = None
    usd_beta_cap: float | None = None
    var95_cap: float | None = None
    sector_limits: dict[str, float] | None = None
    asset_class_limits: dict[str, float] | None = None
    instrument_classification: dict[str, dict[str, str]] | None = None

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
            sector_limits=_normalise_limit_mapping(mapping.get("sector_limits")),
            asset_class_limits=_normalise_limit_mapping(
                mapping.get("asset_class_limits")
            ),
            instrument_classification=_normalise_classifications(
                mapping.get("instrument_classification")
            ),
        )

    def classification_for(self, symbol: str) -> dict[str, str]:
        if not self.instrument_classification:
            return {}
        return dict(self.instrument_classification.get(symbol, {}))


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


def _normalise_limit_mapping(value: object | None) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    limits: dict[str, float] = {}
    for key, raw in value.items():
        try:
            numeric = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        limits[str(key)] = numeric
    return limits or None


def _normalise_classifications(
    value: object | None,
) -> dict[str, dict[str, str]] | None:
    if not isinstance(value, Mapping):
        return None
    classifications: dict[str, dict[str, str]] = {}
    for symbol, payload in value.items():
        if not isinstance(payload, Mapping):
            continue
        record: dict[str, str] = {}
        for field in ("sector", "asset_class"):
            raw = payload.get(field)
            if isinstance(raw, str) and raw:
                record[field] = raw
        if record:
            classifications[str(symbol)] = record
    return classifications or None


def _prepare_exposures(
    exposures: Mapping[str, object] | None,
    classifications: Mapping[str, Mapping[str, str]] | None,
) -> tuple[tuple[ExposureBreakdown, ...], float]:
    if not exposures:
        return tuple(), 0.0

    resolved: list[ExposureBreakdown] = []
    classification_map = classifications or {}

    extracted: list[tuple[str, float, dict[str, str]]] = []
    for symbol, raw in exposures.items():
        if isinstance(raw, Mapping):
            candidate = raw.get("notional")
            if candidate is None:
                candidate = raw.get("exposure", raw.get("value", 0.0))
            try:
                notional = float(candidate)
            except (TypeError, ValueError):
                notional = 0.0
            metadata: dict[str, str] = {}
            for key in ("sector", "asset_class"):
                value = raw.get(key)
                if isinstance(value, str) and value:
                    metadata[key] = value
        else:
            try:
                notional = float(raw)
            except (TypeError, ValueError):
                notional = 0.0
            metadata = {}
        extracted.append((symbol, notional, metadata))

    total_abs = float(sum(abs(item[1]) for item in extracted))

    for symbol, notional, metadata in sorted(
        extracted, key=lambda item: abs(item[1]), reverse=True
    ):
        classification = classification_map.get(symbol, {})
        sector = metadata.get("sector") or classification.get("sector")
        asset_class = metadata.get("asset_class") or classification.get("asset_class")
        if total_abs == 0.0:
            pct = 0.0
        else:
            pct = abs(notional) / total_abs * 100.0
        resolved.append(
            ExposureBreakdown(
                symbol=symbol,
                notional=notional,
                percentage=pct,
                sector=sector,
                asset_class=asset_class,
            )
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

    exposures_map = {exposure.symbol: exposure.notional for exposure in exposures}
    classification_map = limits.instrument_classification or {}

    sector_limits = limits.sector_limits or {}
    sector_breaches = check_classification_limits(
        exposures_map,
        classification_map,
        sector_limits,
        classification_key="sector",
    )
    if sector_breaches:
        breaches["sector_limits"] = sector_breaches

    asset_class_limits = limits.asset_class_limits or {}
    asset_class_breaches = check_classification_limits(
        exposures_map,
        classification_map,
        asset_class_limits,
        classification_key="asset_class",
    )
    if asset_class_breaches:
        breaches["asset_class_limits"] = asset_class_breaches

    return breaches


def generate_risk_report(
    returns: Sequence[float] | Iterable[float],
    *,
    confidence: float = 0.99,
    simulations: int = 10_000,
    exposures: Mapping[str, object] | None = None,
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

    classification_map = limits.instrument_classification if limits else None
    exposures_breakdown, total_exposure = _prepare_exposures(
        exposures, classification_map
    )
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
        has_sector = any(exposure.sector for exposure in report.exposures)
        has_asset_class = any(exposure.asset_class for exposure in report.exposures)
        header_cells = ["Symbol", "Notional", "Share"]
        if has_sector:
            header_cells.append("Sector")
        if has_asset_class:
            header_cells.append("Asset Class")
        lines.append("")
        lines.append("| " + " | ".join(header_cells) + " |")
        align_cells = ["---", "---:", "---:"]
        if has_sector:
            align_cells.append("---")
        if has_asset_class:
            align_cells.append("---")
        lines.append("| " + " | ".join(align_cells) + " |")
        for exposure in report.exposures:
            row = [
                exposure.symbol,
                f"{exposure.notional:.4f}",
                f"{exposure.percentage:.2f}%",
            ]
            if has_sector:
                row.append(exposure.sector or "-")
            if has_asset_class:
                row.append(exposure.asset_class or "-")
            lines.append("| " + " | ".join(row) + " |")
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
