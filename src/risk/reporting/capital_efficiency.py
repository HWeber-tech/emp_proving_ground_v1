"""Capital efficiency memo generator aligned with the high-impact roadmap."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from .report_generator import ExposureBreakdown, PortfolioRiskLimits, RiskReport

__all__ = [
    "BudgetUtilisation",
    "generate_capital_efficiency_memo",
]


@dataclass(slots=True)
class BudgetUtilisation:
    """Snapshot of a risk budget utilisation metric."""

    label: str
    realised: float
    target: float | None

    @property
    def utilisation_ratio(self) -> float | None:
        if self.target in (None, 0.0):
            return None
        return self.realised / self.target


def _format_ratio(utilisation: float | None) -> str:
    if utilisation is None:
        return "n/a"
    return f"{utilisation * 100:.1f}%"


def _iter_budget_lines(report: RiskReport) -> Iterable[BudgetUtilisation]:
    limits = report.limits
    yield BudgetUtilisation(
        label="Aggregate exposure",
        realised=report.total_exposure,
        target=limits.aggregate_cap if limits else None,
    )
    yield BudgetUtilisation(
        label=f"Historical VaR ({report.confidence:.0%})",
        realised=report.historical_var,
        target=limits.var95_cap if limits else None,
    )


def _format_notional(value: float) -> str:
    return f"{value:.4f}"


def _per_asset_flag(exposure: ExposureBreakdown, limits: PortfolioRiskLimits | None) -> str:
    if not limits or limits.per_asset_cap is None:
        return ""
    return "⚠️" if abs(exposure.notional) > limits.per_asset_cap else ""


def generate_capital_efficiency_memo(report: RiskReport) -> str:
    """Render a weekly capital efficiency memo for operators."""

    lines: list[str] = [
        "# Weekly Capital Efficiency Memo",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        "",
        "## Risk Budget Utilisation",
        "",
    ]

    for budget in _iter_budget_lines(report):
        target_display = (
            _format_notional(budget.target) if budget.target is not None else "not set"
        )
        lines.append(
            f"- {budget.label}: {_format_notional(budget.realised)} (limit: {target_display}, utilisation: {_format_ratio(budget.utilisation_ratio)})"
        )

    lines.extend(["", "## Exposure Concentration", ""])

    if report.exposures:
        lines.extend(
            [
                "| Symbol | Notional | Share | Limit Flag |",
                "| --- | ---: | ---: | :---: |",
            ]
        )
        for exposure in report.exposures:
            lines.append(
                "| {symbol} | {notional} | {share:.2f}% | {flag} |".format(
                    symbol=exposure.symbol,
                    notional=_format_notional(exposure.notional),
                    share=exposure.percentage,
                    flag=_per_asset_flag(exposure, report.limits),
                )
            )
        if report.limits and report.limits.per_asset_cap is not None:
            lines.append(
                "\n_Per-asset limit: {limit}_".format(
                    limit=_format_notional(report.limits.per_asset_cap)
                )
            )
    else:
        lines.append("_No exposures supplied._")

    lines.extend(["", "## Breach Log", ""])

    if report.breaches:
        for name, payload in sorted(report.breaches.items()):
            lines.append(
                f"- **{name.replace('_', ' ').title()}**: {json.dumps(payload, sort_keys=True)}"
            )
    else:
        lines.append("_No breaches recorded._")

    memo = "\n".join(lines).strip()
    return memo + "\n"
