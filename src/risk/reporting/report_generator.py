"""Generate portfolio risk reports aligned with the high-impact roadmap."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

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
    """Summary of exposure for a single diversification bucket.

    The ``symbol`` attribute historically referenced an instrument identifier.
    Portfolio-level antifragility now aggregates exposures by regime correlation
    buckets, so the value may represent a regime label such as
    ``"regime:carry-balanced"``.
    """

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

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RiskReport":
        """Rehydrate a :class:`RiskReport` from a JSON-compatible mapping."""

        def _get_float(key: str, *, default: float | None = None) -> float:
            try:
                value = mapping.get(key, default)
                if value is None:
                    raise KeyError(key)
                return float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                raise ValueError(f"risk report field '{key}' must be numeric") from None

        def _get_int(key: str) -> int:
            value = mapping.get(key)
            if value is None:
                raise ValueError(f"risk report field '{key}' is required")
            try:
                return int(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"risk report field '{key}' must be an integer") from exc

        generated_at_raw = mapping.get("generated_at")
        if not isinstance(generated_at_raw, str):
            raise ValueError("risk report field 'generated_at' must be an ISO timestamp")
        try:
            generated_at = datetime.fromisoformat(generated_at_raw)
        except ValueError as exc:  # pragma: no cover - invalid timestamp guard
            raise ValueError("risk report field 'generated_at' is not a valid ISO timestamp") from exc

        exposures_payload = mapping.get("exposures", [])
        exposures: list[ExposureBreakdown] = []
        if exposures_payload:
            if not isinstance(exposures_payload, Sequence):
                raise ValueError("risk report field 'exposures' must be a sequence")
            for entry in exposures_payload:
                if not isinstance(entry, Mapping):
                    raise ValueError("each exposure entry must be a mapping")
                symbol = str(entry.get("symbol"))
                notional_raw = entry.get("notional")
                percentage_raw = entry.get("percentage")
                if notional_raw is None or percentage_raw is None:
                    raise ValueError("exposure entries require 'notional' and 'percentage'")
                try:
                    notional = float(notional_raw)
                    percentage = float(percentage_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError("exposure fields must be numeric") from exc
                exposures.append(
                    ExposureBreakdown(symbol=symbol, notional=notional, percentage=percentage)
                )

        limits_payload = mapping.get("limits")
        limits: PortfolioRiskLimits | None = None
        if isinstance(limits_payload, Mapping):
            limits = PortfolioRiskLimits.from_mapping(limits_payload)

        breaches_payload = mapping.get("breaches", {})
        if breaches_payload is None:
            breaches = {}
        elif isinstance(breaches_payload, Mapping):
            breaches = dict(breaches_payload)
        else:
            raise ValueError("risk report field 'breaches' must be a mapping")

        return cls(
            generated_at=generated_at,
            confidence=float(mapping.get("confidence", 0.0)),
            sample_size=_get_int("sample_size"),
            historical_var=_get_float("historical_var"),
            parametric_var=_get_float("parametric_var"),
            monte_carlo_var=_get_float("monte_carlo_var"),
            monte_carlo_simulations=_get_int("monte_carlo_simulations"),
            historical_expected_shortfall=_get_float("historical_expected_shortfall"),
            parametric_expected_shortfall=_get_float("parametric_expected_shortfall"),
            total_exposure=_get_float("total_exposure", default=0.0),
            exposures=tuple(exposures),
            breaches=breaches,
            limits=limits,
        )


def _normalise_returns(returns: Sequence[float] | Iterable[float]) -> np.ndarray:
    array = np.asarray(list(returns), dtype=float)
    if array.size == 0:
        raise ValueError("returns must contain at least one observation")
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("returns must contain at least one finite observation")
    return array


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _stringify_label(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        candidate = raw.strip()
        return candidate or None
    value_attr = getattr(raw, "value", None)
    if isinstance(value_attr, str):
        candidate = value_attr.strip()
        return candidate or None
    return None


def _prefix_regime(label: str) -> str:
    stripped = label.strip()
    if not stripped:
        return stripped
    lower = stripped.lower()
    if lower.startswith("regime:"):
        return stripped
    return f"regime:{stripped}"


def _resolve_notional(entry: Mapping[str, object]) -> float | None:
    for key in ("notional", "exposure", "value", "size", "amount"):
        if key in entry:
            maybe = _coerce_float(entry[key])
            if maybe is not None:
                return maybe
    return None


def _resolve_bucket_label(
    entry: Mapping[str, object], fallback: object | None = None
) -> str | None:
    regime_keys = (
        "regime_correlation",
        "correlation_regime",
        "regime_cluster",
        "regime_bucket",
        "regime_group",
        "regime",
    )
    correlation_keys = (
        "correlation_cluster",
        "correlation_group",
        "correlation_bucket",
        "correlation_id",
    )
    symbol_like_keys = ("bucket", "group", "symbol", "instrument", "asset", "name", "id")

    for key in regime_keys:
        if key in entry:
            candidate = _stringify_label(entry[key])
            if candidate:
                return _prefix_regime(candidate)

    for key in correlation_keys:
        if key in entry:
            candidate = _stringify_label(entry[key])
            if candidate:
                return _prefix_regime(candidate)

    for key in symbol_like_keys:
        if key in entry:
            candidate = _stringify_label(entry[key])
            if candidate:
                return candidate

    fallback_label = _stringify_label(fallback)
    if fallback_label:
        return fallback_label
    return None


def _normalise_bucket_label(label: str) -> str:
    stripped = label.strip()
    if not stripped:
        return stripped
    lower = stripped.lower()
    if lower.startswith("regime:"):
        _, _, remainder = stripped.partition(":")
        remainder = remainder.strip()
        return f"regime:{remainder}" if remainder else "regime"
    return stripped


def _iter_exposure_entries(
    exposures: Mapping[str, object] | Sequence[object],
) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []

    if isinstance(exposures, Mapping):
        for key, value in exposures.items():
            fallback_label = key
            if isinstance(value, Mapping):
                notional = _resolve_notional(value)
                if notional is None:
                    continue
                label = _resolve_bucket_label(value, fallback=fallback_label)
            else:
                notional = _coerce_float(value)
                if notional is None:
                    continue
                label = _stringify_label(fallback_label)
            if notional is None or label is None:
                continue
            entries.append((_normalise_bucket_label(label), float(notional)))
        return entries

    if isinstance(exposures, Sequence) and not isinstance(exposures, (str, bytes)):
        for entry in exposures:
            label: str | None
            notional: float | None
            if isinstance(entry, Mapping):
                notional = _resolve_notional(entry)
                if notional is None:
                    continue
                label = _resolve_bucket_label(entry)
            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                label = _stringify_label(entry[0])
                notional = _coerce_float(entry[1])
            else:
                continue
            if label is None or notional is None:
                continue
            entries.append((_normalise_bucket_label(label), float(notional)))

    return entries


def _prepare_exposures(
    exposures: Mapping[str, object] | Sequence[object] | None,
) -> tuple[tuple[ExposureBreakdown, ...], float]:
    if not exposures:
        return tuple(), 0.0

    entries = _iter_exposure_entries(exposures)
    if not entries:
        return tuple(), 0.0

    aggregated: dict[str, float] = {}
    for label, notional in entries:
        if not label:
            continue
        aggregated[label] = aggregated.get(label, 0.0) + float(notional)

    if not aggregated:
        return tuple(), 0.0

    total_abs = float(sum(abs(value) for value in aggregated.values()))
    resolved: list[ExposureBreakdown] = []

    for label, notional in sorted(
        aggregated.items(), key=lambda item: (abs(item[1]), item[0]), reverse=True
    ):
        percentage = 0.0 if total_abs == 0.0 else abs(notional) / total_abs * 100.0
        resolved.append(
            ExposureBreakdown(symbol=label, notional=float(notional), percentage=float(percentage))
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
    exposures: Mapping[str, object] | Sequence[object] | None = None,
    limits: PortfolioRiskLimits | None = None,
) -> RiskReport:
    """Compute a risk report from historical returns and exposures.

    Parameters
    ----------
    returns:
        Historical return series used to estimate distributional risk metrics.
    exposures:
        Optional mapping or sequence describing current portfolio exposures.
        Entries can be raw floats keyed by identifier or structured mappings
        containing ``notional`` and regime correlation metadata (for example,
        ``{"notional": 1.2, "regime_correlation": "carry-balanced"}``).
        When regime metadata is supplied, exposures are aggregated by the
        regime correlation bucket rather than instrument name.
    limits:
        Optional portfolio risk limits used for breach detection.
    """

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
                "| Bucket | Notional | Share |",
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
