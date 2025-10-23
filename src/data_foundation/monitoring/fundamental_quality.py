"""Fundamental data quality monitoring heuristics.

This module implements the roadmap item "Build fundamental data quality
monitoring" by grading per-symbol fundamentals snapshots across freshness,
coverage, and sanity constraints.  The implementation intentionally remains
self-contained so it can execute inside CI and offline diagnostic tooling
without requiring live database access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "FundamentalQualityStatus",
    "FundamentalQualityCheck",
    "FundamentalQualityReport",
    "evaluate_fundamental_quality",
]


class FundamentalQualityStatus(StrEnum):
    """Discrete severity grades for fundamental quality evaluations."""

    ok = "ok"
    warn = "warn"
    error = "error"


@dataclass(frozen=True)
class FundamentalQualityCheck:
    """Per-symbol quality outcome."""

    symbol: str
    status: FundamentalQualityStatus
    score: float
    coverage_ratio: float
    staleness_hours: float | None
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    issues: tuple[str, ...] = field(default_factory=tuple)
    messages: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "status": self.status.value,
            "score": self.score,
            "coverage_ratio": self.coverage_ratio,
            "staleness_hours": self.staleness_hours,
            "missing_fields": list(self.missing_fields),
            "issues": list(self.issues),
            "messages": list(self.messages),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FundamentalQualityReport:
    """Aggregated quality grading across all evaluated symbols."""

    status: FundamentalQualityStatus
    score: float
    generated_at: datetime
    checks: tuple[FundamentalQualityCheck, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "score": self.score,
            "generated_at": self.generated_at.isoformat(),
            "checks": [check.as_dict() for check in self.checks],
            "metadata": dict(self.metadata),
        }


_DEFAULT_REQUIRED_FIELDS: tuple[str, ...] = (
    "price",
    "eps",
    "book_value_per_share",
    "free_cash_flow_per_share",
    "growth_rate",
    "discount_rate",
    "revenue",
    "net_income",
)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            numeric = float(token)
        except ValueError:
            return None
    else:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    if not pd.notna(numeric):
        return None
    return float(numeric)


def _coerce_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    try:
        converted = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if isinstance(converted, pd.Series):  # pragma: no cover - defensive
        if converted.empty:
            return None
        converted = converted.iloc[0]
    if not isinstance(converted, pd.Timestamp):
        return None
    return converted.to_pydatetime()


def _normalise_symbol(value: object, default: str | None = None) -> str | None:
    if isinstance(value, str):
        token = value.strip().upper()
        return token or None
    if value is None:
        return default
    return _normalise_symbol(str(value), default=default)


def _normalise_fields(mapping: Mapping[str, object] | None) -> dict[str, float | None]:
    if not mapping:
        return {}
    fields: dict[str, float | None] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        fields[key.strip().lower()] = _coerce_float(value)
    return fields


def _extract_fields(payload: Mapping[str, object]) -> dict[str, float | None]:
    for key in ("metrics", "snapshot", "fundamentals", "data"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return _normalise_fields(candidate)
    # fallback: treat payload sans control keys as values
    ignored = {
        "symbol",
        "ticker",
        "as_of",
        "timestamp",
        "updated_at",
        "report_date",
        "source",
        "provider",
        "metadata",
    }
    data_fields: dict[str, object] = {
        key: value for key, value in payload.items() if key not in ignored and isinstance(key, str)
    }
    return _normalise_fields(data_fields)


def _iter_records(
    records: Sequence[Mapping[str, object]] | Mapping[str, object] | None,
    expected_symbols: Sequence[str] | None,
) -> dict[str, dict[str, object]]:
    """Normalise raw records into a symbol keyed mapping."""

    normalised: dict[str, dict[str, object]] = {}
    entries: Iterable[tuple[str | None, Mapping[str, object]]]

    if records is None:
        entries = []
    elif isinstance(records, Mapping) and "symbol" not in records:
        entries = ((symbol, payload) for symbol, payload in records.items() if isinstance(payload, Mapping))
    else:
        sequence: Iterable[Mapping[str, object]]
        if isinstance(records, Mapping):
            sequence = (records,)  # type: ignore[assignment]
        else:
            sequence = records
        entries = ((None, payload) for payload in sequence if isinstance(payload, Mapping))

    for default_symbol, payload in entries:
        symbol_value = payload.get("symbol") or payload.get("ticker") or default_symbol
        symbol = _normalise_symbol(symbol_value)
        if not symbol:
            continue
        as_of = _coerce_datetime(
            payload.get("as_of")
            or payload.get("timestamp")
            or payload.get("updated_at")
            or payload.get("report_date")
        )
        fields = _extract_fields(payload)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        record = {
            "symbol": symbol,
            "as_of": as_of,
            "fields": fields,
            "source": payload.get("source") or payload.get("provider"),
            "metadata": dict(metadata),
        }
        existing = normalised.get(symbol)
        if existing is None:
            normalised[symbol] = record
            continue
        existing_as_of = existing.get("as_of")
        if as_of is None:
            continue
        if existing_as_of is None or (isinstance(existing_as_of, datetime) and as_of > existing_as_of):
            normalised[symbol] = record

    if expected_symbols:
        for symbol_value in expected_symbols:
            symbol = _normalise_symbol(symbol_value)
            if not symbol:
                continue
            normalised.setdefault(symbol, {
                "symbol": symbol,
                "as_of": None,
                "fields": {},
                "source": None,
                "metadata": {},
            })
    return normalised


def _status_from_score(score: float, warn_threshold: float, error_threshold: float) -> FundamentalQualityStatus:
    if score < error_threshold:
        return FundamentalQualityStatus.error
    if score < warn_threshold:
        return FundamentalQualityStatus.warn
    return FundamentalQualityStatus.ok


def _escalate(current: FundamentalQualityStatus, candidate: FundamentalQualityStatus) -> FundamentalQualityStatus:
    order = {
        FundamentalQualityStatus.ok: 0,
        FundamentalQualityStatus.warn: 1,
        FundamentalQualityStatus.error: 2,
    }
    if order[candidate] > order[current]:
        return candidate
    return current


def evaluate_fundamental_quality(
    records: Sequence[Mapping[str, object]] | Mapping[str, object] | None,
    *,
    expected_symbols: Sequence[str] | None = None,
    required_fields: Sequence[str] | None = None,
    warn_threshold: float = 0.85,
    error_threshold: float = 0.6,
    freshness_warn_hours: float = 168.0,
    freshness_error_hours: float = 336.0,
    now: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
) -> FundamentalQualityReport:
    """Evaluate fundamental dataset quality and return a report."""

    required = tuple(field.lower() for field in (required_fields or _DEFAULT_REQUIRED_FIELDS))
    timestamp = now or datetime.now(tz=UTC)
    symbol_map = _iter_records(records, expected_symbols)

    checks: list[FundamentalQualityCheck] = []
    overall_status = FundamentalQualityStatus.ok
    overall_score = 1.0
    total_required_fields = len(required)
    required_present = 0
    missing_symbols: list[str] = []

    for symbol in sorted(symbol_map.keys()):
        record = symbol_map[symbol]
        fields = record["fields"]
        source = record.get("source")
        as_of: datetime | None = record.get("as_of")
        staleness_hours: float | None = None
        if as_of is not None:
            delta = (timestamp - as_of).total_seconds() / 3600.0
            staleness_hours = max(delta, 0.0)
        present_fields = [field for field in required if fields.get(field) is not None]
        missing_fields = [field for field in required if field not in present_fields]
        coverage_ratio = (
            len(present_fields) / total_required_fields if total_required_fields else 1.0
        )
        required_present += len(present_fields)

        score = 1.0
        issues: list[str] = []
        messages: list[str] = []

        if not fields:
            issues.append("no_fundamentals")
            messages.append("No fundamental data available for symbol")
            score = 0.0
        else:
            score = min(score, coverage_ratio)
            if missing_fields:
                issues.append("missing_fields")
                messages.append(
                    "Missing fundamental fields: " + ", ".join(sorted(missing_fields))
                )

            price = fields.get("price")
            eps = fields.get("eps")
            bvps = fields.get("book_value_per_share")
            fcf_ps = fields.get("free_cash_flow_per_share")
            growth = fields.get("growth_rate")
            discount = fields.get("discount_rate")
            dividend_yield = fields.get("dividend_yield")

            if price is None or price <= 0:
                issues.append("price_invalid")
                messages.append("Price missing or non-positive")
                score = min(score, 0.1 if price is not None else 0.0)
            if eps is None:
                issues.append("eps_missing")
            elif eps == 0:
                issues.append("eps_zero")
                score = min(score, 0.35)
            elif price and eps:
                pe_ratio = price / eps if eps != 0 else None
                if pe_ratio is not None:
                    if pe_ratio < 0:
                        issues.append("negative_pe")
                        score = min(score, 0.25)
                    elif pe_ratio > 140:
                        issues.append("extreme_pe")
                        score = min(score, 0.55)
            if bvps is not None and bvps <= 0:
                issues.append("book_value_non_positive")
                score = min(score, 0.4)
            if fcf_ps is not None and price and price > 0:
                fcf_yield = fcf_ps / price if price else None
                if fcf_yield is not None and fcf_yield < -0.5:
                    issues.append("negative_fcf_yield")
                    score = min(score, 0.35)
            if growth is not None and abs(growth) > 2:
                issues.append("growth_outlier")
                score = min(score, 0.5)
            if discount is not None and discount <= 0:
                issues.append("discount_invalid")
                score = min(score, 0.45)
            if dividend_yield is not None and dividend_yield < 0:
                issues.append("negative_dividend_yield")
                score = min(score, 0.5)

        if staleness_hours is None:
            issues.append("missing_timestamp")
            messages.append("No as_of timestamp provided")
            score = min(score, 0.6)
        else:
            if freshness_error_hours and staleness_hours > freshness_error_hours:
                issues.append("stale_data_error")
                messages.append(
                    f"Fundamental snapshot stale by {staleness_hours:.1f}h (error threshold {freshness_error_hours:.1f}h)"
                )
                score = min(score, 0.25)
            elif freshness_warn_hours and staleness_hours > freshness_warn_hours:
                issues.append("stale_data_warn")
                messages.append(
                    f"Fundamental snapshot stale by {staleness_hours:.1f}h (warn threshold {freshness_warn_hours:.1f}h)"
                )
                score = min(score, 0.55)

        status = _status_from_score(score, warn_threshold, error_threshold)

        if score <= 0.0 and not fields:
            missing_symbols.append(symbol)

        check_metadata = {
            "source": source,
            "fields_present": sorted(present_fields),
            "as_of": as_of.isoformat() if as_of is not None else None,
            "raw_field_count": len(fields),
        }
        if record.get("metadata"):
            check_metadata["metadata"] = record["metadata"]

        check = FundamentalQualityCheck(
            symbol=symbol,
            status=status,
            score=round(max(score, 0.0), 4),
            coverage_ratio=round(coverage_ratio, 4),
            staleness_hours=None if staleness_hours is None else round(staleness_hours, 4),
            missing_fields=tuple(sorted(missing_fields)),
            issues=tuple(sorted(set(issues))),
            messages=tuple(messages),
            metadata=check_metadata,
        )
        checks.append(check)
        overall_status = _escalate(overall_status, status)
        overall_score = min(overall_score, check.score)

    overall_score = round(overall_score if checks else 0.0, 4)

    report_metadata: dict[str, object] = {
        "warn_threshold": warn_threshold,
        "error_threshold": error_threshold,
        "freshness_warn_hours": freshness_warn_hours,
        "freshness_error_hours": freshness_error_hours,
        "required_fields": list(required),
    }
    if metadata:
        report_metadata.update({str(key): value for key, value in metadata.items()})
    if missing_symbols:
        report_metadata["missing_symbols"] = sorted(missing_symbols)
    if total_required_fields and checks:
        coverage_possible = total_required_fields * len(checks)
        report_metadata["required_field_coverage"] = round(required_present / coverage_possible, 4)

    report = FundamentalQualityReport(
        status=overall_status if checks else FundamentalQualityStatus.error,
        score=overall_score,
        generated_at=timestamp,
        checks=tuple(checks),
        metadata=report_metadata,
    )
    return report
