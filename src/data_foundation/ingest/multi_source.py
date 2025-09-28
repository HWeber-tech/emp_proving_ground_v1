"""Multi-source market data aggregation with quality validators.

This module fulfils the high-impact roadmap requirement to consolidate the
free Tier-0 data vendors (Yahoo Finance, Alpha Vantage, FRED) behind a single
aggregation interface.  The aggregator normalises per-vendor datasets into a
canonical bar schema, stitches gaps across sources, and executes a suite of
data-quality validators so downstream components can reason about coverage and
alignment before the data enters trading or research workflows.

The implementation intentionally stays dependency-light (``pandas`` only) so it
can run in CI and offline notebooks without access to premium feeds.  Providers
are supplied as callables, enabling deterministic testing and local fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

ProviderFetch = Callable[[Sequence[str], datetime, datetime], pd.DataFrame]


_BAR_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")


class DataQualitySeverity(str, Enum):
    """Enumeration of severity levels for quality findings."""

    ok = "ok"
    warning = "warning"
    error = "error"


@dataclass(slots=True, frozen=True)
class ProviderSpec:
    """Specification describing how to fetch data from a provider."""

    name: str
    fetch: ProviderFetch
    required: bool = False


@dataclass(slots=True, frozen=True)
class ProviderSnapshot:
    """Normalised snapshot returned by a provider fetch."""

    name: str
    data: pd.DataFrame
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ProviderContribution:
    """Value contributed by a provider for a symbol/timestamp pair."""

    source: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None


@dataclass(slots=True, frozen=True)
class AggregationMetadata:
    """Metadata describing the aggregated dataset."""

    symbols: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    frequency: str


@dataclass(slots=True, frozen=True)
class DataQualityFinding:
    """Outcome returned by a quality validator."""

    name: str
    severity: DataQualitySeverity
    message: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class AggregationResult:
    """Aggregated market data accompanied by quality diagnostics."""

    data: pd.DataFrame
    metadata: AggregationMetadata
    provider_snapshots: tuple[ProviderSnapshot, ...]
    contributions: Mapping[tuple[str, pd.Timestamp], tuple[ProviderContribution, ...]]
    quality_findings: tuple[DataQualityFinding, ...] = ()

    @property
    def status(self) -> DataQualitySeverity:
        """Return the highest severity across quality findings."""

        if not self.quality_findings:
            return DataQualitySeverity.ok
        precedence = {
            DataQualitySeverity.ok: 0,
            DataQualitySeverity.warning: 1,
            DataQualitySeverity.error: 2,
        }
        max_level = max(precedence[finding.severity] for finding in self.quality_findings)
        for severity, level in precedence.items():
            if level == max_level:
                return severity
        return DataQualitySeverity.ok


class DataQualityValidator:
    """Interface implemented by quality validators."""

    name: str

    def evaluate(self, result: AggregationResult) -> DataQualityFinding:  # pragma: no cover - protocol
        raise NotImplementedError


def _as_utc(timestamp: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _nan_to_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)) and np.isnan(value):
        return None
    return float(value)


def _normalise_frame(
    frame: pd.DataFrame, *, provider: str, symbols: set[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=_BAR_COLUMNS)
    missing = [column for column in _BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Provider {provider} missing required columns: {missing}")

    normalised = frame.copy()
    normalised["timestamp"] = pd.to_datetime(normalised["timestamp"], utc=True, errors="coerce")
    normalised = normalised.dropna(subset=["timestamp", "symbol"])
    normalised["symbol"] = normalised["symbol"].astype(str).str.upper().str.strip()
    normalised = normalised[normalised["symbol"].isin(symbols)]
    normalised = normalised[(normalised["timestamp"] >= start) & (normalised["timestamp"] <= end)]
    for column in _NUMERIC_COLUMNS:
        normalised[column] = pd.to_numeric(normalised[column], errors="coerce")
    normalised = normalised.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return normalised


def _combine_snapshots(
    snapshots: Sequence[ProviderSnapshot],
) -> tuple[pd.DataFrame, dict[tuple[str, pd.Timestamp], tuple[ProviderContribution, ...]]]:
    frames: list[pd.DataFrame] = []
    contributions: dict[tuple[str, pd.Timestamp], tuple[ProviderContribution, ...]] = {}

    for priority, snapshot in enumerate(snapshots):
        if snapshot.data.empty:
            continue
        frame = snapshot.data.copy()
        frame["source"] = snapshot.name
        frame["_priority"] = priority
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=(*_BAR_COLUMNS, "primary_source", "sources")), contributions

    stacked = pd.concat(frames, ignore_index=True)
    stacked = stacked.sort_values(["symbol", "timestamp", "_priority"]).reset_index(drop=True)

    combined_rows: list[pd.Series] = []
    for (symbol, timestamp), group in stacked.groupby(["symbol", "timestamp"], sort=False):
        sources_used: list[str] = []
        for row in group.itertuples(index=False):
            if any(not pd.isna(getattr(row, column)) for column in _NUMERIC_COLUMNS):
                if row.source not in sources_used:
                    sources_used.append(row.source)
        if not sources_used:
            sources_used = list(dict.fromkeys(group["source"].tolist()))
        base = group.iloc[0].copy()
        primary_source = None
        for row in group.itertuples(index=False):
            if not pd.isna(row.close):
                primary_source = row.source
                break
        if primary_source is None:
            primary_source = base["source"]

        for column in _NUMERIC_COLUMNS:
            if pd.isna(base[column]):
                for _, row in group.iloc[1:].iterrows():
                    if not pd.isna(row[column]):
                        base[column] = row[column]
                        break

        base["source"] = primary_source
        base["primary_source"] = primary_source
        base["sources"] = tuple(sources_used)
        base = base.drop(labels="_priority")
        combined_rows.append(base)

        contributions[(symbol, timestamp)] = tuple(
            ProviderContribution(
                source=row.source,
                open=_nan_to_none(row.open),
                high=_nan_to_none(row.high),
                low=_nan_to_none(row.low),
                close=_nan_to_none(row.close),
                volume=_nan_to_none(row.volume),
            )
            for row in group.itertuples(index=False)
        )

    aggregated = pd.DataFrame(combined_rows)
    aggregated = aggregated.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    aggregated["timestamp"] = pd.to_datetime(aggregated["timestamp"], utc=True)
    return aggregated, contributions


class CoverageValidator(DataQualityValidator):
    """Ensure datasets meet minimum coverage targets."""

    def __init__(
        self,
        *,
        frequency: str = "1D",
        warn_ratio: float = 0.9,
        error_ratio: float = 0.75,
    ) -> None:
        self.name = "coverage"
        self.frequency = frequency
        self.warn_ratio = warn_ratio
        self.error_ratio = error_ratio

    def evaluate(self, result: AggregationResult) -> DataQualityFinding:
        metadata = result.metadata
        if not metadata.symbols:
            return DataQualityFinding(
                name=self.name,
                severity=DataQualitySeverity.ok,
                message="no symbols requested",
            )

        periods = pd.date_range(metadata.start, metadata.end, freq=self.frequency)
        expected_per_symbol = max(len(periods), 1)

        ratios: MutableMapping[str, float] = {}
        for symbol in metadata.symbols:
            observed = result.data.loc[result.data["symbol"] == symbol, "timestamp"].nunique()
            ratios[symbol] = observed / expected_per_symbol if expected_per_symbol else 1.0

        overall_ratio = sum(ratios.values()) / len(ratios)
        severity = DataQualitySeverity.ok
        breaches = {symbol: ratio for symbol, ratio in ratios.items() if ratio < self.warn_ratio}
        if breaches:
            worst_ratio = min(breaches.values())
            severity = (
                DataQualitySeverity.error
                if worst_ratio < self.error_ratio
                else DataQualitySeverity.warning
            )
        message = (
            "coverage within thresholds"
            if severity is DataQualitySeverity.ok
            else "coverage below target for: " + ", ".join(sorted(breaches))
        )
        return DataQualityFinding(
            name=self.name,
            severity=severity,
            message=message,
            metrics={
                "coverage_ratio": overall_ratio,
                "expected_per_symbol": float(expected_per_symbol),
            },
            details={"per_symbol": ratios},
        )


class StalenessValidator(DataQualityValidator):
    """Check that the latest observation per symbol is recent enough."""

    def __init__(
        self,
        *,
        max_staleness: timedelta,
        warn_staleness: timedelta | None = None,
    ) -> None:
        self.name = "staleness"
        self.max_staleness = max_staleness
        self.warn_staleness = warn_staleness or max_staleness

    def evaluate(self, result: AggregationResult) -> DataQualityFinding:
        metadata = result.metadata
        if result.data.empty:
            return DataQualityFinding(
                name=self.name,
                severity=DataQualitySeverity.error,
                message="no data available",
            )

        end_timestamp = metadata.end.to_pydatetime()
        stale_symbols: dict[str, float] = {}
        warn_symbols: dict[str, float] = {}
        for symbol in metadata.symbols:
            subset = result.data.loc[result.data["symbol"] == symbol, "timestamp"]
            if subset.empty:
                stale_symbols[symbol] = float(self.max_staleness.total_seconds())
                continue
            latest = subset.max().to_pydatetime()
            age = (end_timestamp - latest).total_seconds()
            if age > self.max_staleness.total_seconds():
                stale_symbols[symbol] = age
            elif age > self.warn_staleness.total_seconds():
                warn_symbols[symbol] = age

        if stale_symbols:
            severity = DataQualitySeverity.error
            message = "stale data for: " + ", ".join(sorted(stale_symbols))
            metrics = {"max_staleness_seconds": max(stale_symbols.values())}
            details = {"stale_symbols": stale_symbols}
        elif warn_symbols:
            severity = DataQualitySeverity.warning
            message = "approaching staleness threshold: " + ", ".join(sorted(warn_symbols))
            metrics = {"max_staleness_seconds": max(warn_symbols.values())}
            details = {"warn_symbols": warn_symbols}
        else:
            severity = DataQualitySeverity.ok
            message = "latest observations within staleness threshold"
            metrics = {"max_staleness_seconds": 0.0}
            details = {}

        return DataQualityFinding(
            name=self.name,
            severity=severity,
            message=message,
            metrics=metrics,
            details=details,
        )


class CrossSourceDriftValidator(DataQualityValidator):
    """Detect price drift between providers for overlapping candles."""

    def __init__(
        self,
        *,
        tolerance: float,
        warn_tolerance: float | None = None,
    ) -> None:
        self.name = "cross_source_drift"
        self.tolerance = tolerance
        self.warn_tolerance = warn_tolerance or tolerance

    def evaluate(self, result: AggregationResult) -> DataQualityFinding:
        breaches: dict[str, float] = {}
        max_drift = 0.0

        for (symbol, timestamp), contributions in result.contributions.items():
            closes = [contrib.close for contrib in contributions if contrib.close is not None]
            if len(closes) <= 1:
                continue
            max_close = max(closes)
            min_close = min(closes)
            if max_close == 0:
                continue
            drift = abs(max_close - min_close) / max_close
            max_drift = max(max_drift, drift)
            if drift > self.warn_tolerance:
                key = f"{symbol}@{timestamp.isoformat()}"
                breaches[key] = drift

        if not breaches:
            return DataQualityFinding(
                name=self.name,
                severity=DataQualitySeverity.ok,
                message="no cross-source drift detected",
                metrics={"max_relative_drift": max_drift},
            )

        worst = max(breaches.values())
        severity = (
            DataQualitySeverity.error
            if worst > self.tolerance
            else DataQualitySeverity.warning
        )
        message = "cross-source drift above threshold"
        return DataQualityFinding(
            name=self.name,
            severity=severity,
            message=message,
            metrics={"max_relative_drift": worst},
            details={"breaches": breaches},
        )


class MultiSourceAggregator:
    """Aggregate market data across multiple vendors with validation."""

    def __init__(
        self,
        providers: Sequence[ProviderSpec],
        *,
        validators: Sequence[DataQualityValidator] | None = None,
        frequency: str = "1D",
    ) -> None:
        if not providers:
            raise ValueError("at least one provider must be supplied")
        self._providers = tuple(providers)
        self._validators = tuple(validators or ())
        self._frequency = frequency

    def aggregate(
        self,
        symbols: Iterable[str],
        *,
        start: datetime,
        end: datetime,
    ) -> AggregationResult:
        if start > end:
            raise ValueError("start must be before end")
        symbol_list = tuple({sym.strip().upper() for sym in symbols if sym})
        if not symbol_list:
            raise ValueError("at least one symbol must be provided")

        start_ts = _as_utc(start)
        end_ts = _as_utc(end)
        if start_ts > end_ts:
            raise ValueError("start must be before end")

        snapshots: list[ProviderSnapshot] = []
        for spec in self._providers:
            frame = spec.fetch(symbol_list, start_ts.to_pydatetime(), end_ts.to_pydatetime())
            normalised = _normalise_frame(frame, provider=spec.name, symbols=set(symbol_list), start=start_ts, end=end_ts)
            metadata = {
                "rows": int(normalised.shape[0]),
                "columns": list(normalised.columns),
                "required": spec.required,
            }
            if spec.required and normalised.empty:
                raise ValueError(f"required provider {spec.name} returned no data")
            snapshots.append(ProviderSnapshot(name=spec.name, data=normalised, metadata=metadata))

        aggregated_frame, contributions = _combine_snapshots(snapshots)
        metadata = AggregationMetadata(
            symbols=symbol_list,
            start=start_ts,
            end=end_ts,
            frequency=self._frequency,
        )
        result = AggregationResult(
            data=aggregated_frame,
            metadata=metadata,
            provider_snapshots=tuple(snapshots),
            contributions=contributions,
        )

        if self._validators:
            findings = tuple(validator.evaluate(result) for validator in self._validators)
            result = replace(result, quality_findings=findings)
        return result


__all__ = [
    "AggregationMetadata",
    "AggregationResult",
    "CoverageValidator",
    "CrossSourceDriftValidator",
    "DataQualityFinding",
    "DataQualitySeverity",
    "MultiSourceAggregator",
    "ProviderContribution",
    "ProviderSnapshot",
    "ProviderSpec",
    "StalenessValidator",
]

