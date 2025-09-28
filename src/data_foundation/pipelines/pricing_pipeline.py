"""Normalize historical pricing data across vendor APIs.

The high-impact roadmap calls for a canonical pricing pipeline that can stitch
multiple data vendors into the EMP stack without hand-written glue code.  This
module provides a light orchestrator that:

* Accepts a declarative :class:`PricingPipelineConfig` describing the desired
  universe, window and vendor.
* Delegates fetching to pluggable vendor adapters (Yahoo, Alpha Vantage, FRED).
* Normalises the resulting frames into a canonical OHLCV schema with
  deterministic column names.
* Evaluates basic data-quality heuristics so downstream sensors can reject
  stale or incomplete datasets before backtesting.

The implementation intentionally avoids heavy runtime dependencies.  The default
Yahoo adapter piggybacks on the existing ``yahoo_ingest`` helper while the
Alpha Vantage and FRED adapters operate on dependency-free callables.  In
production these adapters can be replaced with richer implementations that hit
vendor APIs or internal data lakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Callable, Mapping, MutableMapping, Protocol, Sequence

import pandas as pd

from src.data_foundation.ingest.yahoo_ingest import fetch_daily_bars

# ---------------------------------------------------------------------------
# Public configuration and result models


@dataclass(frozen=True)
class PricingPipelineConfig:
    """Describe the pricing universe to normalise."""

    symbols: Sequence[str]
    vendor: str = "yahoo"
    interval: str = "1d"
    lookback_days: int = 60
    start: datetime | None = None
    end: datetime | None = None
    minimum_coverage_ratio: float = 0.6

    def normalised_symbols(self) -> list[str]:
        return [sym.strip() for sym in self.symbols if str(sym).strip()]

    def window_start(self) -> datetime:
        if self.start is not None:
            return self._coerce_ts(self.start)
        return self.window_end() - timedelta(days=max(self.lookback_days, 1))

    def window_end(self) -> datetime:
        if self.end is not None:
            return self._coerce_ts(self.end)
        return datetime.now(tz=UTC)

    def candles_per_symbol_hint(self) -> int:
        delta = self.window_end() - self.window_start()
        if delta <= timedelta(0):
            return max(self.lookback_days, 1)
        return max(int(delta.total_seconds() // 86400), 1)

    @staticmethod
    def _coerce_ts(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


@dataclass(frozen=True)
class PricingQualityIssue:
    """Represent a detected data-quality issue."""

    code: str
    severity: str
    message: str
    symbol: str | None = None
    context: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PricingPipelineResult:
    """Canonical result returned by :class:`PricingPipeline`."""

    data: pd.DataFrame
    issues: tuple[PricingQualityIssue, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def symbols(self) -> tuple[str, ...]:
        if self.data.empty:
            return ()
        return tuple(sorted(self.data["symbol"].dropna().unique()))


# ---------------------------------------------------------------------------
# Vendor abstractions


class PricingVendor(Protocol):
    """Minimal protocol every vendor adapter must implement."""

    def fetch(self, config: PricingPipelineConfig) -> pd.DataFrame: ...


class YahooPricingVendor:
    """Adapter delegating to the lightweight Yahoo ingest helper."""

    def __init__(self, fetcher: Callable[[list[str], int], pd.DataFrame] = fetch_daily_bars) -> None:
        self._fetcher = fetcher

    def fetch(self, config: PricingPipelineConfig) -> pd.DataFrame:
        if config.interval not in {"1d", "D"}:
            raise ValueError("Yahoo vendor currently supports daily candles only")
        symbols = config.normalised_symbols()
        if not symbols:
            return pd.DataFrame()
        days = max(config.candles_per_symbol_hint(), 1)
        return self._fetcher(symbols, days=days)


class CallablePricingVendor:
    """Wrap a simple callable to comply with :class:`PricingVendor`."""

    def __init__(self, fetcher: Callable[[PricingPipelineConfig], pd.DataFrame]) -> None:
        self._fetcher = fetcher

    def fetch(self, config: PricingPipelineConfig) -> pd.DataFrame:
        return self._fetcher(config)


# ---------------------------------------------------------------------------
# Pipeline implementation


class PricingPipeline:
    """Normalise OHLCV datasets across vendor implementations."""

    def __init__(
        self,
        *,
        vendor_registry: Mapping[str, PricingVendor] | None = None,
    ) -> None:
        registry: MutableMapping[str, PricingVendor] = {}
        if vendor_registry is not None:
            registry.update(vendor_registry)
        # Always provide the Yahoo adapter unless explicitly overridden
        registry.setdefault("yahoo", YahooPricingVendor())
        self._vendors = dict(registry)

    # ------------------------------------------------------------------
    def run(self, config: PricingPipelineConfig) -> PricingPipelineResult:
        vendor = self._vendors.get(config.vendor)
        if vendor is None:
            raise ValueError(f"Unknown pricing vendor: {config.vendor}")

        raw = vendor.fetch(config)
        frame = self._normalise_frame(raw, source=config.vendor)
        issues = self._validate_frame(frame, config)
        metadata = {
            "vendor": config.vendor,
            "symbol_count": len(config.normalised_symbols()),
            "row_count": int(len(frame)),
            "window_start": config.window_start().isoformat(),
            "window_end": config.window_end().isoformat(),
        }
        return PricingPipelineResult(data=frame, issues=issues, metadata=metadata)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_frame(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "source",
                ]
            )

        df = frame.copy()
        rename_map: dict[str, str] = {}
        for column in list(df.columns):
            lower = column.lower()
            if lower in {"date", "datetime", "timestamp"}:
                rename_map[column] = "timestamp"
            elif lower in {"adj close", "adj_close"}:
                rename_map[column] = "adj_close"
            else:
                rename_map[column] = lower
        df = df.rename(columns=rename_map)

        required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        for column in required:
            if column not in df:
                df[column] = pd.NA

        if "adj_close" not in df:
            df["adj_close"] = df["close"]

        df = df.dropna(subset=["timestamp", "symbol"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["symbol"] = df["symbol"].astype(str)
        numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        for column in numeric_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df["source"] = source
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        return df[[
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "source",
        ]]

    # ------------------------------------------------------------------
    def _validate_frame(
        self, df: pd.DataFrame, config: PricingPipelineConfig
    ) -> tuple[PricingQualityIssue, ...]:
        issues: list[PricingQualityIssue] = []

        if df.empty:
            issues.append(
                PricingQualityIssue(
                    code="no_data",
                    severity="error",
                    message="Vendor returned no pricing rows",
                )
            )
            return tuple(issues)

        duplicated = df[df.duplicated(subset=["symbol", "timestamp"], keep=False)]
        if not duplicated.empty:
            issues.append(
                PricingQualityIssue(
                    code="duplicate_rows",
                    severity="warning",
                    message="Duplicate candles detected",
                    context={"rows": int(len(duplicated))},
                )
            )

        expected_per_symbol = config.candles_per_symbol_hint()
        min_required = max(int(expected_per_symbol * config.minimum_coverage_ratio), 1)
        window_end = config.window_end()
        staleness_cutoff = window_end - timedelta(days=2)

        for symbol, group in df.groupby("symbol"):
            group_count = int(len(group))
            if group_count < min_required:
                issues.append(
                    PricingQualityIssue(
                        code="missing_rows",
                        severity="warning",
                        message="Observed candles below coverage threshold",
                        symbol=symbol,
                        context={
                            "observed": group_count,
                            "expected_hint": expected_per_symbol,
                            "minimum_required": min_required,
                        },
                    )
                )

            unique_close = group["close"].dropna().nunique()
            if group_count > 1 and unique_close <= 1:
                issues.append(
                    PricingQualityIssue(
                        code="flat_prices",
                        severity="warning",
                        message="Close prices are constant across the window",
                        symbol=symbol,
                    )
                )

            latest_ts = group["timestamp"].max()
            if pd.isna(latest_ts) or latest_ts.tzinfo is None:
                continue
            if latest_ts < staleness_cutoff:
                issues.append(
                    PricingQualityIssue(
                        code="stale_series",
                        severity="warning",
                        message="Latest candle predates staleness cutoff",
                        symbol=symbol,
                        context={"latest": latest_ts.isoformat()},
                    )
                )

        return tuple(issues)


__all__ = [
    "PricingPipeline",
    "PricingPipelineConfig",
    "PricingPipelineResult",
    "PricingQualityIssue",
    "PricingVendor",
    "YahooPricingVendor",
    "CallablePricingVendor",
]
