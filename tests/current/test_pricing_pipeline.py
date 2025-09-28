from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from src.data_foundation.pipelines import (
    CallablePricingVendor,
    PricingPipeline,
    PricingPipelineConfig,
)


class _StubVendor:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.requested_config: PricingPipelineConfig | None = None

    def fetch(self, config: PricingPipelineConfig) -> pd.DataFrame:  # pragma: no cover - protocol shim
        self.requested_config = config
        return self._frame


def _frame_for(symbol: str, start: datetime, days: int) -> pd.DataFrame:
    dates = [start + timedelta(days=offset) for offset in range(days)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100 + offset for offset in range(days)],
            "high": [101 + offset for offset in range(days)],
            "low": [99 + offset for offset in range(days)],
            "close": [100.5 + offset for offset in range(days)],
            "volume": [1_000 + offset for offset in range(days)],
            "symbol": [symbol] * days,
        }
    )


def test_pipeline_normalises_vendor_frame_and_reports_metadata() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    frame = _frame_for("EURUSD", start, 3)
    vendor = _StubVendor(frame)
    pipeline = PricingPipeline(vendor_registry={"stub": vendor})
    config = PricingPipelineConfig(symbols=["EURUSD"], vendor="stub", start=start, end=start + timedelta(days=3))

    result = pipeline.run(config)

    assert set(result.data.columns) == {
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "source",
    }
    assert list(result.data["symbol"].unique()) == ["EURUSD"]
    assert result.metadata["vendor"] == "stub"
    assert result.metadata["row_count"] == 3
    assert result.issues == ()
    assert vendor.requested_config is config


def test_pipeline_flags_missing_rows_flat_prices_and_staleness() -> None:
    start = datetime(2024, 2, 1, tzinfo=UTC)
    stale = start - timedelta(days=10)
    frame = pd.DataFrame(
        {
            "timestamp": [stale, stale],
            "symbol": ["ES", "ES"],
            "open": [4500.0, 4500.0],
            "high": [4501.0, 4501.0],
            "low": [4499.0, 4499.0],
            "close": [4500.5, 4500.5],
            "volume": [100, 110],
        }
    )
    vendor = _StubVendor(frame)
    config = PricingPipelineConfig(
        symbols=["ES"],
        vendor="stub",
        start=start - timedelta(days=5),
        end=start,
        minimum_coverage_ratio=0.8,
    )

    pipeline = PricingPipeline(vendor_registry={"stub": vendor})
    result = pipeline.run(config)

    codes = {issue.code for issue in result.issues}
    assert {"missing_rows", "flat_prices", "stale_series"}.issubset(codes)
    stale_issue = next(issue for issue in result.issues if issue.code == "stale_series")
    assert stale_issue.symbol == "ES"
    assert stale_issue.context["latest"].startswith(str(stale.year))



def test_pipeline_detects_duplicate_rows() -> None:
    ts = datetime(2024, 3, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "symbol": ["AAPL", "AAPL"],
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [100, 100],
        }
    )
    pipeline = PricingPipeline(vendor_registry={"stub": _StubVendor(frame)})
    config = PricingPipelineConfig(symbols=["AAPL"], vendor="stub", start=ts - timedelta(days=1), end=ts)

    result = pipeline.run(config)

    assert any(issue.code == "duplicate_rows" for issue in result.issues)



def test_pipeline_handles_empty_vendor_frame() -> None:
    empty_vendor = CallablePricingVendor(lambda _: pd.DataFrame())
    pipeline = PricingPipeline(vendor_registry={"stub": empty_vendor})
    config = PricingPipelineConfig(symbols=["BTCUSD"], vendor="stub")

    result = pipeline.run(config)

    assert result.has_errors()
    assert result.issues[0].code == "no_data"



def test_pipeline_unknown_vendor_raises_value_error() -> None:
    pipeline = PricingPipeline(vendor_registry={})
    config = PricingPipelineConfig(symbols=["EURUSD"], vendor="missing")

    try:
        pipeline.run(config)
    except ValueError as exc:  # pragma: no cover - explicit assertion path
        assert "Unknown pricing vendor" in str(exc)
    else:  # pragma: no cover - ensures failure triggers
        raise AssertionError("Expected ValueError for missing vendor")
