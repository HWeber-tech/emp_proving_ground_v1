from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackbonePlan,
)
from src.runtime.runtime_builder import _normalise_ingest_plan_metadata, _plan_dimensions


def test_plan_dimensions_lists_enabled_slices() -> None:
    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=["EURUSD", " GBPUSD "]),
        intraday=IntradayTradeIngestPlan(symbols=["EURUSD"], lookback_days=1, interval="5m"),
        macro=MacroEventIngestPlan(events=({"event": "NFP"},), start="2024-01-01", end="2024-01-02"),
    )

    assert _plan_dimensions(plan) == ["daily_bars", "intraday_trades", "macro_events"]


def test_normalise_ingest_plan_metadata_handles_structures() -> None:
    mapping_result = _normalise_ingest_plan_metadata({"daily_bars": {"symbols": ["EURUSD"]}})
    sequence_result = _normalise_ingest_plan_metadata(["intraday_trades", "macro_events"])
    scalar_result = _normalise_ingest_plan_metadata("daily_bars")
    none_result = _normalise_ingest_plan_metadata(None)

    assert mapping_result == ["daily_bars"]
    assert sequence_result == ["intraday_trades", "macro_events"]
    assert scalar_result == ["daily_bars"]
    assert none_result == []
