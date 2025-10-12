"""Utilities for ingesting a real-market slice and emitting belief/regime state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.core.event_bus import Event
from src.data_foundation.ingest.timescale_pipeline import (
    DailyBarIngestPlan,
    TimescaleBackboneOrchestrator,
    TimescaleBackbonePlan,
)
from src.data_foundation.persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestResult,
)
from src.data_integration.real_data_integration import RealDataManager
from src.sensory.real_sensory_organ import RealSensoryOrgan
from src.understanding.belief import BeliefBuffer, BeliefEmitter, BeliefState, RegimeFSM, RegimeSignal
from src.understanding.belief_regime_calibrator import (
    BeliefRegimeCalibration,
    build_calibrated_belief_components,
)
from src.understanding.belief_real_data_utils import (
    calibrate_from_market_data,
    extract_snapshot_volatility,
    resolve_threshold_scale,
)

__all__ = [
    "RealDataSliceConfig",
    "RealDataSliceOutcome",
    "ingest_daily_slice_from_csv",
    "fetch_enriched_market_frame",
    "build_belief_from_market_data",
    "run_real_data_slice",
]


@dataclass(slots=True, frozen=True)
class RealDataSliceConfig:
    """Configuration describing the real-market slice to ingest."""

    csv_path: Path
    symbol: str
    source: str = "fixture"
    lookback_days: int | None = None
    belief_id: str = "real-data-slice"


@dataclass(slots=True, frozen=True)
class RealDataSliceOutcome:
    """Result bundle produced by :func:`run_real_data_slice`."""

    ingest_result: TimescaleIngestResult
    market_data: pd.DataFrame
    sensory_snapshot: Mapping[str, object]
    belief_state: BeliefState
    regime_signal: RegimeSignal
    calibration: BeliefRegimeCalibration | None


class _InMemoryEventBus:
    """Minimal synchronous bus satisfying the failover helper contract."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def is_running(self) -> bool:
        return True

    def publish_from_sync(self, event: Event) -> int:
        self.events.append(event)
        return 1



def _load_csv(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"CSV file {csv_path} is empty")
    lower = {str(column).strip().lower(): column for column in frame.columns}
    rename_map: dict[str, str] = {}
    for target in ("date", "timestamp"):
        if target in lower:
            rename_map[lower[target]] = "date"
            break
    if "symbol" not in frame.columns:
        raise ValueError("CSV must contain a 'symbol' column")
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        source = lower.get(column)
        if source is not None and source != column:
            rename_map[source] = column
    if rename_map:
        frame = frame.rename(columns=rename_map)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str)
    if frame["date"].isna().all():
        raise ValueError("CSV must contain at least one valid date/timestamp value")
    return frame


def ingest_daily_slice_from_csv(
    *,
    csv_path: Path,
    symbol: str,
    settings: TimescaleConnectionSettings,
    source: str = "fixture",
    lookback_days: int | None = None,
) -> TimescaleIngestResult:
    """Load a CSV of daily bars and persist it into Timescale."""

    frame = _load_csv(csv_path)
    normalised_symbol = symbol.strip().upper()
    frame = frame.loc[frame["symbol"].str.upper() == normalised_symbol].copy()
    if frame.empty:
        raise ValueError(f"CSV file {csv_path} does not contain data for {normalised_symbol}")

    frame = frame.sort_values("date").reset_index(drop=True)
    frame = frame.rename(columns={"date": "timestamp"})

    days = lookback_days if lookback_days is not None else int(frame.shape[0])
    days = max(days, 1)

    def _fetch_daily(symbols: list[str], _days: int) -> pd.DataFrame:
        if symbols:
            allowed = {value.strip().upper() for value in symbols}
            subset = frame.loc[frame["symbol"].str.upper().isin(allowed)].copy()
        else:
            subset = frame.copy()
        if _days > 0:
            subset = subset.tail(_days)
        subset = subset.rename(columns={"timestamp": "date"})
        return subset

    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(
            symbols=[normalised_symbol],
            lookback_days=days,
            source=source,
        )
    )
    orchestrator = TimescaleBackboneOrchestrator(settings)
    results = orchestrator.run(plan=plan, fetch_daily=_fetch_daily)
    ingest_result = results.get(
        "daily_bars",
        TimescaleIngestResult.empty(dimension="daily_bars", source=source),
    )
    return ingest_result


def fetch_enriched_market_frame(
    *,
    settings: TimescaleConnectionSettings,
    symbol: str,
    period: str | None = None,
) -> pd.DataFrame:
    """Retrieve the Timescale slice and enrich it for sensory processing."""

    manager = RealDataManager(timescale_settings=settings)
    try:
        frame = manager.fetch_data(symbol=symbol, period=period)
    finally:
        manager.close()
    if frame.empty:
        return frame

    enriched = frame.copy()
    enriched = enriched.sort_values("timestamp").reset_index(drop=True)
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], utc=True, errors="coerce")
    enriched["symbol"] = enriched["symbol"].astype(str)
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    enriched["volume"] = enriched["volume"].fillna(0.0)
    enriched["volume"] = enriched["volume"].where(enriched["volume"] > 0.0, 1.0)

    returns = enriched["close"].pct_change().fillna(0.0)
    volatility = returns.rolling(window=5, min_periods=1).std().fillna(0.0)
    spread = (enriched["high"] - enriched["low"]).abs().fillna(0.0)
    fallback_spread = (enriched["close"].abs() * 0.0001).fillna(0.0)
    enriched["volatility"] = volatility.clip(lower=0.0)
    enriched["spread"] = spread.where(spread > 0.0, fallback_spread)
    enriched["depth"] = (1_000_000.0 * (1.0 + returns.abs())).fillna(1_000_000.0)
    enriched["order_imbalance"] = returns.rolling(window=3, min_periods=1).mean().fillna(0.0)
    mean_window = enriched["close"].rolling(window=20, min_periods=1).mean()
    enriched["macro_bias"] = (enriched["close"] / mean_window - 1.0).fillna(0.0)
    enriched["data_quality"] = 0.98

    yield_curves: list[Mapping[str, float]] = []
    for bias in enriched["macro_bias"]:
        direction = float(bias)
        yield_curves.append(
            {
                "2Y": 0.045,
                "10Y": 0.047 + direction * 0.01,
                "30Y": 0.050 + direction * 0.005,
                "direction": direction,
                "confidence": 0.6,
                "slope_2s10s": -0.002 + direction * 0.001,
            }
        )
    enriched["yield_curve"] = yield_curves

    enriched = enriched.dropna(subset=["timestamp", "close"])
    return enriched


def build_belief_from_market_data(
    *,
    market_data: pd.DataFrame,
    symbol: str,
    belief_id: str,
    event_bus: _InMemoryEventBus | None = None,
) -> tuple[Mapping[str, object], BeliefState, RegimeSignal, BeliefRegimeCalibration | None]:
    """Execute the sensory organ and belief/regime buffers for the provided frame."""

    if market_data.empty:
        raise ValueError("Market data frame is empty; cannot build belief state")

    organ = RealSensoryOrgan(event_bus=None)
    snapshot = organ.observe(market_data, symbol=symbol)

    bus = event_bus or _InMemoryEventBus()
    calibration = calibrate_from_market_data(market_data)

    if calibration is not None:
        buffer, emitter, regime_fsm = build_calibrated_belief_components(
            calibration,
            belief_id=belief_id,
            regime_signal_id=f"{belief_id}-regime",
            event_bus=bus,
        )
    else:
        buffer = BeliefBuffer(belief_id=belief_id)
        emitter = BeliefEmitter(buffer=buffer, event_bus=bus)
        regime_fsm = RegimeFSM(event_bus=bus, signal_id=f"{belief_id}-regime")

    volatility_hint = extract_snapshot_volatility(
        snapshot,
        calibration.volatility_feature if calibration is not None else "HOW_signal",
    )
    scale = resolve_threshold_scale(calibration, volatility_hint)
    if scale is not None:
        regime_fsm.apply_threshold_scale(scale)

    state = emitter.emit(snapshot)
    regime_signal = regime_fsm.publish(state)
    return snapshot, state, regime_signal, calibration


def run_real_data_slice(
    *,
    config: RealDataSliceConfig,
    settings: TimescaleConnectionSettings,
) -> RealDataSliceOutcome:
    """Ingest a CSV slice, retrieve enriched data, and emit a belief state."""

    ingest_result = ingest_daily_slice_from_csv(
        csv_path=config.csv_path,
        symbol=config.symbol,
        settings=settings,
        source=config.source,
        lookback_days=config.lookback_days,
    )

    lookback = (
        config.lookback_days
        if config.lookback_days is not None
        else max(int(ingest_result.rows_written), 1)
    )
    period = f"{lookback}d" if lookback else None
    market_data = fetch_enriched_market_frame(
        settings=settings,
        symbol=config.symbol,
        period=period,
    )

    if market_data.empty:
        raise RuntimeError("Timescale returned no rows after ingesting the CSV slice")

    snapshot, belief_state, regime_signal, calibration = build_belief_from_market_data(
        market_data=market_data,
        symbol=config.symbol,
        belief_id=config.belief_id,
    )

    return RealDataSliceOutcome(
        ingest_result=ingest_result,
        market_data=market_data,
        sensory_snapshot=snapshot,
        belief_state=belief_state,
        regime_signal=regime_signal,
        calibration=calibration,
    )
