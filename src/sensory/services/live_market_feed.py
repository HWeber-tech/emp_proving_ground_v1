"""Live-market feed monitor bridging RealDataManager into the sensory cortex."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

from src.data_foundation.monitoring.feed_anomaly import (
    FeedAnomalyConfig,
    FeedAnomalyReport,
    Tick,
    analyse_feed,
)
from src.data_integration.real_data_integration import (
    BackboneConnectivityReport,
    RealDataManager,
)
from src.sensory.monitoring.live_diagnostics import (
    LiveSensoryDiagnostics,
    build_live_sensory_diagnostics,
)
from src.sensory.real_sensory_organ import RealSensoryOrgan

logger = logging.getLogger(__name__)

__all__ = [
    "LiveMarketSnapshot",
    "LiveMarketFeedMonitor",
]


@dataclass(slots=True)
class LiveMarketSnapshot:
    """Aggregated snapshot combining market data, sensory diagnostics, and feed health."""

    symbol: str
    market_data: pd.DataFrame
    diagnostics: LiveSensoryDiagnostics
    feed_report: FeedAnomalyReport
    connectivity: BackboneConnectivityReport | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialise the snapshot into JSON-friendly primitives."""

        return {
            "symbol": self.symbol,
            "market_data": _serialise_frame(self.market_data),
            "diagnostics": self.diagnostics.as_dict(),
            "feed_report": self.feed_report.as_dict(),
            "connectivity": self.connectivity.as_dict() if self.connectivity else None,
        }


class LiveMarketFeedMonitor:
    """Coordinate live market fetches, sensory fusion, and feed anomaly checks."""

    def __init__(
        self,
        *,
        manager: RealDataManager,
        organ: RealSensoryOrgan | None = None,
        feed_config: FeedAnomalyConfig | None = None,
    ) -> None:
        self._manager = manager
        self._organ = organ or RealSensoryOrgan()
        self._feed_config = feed_config or FeedAnomalyConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(
        self,
        symbol: str,
        *,
        period: str | None = None,
        interval: str | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        narrative_events: Sequence[Any] | None = None,
        macro_regime_flags: Mapping[str, float] | None = None,
        include_connectivity: bool = True,
    ) -> LiveMarketSnapshot:
        """Fetch market data synchronously and build a live sensory snapshot."""

        frame = self._manager.fetch_data(
            symbol,
            period=period,
            interval=interval,
            start=start,
            end=end,
        )
        return self._build_snapshot(
            symbol,
            frame,
            narrative_events=narrative_events,
            macro_regime_flags=macro_regime_flags,
            include_connectivity=include_connectivity,
        )

    async def capture_async(
        self,
        symbol: str,
        *,
        period: str | None = None,
        interval: str | None = None,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        narrative_events: Sequence[Any] | None = None,
        macro_regime_flags: Mapping[str, float] | None = None,
        include_connectivity: bool = True,
    ) -> LiveMarketSnapshot:
        """Fetch market data asynchronously and build a live sensory snapshot."""

        async_fetch = getattr(self._manager, "get_market_data", None)
        if callable(async_fetch):
            frame = await async_fetch(
                symbol,
                period=period,
                interval=interval,
                start=start,
                end=end,
            )
        else:
            frame = await asyncio.to_thread(
                self._manager.fetch_data,
                symbol,
                period,
                interval,
                start,
                end,
            )
        return self._build_snapshot(
            symbol,
            frame,
            narrative_events=narrative_events,
            macro_regime_flags=macro_regime_flags,
            include_connectivity=include_connectivity,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_snapshot(
        self,
        symbol: str,
        frame: Any,
        *,
        narrative_events: Sequence[Any] | None,
        macro_regime_flags: Mapping[str, float] | None,
        include_connectivity: bool,
    ) -> LiveMarketSnapshot:
        if frame is None:
            raise ValueError("RealDataManager returned no market data")
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        if frame.empty:
            raise ValueError("Market data frame is empty; live feed unavailable")

        market_frame = _normalise_market_frame(frame, symbol=symbol)
        if market_frame.empty:
            raise ValueError("Normalised market data frame is empty; cannot build snapshot")

        diagnostics = build_live_sensory_diagnostics(
            market_frame,
            symbol=symbol,
            organ=self._organ,
            narrative_events=narrative_events,
            macro_regime_flags=macro_regime_flags,
        )

        ticks = _dataframe_to_ticks(market_frame)
        feed_report = analyse_feed(
            symbol,
            ticks,
            config=self._feed_config,
        )

        connectivity: BackboneConnectivityReport | None = None
        if include_connectivity:
            connectivity_fn = getattr(self._manager, "connectivity_report", None)
            if callable(connectivity_fn):
                try:
                    connectivity = connectivity_fn()
                except Exception:  # pragma: no cover - defensive logging only
                    logger.debug("Failed to obtain backbone connectivity report", exc_info=True)

        return LiveMarketSnapshot(
            symbol=symbol,
            market_data=market_frame,
            diagnostics=diagnostics,
            feed_report=feed_report,
            connectivity=connectivity,
        )


# ----------------------------------------------------------------------
# Serialisation helpers
# ----------------------------------------------------------------------


def _serialise_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    serialised: list[dict[str, Any]] = []
    for row in records:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                resolved = value.tz_convert(timezone.utc) if value.tzinfo else value.tz_localize(timezone.utc)
                converted[key] = resolved.isoformat()
            elif isinstance(value, datetime):
                resolved_dt = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
                converted[key] = resolved_dt.isoformat()
            else:
                converted[key] = value
        serialised.append(converted)
    return serialised


def _normalise_market_frame(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    df = frame.copy()
    columns_lower = {str(column).lower(): column for column in df.columns}

    timestamp_column = None
    for candidate in ("timestamp", "date", "datetime"):
        if candidate in columns_lower:
            timestamp_column = columns_lower[candidate]
            break

    if timestamp_column is None:
        raise ValueError("Market data frame missing timestamp column")

    if timestamp_column != "timestamp":
        df = df.rename(columns={timestamp_column: "timestamp"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.loc[~df["timestamp"].isna()].copy()

    price_source = None
    for candidate in ("close", "adj_close", "price"):
        if candidate in df.columns:
            price_source = candidate
            break
    if price_source is None:
        raise ValueError("Market data frame missing price column (close/adj_close/price)")

    if "close" not in df.columns:
        df["close"] = df[price_source]

    for column in ("open", "high", "low"):
        if column not in df.columns:
            df[column] = df[price_source]

    if "volume" not in df.columns:
        if "size" in df.columns:
            df["volume"] = df["size"]
        else:
            df["volume"] = 0.0

    if "symbol" not in df.columns:
        df["symbol"] = symbol
    else:
        df["symbol"] = df["symbol"].fillna(symbol).astype(str)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _dataframe_to_ticks(frame: pd.DataFrame) -> list[Tick]:
    if frame.empty:
        return []

    price_column = None
    for candidate in ("price", "close", "adj_close"):
        if candidate in frame.columns:
            price_column = candidate
            break
    if price_column is None:
        return []

    volume_column = None
    for candidate in ("size", "volume"):
        if candidate in frame.columns:
            volume_column = candidate
            break

    ticks: list[Tick] = []
    for row in frame.itertuples(index=False):
        timestamp_value = getattr(row, "timestamp", None)
        if timestamp_value is None:
            continue
        timestamp = _coerce_datetime(timestamp_value)
        if timestamp is None:
            continue

        price_value = getattr(row, price_column)
        if price_value is None or pd.isna(price_value):
            continue
        try:
            price = float(price_value)
        except (TypeError, ValueError):
            continue

        volume: float | None = None
        if volume_column is not None:
            volume_value = getattr(row, volume_column)
            if volume_value is not None and not pd.isna(volume_value):
                try:
                    volume = float(volume_value)
                except (TypeError, ValueError):
                    volume = None

        ticks.append(Tick(timestamp=timestamp, price=price, volume=volume))

    ticks.sort(key=lambda tick: tick.timestamp)
    return ticks


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    try:
        converted = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if isinstance(converted, pd.Timestamp):
        if pd.isna(converted):
            return None
        return converted.to_pydatetime()
    if isinstance(converted, datetime):
        return converted if converted.tzinfo else converted.replace(tzinfo=timezone.utc)
    return None
