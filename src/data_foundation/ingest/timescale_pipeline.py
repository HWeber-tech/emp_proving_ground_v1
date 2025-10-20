"""Orchestrators that load bootstrap data into TimescaleDB."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Protocol

import pandas as pd

from .fred_calendar import fetch_fred_calendar
from .yahoo_ingest import fetch_daily_bars, fetch_intraday_trades
from ..macro_event_enrichment import enrich_macro_event_payload
from ..persist.timescale import (
    TimescaleConnectionSettings,
    TimescaleIngestResult,
    TimescaleIngestor,
    TimescaleMigrator,
)
from ..schemas import MacroEvent


def _default_fetch_fred_calendar(start: str, end: str) -> list[MacroEvent]:
    """Return placeholder macro events when no calendar provider is available."""

    return []


def _normalise_symbols(symbols: Sequence[str]) -> list[str]:
    return [str(symbol).strip() for symbol in symbols if str(symbol).strip()]


def _macro_events_to_frame(events: Sequence[MacroEvent | Mapping[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events:
        if isinstance(event, MacroEvent):
            payload = event.dict()
        else:
            payload = dict(event)
        rows.append(enrich_macro_event_payload(payload))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class DailyBarIngestPlan:
    symbols: Sequence[str]
    lookback_days: int = 60
    source: str = "yahoo"

    def normalised_symbols(self) -> list[str]:
        return _normalise_symbols(self.symbols)


@dataclass(frozen=True)
class IntradayTradeIngestPlan:
    symbols: Sequence[str]
    lookback_days: int = 2
    interval: str = "1m"
    source: str = "yahoo"

    def normalised_symbols(self) -> list[str]:
        return _normalise_symbols(self.symbols)


@dataclass(frozen=True)
class MacroEventIngestPlan:
    start: str | None = None
    end: str | None = None
    events: Sequence[MacroEvent | Mapping[str, object]] | None = None
    source: str = "fred"

    def has_window(self) -> bool:
        return bool(self.start and self.end)


@dataclass(frozen=True)
class TimescaleBackbonePlan:
    daily: DailyBarIngestPlan | None = None
    intraday: IntradayTradeIngestPlan | None = None
    macro: MacroEventIngestPlan | None = None

    def is_empty(self) -> bool:
        return not any((self.daily, self.intraday, self.macro))


class IngestResultPublisher(Protocol):
    """Publish ingest outcomes to downstream systems (e.g. Kafka, EventBus)."""

    def publish(
        self,
        result: TimescaleIngestResult,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None: ...


class TimescaleBackboneOrchestrator:
    """Apply migrations and run ingest slices according to a plan."""

    def __init__(
        self,
        settings: TimescaleConnectionSettings,
        *,
        migrator_cls: type[TimescaleMigrator] = TimescaleMigrator,
        ingestor_cls: type[TimescaleIngestor] = TimescaleIngestor,
        event_publisher: IngestResultPublisher | None = None,
    ) -> None:
        self._settings = settings
        self._migrator_cls = migrator_cls
        self._ingestor_cls = ingestor_cls
        self._logger = logging.getLogger(f"{__name__}.TimescaleBackboneOrchestrator")
        self._event_publisher = event_publisher

    @staticmethod
    def _attach_result_metadata(
        metadata: dict[str, object], result: TimescaleIngestResult
    ) -> None:
        """Augment slice metadata with the persisted ingest result details."""

        payload = result.as_dict()
        metadata["result"] = payload
        metadata["rows_written"] = payload["rows_written"]
        metadata["ingested_symbols"] = payload["symbols"]
        metadata["result_start_ts"] = payload["start_ts"]
        metadata["result_end_ts"] = payload["end_ts"]
        metadata["result_duration_seconds"] = payload["ingest_duration_seconds"]
        metadata["result_freshness_seconds"] = payload["freshness_seconds"]

    def _publish_result(
        self,
        result: TimescaleIngestResult,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if self._event_publisher is None:
            return
        try:
            self._event_publisher.publish(result, metadata=metadata)
        except Exception:  # pragma: no cover - defensive logging
            self._logger.exception(
                "Failed to publish Timescale ingest result for %s", result.dimension
            )

    def run(
        self,
        *,
        plan: TimescaleBackbonePlan,
        fetch_daily: Callable[[list[str], int], pd.DataFrame] = fetch_daily_bars,
        fetch_intraday: Callable[[list[str], int, str], pd.DataFrame]
        | None = fetch_intraday_trades,
        fetch_macro: Callable[[str, str], Sequence[MacroEvent]] = _default_fetch_fred_calendar,
    ) -> dict[str, TimescaleIngestResult]:
        if plan.is_empty():
            self._logger.info("Timescale backbone ingest: nothing to execute")
            return {}

        engine = self._settings.create_engine()
        try:
            migrator = self._migrator_cls(engine)
            migrator.apply()
            ingestor = self._ingestor_cls(engine)
            results: dict[str, TimescaleIngestResult] = {}

            if plan.daily is not None:
                symbols = plan.daily.normalised_symbols()
                metadata: dict[str, object] = {
                    "plan": "daily_bars",
                    "source": plan.daily.source,
                    "requested_symbols": symbols,
                    "lookback_days": plan.daily.lookback_days,
                    "fetched_rows": 0,
                }

                if not symbols:
                    result = TimescaleIngestResult.empty(
                        dimension="daily_bars", source=plan.daily.source
                    )
                else:
                    df = fetch_daily(symbols, plan.daily.lookback_days)
                    metadata["fetched_rows"] = int(len(df))
                    if df.empty:
                        result = TimescaleIngestResult.empty(
                            dimension="daily_bars", source=plan.daily.source
                        )
                    else:
                        result = ingestor.upsert_daily_bars(df, source=plan.daily.source)

                self._attach_result_metadata(metadata, result)
                results["daily_bars"] = result
                self._publish_result(result, metadata)

            if plan.intraday is not None:
                if fetch_intraday is None:
                    raise ValueError("Intraday ingest requested but no fetcher was provided")
                symbols = plan.intraday.normalised_symbols()
                metadata = {
                    "plan": "intraday_trades",
                    "source": plan.intraday.source,
                    "requested_symbols": symbols,
                    "lookback_days": plan.intraday.lookback_days,
                    "interval": plan.intraday.interval,
                    "fetched_rows": 0,
                }
                if not symbols:
                    result = TimescaleIngestResult.empty(
                        dimension="intraday_trades", source=plan.intraday.source
                    )
                else:
                    trades = fetch_intraday(
                        symbols, plan.intraday.lookback_days, plan.intraday.interval
                    )
                    metadata["fetched_rows"] = int(len(trades))
                    if trades.empty:
                        result = TimescaleIngestResult.empty(
                            dimension="intraday_trades", source=plan.intraday.source
                        )
                    else:
                        result = ingestor.upsert_intraday_trades(
                            trades, source=plan.intraday.source
                        )

                self._attach_result_metadata(metadata, result)
                results["intraday_trades"] = result
                self._publish_result(result, metadata)

            if plan.macro is not None:
                events = plan.macro.events
                metadata = {
                    "plan": "macro_events",
                    "source": plan.macro.source,
                    "window": {"start": plan.macro.start, "end": plan.macro.end}
                    if plan.macro.has_window()
                    else None,
                    "provided_events": len(events) if events is not None else 0,
                    "fetched_events": 0,
                    "frame_rows": 0,
                }

                if events is None:
                    if plan.macro.has_window():
                        events = fetch_macro(plan.macro.start or "", plan.macro.end or "")
                        metadata["fetched_via_window"] = True
                        metadata["fetched_events"] = len(events)
                    else:
                        self._logger.warning(
                            "Macro ingest requested without events or a start/end window; skipping"
                        )
                        result = TimescaleIngestResult.empty(
                            dimension="macro_events", source=plan.macro.source
                        )
                        self._attach_result_metadata(metadata, result)
                        results["macro_events"] = result
                        self._publish_result(result, metadata)
                        return results

                if events:
                    frame = _macro_events_to_frame(events)
                    metadata["frame_rows"] = int(len(frame))
                    if frame.empty:
                        result = TimescaleIngestResult.empty(
                            dimension="macro_events", source=plan.macro.source
                        )
                    else:
                        result = ingestor.upsert_macro_events(frame, source=plan.macro.source)
                else:
                    result = TimescaleIngestResult.empty(
                        dimension="macro_events", source=plan.macro.source
                    )

                self._attach_result_metadata(metadata, result)
                results["macro_events"] = result
                self._publish_result(result, metadata)

            return results
        finally:
            engine.dispose()


def ingest_yahoo_daily_bars(
    *,
    settings: TimescaleConnectionSettings,
    symbols: Sequence[str],
    lookback_days: int = 60,
) -> TimescaleIngestResult:
    """Fetch Yahoo daily bars and upsert them into TimescaleDB."""

    plan = TimescaleBackbonePlan(
        daily=DailyBarIngestPlan(symbols=symbols, lookback_days=lookback_days, source="yahoo")
    )
    orchestrator = TimescaleBackboneOrchestrator(settings)
    results = orchestrator.run(plan=plan, fetch_daily=fetch_daily_bars)
    return results.get(
        "daily_bars",
        TimescaleIngestResult.empty(dimension="daily_bars", source="yahoo"),
    )


def ingest_yahoo_intraday_trades(
    *,
    settings: TimescaleConnectionSettings,
    symbols: Sequence[str],
    lookback_days: int = 2,
    interval: str = "1m",
) -> TimescaleIngestResult:
    """Fetch Yahoo intraday data and persist it as trade records."""

    plan = TimescaleBackbonePlan(
        intraday=IntradayTradeIngestPlan(
            symbols=symbols,
            lookback_days=lookback_days,
            interval=interval,
            source="yahoo",
        )
    )
    orchestrator = TimescaleBackboneOrchestrator(settings)
    results = orchestrator.run(plan=plan, fetch_intraday=fetch_intraday_trades)
    return results.get(
        "intraday_trades",
        TimescaleIngestResult.empty(dimension="intraday_trades", source="yahoo"),
    )


def ingest_macro_events(
    *,
    settings: TimescaleConnectionSettings,
    events: Sequence[MacroEvent | Mapping[str, object]],
    source: str = "fred",
) -> TimescaleIngestResult:
    """Persist macro events into Timescale."""

    plan = TimescaleBackbonePlan(macro=MacroEventIngestPlan(events=events, source=source))
    orchestrator = TimescaleBackboneOrchestrator(settings)
    results = orchestrator.run(plan=plan)
    return results.get(
        "macro_events",
        TimescaleIngestResult.empty(dimension="macro_events", source=source),
    )


def ingest_fred_macro_events(
    *,
    settings: TimescaleConnectionSettings,
    start: str,
    end: str,
) -> TimescaleIngestResult:
    """Fetch macro events using the FRED fetcher and ingest them."""

    plan = TimescaleBackbonePlan(macro=MacroEventIngestPlan(start=start, end=end, source="fred"))
    orchestrator = TimescaleBackboneOrchestrator(settings)
    results = orchestrator.run(plan=plan, fetch_macro=fetch_fred_calendar)
    return results.get(
        "macro_events",
        TimescaleIngestResult.empty(dimension="macro_events", source="fred"),
    )


__all__ = [
    "DailyBarIngestPlan",
    "IntradayTradeIngestPlan",
    "MacroEventIngestPlan",
    "TimescaleBackbonePlan",
    "IngestResultPublisher",
    "TimescaleBackboneOrchestrator",
    "ingest_yahoo_daily_bars",
    "ingest_yahoo_intraday_trades",
    "ingest_macro_events",
    "ingest_fred_macro_events",
]
