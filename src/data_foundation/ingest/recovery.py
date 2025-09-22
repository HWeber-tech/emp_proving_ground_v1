"""Helpers that propose Timescale ingest recovery plans after degraded runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .configuration import TimescaleIngestRecoverySettings
from .health import IngestHealthReport, IngestHealthStatus
from .timescale_pipeline import (
    DailyBarIngestPlan,
    IntradayTradeIngestPlan,
    MacroEventIngestPlan,
    TimescaleBackbonePlan,
)
from ..persist.timescale import TimescaleIngestResult


@dataclass(frozen=True)
class IngestRecoveryRecommendation:
    """Proposed recovery actions derived from ingest health findings."""

    plan: TimescaleBackbonePlan
    reasons: dict[str, str] = field(default_factory=dict)
    missing_symbols: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return self.plan.is_empty()

    def summary(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        plan_summary = _summarise_plan(self.plan)
        if plan_summary:
            payload["plan"] = plan_summary
        if self.reasons:
            payload["reasons"] = dict(self.reasons)
        if self.missing_symbols:
            payload["missing_symbols"] = {
                dimension: list(symbols) for dimension, symbols in self.missing_symbols.items()
            }
        return payload


def plan_ingest_recovery(
    report: IngestHealthReport,
    *,
    original_plan: TimescaleBackbonePlan,
    results: Mapping[str, TimescaleIngestResult],
    settings: TimescaleIngestRecoverySettings,
    attempt: int = 1,
) -> IngestRecoveryRecommendation:
    """Return a targeted recovery plan for degraded ingest slices."""

    if not settings.should_attempt():
        return IngestRecoveryRecommendation(plan=TimescaleBackbonePlan())

    attempt_index = max(1, int(attempt))
    multiplier = max(1.0, settings.lookback_multiplier) ** attempt_index

    daily_plan: DailyBarIngestPlan | None = None
    intraday_plan: IntradayTradeIngestPlan | None = None
    macro_plan: MacroEventIngestPlan | None = None
    reasons: dict[str, str] = {}
    missing: dict[str, tuple[str, ...]] = {}

    for check in report.checks:
        if check.status is IngestHealthStatus.ok:
            continue

        dimension = check.dimension
        baseline = results.get(dimension) or TimescaleIngestResult.empty(dimension=dimension)
        reasons[dimension] = check.message
        if check.missing_symbols:
            missing[dimension] = tuple(dict.fromkeys(check.missing_symbols))

        if dimension == "daily_bars" and original_plan.daily is not None:
            symbols = list(original_plan.daily.normalised_symbols())
            if settings.target_missing_symbols and check.missing_symbols:
                symbols = list(dict.fromkeys(check.missing_symbols))
            elif not symbols:
                symbols = list(baseline.symbols)
            if not symbols:
                continue
            lookback = int(round(original_plan.daily.lookback_days * multiplier))
            lookback = max(original_plan.daily.lookback_days, lookback)
            daily_plan = DailyBarIngestPlan(
                symbols=symbols,
                lookback_days=lookback,
                source=original_plan.daily.source,
            )
        elif dimension == "intraday_trades" and original_plan.intraday is not None:
            symbols = list(original_plan.intraday.normalised_symbols())
            if settings.target_missing_symbols and check.missing_symbols:
                symbols = list(dict.fromkeys(check.missing_symbols))
            elif not symbols:
                symbols = list(baseline.symbols)
            if not symbols:
                continue
            lookback = int(round(original_plan.intraday.lookback_days * multiplier))
            lookback = max(original_plan.intraday.lookback_days, lookback)
            intraday_plan = IntradayTradeIngestPlan(
                symbols=symbols,
                lookback_days=lookback,
                interval=original_plan.intraday.interval,
                source=original_plan.intraday.source,
            )
        elif dimension == "macro_events" and original_plan.macro is not None:
            macro_plan = _build_macro_recovery(
                original_plan.macro,
                check.missing_symbols if settings.target_missing_symbols else tuple(),
            )
        else:
            reasons.pop(dimension, None)
            missing.pop(dimension, None)

    plan = TimescaleBackbonePlan(daily=daily_plan, intraday=intraday_plan, macro=macro_plan)
    if plan.is_empty():
        return IngestRecoveryRecommendation(plan=plan)

    filtered_reasons = {
        dim: msg
        for dim, msg in reasons.items()
        if dim in {"daily_bars", "intraday_trades", "macro_events"}
    }
    filtered_missing = {
        dim: symbols
        for dim, symbols in missing.items()
        if dim in {"daily_bars", "intraday_trades", "macro_events"}
    }
    return IngestRecoveryRecommendation(
        plan=plan,
        reasons=filtered_reasons,
        missing_symbols=filtered_missing,
    )


def _build_macro_recovery(
    plan: MacroEventIngestPlan,
    missing_symbols: Sequence[str] | tuple[str, ...],
) -> MacroEventIngestPlan:
    events: Sequence[object] | None = plan.events
    if events is not None and missing_symbols:
        targets = {symbol for symbol in missing_symbols if symbol}
        if targets:
            filtered: list[object] = []
            for event in events:
                name = _extract_event_name(event)
                if not targets or name in targets:
                    filtered.append(event)
            if filtered:
                events = tuple(filtered)
    return MacroEventIngestPlan(
        start=plan.start,
        end=plan.end,
        events=events,
        source=plan.source,
    )


def _extract_event_name(event: object) -> str | None:
    if hasattr(event, "event_name"):
        value = getattr(event, "event_name")
        return str(value) if value is not None else None
    if isinstance(event, Mapping):
        candidate = event.get("event_name") or event.get("name")
        return str(candidate) if candidate is not None else None
    return None


def _summarise_plan(plan: TimescaleBackbonePlan) -> dict[str, object]:
    summary: dict[str, object] = {}
    if plan.daily is not None:
        summary["daily_bars"] = {
            "symbols": plan.daily.normalised_symbols(),
            "lookback_days": plan.daily.lookback_days,
            "source": plan.daily.source,
        }
    if plan.intraday is not None:
        summary["intraday_trades"] = {
            "symbols": plan.intraday.normalised_symbols(),
            "lookback_days": plan.intraday.lookback_days,
            "interval": plan.intraday.interval,
            "source": plan.intraday.source,
        }
    if plan.macro is not None:
        macro_summary: dict[str, object] = {
            "source": plan.macro.source,
        }
        if plan.macro.events is not None:
            macro_summary["events"] = len(plan.macro.events)
        if plan.macro.start:
            macro_summary["start"] = plan.macro.start
        if plan.macro.end:
            macro_summary["end"] = plan.macro.end
        summary["macro_events"] = macro_summary
    return summary


__all__ = [
    "IngestRecoveryRecommendation",
    "plan_ingest_recovery",
]
