"""Evolution KPI telemetry aggregation aligned with the roadmap deliverable.

This module crunches experimentation and governance artefacts to expose
time-to-candidate SLAs, promotion posture, exploration budget usage, and
rollback responsiveness as a single snapshot.  Callers can feed the resulting
summary into observability surfaces or Prometheus exporters so operators track
adaptive-loop performance without reopening the roadmap document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from src.governance.policy_ledger import PolicyLedgerRecord, PolicyLedgerStage
from src.operational import metrics as operational_metrics

try:  # Optional dependency for experimentation telemetry
    from emp.core.findings_memory import TimeToCandidateStats
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    TimeToCandidateStats = None  # type: ignore[assignment]


class EvolutionKpiStatus(StrEnum):
    """Severity grading for the aggregated evolution KPI snapshot."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


_STATUS_ORDER: Mapping[EvolutionKpiStatus, int] = {
    EvolutionKpiStatus.ok: 0,
    EvolutionKpiStatus.warn: 1,
    EvolutionKpiStatus.fail: 2,
}


def _escalate(
    current: EvolutionKpiStatus, candidate: EvolutionKpiStatus
) -> EvolutionKpiStatus:
    if _STATUS_ORDER[candidate] > _STATUS_ORDER[current]:
        return candidate
    return current


@dataclass(frozen=True)
class TimeToCandidateKpi:
    """Snapshot of experimentation turnaround metrics."""

    count: int
    average_hours: float | None
    median_hours: float | None
    p90_hours: float | None
    max_hours: float | None
    threshold_hours: float | None
    sla_met: bool | None
    breaches: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "count": self.count,
            "avg_hours": self.average_hours,
            "median_hours": self.median_hours,
            "p90_hours": self.p90_hours,
            "max_hours": self.max_hours,
            "threshold_hours": self.threshold_hours,
            "sla_met": self.sla_met,
            "breaches": [dict(breach) for breach in self.breaches],
        }
        return payload


@dataclass(frozen=True)
class PromotionKpi:
    """Promotion and demotion rates derived from policy ledger history."""

    promotions: int
    demotions: int
    transitions: int
    promotion_rate: float | None
    recent_promotions: int
    window_days: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "promotions": self.promotions,
            "demotions": self.demotions,
            "transitions": self.transitions,
            "promotion_rate": self.promotion_rate,
            "recent_promotions": self.recent_promotions,
        }
        if self.window_days is not None:
            payload["window_days"] = self.window_days
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class BudgetUsageKpi:
    """Exploration budget utilisation derived from router decisions."""

    samples: int
    average_share: float | None
    max_share: float | None
    average_usage_ratio: float | None
    max_usage_ratio: float | None
    blocked_attempts: int
    forced_decisions: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "samples": self.samples,
            "average_share": self.average_share,
            "max_share": self.max_share,
            "average_usage_ratio": self.average_usage_ratio,
            "max_usage_ratio": self.max_usage_ratio,
            "blocked_attempts": self.blocked_attempts,
            "forced_decisions": self.forced_decisions,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class RollbackLatencyKpi:
    """Latency between promotions and subsequent demotions."""

    samples: int
    average_hours: float | None
    median_hours: float | None
    p95_hours: float | None
    max_hours: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "samples": self.samples,
            "average_hours": self.average_hours,
            "median_hours": self.median_hours,
            "p95_hours": self.p95_hours,
            "max_hours": self.max_hours,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class EvolutionKpiSnapshot:
    """Aggregated evolution KPIs exported for observability surfaces."""

    generated_at: datetime
    status: EvolutionKpiStatus
    time_to_candidate: TimeToCandidateKpi | None
    promotion: PromotionKpi | None
    budget: BudgetUsageKpi | None
    rollback: RollbackLatencyKpi | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }
        if self.time_to_candidate is not None:
            payload["time_to_candidate"] = self.time_to_candidate.as_dict()
        if self.promotion is not None:
            payload["promotion"] = self.promotion.as_dict()
        if self.budget is not None:
            payload["budget"] = self.budget.as_dict()
        if self.rollback is not None:
            payload["rollback"] = self.rollback.as_dict()
        return payload


_DEFAULT_PROMOTION_WINDOW_DAYS = 30.0
_ROLLBACK_WARN_HOURS = 6.0


def _parse_datetime(value: object, *, default: datetime | None = None) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return default
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return default


def _stage_rank(stage: PolicyLedgerStage | None) -> int:
    if stage is None:
        return -1
    return {
        PolicyLedgerStage.EXPERIMENT: 0,
        PolicyLedgerStage.PAPER: 1,
        PolicyLedgerStage.PILOT: 2,
        PolicyLedgerStage.LIMITED_LIVE: 3,
    }[stage]


def _coerce_stage(value: object | None) -> PolicyLedgerStage | None:
    if value is None:
        return None
    try:
        return PolicyLedgerStage.from_value(value)
    except (ValueError, TypeError):
        return None


def _coerce_time_to_candidate(
    stats: TimeToCandidateStats | Mapping[str, Any] | None,
) -> TimeToCandidateKpi | None:
    if stats is None:
        return None

    if TimeToCandidateStats is not None and isinstance(stats, TimeToCandidateStats):
        breaches = tuple(
            {
                "id": breach.id,
                "stage": breach.stage,
                "created_at": breach.created_at,
                "tested_at": breach.tested_at,
                "hours": breach.hours,
            }
            for breach in getattr(stats, "breaches", ())
        )
        return TimeToCandidateKpi(
            count=stats.count,
            average_hours=stats.average_hours,
            median_hours=stats.median_hours,
            p90_hours=stats.p90_hours,
            max_hours=stats.max_hours,
            threshold_hours=stats.threshold_hours,
            sla_met=stats.sla_met,
            breaches=breaches,
        )

    if isinstance(stats, Mapping):
        def _get_float(name: str) -> float | None:
            value = stats.get(name)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        count = int(stats.get("count", 0) or 0)
        breaches_raw = stats.get("breaches")
        breaches: tuple[Mapping[str, Any], ...]
        if isinstance(breaches_raw, Sequence):
            breaches = tuple(
                dict(entry) for entry in breaches_raw if isinstance(entry, Mapping)
            )
        else:
            breaches = tuple()

        sla_met_raw = stats.get("sla_met")
        sla_met = bool(sla_met_raw) if sla_met_raw is not None else None

        return TimeToCandidateKpi(
            count=count,
            average_hours=_get_float("avg_hours") or _get_float("average_hours"),
            median_hours=_get_float("median_hours"),
            p90_hours=_get_float("p90_hours"),
            max_hours=_get_float("max_hours"),
            threshold_hours=_get_float("threshold_hours"),
            sla_met=sla_met,
            breaches=breaches,
        )

    return None


def _iter_transition_history(
    record: PolicyLedgerRecord | Mapping[str, Any]
) -> Iterable[tuple[PolicyLedgerStage | None, PolicyLedgerStage | None, datetime | None]]:
    if isinstance(record, PolicyLedgerRecord):
        history = record.history
    elif isinstance(record, Mapping):
        history = record.get("history", ())
    else:
        history = ()

    for entry in history:
        if not isinstance(entry, Mapping):
            continue
        prior = _coerce_stage(entry.get("prior_stage"))
        nxt = _coerce_stage(entry.get("next_stage"))
        timestamp = _parse_datetime(entry.get("applied_at"))
        yield prior, nxt, timestamp


def _compute_promotion_stats(
    records: Sequence[PolicyLedgerRecord | Mapping[str, Any]] | None,
    *,
    now: datetime,
    window_days: float,
) -> PromotionKpi | None:
    if not records:
        return None

    promotions = 0
    demotions = 0
    total = 0
    recent_promotions = 0
    window_delta = timedelta(days=max(window_days, 0.0))
    cutoff = now - window_delta

    for record in records:
        for prior, nxt, timestamp in _iter_transition_history(record):
            if prior is None or prior is nxt:
                continue
            total += 1
            prior_rank = _stage_rank(prior)
            next_rank = _stage_rank(nxt)
            if next_rank > prior_rank:
                promotions += 1
                if timestamp is not None and timestamp >= cutoff:
                    recent_promotions += 1
            elif next_rank < prior_rank:
                demotions += 1

    promotion_rate = (promotions / total) if total else None

    return PromotionKpi(
        promotions=promotions,
        demotions=demotions,
        transitions=total,
        promotion_rate=promotion_rate,
        recent_promotions=recent_promotions,
        window_days=window_days,
        metadata={},
    )


def _compute_rollback_kpi(
    records: Sequence[PolicyLedgerRecord | Mapping[str, Any]] | None,
) -> RollbackLatencyKpi | None:
    if not records:
        return None

    latencies: list[float] = []
    promotion_times: dict[str, datetime] = {}

    for record in records:
        promotion_times.clear()
        for prior, nxt, timestamp in _iter_transition_history(record):
            if timestamp is None or prior is None or nxt is None or prior is nxt:
                continue
            prior_key = prior.value
            next_key = nxt.value
            prior_rank = _stage_rank(prior)
            next_rank = _stage_rank(nxt)
            if next_rank > prior_rank:
                promotion_times[next_key] = timestamp
            elif next_rank < prior_rank and prior_key in promotion_times:
                delta = (timestamp - promotion_times[prior_key]).total_seconds()
                if delta >= 0:
                    latencies.append(delta / 3600.0)

    if not latencies:
        return RollbackLatencyKpi(
            samples=0,
            average_hours=None,
            median_hours=None,
            p95_hours=None,
            max_hours=None,
            metadata={},
        )

    sorted_latencies = sorted(latencies)
    samples = len(sorted_latencies)
    average_hours = sum(sorted_latencies) / samples
    median_hours = sorted_latencies[samples // 2] if samples % 2 else (
        (sorted_latencies[samples // 2 - 1] + sorted_latencies[samples // 2]) / 2
    )
    p95_index = max(0, min(samples - 1, int(round(0.95 * (samples - 1)))))
    p95_hours = sorted_latencies[p95_index]
    max_hours = sorted_latencies[-1]

    return RollbackLatencyKpi(
        samples=samples,
        average_hours=average_hours,
        median_hours=median_hours,
        p95_hours=p95_hours,
        max_hours=max_hours,
        metadata={},
    )


def _coerce_budget_metadata(entry: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if entry is None:
        return {}
    if isinstance(entry, Mapping):
        return entry
    return {}


def _iter_budget_snapshots(
    loop_results: Sequence[Mapping[str, Any] | Any] | None,
) -> Iterable[Mapping[str, Any]]:
    if not loop_results:
        return []
    snapshots: list[Mapping[str, Any]] = []
    for result in loop_results:
        decision = getattr(result, "decision", None)
        exploration_metadata = None
        if decision is not None:
            exploration_metadata = getattr(decision, "exploration_metadata", None)
        elif isinstance(result, Mapping):
            candidate = result.get("decision")
            if isinstance(candidate, Mapping):
                exploration_metadata = candidate.get("exploration_metadata")

        metadata = _coerce_budget_metadata(exploration_metadata)
        if not metadata:
            continue
        budget_after = metadata.get("budget_after")
        budget_snapshot = budget_after if isinstance(budget_after, Mapping) else None
        if budget_snapshot is None:
            continue
        snapshots.append(dict(budget_snapshot))
    return snapshots


def _compute_budget_usage(
    loop_results: Sequence[Mapping[str, Any] | Any] | None,
) -> BudgetUsageKpi | None:
    snapshots = list(_iter_budget_snapshots(loop_results))
    if not snapshots:
        return None

    shares: list[float] = []
    usage_ratios: list[float] = []
    blocked = 0
    forced = 0

    last_max_fraction: float | None = None

    for snapshot in snapshots:
        share_value = snapshot.get("exploration_share")
        try:
            share = float(share_value)
        except (TypeError, ValueError):
            share = 0.0
        shares.append(share)

        raw_max = snapshot.get("max_fraction")
        try:
            max_fraction_value = float(raw_max)
        except (TypeError, ValueError):
            max_fraction_value = last_max_fraction
        else:
            if max_fraction_value <= 0:
                max_fraction_value = None
            else:
                last_max_fraction = max_fraction_value

        if max_fraction_value is not None and max_fraction_value > 0:
            usage_ratios.append(min(share, max_fraction_value) / max_fraction_value)
        else:
            usage_ratios.append(share)

        blocked_attempts = snapshot.get("blocked_attempts")
        forced_decisions = snapshot.get("forced_decisions")

        try:
            blocked += int(blocked_attempts or 0)
        except (TypeError, ValueError):
            pass
        try:
            forced += int(forced_decisions or 0)
        except (TypeError, ValueError):
            pass

    average_share = sum(shares) / len(shares) if shares else None
    max_share = max(shares) if shares else None
    average_usage = sum(usage_ratios) / len(usage_ratios) if usage_ratios else None
    max_usage = max(usage_ratios) if usage_ratios else None

    metadata: dict[str, Any] = {}
    if last_max_fraction is not None:
        metadata["max_fraction"] = last_max_fraction

    return BudgetUsageKpi(
        samples=len(snapshots),
        average_share=average_share,
        max_share=max_share,
        average_usage_ratio=average_usage,
        max_usage_ratio=max_usage,
        blocked_attempts=blocked,
        forced_decisions=forced,
        metadata=metadata,
    )


def _record_metrics(
    *,
    time_to_candidate: TimeToCandidateKpi | None,
    promotion: PromotionKpi | None,
    budget: BudgetUsageKpi | None,
    rollback: RollbackLatencyKpi | None,
) -> None:
    if time_to_candidate is not None:
        operational_metrics.set_evolution_time_to_candidate_stat(
            "avg", time_to_candidate.average_hours
        )
        operational_metrics.set_evolution_time_to_candidate_stat(
            "median", time_to_candidate.median_hours
        )
        operational_metrics.set_evolution_time_to_candidate_stat(
            "p90", time_to_candidate.p90_hours
        )
        operational_metrics.set_evolution_time_to_candidate_stat(
            "max", time_to_candidate.max_hours
        )
        if time_to_candidate.threshold_hours is not None:
            operational_metrics.set_evolution_time_to_candidate_stat(
                "threshold", time_to_candidate.threshold_hours
            )
        operational_metrics.set_evolution_time_to_candidate_total(time_to_candidate.count)
        operational_metrics.set_evolution_time_to_candidate_breaches(
            float(len(time_to_candidate.breaches))
        )

    if promotion is not None:
        operational_metrics.set_evolution_promotion_counts(
            float(promotion.promotions), float(promotion.demotions)
        )
        operational_metrics.set_evolution_promotion_transitions(float(promotion.transitions))
        operational_metrics.set_evolution_promotion_rate(
            promotion.promotion_rate if promotion.promotion_rate is not None else 0.0
        )

    if budget is not None:
        operational_metrics.set_evolution_budget_usage(
            "avg_share", budget.average_share
        )
        operational_metrics.set_evolution_budget_usage("max_share", budget.max_share)
        operational_metrics.set_evolution_budget_usage(
            "avg_usage_ratio", budget.average_usage_ratio
        )
        operational_metrics.set_evolution_budget_usage(
            "max_usage_ratio", budget.max_usage_ratio
        )
        operational_metrics.set_evolution_budget_blocked(float(budget.blocked_attempts))
        operational_metrics.set_evolution_budget_forced(float(budget.forced_decisions))
        operational_metrics.set_evolution_budget_samples(float(budget.samples))

    if rollback is not None:
        operational_metrics.set_evolution_rollback_latency(
            "avg", rollback.average_hours
        )
        operational_metrics.set_evolution_rollback_latency(
            "median", rollback.median_hours
        )
        operational_metrics.set_evolution_rollback_latency(
            "p95", rollback.p95_hours
        )
        operational_metrics.set_evolution_rollback_latency(
            "max", rollback.max_hours
        )
        operational_metrics.set_evolution_rollback_events(float(rollback.samples))


def evaluate_evolution_kpis(
    *,
    time_to_candidate: TimeToCandidateStats | Mapping[str, Any] | None = None,
    ledger_records: Sequence[PolicyLedgerRecord | Mapping[str, Any]] | None = None,
    loop_results: Sequence[Mapping[str, Any] | Any] | None = None,
    now: datetime | None = None,
    promotion_window_days: float = _DEFAULT_PROMOTION_WINDOW_DAYS,
) -> EvolutionKpiSnapshot:
    """Fuse experimentation, governance, and routing telemetry into KPIs."""

    resolved_now = now or datetime.now(tz=UTC)

    time_summary = _coerce_time_to_candidate(time_to_candidate)
    promotion_summary = _compute_promotion_stats(
        ledger_records,
        now=resolved_now,
        window_days=promotion_window_days,
    )
    rollback_summary = _compute_rollback_kpi(ledger_records)
    budget_summary = _compute_budget_usage(loop_results)

    status = EvolutionKpiStatus.ok
    if time_summary is not None and time_summary.sla_met is False:
        status = EvolutionKpiStatus.fail

    if promotion_summary is not None:
        if promotion_summary.promotions == 0 and promotion_summary.transitions > 0:
            status = _escalate(status, EvolutionKpiStatus.warn)

    if budget_summary is not None:
        max_usage = budget_summary.max_usage_ratio
        if max_usage is not None:
            if max_usage > 1.0 + 1e-6:
                status = EvolutionKpiStatus.fail
            elif max_usage > 0.9:
                status = _escalate(status, EvolutionKpiStatus.warn)

    if rollback_summary is not None and rollback_summary.median_hours is not None:
        if rollback_summary.median_hours > _ROLLBACK_WARN_HOURS:
            status = _escalate(status, EvolutionKpiStatus.warn)

    metadata: MutableMapping[str, Any] = {
        "promotion_window_days": promotion_window_days,
    }

    snapshot = EvolutionKpiSnapshot(
        generated_at=resolved_now,
        status=status,
        time_to_candidate=time_summary,
        promotion=promotion_summary,
        budget=budget_summary,
        rollback=rollback_summary,
        metadata=metadata,
    )

    _record_metrics(
        time_to_candidate=time_summary,
        promotion=promotion_summary,
        budget=budget_summary,
        rollback=rollback_summary,
    )

    return snapshot


__all__ = [
    "EvolutionKpiSnapshot",
    "EvolutionKpiStatus",
    "TimeToCandidateKpi",
    "PromotionKpi",
    "BudgetUsageKpi",
    "RollbackLatencyKpi",
    "evaluate_evolution_kpis",
]
