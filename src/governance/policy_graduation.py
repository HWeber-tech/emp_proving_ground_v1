"""Assess AlphaTrade policy readiness for stage promotions.

This module fulfils the roadmap deliverable that graduates the AlphaTrade
live-shadow pilot into the staged execution pathway (experiment -> paper ->
pilot -> limited live). It inspects DecisionDiary evidence alongside the
policy ledger to recommend the highest release stage that current telemetry
supports, highlighting outstanding blockers so operators can promote tactics
with confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, MutableMapping, Sequence

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage
from src.understanding.decision_diary import DecisionDiaryEntry, DecisionDiaryStore

__all__ = [
    "DiaryStageMetrics",
    "DiaryMetrics",
    "PolicyGraduationAssessment",
    "PolicyGraduationEvaluator",
]


_UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(tz=_UTC)


@dataclass(slots=True, frozen=True)
class DiaryStageMetrics:
    """Aggregated telemetry for a specific policy ledger stage."""

    stage: PolicyLedgerStage
    total: int
    forced: int
    severity_counts: Mapping[str, int]

    @property
    def normal(self) -> int:
        return self.severity_counts.get("normal", 0)

    @property
    def warn(self) -> int:
        return self.severity_counts.get("warn", 0)

    @property
    def alert(self) -> int:
        return self.severity_counts.get("alert", 0)

    @property
    def forced_ratio(self) -> float:
        return (self.forced / self.total) if self.total else 0.0

    @property
    def normal_ratio(self) -> float:
        return (self.normal / self.total) if self.total else 0.0

    @property
    def warn_ratio(self) -> float:
        return (self.warn / self.total) if self.total else 0.0

    @property
    def alert_ratio(self) -> float:
        return (self.alert / self.total) if self.total else 0.0

    def to_dict(self) -> Mapping[str, object]:
        return {
            "stage": self.stage.value,
            "total": self.total,
            "forced": self.forced,
            "severity": dict(self.severity_counts),
            "forced_ratio": round(self.forced_ratio, 5),
            "normal_ratio": round(self.normal_ratio, 5),
            "warn_ratio": round(self.warn_ratio, 5),
            "alert_ratio": round(self.alert_ratio, 5),
        }


@dataclass(slots=True, frozen=True)
class DiaryMetrics:
    """Aggregated DecisionDiary telemetry for a policy across stages."""

    policy_id: str
    total_decisions: int
    stage_metrics: Mapping[PolicyLedgerStage, DiaryStageMetrics]
    release_routes: Mapping[str, int]
    severity_counts: Mapping[str, int]
    forced_total: int
    consecutive_normal_latest_stage: int
    first_entry_at: datetime | None
    last_entry_at: datetime | None
    latest_stage: PolicyLedgerStage | None

    @property
    def normal(self) -> int:
        return self.severity_counts.get("normal", 0)

    @property
    def warn(self) -> int:
        return self.severity_counts.get("warn", 0)

    @property
    def alert(self) -> int:
        return self.severity_counts.get("alert", 0)

    @property
    def forced_ratio(self) -> float:
        return (self.forced_total / self.total_decisions) if self.total_decisions else 0.0

    def stage(self, stage: PolicyLedgerStage) -> DiaryStageMetrics | None:
        return self.stage_metrics.get(stage)

    def to_dict(self) -> Mapping[str, object]:
        return {
            "policy_id": self.policy_id,
            "total_decisions": self.total_decisions,
            "stage_metrics": {stage.value: metrics.to_dict() for stage, metrics in self.stage_metrics.items()},
            "release_routes": dict(self.release_routes),
            "severity": dict(self.severity_counts),
            "forced_total": self.forced_total,
            "forced_ratio": round(self.forced_ratio, 5),
            "consecutive_normal_latest_stage": self.consecutive_normal_latest_stage,
            "first_entry_at": self.first_entry_at.isoformat() if self.first_entry_at else None,
            "last_entry_at": self.last_entry_at.isoformat() if self.last_entry_at else None,
            "latest_stage": self.latest_stage.value if self.latest_stage else None,
        }


@dataclass(slots=True, frozen=True)
class PolicyGraduationAssessment:
    """Recommendation describing the highest justified release stage."""

    policy_id: str
    current_stage: PolicyLedgerStage
    declared_stage: PolicyLedgerStage | None
    audit_stage: PolicyLedgerStage | None
    recommended_stage: PolicyLedgerStage
    metrics: DiaryMetrics
    approvals: tuple[str, ...]
    evidence_id: str | None
    audit_gaps: tuple[str, ...]
    stage_blockers: Mapping[PolicyLedgerStage, tuple[str, ...]]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "policy_id": self.policy_id,
            "current_stage": self.current_stage.value,
            "declared_stage": self.declared_stage.value if self.declared_stage else None,
            "audit_stage": self.audit_stage.value if self.audit_stage else None,
            "recommended_stage": self.recommended_stage.value,
            "approvals": list(self.approvals),
            "evidence_id": self.evidence_id,
            "audit_gaps": list(self.audit_gaps),
            "metrics": self.metrics.to_dict(),
            "stage_blockers": {
                stage.value: list(blockers) for stage, blockers in self.stage_blockers.items()
            },
        }


class PolicyGraduationEvaluator:
    """Compute AlphaTrade stage readiness recommendations from diary evidence."""

    # Heuristic thresholds tuned against roadmap expectations.
    _MIN_DECISIONS_PAPER = 20
    _MIN_DECISIONS_PILOT = 40
    _MIN_DECISIONS_LIVE = 60
    _MAX_FORCE_RATIO_PAPER = 0.35
    _MAX_WARN_RATIO_PAPER = 0.40
    _MAX_FORCE_RATIO_PILOT = 0.25
    _MAX_WARN_RATIO_PILOT = 0.20
    _MAX_FORCE_RATIO_LIVE = 0.10
    _MAX_WARN_RATIO_LIVE = 0.10
    _MIN_NORMAL_RATIO_PAPER = 0.50
    _MIN_NORMAL_RATIO_PILOT = 0.65
    _MIN_NORMAL_RATIO_LIVE = 0.80
    _MIN_NORMAL_STREAK_PILOT = 8
    _MIN_NORMAL_STREAK_LIVE = 15
    _MIN_PAPER_GREEN_DAYS = 14.0

    def __init__(
        self,
        release_manager: LedgerReleaseManager,
        diary_store: DecisionDiaryStore,
        *,
        window: timedelta | None = None,
    ) -> None:
        self._release_manager = release_manager
        self._diary_store = diary_store
        self._window = window

    def assess(self, policy_id: str) -> PolicyGraduationAssessment:
        """Return the recommended stage and blockers for the policy."""

        entries = self._policy_entries(policy_id)
        metrics = self._compute_metrics(policy_id, entries)

        summary = self._release_manager.describe(policy_id)
        current_stage = PolicyLedgerStage.from_value(summary.get("stage"))
        declared_stage_raw = summary.get("declared_stage")
        declared_stage = (
            PolicyLedgerStage.from_value(declared_stage_raw)
            if declared_stage_raw
            else None
        )
        audit_stage_raw = summary.get("audit_stage")
        audit_stage = (
            PolicyLedgerStage.from_value(audit_stage_raw)
            if audit_stage_raw
            else None
        )
        approvals = tuple(sorted(str(value) for value in summary.get("approvals", ()) if value))
        evidence_id = summary.get("evidence_id")
        audit_gaps = tuple(str(gap) for gap in summary.get("audit_gaps", ()) if gap)

        paper_green_span_days = self._longest_paper_green_span_days(entries)

        recommended_stage, stage_blockers = self._recommend_stage(
            metrics,
            approvals=approvals,
            audit_gaps=audit_gaps,
            paper_green_span_days=paper_green_span_days,
        )

        return PolicyGraduationAssessment(
            policy_id=policy_id,
            current_stage=current_stage,
            declared_stage=declared_stage,
            audit_stage=audit_stage,
            recommended_stage=recommended_stage,
            metrics=metrics,
            approvals=approvals,
            evidence_id=str(evidence_id) if evidence_id else None,
            audit_gaps=audit_gaps,
            stage_blockers={stage: tuple(notes) for stage, notes in stage_blockers.items()},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _policy_entries(self, policy_id: str) -> tuple[DecisionDiaryEntry, ...]:
        cutoff: datetime | None = None
        if self._window is not None:
            cutoff = _now() - self._window

        selected: list[DecisionDiaryEntry] = []
        for entry in self._diary_store.entries():
            if entry.policy_id != policy_id:
                continue
            if cutoff is not None and entry.recorded_at < cutoff:
                continue
            selected.append(entry)
        return tuple(selected)

    def _compute_metrics(
        self,
        policy_id: str,
        entries: Sequence[DecisionDiaryEntry],
    ) -> DiaryMetrics:
        total = len(entries)
        severity_counts: MutableMapping[str, int] = {}
        stage_totals: MutableMapping[PolicyLedgerStage, int] = {}
        stage_forced: MutableMapping[PolicyLedgerStage, int] = {}
        stage_severity: MutableMapping[PolicyLedgerStage, MutableMapping[str, int]] = {}
        release_routes: MutableMapping[str, int] = {}

        forced_total = 0
        first_entry_at: datetime | None = None
        last_entry_at: datetime | None = None
        latest_stage: PolicyLedgerStage | None = None

        for entry in entries:
            recorded_at = entry.recorded_at.astimezone(_UTC)
            if first_entry_at is None or recorded_at < first_entry_at:
                first_entry_at = recorded_at
            if last_entry_at is None or recorded_at > last_entry_at:
                last_entry_at = recorded_at

            stage = _extract_stage(entry)
            severity = _extract_severity(entry)
            forced = _is_forced(entry)
            route = _extract_route(entry)

            if stage is not None:
                stage_totals[stage] = stage_totals.get(stage, 0) + 1
                if forced:
                    stage_forced[stage] = stage_forced.get(stage, 0) + 1
                stage_severity.setdefault(stage, {})
                stage_severity[stage][severity] = stage_severity[stage].get(severity, 0) + 1
                latest_stage = stage  # entries() are chronological, so last assignment wins

            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            if forced:
                forced_total += 1
            if route:
                release_routes[route] = release_routes.get(route, 0) + 1

        stage_metrics: MutableMapping[PolicyLedgerStage, DiaryStageMetrics] = {}
        for stage, total_count in stage_totals.items():
            stage_metrics[stage] = DiaryStageMetrics(
                stage=stage,
                total=total_count,
                forced=stage_forced.get(stage, 0),
                severity_counts={k: v for k, v in stage_severity.get(stage, {}).items()},
            )

        consecutive_normal = self._consecutive_normal_latest_stage(entries, latest_stage)

        return DiaryMetrics(
            policy_id=policy_id,
            total_decisions=total,
            stage_metrics=stage_metrics,
            release_routes=release_routes,
            severity_counts=severity_counts,
            forced_total=forced_total,
            consecutive_normal_latest_stage=consecutive_normal,
            first_entry_at=first_entry_at,
            last_entry_at=last_entry_at,
            latest_stage=latest_stage,
        )

    def _consecutive_normal_latest_stage(
        self,
        entries: Sequence[DecisionDiaryEntry],
        latest_stage: PolicyLedgerStage | None,
    ) -> int:
        if latest_stage is None:
            return 0
        streak = 0
        for entry in reversed(entries):
            stage = _extract_stage(entry)
            if stage != latest_stage:
                break
            severity = _extract_severity(entry)
            if severity == "normal" and not _is_forced(entry):
                streak += 1
            else:
                break
        return streak

    def _recommend_stage(
        self,
        metrics: DiaryMetrics,
        *,
        approvals: Iterable[str],
        audit_gaps: Iterable[str],
        paper_green_span_days: float,
    ) -> tuple[PolicyLedgerStage, Mapping[PolicyLedgerStage, list[str]]]:
        approvals_tuple = tuple(approvals)
        audit_gaps_tuple = tuple(str(gap) for gap in audit_gaps)

        blockers: MutableMapping[PolicyLedgerStage, list[str]] = {
            PolicyLedgerStage.PAPER: [],
            PolicyLedgerStage.PILOT: [],
            PolicyLedgerStage.LIMITED_LIVE: [],
        }

        recommended = PolicyLedgerStage.EXPERIMENT

        experiment_metrics = metrics.stage(PolicyLedgerStage.EXPERIMENT) or metrics.stage(metrics.latest_stage or PolicyLedgerStage.EXPERIMENT)
        if experiment_metrics is None or experiment_metrics.total < self._MIN_DECISIONS_PAPER:
            blockers[PolicyLedgerStage.PAPER].append(
                f"experiment_stage_decisions_insufficient:{experiment_metrics.total if experiment_metrics else 0}/{self._MIN_DECISIONS_PAPER}"
            )
        else:
            if experiment_metrics.alert > 0:
                blockers[PolicyLedgerStage.PAPER].append(
                    f"experiment_stage_alerts_present:{experiment_metrics.alert}"
                )
            if experiment_metrics.forced_ratio > self._MAX_FORCE_RATIO_PAPER:
                blockers[PolicyLedgerStage.PAPER].append(
                    f"experiment_forced_ratio_exceeds:{experiment_metrics.forced_ratio:.2f}>{self._MAX_FORCE_RATIO_PAPER:.2f}"
                )
            if experiment_metrics.warn_ratio > self._MAX_WARN_RATIO_PAPER:
                blockers[PolicyLedgerStage.PAPER].append(
                    f"experiment_warn_ratio_exceeds:{experiment_metrics.warn_ratio:.2f}>{self._MAX_WARN_RATIO_PAPER:.2f}"
                )
            if experiment_metrics.normal_ratio < self._MIN_NORMAL_RATIO_PAPER:
                blockers[PolicyLedgerStage.PAPER].append(
                    f"experiment_normal_ratio_below:{experiment_metrics.normal_ratio:.2f}<{self._MIN_NORMAL_RATIO_PAPER:.2f}"
                )

        if not blockers[PolicyLedgerStage.PAPER]:
            recommended = PolicyLedgerStage.PAPER
        else:
            return recommended, blockers

        paper_metrics = metrics.stage(PolicyLedgerStage.PAPER)
        if paper_metrics is None or paper_metrics.total < self._MIN_DECISIONS_PILOT:
            blockers[PolicyLedgerStage.PILOT].append(
                f"paper_stage_decisions_insufficient:{paper_metrics.total if paper_metrics else 0}/{self._MIN_DECISIONS_PILOT}"
            )
        else:
            if paper_metrics.alert > 0:
                blockers[PolicyLedgerStage.PILOT].append(
                    f"paper_stage_alerts_present:{paper_metrics.alert}"
                )
            if paper_metrics.forced_ratio > self._MAX_FORCE_RATIO_PILOT:
                blockers[PolicyLedgerStage.PILOT].append(
                    f"paper_forced_ratio_exceeds:{paper_metrics.forced_ratio:.2f}>{self._MAX_FORCE_RATIO_PILOT:.2f}"
                )
            if paper_metrics.warn_ratio > self._MAX_WARN_RATIO_PILOT:
                blockers[PolicyLedgerStage.PILOT].append(
                    f"paper_warn_ratio_exceeds:{paper_metrics.warn_ratio:.2f}>{self._MAX_WARN_RATIO_PILOT:.2f}"
                )
            if paper_metrics.normal_ratio < self._MIN_NORMAL_RATIO_PILOT:
                blockers[PolicyLedgerStage.PILOT].append(
                    f"paper_normal_ratio_below:{paper_metrics.normal_ratio:.2f}<{self._MIN_NORMAL_RATIO_PILOT:.2f}"
                )
            if metrics.latest_stage is PolicyLedgerStage.PAPER and metrics.consecutive_normal_latest_stage < self._MIN_NORMAL_STREAK_PILOT:
                blockers[PolicyLedgerStage.PILOT].append(
                    f"paper_normal_streak_below:{metrics.consecutive_normal_latest_stage}/{self._MIN_NORMAL_STREAK_PILOT}"
                )

        if not blockers[PolicyLedgerStage.PILOT]:
            recommended = PolicyLedgerStage.PILOT
        else:
            return recommended, blockers

        pilot_metrics = metrics.stage(PolicyLedgerStage.PILOT)
        if pilot_metrics is None or pilot_metrics.total < self._MIN_DECISIONS_LIVE:
            blockers[PolicyLedgerStage.LIMITED_LIVE].append(
                f"pilot_stage_decisions_insufficient:{pilot_metrics.total if pilot_metrics else 0}/{self._MIN_DECISIONS_LIVE}"
            )
        else:
            if pilot_metrics.alert > 0:
                blockers[PolicyLedgerStage.LIMITED_LIVE].append(
                    f"pilot_stage_alerts_present:{pilot_metrics.alert}"
                )
            if pilot_metrics.forced_ratio > self._MAX_FORCE_RATIO_LIVE:
                blockers[PolicyLedgerStage.LIMITED_LIVE].append(
                    f"pilot_forced_ratio_exceeds:{pilot_metrics.forced_ratio:.2f}>{self._MAX_FORCE_RATIO_LIVE:.2f}"
                )
            if pilot_metrics.warn_ratio > self._MAX_WARN_RATIO_LIVE:
                blockers[PolicyLedgerStage.LIMITED_LIVE].append(
                    f"pilot_warn_ratio_exceeds:{pilot_metrics.warn_ratio:.2f}>{self._MAX_WARN_RATIO_LIVE:.2f}"
                )
            if pilot_metrics.normal_ratio < self._MIN_NORMAL_RATIO_LIVE:
                blockers[PolicyLedgerStage.LIMITED_LIVE].append(
                    f"pilot_normal_ratio_below:{pilot_metrics.normal_ratio:.2f}<{self._MIN_NORMAL_RATIO_LIVE:.2f}"
                )
            if metrics.latest_stage is PolicyLedgerStage.PILOT and metrics.consecutive_normal_latest_stage < self._MIN_NORMAL_STREAK_LIVE:
                blockers[PolicyLedgerStage.LIMITED_LIVE].append(
                    f"pilot_normal_streak_below:{metrics.consecutive_normal_latest_stage}/{self._MIN_NORMAL_STREAK_LIVE}"
                )
            if len(approvals_tuple) < 2:
                blockers[PolicyLedgerStage.LIMITED_LIVE].append(
                    f"approvals_required:2_have:{len(approvals_tuple)}"
                )
            if audit_gaps_tuple:
                blockers[PolicyLedgerStage.LIMITED_LIVE].append(
                    "audit_gaps_present:" + ",".join(audit_gaps_tuple)
                )

        paper_green_span = paper_green_span_days
        if paper_green_span < self._MIN_PAPER_GREEN_DAYS:
            blockers[PolicyLedgerStage.LIMITED_LIVE].append(
                "paper_green_gate_duration_below:" +
                f"{paper_green_span:.2f}<{self._MIN_PAPER_GREEN_DAYS:.2f}"
            )

        if not blockers[PolicyLedgerStage.LIMITED_LIVE]:
            recommended = PolicyLedgerStage.LIMITED_LIVE

        return recommended, blockers

    def _longest_paper_green_span_days(
        self,
        entries: Sequence[DecisionDiaryEntry],
    ) -> float:
        current_span_days = 0.0
        current_start: datetime | None = None

        for entry in entries:
            if _extract_stage(entry) is not PolicyLedgerStage.PAPER:
                continue

            if self._is_paper_gate_green(entry):
                timestamp = entry.recorded_at.astimezone(_UTC)
                if current_start is None:
                    current_start = timestamp
                current_span_days = (timestamp - current_start).total_seconds() / 86400.0
            else:
                current_start = None
                current_span_days = 0.0

        return current_span_days

    @staticmethod
    def _is_paper_gate_green(entry: DecisionDiaryEntry) -> bool:
        return _extract_severity(entry) == "normal" and not _is_forced(entry)


def _extract_stage(entry: DecisionDiaryEntry) -> PolicyLedgerStage | None:
    metadata = entry.metadata or {}
    stage_candidate = metadata.get("release_stage")
    if not stage_candidate:
        release_meta = metadata.get("release_execution")
        if isinstance(release_meta, Mapping):
            stage_candidate = release_meta.get("stage") or release_meta.get("release_stage")
    if not stage_candidate:
        return None
    try:
        return PolicyLedgerStage.from_value(stage_candidate)
    except Exception:
        return None


def _extract_route(entry: DecisionDiaryEntry) -> str | None:
    metadata = entry.metadata or {}
    release_meta = metadata.get("release_execution")
    if isinstance(release_meta, Mapping):
        route = release_meta.get("route")
        if route:
            return str(route)
    route = metadata.get("release_execution_route")
    return str(route) if route else None


def _extract_severity(entry: DecisionDiaryEntry) -> str:
    metadata = entry.metadata or {}
    drift_meta = metadata.get("drift_decision")
    if isinstance(drift_meta, Mapping):
        severity = drift_meta.get("severity")
        if severity:
            return str(severity).strip().lower()
    return "unknown"


def _is_forced(entry: DecisionDiaryEntry) -> bool:
    metadata = entry.metadata or {}
    drift_meta = metadata.get("drift_decision")
    if isinstance(drift_meta, Mapping):
        forced = drift_meta.get("force_paper")
        if isinstance(forced, bool):
            return forced
        if isinstance(forced, str):
            lowered = forced.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
    release_meta = metadata.get("release_execution")
    if isinstance(release_meta, Mapping):
        forced = release_meta.get("forced")
        if isinstance(forced, bool):
            return forced
        forced_reason = release_meta.get("forced_reason")
        if forced_reason:
            return True
    forced_summary = metadata.get("release_execution_forced")
    return bool(forced_summary)
