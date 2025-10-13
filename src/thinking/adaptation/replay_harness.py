"""Replay evaluation harness for policy tactics with governance transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Iterable, Mapping, Sequence, Tuple, TYPE_CHECKING

from src.evolution.evaluation.recorded_replay import (
    RecordedEvaluationResult,
    RecordedSensoryEvaluator,
    RecordedSensorySnapshot,
)
from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage
from src.thinking.adaptation.policy_router import PolicyTactic

if TYPE_CHECKING:  # pragma: no cover - optional import for typing only
    from src.thinking.adaptation.policy_router import PolicyRouter

__all__ = [
    "StageThresholds",
    "StageDecision",
    "TacticEvaluationResult",
    "TacticReplayHarness",
]


@dataclass(frozen=True)
class StageThresholds:
    """Threshold bands that govern tactic promotions and demotions."""

    promote_total_return: float
    promote_win_rate: float
    promote_sharpe: float
    promote_max_drawdown: float
    demote_total_return: float
    demote_win_rate: float
    demote_sharpe: float
    demote_max_drawdown: float
    min_trades: int = 5
    min_confidence: float = 0.55

    def as_dict(self) -> Mapping[str, float]:  # pragma: no cover - trivial getter
        return {
            "promote_total_return": self.promote_total_return,
            "promote_win_rate": self.promote_win_rate,
            "promote_sharpe": self.promote_sharpe,
            "promote_max_drawdown": self.promote_max_drawdown,
            "demote_total_return": self.demote_total_return,
            "demote_win_rate": self.demote_win_rate,
            "demote_sharpe": self.demote_sharpe,
            "demote_max_drawdown": self.demote_max_drawdown,
            "min_trades": float(self.min_trades),
            "min_confidence": self.min_confidence,
        }


class StageDecision(StrEnum):
    """Decision emitted by the harness for governance gates."""

    promote = "promote"
    demote = "demote"
    maintain = "maintain"


@dataclass(frozen=True)
class TacticEvaluationResult:
    """Summary describing a tactic's replay performance and governance action."""

    tactic_id: str
    policy_id: str
    execution_topology: str | None
    current_stage: PolicyLedgerStage
    target_stage: PolicyLedgerStage
    decision: StageDecision
    metrics: RecordedEvaluationResult
    thresholds: StageThresholds
    reason: str
    snapshot_count: int
    evaluated_at: datetime

    def metrics_summary(self) -> Mapping[str, float]:
        """Return a compact view of the key replay metrics."""

        return {
            "total_return": self.metrics.total_return,
            "max_drawdown": self.metrics.max_drawdown,
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "volatility": self.metrics.volatility,
            "win_rate": self.metrics.win_rate,
            "trades": float(self.metrics.trades),
            "wins": float(self.metrics.wins),
            "losses": float(self.metrics.losses),
        }

    def thresholds_summary(self) -> Mapping[str, float]:  # pragma: no cover - simple proxy
        return dict(self.thresholds.as_dict())


_DEFAULT_STAGE_THRESHOLDS: Mapping[PolicyLedgerStage, StageThresholds] = {
    PolicyLedgerStage.EXPERIMENT: StageThresholds(
        promote_total_return=0.03,
        promote_win_rate=0.53,
        promote_sharpe=0.35,
        promote_max_drawdown=0.2,
        demote_total_return=-0.05,
        demote_win_rate=0.45,
        demote_sharpe=-0.1,
        demote_max_drawdown=0.35,
        min_trades=4,
        min_confidence=0.55,
    ),
    PolicyLedgerStage.PAPER: StageThresholds(
        promote_total_return=0.06,
        promote_win_rate=0.56,
        promote_sharpe=0.55,
        promote_max_drawdown=0.18,
        demote_total_return=-0.02,
        demote_win_rate=0.48,
        demote_sharpe=0.1,
        demote_max_drawdown=0.28,
        min_trades=6,
        min_confidence=0.6,
    ),
    PolicyLedgerStage.PILOT: StageThresholds(
        promote_total_return=0.09,
        promote_win_rate=0.6,
        promote_sharpe=0.75,
        promote_max_drawdown=0.15,
        demote_total_return=0.0,
        demote_win_rate=0.52,
        demote_sharpe=0.3,
        demote_max_drawdown=0.22,
        min_trades=8,
        min_confidence=0.62,
    ),
    PolicyLedgerStage.LIMITED_LIVE: StageThresholds(
        promote_total_return=0.12,
        promote_win_rate=0.62,
        promote_sharpe=0.9,
        promote_max_drawdown=0.12,
        demote_total_return=0.02,
        demote_win_rate=0.55,
        demote_sharpe=0.45,
        demote_max_drawdown=0.18,
        min_trades=10,
        min_confidence=0.65,
    ),
}

_STAGE_SEQUENCE: Tuple[PolicyLedgerStage, ...] = (
    PolicyLedgerStage.EXPERIMENT,
    PolicyLedgerStage.PAPER,
    PolicyLedgerStage.PILOT,
    PolicyLedgerStage.LIMITED_LIVE,
)


def _normalise_snapshots(
    snapshots: Sequence[RecordedSensorySnapshot | Mapping[str, object]] | Iterable[RecordedSensorySnapshot | Mapping[str, object]],
) -> Tuple[RecordedSensorySnapshot, ...]:
    result: list[RecordedSensorySnapshot] = []
    for entry in snapshots:
        if isinstance(entry, RecordedSensorySnapshot):
            result.append(entry)
            continue
        if isinstance(entry, Mapping):
            result.append(RecordedSensorySnapshot.from_snapshot(entry))
            continue
        raise TypeError(
            "Replay harness snapshots must be mappings or RecordedSensorySnapshot instances",
        )
    return tuple(result)


def _next_stage(stage: PolicyLedgerStage) -> PolicyLedgerStage | None:
    try:
        index = _STAGE_SEQUENCE.index(stage)
    except ValueError:  # pragma: no cover - defensive
        return None
    if index + 1 >= len(_STAGE_SEQUENCE):
        return None
    return _STAGE_SEQUENCE[index + 1]


def _previous_stage(stage: PolicyLedgerStage) -> PolicyLedgerStage | None:
    try:
        index = _STAGE_SEQUENCE.index(stage)
    except ValueError:  # pragma: no cover - defensive
        return None
    if index == 0:
        return None
    return _STAGE_SEQUENCE[index - 1]


class TacticReplayHarness:
    """Evaluate tactics via recorded sensory replay and derive stage decisions."""

    def __init__(
        self,
        *,
        snapshots: Sequence[RecordedSensorySnapshot | Mapping[str, object]]
        | Iterable[RecordedSensorySnapshot | Mapping[str, object]],
        release_manager: LedgerReleaseManager,
        stage_thresholds: Mapping[PolicyLedgerStage, StageThresholds] | None = None,
    ) -> None:
        self._snapshots = _normalise_snapshots(snapshots)
        self._evaluator = RecordedSensoryEvaluator(self._snapshots)
        self._release_manager = release_manager
        thresholds = {stage: value for stage, value in _DEFAULT_STAGE_THRESHOLDS.items()}
        if stage_thresholds:
            for stage, value in stage_thresholds.items():
                thresholds[PolicyLedgerStage.from_value(stage)] = value
        self._thresholds = thresholds

    def evaluate_router(
        self,
        router: "PolicyRouter",
        *,
        policy_ids: Mapping[str, str] | None = None,
        min_confidence: float | None = None,
    ) -> tuple[TacticEvaluationResult, ...]:
        results: list[TacticEvaluationResult] = []
        for tactic_id, tactic in router.tactics().items():
            policy_id = policy_ids.get(tactic_id, tactic_id) if policy_ids else tactic_id
            results.append(
                self.evaluate_tactic(
                    tactic,
                    policy_id=policy_id,
                    min_confidence=min_confidence,
                )
            )
        return tuple(results)

    def evaluate_tactic(
        self,
        tactic: PolicyTactic,
        *,
        policy_id: str | None = None,
        min_confidence: float | None = None,
    ) -> TacticEvaluationResult:
        policy_key = policy_id or tactic.tactic_id
        current_stage = self._release_manager.resolve_stage(policy_key)
        thresholds = self._thresholds.get(current_stage, _DEFAULT_STAGE_THRESHOLDS[current_stage])
        confidence_floor = min_confidence if min_confidence is not None else thresholds.min_confidence

        metrics = self._evaluator.evaluate(
            tactic.parameters,
            min_confidence=confidence_floor,
        )
        decision, target_stage, reason = self._decide_stage(
            current_stage=current_stage,
            thresholds=thresholds,
            metrics=metrics,
        )
        return TacticEvaluationResult(
            tactic_id=tactic.tactic_id,
            policy_id=policy_key,
            execution_topology=tactic.topology,
            current_stage=current_stage,
            target_stage=target_stage,
            decision=decision,
            metrics=metrics,
            thresholds=thresholds,
            reason=reason,
            snapshot_count=len(self._snapshots),
            evaluated_at=datetime.now(tz=timezone.utc),
        )

    def _decide_stage(
        self,
        *,
        current_stage: PolicyLedgerStage,
        thresholds: StageThresholds,
        metrics: RecordedEvaluationResult,
    ) -> tuple[StageDecision, PolicyLedgerStage, str]:
        if metrics.trades < thresholds.min_trades:
            return (
                StageDecision.maintain,
                current_stage,
                f"insufficient trades ({metrics.trades} < {thresholds.min_trades})",
            )

        next_stage = _next_stage(current_stage)
        previous_stage = _previous_stage(current_stage)

        if next_stage is not None:
            if (
                metrics.total_return >= thresholds.promote_total_return
                and metrics.win_rate >= thresholds.promote_win_rate
                and metrics.sharpe_ratio >= thresholds.promote_sharpe
                and metrics.max_drawdown <= thresholds.promote_max_drawdown
            ):
                return (
                    StageDecision.promote,
                    next_stage,
                    "promotion thresholds satisfied",
                )

        if previous_stage is not None:
            if (
                metrics.total_return <= thresholds.demote_total_return
                or metrics.win_rate <= thresholds.demote_win_rate
                or metrics.sharpe_ratio <= thresholds.demote_sharpe
                or metrics.max_drawdown >= thresholds.demote_max_drawdown
            ):
                return (
                    StageDecision.demote,
                    previous_stage,
                    "performance breached demotion thresholds",
                )

        return (
            StageDecision.maintain,
            current_stage,
            "within governance guard band",
        )
