from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

_EPSILON = 1e-9


@dataclass(frozen=True)
class PerformanceRule:
    """Configuration describing how a performance metric contributes to fitness."""

    weight: float
    target: float
    higher_is_better: bool = True
    max_contribution: Optional[float] = None

    def contribution(self, value: float) -> float:
        """Return the weighted contribution for the supplied metric value."""

        if self.target == 0:
            scaled = value
        elif self.higher_is_better:
            scaled = value / self.target
        else:
            scaled = self.target / max(value, _EPSILON)

        contribution = scaled * self.weight
        if self.max_contribution is not None:
            contribution = max(-self.max_contribution, min(contribution, self.max_contribution))
        return contribution


@dataclass(frozen=True)
class PenaltyRule:
    """Configuration describing how a metric introduces a penalty."""

    weight: float
    target: float
    exponent: float = 1.0
    max_penalty: Optional[float] = None

    def penalty(self, value: float) -> float:
        """Return the penalty for the supplied metric value."""

        if self.target <= 0:
            return 0.0

        excess = max(0.0, value - self.target)
        if excess == 0:
            return 0.0

        normalized = (excess / self.target) ** self.exponent
        penalty_value = normalized * self.weight
        if self.max_penalty is not None:
            penalty_value = min(penalty_value, self.max_penalty)
        return penalty_value


@dataclass(frozen=True)
class FitnessReport:
    """Detailed report describing a fitness evaluation."""

    total_score: float
    performance_score: float
    risk_adjusted_return: float
    risk_penalty: float
    behaviour_penalty: float
    detail: Dict[str, float] = field(default_factory=dict)


class FitnessCalculator:
    """Compute comprehensive fitness scores for trading strategies.

    The calculator combines three dimensions:
      * Performance contributions (returns, Sharpe, win-rate, ...)
      * Risk adjustments and penalties (volatility, drawdown)
      * Behavioural penalties (excessive trading, turnover)

    The defaults provide sensible scoring for typical strategy metrics while
    still allowing custom rules to be supplied.
    """

    def __init__(
        self,
        performance_rules: Optional[Mapping[str, PerformanceRule]] = None,
        risk_penalties: Optional[Mapping[str, PenaltyRule]] = None,
        behaviour_penalties: Optional[Mapping[str, PenaltyRule]] = None,
        *,
        risk_volatility_target: float = 0.2,
        risk_drawdown_target: float = 0.2,
        drawdown_penalty_weight: float = 1.0,
    ) -> None:
        self.performance_rules: Dict[str, PerformanceRule] = dict(
            performance_rules or {
                "return": PerformanceRule(weight=0.5, target=0.1, higher_is_better=True),
                "sharpe": PerformanceRule(weight=0.3, target=1.0, higher_is_better=True),
                "win_rate": PerformanceRule(weight=0.2, target=0.55, higher_is_better=True),
            }
        )
        self.risk_penalties: Dict[str, PenaltyRule] = dict(
            risk_penalties or {
                "volatility": PenaltyRule(weight=1.0, target=0.2, exponent=1.0),
                "max_drawdown": PenaltyRule(weight=1.5, target=0.15, exponent=1.0),
            }
        )
        self.behaviour_penalties: Dict[str, PenaltyRule] = dict(
            behaviour_penalties or {
                "trade_count": PenaltyRule(weight=0.5, target=100.0, exponent=1.0),
                "turnover": PenaltyRule(weight=0.5, target=1.0, exponent=1.0),
            }
        )
        self.risk_volatility_target = risk_volatility_target
        self.risk_drawdown_target = risk_drawdown_target
        self.drawdown_penalty_weight = drawdown_penalty_weight

    def evaluate(
        self,
        performance_metrics: Mapping[str, float],
        risk_metrics: Optional[Mapping[str, float]] = None,
        behaviour_metrics: Optional[Mapping[str, float]] = None,
    ) -> FitnessReport:
        """Return the composite fitness score for the supplied metrics."""

        performance_score, performance_detail = self._compute_performance(performance_metrics)
        risk_adjusted = self._risk_adjusted_return(performance_metrics, risk_metrics)
        risk_penalty, risk_detail = self._compute_penalties(risk_metrics, self.risk_penalties)
        behaviour_penalty, behaviour_detail = self._compute_penalties(
            behaviour_metrics, self.behaviour_penalties
        )

        detail: Dict[str, float] = {}
        for name, contribution in performance_detail.items():
            detail[f"performance:{name}"] = contribution
        detail["risk_adjusted_return"] = risk_adjusted
        for name, penalty_value in risk_detail.items():
            detail[f"risk_penalty:{name}"] = -penalty_value
        for name, penalty_value in behaviour_detail.items():
            detail[f"behaviour_penalty:{name}"] = -penalty_value

        total = performance_score + risk_adjusted - risk_penalty - behaviour_penalty
        return FitnessReport(
            total_score=total,
            performance_score=performance_score,
            risk_adjusted_return=risk_adjusted,
            risk_penalty=risk_penalty,
            behaviour_penalty=behaviour_penalty,
            detail=detail,
        )

    def _compute_performance(
        self, performance_metrics: Mapping[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        score = 0.0
        detail: Dict[str, float] = {}
        for name, rule in self.performance_rules.items():
            value = performance_metrics.get(name)
            if value is None:
                continue
            contribution = rule.contribution(value)
            detail[name] = contribution
            score += contribution
        return score, detail

    def _compute_penalties(
        self,
        metrics: Optional[Mapping[str, float]],
        rules: Mapping[str, PenaltyRule],
    ) -> Tuple[float, Dict[str, float]]:
        if metrics is None:
            return 0.0, {}

        penalty_total = 0.0
        detail: Dict[str, float] = {}
        for name, rule in rules.items():
            value = metrics.get(name)
            if value is None:
                continue
            penalty_value = rule.penalty(value)
            detail[name] = penalty_value
            penalty_total += penalty_value
        return penalty_total, detail

    def _risk_adjusted_return(
        self,
        performance_metrics: Mapping[str, float],
        risk_metrics: Optional[Mapping[str, float]],
    ) -> float:
        base_return = performance_metrics.get("return", 0.0)
        if not risk_metrics:
            return base_return

        adjusted_return = base_return
        volatility = risk_metrics.get("volatility")
        if volatility is not None and volatility > 0:
            if self.risk_volatility_target > 0:
                scale = self.risk_volatility_target / max(volatility, _EPSILON)
                adjusted_return *= scale

        max_drawdown = risk_metrics.get("max_drawdown")
        if max_drawdown is not None:
            drawdown_excess = max(0.0, max_drawdown - self.risk_drawdown_target)
            if drawdown_excess:
                adjusted_return -= drawdown_excess * self.drawdown_penalty_weight

        return adjusted_return


__all__ = [
    "FitnessCalculator",
    "FitnessReport",
    "PerformanceRule",
    "PenaltyRule",
]
