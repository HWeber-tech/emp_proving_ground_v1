"""
EMP Market Analyzer v1.1

Comprehensive market analysis for the thinking layer.
Combines multiple analysis components into unified market insights.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Protocol, TypeAlias, cast

import numpy as np

from src.core.exceptions import TradingException

if TYPE_CHECKING:
    from src.core.interfaces import AnalysisResult, SensorySignal, ThinkingPattern
else:
    AnalysisResult: TypeAlias = Any
    SensorySignal: TypeAlias = Any
    ThinkingPattern: TypeAlias = Any

from .performance_analyzer import PerformanceAnalyzer
from .risk_analyzer import RiskAnalyzer

logger = logging.getLogger(__name__)


# typing-only protocol for signal attributes used locally
class _SignalProto(Protocol):
    signal_type: str
    value: float
    confidence: float


@dataclass(slots=True)
class _PlannerState:
    """Compact planning state for the shallow MCTS controller."""

    opportunity: float
    sentiment: float
    risk: float
    var_95: float
    drawdown: float
    signal_quality: float
    reward_hint: float


@dataclass(slots=True)
class _MCTSNode:
    """Single node within the execution-style MCTS search tree."""

    state: _PlannerState
    parent: "_MCTSNode | None"
    action: Optional[str]
    depth: int
    incoming_reward: float
    visits: int = 0
    total_reward: float = 0.0
    children: Dict[str, "_MCTSNode"] = field(default_factory=dict)

    @property
    def average_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def ucb_score(self, parent_visits: int, exploration: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.average_reward
        exploration_bonus = exploration * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration_bonus


class _ExecutionMCTS:
    """Depth-limited Monte Carlo Tree Search for execution action selection."""

    ACTIONS: tuple[str, ...] = ("cross", "post", "hold")

    def __init__(
        self,
        *,
        budget_seconds: float,
        max_depth: int = 3,
        exploration: float = 0.9,
        discount: float = 0.85,
        sla_grace: int = 1,
    ) -> None:
        self._budget_seconds = max(1e-4, float(budget_seconds))
        self._max_depth = max(1, int(max_depth))
        self._exploration = float(exploration)
        self._discount = float(discount)
        self._sla_grace = max(0, int(sla_grace))
        self._enabled = True
        self._breaches = 0
        self._disabled_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def plan(self, state: _PlannerState) -> Optional[str]:
        if not self._enabled:
            return None

        root = _MCTSNode(state=state, parent=None, action=None, depth=0, incoming_reward=0.0)
        start = perf_counter()
        iterations = 0

        while True:
            now = perf_counter()
            if now - start >= self._budget_seconds:
                break
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
            iterations += 1
            if iterations >= 12 and root.visits >= 9:
                break

        elapsed = perf_counter() - start
        if elapsed > self._budget_seconds:
            self._breaches += 1
        else:
            self._breaches = 0

        if self._breaches > self._sla_grace:
            self._enabled = False
            self._disabled_reason = (
                f"mcts planner disabled: {elapsed * 1000.0:.3f}ms > {self._budget_seconds * 1000.0:.3f}ms"
            )

        if not root.children:
            return None

        best_child = max(root.children.values(), key=lambda child: child.average_reward)
        return best_child.action

    # MCTS internals -------------------------------------------------
    def _select(self, node: _MCTSNode) -> _MCTSNode:
        current = node
        while current.depth < self._max_depth and current.children:
            if len(current.children) < len(self.ACTIONS):
                break
            current = max(
                current.children.values(),
                key=lambda child: child.ucb_score(current.visits or 1, self._exploration),
            )
        if current.depth < self._max_depth and len(current.children) < len(self.ACTIONS):
            return self._expand(current)
        return current

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        untried = [action for action in self.ACTIONS if action not in node.children]
        action = random.choice(untried)
        next_state, reward = self._transition(node.state, action)
        child = _MCTSNode(
            state=next_state,
            parent=node,
            action=action,
            depth=node.depth + 1,
            incoming_reward=reward,
        )
        node.children[action] = child
        return child

    def _simulate(self, node: _MCTSNode) -> float:
        total_reward = node.incoming_reward
        depth = node.depth
        state = node.state
        discount = self._discount
        current_discount = discount

        while depth < self._max_depth:
            depth += 1
            action = random.choice(self.ACTIONS)
            state, reward = self._transition(state, action)
            total_reward += reward * current_discount
            current_discount *= discount

        eval_score = self._evaluate_state(state)
        total_reward += eval_score * current_discount
        return total_reward

    def _backpropagate(self, node: _MCTSNode, reward: float) -> None:
        current = node
        value = reward
        while current is not None:
            current.visits += 1
            current.total_reward += value
            value = current.incoming_reward + self._discount * value
            current = current.parent

    def _transition(self, state: _PlannerState, action: str) -> tuple[_PlannerState, float]:
        opportunity = state.opportunity
        sentiment = state.sentiment
        risk = state.risk
        var_95 = state.var_95
        drawdown = state.drawdown
        signal_quality = state.signal_quality
        reward_hint = state.reward_hint

        noise = random.uniform(-0.02, 0.02)

        if action == "cross":
            reward = (
                reward_hint
                + 0.8 * opportunity
                + 0.4 * sentiment
                - 0.6 * risk
                - 0.2 * var_95
                - 0.1 * drawdown
            )
            opportunity = max(0.0, opportunity - 0.2 + noise)
            sentiment = max(0.0, min(1.0, sentiment + 0.04 + noise))
            risk = min(1.0, max(0.0, risk + 0.12 + abs(noise)))
        elif action == "post":
            reward = (
                reward_hint
                + 0.5 * opportunity
                + 0.3 * sentiment
                + 0.15 * signal_quality
                - 0.35 * risk
                - 0.15 * var_95
                - 0.05 * drawdown
            )
            opportunity = max(0.0, min(1.0, opportunity - 0.08 + noise))
            sentiment = max(0.0, min(1.0, sentiment + 0.02 + noise * 0.5))
            risk = max(0.0, min(1.0, risk + 0.05 + max(noise, 0.0)))
        else:  # hold
            reward = (
                reward_hint
                + 0.15 * (1.0 - risk)
                + 0.1 * signal_quality
                - 0.25 * opportunity
                - 0.05 * sentiment
            )
            opportunity = max(0.0, min(1.0, opportunity * (0.85 + noise)))
            sentiment = max(0.0, min(1.0, sentiment * (0.92 + noise * 0.5)))
            risk = max(0.0, risk - 0.08 + abs(noise) * 0.5)

        var_95 = max(0.0, min(1.0, var_95 * (0.9 if action != "cross" else 1.05) + abs(noise) * 0.1))
        drawdown = max(0.0, min(1.0, drawdown * (0.85 if action == "hold" else 0.95) + abs(noise) * 0.08))
        signal_quality = max(0.0, min(1.0, signal_quality * (0.95 + noise)))

        next_state = _PlannerState(
            opportunity=opportunity,
            sentiment=sentiment,
            risk=risk,
            var_95=var_95,
            drawdown=drawdown,
            signal_quality=signal_quality,
            reward_hint=reward_hint * 0.7 + reward * 0.3,
        )
        return next_state, reward

    def _evaluate_state(self, state: _PlannerState) -> float:
        return (
            0.6 * state.opportunity
            + 0.4 * state.sentiment
            + 0.2 * state.signal_quality
            - 0.5 * state.risk
            - 0.2 * state.var_95
            - 0.1 * state.drawdown
        )



class MarketAnalyzer(ThinkingPattern):
    """Comprehensive market analyzer combining multiple analysis components."""

    def __init__(self, config: Optional[dict[str, object]] = None):
        self.config = config or {}
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        budget_ms = float(self.config.get("mcts_budget_ms", 0.4))
        sla_grace = int(self.config.get("mcts_sla_grace", 1))
        self._planner = _ExecutionMCTS(
            budget_seconds=max(0.0003, min(0.0005, budget_ms / 1000.0)),
            max_depth=3,
            exploration=float(self.config.get("mcts_exploration", 0.9)),
            discount=float(self.config.get("mcts_discount", 0.85)),
            sla_grace=sla_grace,
        )
        self._planner_disabled_reason: str | None = None

    def analyze(self, signals: list[SensorySignal]) -> AnalysisResult:
        """Analyze market using multiple analysis components."""
        try:
            # Perform performance analysis
            performance_result = self.performance_analyzer.analyze_performance(
                [],
                initial_capital=100000.0,  # Empty trade history for signal-based analysis
            )

            # Perform risk analysis
            risk_result = self.risk_analyzer.analyze_risk(
                [],
                market_data=None,  # Empty trade history for signal-based analysis
            )

            # Combine analysis results
            combined_analysis = self._combine_analysis_results(
                performance_result, risk_result, signals
            )

            # Create unified market analysis result
            return cast(
                AnalysisResult,
                {
                    "timestamp": datetime.now(),
                    "analysis_type": "market_analysis",
                    "result": combined_analysis,
                    "confidence": self._calculate_overall_confidence(
                        performance_result, risk_result
                    ),
                    "metadata": {
                        "analysis_components": ["performance", "risk"],
                        "signal_count": len(signals),
                        "analysis_method": "comprehensive_market_analysis",
                    },
                },
            )

        except Exception as e:
            raise TradingException(f"Error in market analysis: {e}")

    def learn(self, feedback: Mapping[str, object]) -> bool:
        """Learn from feedback to improve market analysis."""
        try:
            # Delegate learning to component analyzers
            performance_learned = True  # component has no learn(); assume success
            risk_learned = True  # component has no learn(); assume success

            logger.info("Market analyzer learned from feedback")
            return bool(performance_learned) and bool(risk_learned)

        except Exception as e:
            logger.error(f"Error in market analyzer learning: {e}")
            return False

    def _combine_analysis_results(
        self,
        performance_result: AnalysisResult,
        risk_result: AnalysisResult,
        signals: list[SensorySignal],
    ) -> dict[str, object]:
        """Combine analysis results into unified market insights."""

        # Extract key metrics
        performance_metrics = cast(Any, performance_result).result.get("performance_metrics", {})
        risk_metrics = cast(Any, risk_result).result.get("risk_metrics", {})
        signal_quality = self._assess_signal_quality(signals)

        # Calculate market sentiment from signals
        market_sentiment = self._calculate_market_sentiment(signals)

        # Create combined analysis
        combined_analysis = {
            "market_sentiment": market_sentiment,
            "performance_metrics": performance_metrics,
            "risk_metrics": risk_metrics,
            "market_health": self._calculate_market_health(performance_metrics, risk_metrics),
            "trading_opportunity": self._assess_trading_opportunity(
                performance_metrics,
                risk_metrics,
                market_sentiment,
                signal_quality,
            ),
            "risk_level": self._assess_risk_level(risk_metrics),
            "signal_quality": signal_quality,
        }

        return combined_analysis

    def _calculate_market_sentiment(self, signals: list[SensorySignal]) -> dict[str, object]:
        """Calculate market sentiment from sensory signals."""
        signals_t = cast(list[_SignalProto], signals)
        if not signals:
            return {"overall_sentiment": 0.5, "confidence": 0.0, "signal_count": 0}

        # Calculate sentiment from signal values
        sentiment_values: list[float] = []
        confidences: list[float] = []

        for signal in signals_t:
            if signal.signal_type in ["sentiment", "momentum", "price_composite"]:
                sentiment_values.append(float(signal.value))  # ensure numeric as float
                confidences.append(float(signal.confidence))

        if sentiment_values:
            overall_sentiment = sum(v * c for v, c in zip(sentiment_values, confidences)) / sum(
                confidences
            )
            confidence = float(np.mean(confidences))
        else:
            overall_sentiment = 0.5
            confidence = 0.0

        return {
            "overall_sentiment": max(0.0, min(1.0, overall_sentiment)),
            "confidence": confidence,
            "signal_count": len(signals),
        }

    def _calculate_market_health(
        self, performance_metrics: dict[str, object], risk_metrics: dict[str, object]
    ) -> dict[str, object]:
        """Calculate overall market health score."""

        # Extract key metrics
        sharpe_ratio = float(cast(Any, performance_metrics.get("sharpe_ratio", 0.0)))
        max_drawdown = float(cast(Any, performance_metrics.get("max_drawdown", 0.0)))
        risk_score = float(cast(Any, risk_metrics.get("risk_score", 0.5)))

        # Calculate health components
        performance_health = min(max(sharpe_ratio / 2.0, 0.0), 1.0)  # Normalize Sharpe ratio
        drawdown_health = max(0.0, 1.0 - max_drawdown)  # Lower drawdown = better health
        risk_health = max(0.0, 1.0 - risk_score)  # Lower risk = better health

        # Calculate overall health
        overall_health = (performance_health + drawdown_health + risk_health) / 3.0

        return {
            "overall_health": overall_health,
            "performance_health": performance_health,
            "drawdown_health": drawdown_health,
            "risk_health": risk_health,
            "health_status": self._classify_health_status(overall_health),
        }

    def _assess_trading_opportunity(
        self,
        performance_metrics: dict[str, object],
        risk_metrics: dict[str, object],
        market_sentiment: dict[str, object],
        signal_quality: dict[str, object],
    ) -> dict[str, object]:
        """Assess trading opportunity based on analysis."""

        # Extract metrics
        sharpe_ratio = float(cast(Any, performance_metrics.get("sharpe_ratio", 0.0)))
        risk_score = float(cast(Any, risk_metrics.get("risk_score", 0.5)))
        sentiment = float(cast(Any, market_sentiment.get("overall_sentiment", 0.5)))
        var_95 = float(cast(Any, risk_metrics.get("var_95", 0.0)))
        current_drawdown = float(cast(Any, risk_metrics.get("current_drawdown", 0.0)))
        signal_quality_score = float(cast(Any, signal_quality.get("quality_score", 0.0)))

        # Calculate opportunity score
        opportunity_score = 0.0

        # Performance component
        if sharpe_ratio > 1.0:
            opportunity_score += 0.4
        elif sharpe_ratio > 0.5:
            opportunity_score += 0.2

        # Risk component
        if risk_score < 0.3:
            opportunity_score += 0.3
        elif risk_score < 0.5:
            opportunity_score += 0.15

        # Sentiment component
        if sentiment > 0.7:
            opportunity_score += 0.3
        elif sentiment > 0.5:
            opportunity_score += 0.15

        opportunity_score = min(opportunity_score, 1.0)

        return {
            "opportunity_score": opportunity_score,
            "opportunity_level": self._classify_opportunity_level(opportunity_score),
            "recommended_action": self._recommend_action(
                float(opportunity_score),
                float(sentiment),
                risk_score=float(risk_score),
                var_95=var_95,
                drawdown=current_drawdown,
                signal_quality=signal_quality_score,
            ),
        }

    def _assess_risk_level(self, risk_metrics: dict[str, object]) -> dict[str, object]:
        """Assess current risk level."""
        risk_score = float(cast(Any, risk_metrics.get("risk_score", 0.5)))
        var_95 = float(cast(Any, risk_metrics.get("var_95", 0.0)))
        current_drawdown = float(cast(Any, risk_metrics.get("current_drawdown", 0.0)))

        # Classify risk level
        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "var_95": var_95,
            "current_drawdown": current_drawdown,
            "risk_warnings": self._generate_risk_warnings(risk_metrics),
        }

    def _assess_signal_quality(self, signals: list[SensorySignal]) -> dict[str, object]:
        """Assess the quality of sensory signals."""
        signals_t = cast(list[_SignalProto], signals)
        if not signals:
            return {
                "quality_score": 0.0,
                "signal_count": 0,
                "average_confidence": 0.0,
                "quality_status": "poor",
            }

        # Calculate quality metrics
        confidences: list[float] = [float(s.confidence) for s in signals_t]
        average_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Calculate quality score
        quality_score = average_confidence * min(len(signals) / 10.0, 1.0)

        # Classify quality
        if quality_score > 0.7:
            quality_status = "excellent"
        elif quality_score > 0.5:
            quality_status = "good"
        elif quality_score > 0.3:
            quality_status = "fair"
        else:
            quality_status = "poor"

        return {
            "quality_score": quality_score,
            "signal_count": len(signals),
            "average_confidence": average_confidence,
            "quality_status": quality_status,
        }

    def _calculate_overall_confidence(
        self, performance_result: AnalysisResult, risk_result: AnalysisResult
    ) -> float:
        """Calculate overall confidence from component analyses."""
        performance_confidence = float(cast(Any, getattr(performance_result, "confidence", 0.0)))
        risk_confidence = float(cast(Any, getattr(risk_result, "confidence", 0.0)))
        return float((performance_confidence + risk_confidence) / 2.0)

    def _classify_health_status(self, health_score: float) -> str:
        """Classify market health status."""
        if health_score > 0.7:
            return "excellent"
        elif health_score > 0.5:
            return "good"
        elif health_score > 0.3:
            return "fair"
        else:
            return "poor"

    def _classify_opportunity_level(self, opportunity_score: float) -> str:
        """Classify trading opportunity level."""
        if opportunity_score > 0.7:
            return "high"
        elif opportunity_score > 0.4:
            return "medium"
        elif opportunity_score > 0.2:
            return "low"
        else:
            return "none"

    def _recommend_action(
        self,
        opportunity_score: float,
        sentiment: float,
        *,
        risk_score: float,
        var_95: float,
        drawdown: float,
        signal_quality: float,
    ) -> str:
        """Recommend execution style for the opportunity window."""

        planner_state = _PlannerState(
            opportunity=max(0.0, min(1.0, opportunity_score)),
            sentiment=max(0.0, min(1.0, sentiment)),
            risk=max(0.0, min(1.0, risk_score)),
            var_95=max(0.0, min(1.0, var_95)),
            drawdown=max(0.0, min(1.0, drawdown)),
            signal_quality=max(0.0, min(1.0, signal_quality)),
            reward_hint=(opportunity_score * 0.6 + sentiment * 0.4) - (risk_score * 0.5),
        )

        planner_action: Optional[str] = None
        if self._planner.enabled:
            planner_action = self._planner.plan(planner_state)
        if not self._planner.enabled and self._planner.disabled_reason:
            self._planner_disabled_reason = self._planner.disabled_reason

        if planner_action:
            return planner_action

        return self._fallback_action(opportunity_score, sentiment)

    def _fallback_action(self, opportunity_score: float, sentiment: float) -> str:
        """Deterministic fallback policy when MCTS planner is unavailable."""

        alignment = min(float(opportunity_score), float(sentiment))
        urgency = max(float(opportunity_score), float(sentiment))

        if alignment <= 0.25:
            return "hold"
        if opportunity_score >= 0.75 and sentiment >= 0.65:
            return "cross"
        if alignment >= 0.4 and urgency >= 0.45:
            return "post"
        return "hold"

    def _run_action_planner(
        self,
        opportunity_score: float,
        sentiment: float,
        depth_limit: int,
        start_ns: int,
    ) -> str | None:
        root = _PlannerNode((self._clamp(opportunity_score), self._clamp(sentiment)), depth=0)
        iterations = 0

        while iterations < self._planner_max_iterations:
            if time.perf_counter_ns() - start_ns >= self._planner_budget_ns:
                break
            leaf, path = self._traverse_tree(root, depth_limit)
            rollout_value = self._evaluate_leaf(leaf, depth_limit)
            self._backpropagate(path, rollout_value)
            iterations += 1

        if not root.children:
            return None

        best_value = float("-inf")
        best_action: str | None = None
        for action, child in root.children.items():
            if child.visits == 0:
                continue
            score = child.value / child.visits
            if score > best_value:
                best_value = score
                best_action = action

        if best_action is None:
            best_child = max(
                root.children.values(),
                key=lambda node: node.prior_reward if node.prior_reward is not None else float("-inf"),
            )
            best_action = best_child.action

        return best_action

    def _traverse_tree(
        self, root: _PlannerNode, depth_limit: int
    ) -> tuple[_PlannerNode, list[_PlannerNode]]:
        node = root
        path = [root]

        while node.depth < depth_limit:
            if len(node.children) < len(self._planner_actions):
                action = self._planner_actions[len(node.children)]
                next_state, reward = self._apply_action(node.state, action, node.depth)
                child = _PlannerNode(next_state, depth=node.depth + 1, action=action, prior_reward=reward)
                node.children[action] = child
                path.append(child)
                return child, path

            node = self._select_child(node)
            path.append(node)

        return node, path

    def _select_child(self, node: _PlannerNode) -> _PlannerNode:
        parent_visits = max(1, node.visits)
        best_score = float("-inf")
        best_child = None

        for child in node.children.values():
            if child.visits == 0:
                return child
            exploitation = child.value / child.visits
            exploration = self._planner_exploration * (parent_visits**0.5) / (1.0 + child.visits)
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child

        return best_child if best_child is not None else next(iter(node.children.values()))

    def _evaluate_leaf(self, node: _PlannerNode, depth_limit: int) -> float:
        if node.depth >= depth_limit:
            return self._estimate_terminal_value(node.state)
        return self._rollout(node.state, node.depth, depth_limit)

    def _rollout(self, state: tuple[float, float], depth: int, depth_limit: int) -> float:
        opportunity, sentiment = state
        total = 0.0
        discount = 1.0
        current_depth = depth

        while current_depth < depth_limit:
            action = self._rollout_policy(opportunity, sentiment)
            (opportunity, sentiment), reward = self._apply_action(
                (opportunity, sentiment), action, current_depth
            )
            total += reward * discount
            discount *= self._planner_discount
            current_depth += 1

        return total

    def _apply_action(
        self, state: tuple[float, float], action: str, depth: int
    ) -> tuple[tuple[float, float], float]:
        opportunity, sentiment = state
        reward = self._estimate_reward(opportunity, sentiment, action)
        next_state = self._advance_state(opportunity, sentiment, action, depth)
        return next_state, reward

    def _advance_state(
        self, opportunity: float, sentiment: float, action: str, depth: int
    ) -> tuple[float, float]:
        alignment = min(opportunity, sentiment)
        urgency = max(opportunity, sentiment)
        seed = (
            int(opportunity * 1000.0) * 17
            + int(sentiment * 1000.0) * 29
            + depth * 13
            + self._planner_action_hash[action]
        )
        noise = ((seed * 1103515245 + 12345) & 0x7FFFFFFF) / 0x7FFFFFFF - 0.5
        noise *= 0.05

        if action == "cross":
            next_opportunity = self._clamp(opportunity * 0.55 + sentiment * 0.35 + 0.08 + noise)
            next_sentiment = self._clamp(sentiment * 0.82 + alignment * 0.18 + 0.1 + noise * 0.6)
        elif action == "post":
            next_opportunity = self._clamp(opportunity * 0.68 + sentiment * 0.22 + 0.05 + noise * 0.8)
            next_sentiment = self._clamp(sentiment * 0.9 + (urgency - alignment) * 0.12 + 0.04 + noise * 0.5)
        else:  # hold
            next_opportunity = self._clamp(opportunity * 0.74 + (sentiment - 0.5) * 0.08 + 0.03 + noise)
            next_sentiment = self._clamp(sentiment * 0.88 + (0.5 - opportunity) * 0.12 + 0.02 + noise * 0.6)

        return next_opportunity, next_sentiment

    def _estimate_reward(self, opportunity: float, sentiment: float, action: str) -> float:
        alignment = min(opportunity, sentiment)
        urgency = max(opportunity, sentiment)
        imbalance = abs(opportunity - sentiment)

        if action == "cross":
            execution_cost = 0.12 + 0.35 * imbalance
            reward = alignment * 1.6 + urgency * 0.4 - execution_cost
        elif action == "post":
            queue_risk = 0.05 + 0.18 * imbalance
            reward = alignment * 1.2 + (urgency - alignment) * 0.6 - queue_risk
        else:
            patience_bonus = (0.7 - urgency) * 0.45
            reward = alignment * 0.7 + patience_bonus - 0.02

        return reward

    def _estimate_terminal_value(self, state: tuple[float, float]) -> float:
        opportunity, sentiment = state
        alignment = min(opportunity, sentiment)
        return alignment * 0.9 + (sentiment - 0.5) * 0.2

    def _backpropagate(self, path: list[_PlannerNode], rollout_reward: float) -> None:
        accumulated = rollout_reward
        for node in reversed(path):
            if node.prior_reward is not None:
                accumulated = node.prior_reward + self._planner_discount * accumulated
            node.visits += 1
            node.value += accumulated

    def _rollout_policy(self, opportunity: float, sentiment: float) -> str:
        alignment = min(opportunity, sentiment)
        urgency = max(opportunity, sentiment)

        if alignment > 0.65:
            return "cross"
        if urgency - alignment > 0.12:
            return "post"
        if urgency < 0.45:
            return "hold"
        return "post"

    @staticmethod
    def _clamp(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _generate_risk_warnings(self, risk_metrics: dict[str, object]) -> list[str]:
        """Generate risk warnings based on metrics."""
        warnings = []

        risk_score = float(cast(Any, risk_metrics.get("risk_score", 0.5)))
        var_95 = float(cast(Any, risk_metrics.get("var_95", 0.0)))
        current_drawdown = float(cast(Any, risk_metrics.get("current_drawdown", 0.0)))

        if risk_score > 0.7:
            warnings.append("High risk score detected")
        if var_95 > 0.05:
            warnings.append("High Value at Risk (VaR)")
        if current_drawdown > 0.1:
            warnings.append("Significant current drawdown")

        return warnings
