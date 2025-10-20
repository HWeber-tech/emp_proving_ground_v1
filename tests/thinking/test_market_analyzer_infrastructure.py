from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import types


if "src.core.events" not in sys.modules:
    events_module = types.ModuleType("src.core.events")

    @dataclass
    class AnalysisResult:
        timestamp: object
        analysis_type: str
        result: dict[str, object]
        confidence: float
        metadata: dict[str, object]

    @dataclass
    class PerformanceMetrics:
        total_return: float = 0.0
        annualized_return: float = 0.0
        volatility: float = 0.0
        sharpe_ratio: float = 0.0
        sortino_ratio: float = 0.0
        max_drawdown: float = 0.0
        win_rate: float = 0.0
        profit_factor: float = 0.0
        total_trades: int = 0
        avg_trade_duration: float = 0.0
        metadata: dict[str, object] = field(default_factory=dict)

    @dataclass
    class RiskMetrics:
        var_95: float = 0.0
        var_99: float = 0.0
        cvar_95: float = 0.0
        cvar_99: float = 0.0
        beta: float = 0.0
        correlation: float = 0.0
        current_drawdown: float = 0.0
        risk_score: float = 0.0
        metadata: dict[str, object] = field(default_factory=dict)

    @dataclass
    class TradeIntent:
        action: str = "HOLD"
        price: float | None = None
        quantity: float | None = None
        timestamp: object = None

    events_module.AnalysisResult = AnalysisResult
    events_module.PerformanceMetrics = PerformanceMetrics
    events_module.RiskMetrics = RiskMetrics
    events_module.TradeIntent = TradeIntent
    sys.modules["src.core.events"] = events_module

if "src" not in sys.modules:
    src_module = types.ModuleType("src")
    src_module.__path__ = [str(Path(__file__).resolve().parents[2] / "src")]
    sys.modules["src"] = src_module

if "src.thinking" not in sys.modules:
    thinking_module = types.ModuleType("src.thinking")
    thinking_module.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "thinking")]
    sys.modules["src.thinking"] = thinking_module

import pytest

from src.thinking.analysis.market_analyzer import MarketAnalyzer


@dataclass
class _FakeSignal:
    signal_type: str
    value: float
    confidence: float


class _StubPlanner:
    def __init__(self, action: str = "cross") -> None:
        self.enabled = True
        self.disabled_reason: str | None = None
        self._action = action
        self.last_state = None

    def plan(self, state) -> str:
        self.last_state = state
        return self._action


def test_extract_infrastructure_metrics_normalises_signals() -> None:
    analyzer = MarketAnalyzer(config={"planner_latency_threshold_ms": 2.0})
    analyzer.update_decision_latency(0.0015)

    signals = [
        _FakeSignal("infra.latency_ms", 1.0, 1.0),
        _FakeSignal("infra.queue_position", 0.7, 0.8),
        _FakeSignal("infra.execution_certainty", 0.65, 0.9),
    ]

    metrics = analyzer._extract_infrastructure_metrics(signals)

    assert metrics["latency_s"] == pytest.approx(0.001)
    assert metrics["latency_pressure"] == pytest.approx(0.5)
    assert metrics["queue_position"] == pytest.approx(0.7)
    assert metrics["execution_certainty"] == pytest.approx(0.65)


def test_recommend_action_passes_infrastructure_state_to_planner() -> None:
    analyzer = MarketAnalyzer()
    stub = _StubPlanner(action="post")
    analyzer._planner = stub  # type: ignore[assignment]
    analyzer.update_decision_latency(0.0004)

    infrastructure = {
        "latency_s": 0.001,
        "queue_position": 0.8,
        "execution_certainty": 0.3,
    }

    action = analyzer._recommend_action(
        opportunity_score=0.8,
        sentiment=0.7,
        risk_score=0.2,
        var_95=0.05,
        drawdown=0.03,
        signal_quality=0.6,
        infrastructure=infrastructure,
    )

    assert action == "post"
    assert stub.last_state is not None
    assert stub.last_state.latency_pressure == pytest.approx(1.0)
    assert stub.last_state.queue_position == pytest.approx(0.8)
    assert stub.last_state.execution_certainty == pytest.approx(0.3)
