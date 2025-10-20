from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from types import MappingProxyType, SimpleNamespace

from typing import Any, Mapping

import pytest

from src.operations.drift_sentry import DriftSentryConfig, evaluate_drift_sentry
from src.operations.sensory_drift import DriftSeverity
from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage, PolicyLedgerStore
from src.orchestration.alpha_trade_loop import (
    AlphaTradeLoopOrchestrator,
    ComplianceEvent,
    ComplianceEventType,
    ComplianceSeverity,
)
from src.orchestration.alpha_trade_runner import AlphaTradeLoopRunner
from src.trading.gating import DriftSentryGate
from src.trading.trading_manager import TradeIntentOutcome
from src.understanding.belief import BeliefBuffer, BeliefEmitter, RegimeFSM
from src.understanding.decision_diary import DecisionDiaryStore
from src.understanding.router import UnderstandingRouter
from src.thinking.adaptation.feature_toggles import AdaptationFeatureToggles
from src.thinking.adaptation.policy_router import (
    FastWeightExperiment,
    PolicyDecision,
    PolicyRouter,
    PolicyTactic,
)
from tests.util import promotion_checklist_metadata


UTC = timezone.utc


class _FakeTradingManager:
    def __init__(self) -> None:
        self.intents: list[dict[str, object]] = []

    async def on_trade_intent(self, event: dict[str, object]) -> TradeIntentOutcome:
        self.intents.append(dict(event))
        metadata: Mapping[str, object] | dict[str, object]
        metadata = {}
        payload = event.get("metadata") if isinstance(event, dict) else None
        if isinstance(payload, Mapping):
            metadata = dict(payload)
        return TradeIntentOutcome(status="executed", executed=True, metadata=dict(metadata))

    def assess_performance_health(self) -> dict[str, object]:
        return {
            "healthy": True,
            "throughput": {"healthy": True},
            "backlog": {"healthy": True, "evaluated": False},
            "resource": {
                "healthy": True,
                "status": "not_configured",
                "sample": {},
            },
        }


class _ThrottleTradingManager:
    def __init__(self) -> None:
        self.intents: list[dict[str, object]] = []
        message = "Throttled: too many trades in short time"
        self._outcome = TradeIntentOutcome(
            status="throttled",
            executed=False,
            throttle={
                "name": "trade_rate_limit",
                "state": "rate_limited",
                "message": f"{message} (limit 1 trade per minute)",
                "active": True,
            },
            metadata={"message": message},
        )

    async def on_trade_intent(self, event: dict[str, object]) -> TradeIntentOutcome:
        self.intents.append(dict(event))
        return self._outcome

    def assess_performance_health(self) -> dict[str, object]:
        return {
            "healthy": False,
            "throughput": {"healthy": True},
            "backlog": {"healthy": True, "evaluated": False},
            "resource": {
                "healthy": True,
                "status": "not_configured",
                "sample": {},
            },
            "throttle": {"state": "rate_limited", "active": True},
        }


class _RiskRejectTradingManager:
    def __init__(self) -> None:
        self.intents: list[dict[str, object]] = []
        self._outcome = TradeIntentOutcome(
            status="rejected",
            executed=False,
            metadata={"reason": "policy_violation"},
        )

    async def on_trade_intent(self, event: dict[str, object]) -> TradeIntentOutcome:
        self.intents.append(dict(event))
        return self._outcome

    def assess_performance_health(self) -> dict[str, object]:
        return {
            "healthy": True,
            "throughput": {"healthy": True},
            "backlog": {"healthy": True, "evaluated": False},
            "resource": {
                "healthy": True,
                "status": "not_configured",
                "sample": {},
            },
        }


class _GuardrailViolationTradingManager:
    def __init__(self) -> None:
        self.intents: list[dict[str, object]] = []

    async def on_trade_intent(self, event: dict[str, object]) -> TradeIntentOutcome:
        self.intents.append(dict(event))
        incident = {
            "incident_id": "inv-guardrail-001",
            "severity": "violation",
            "reason": "synthetic invariant breach",
            "timestamp": datetime(2025, 2, 3, 12, 30, tzinfo=UTC).isoformat(),
            "metadata": {
                "violations": ["risk.synthetic_invariant_posture"],
                "warnings": [],
                "overrides": False,
            },
            "checks": [
                {
                    "name": "risk.synthetic_invariant_posture",
                    "status": "violation",
                    "threshold": 0.0,
                    "value": 1.0,
                }
            ],
        }
        guardrail_payload = {
            "active": True,
            "force_paper": True,
            "reason": "synthetic invariant breach",
            "severity": "violation",
            "remaining_seconds": 300.0,
            "incident": incident,
        }
        metadata = {
            "guardrails": {
                "risk_guardrail": guardrail_payload,
            }
        }
        return TradeIntentOutcome(status="executed", executed=True, metadata=metadata)

    def assess_performance_health(self) -> dict[str, object]:
        return {
            "healthy": True,
            "throughput": {"healthy": True},
            "backlog": {"healthy": True, "evaluated": False},
            "resource": {
                "healthy": True,
                "status": "not_configured",
                "sample": {},
            },
        }


class _AlphaDecayTradingManager:
    def __init__(self, multiplier: float = 0.65) -> None:
        self.intents: list[dict[str, object]] = []
        self._multiplier = multiplier

    async def on_trade_intent(self, event: dict[str, object]) -> TradeIntentOutcome:
        self.intents.append(dict(event))
        price = float(event.get("price", 1.0))
        quantity_before = float(event.get("quantity", 0.0))
        quantity_after = quantity_before * self._multiplier
        metadata = {
            "latency_ms": 8.0,
            "notional": quantity_after * price,
            "price": price,
            "quantity": quantity_after,
            "throttle_multiplier": self._multiplier,
            "quantity_before_throttle": quantity_before,
            "quantity_after_throttle": quantity_after,
        }
        throttle_snapshot = {
            "name": "alpha_decay",
            "state": "active",
            "active": True,
            "multiplier": self._multiplier,
        }
        return TradeIntentOutcome(
            status="executed",
            executed=True,
            throttle=throttle_snapshot,
            metadata=metadata,
        )

    def assess_performance_health(self) -> dict[str, object]:
        return {
            "healthy": True,
            "throughput": {"healthy": True},
            "backlog": {"healthy": True, "evaluated": False},
            "resource": {
                "healthy": True,
                "status": "not_configured",
                "sample": {},
            },
        }


def _build_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    trading_manager,
    feature_toggles: AdaptationFeatureToggles | None = None,
    runner_kwargs: dict[str, Any] | None = None,
) -> tuple[AlphaTradeLoopRunner, DecisionDiaryStore]:
    buffer = BeliefBuffer(belief_id="alpha-belief")
    belief_emitter = BeliefEmitter(
        buffer=buffer,
        event_bus=object(),
    )
    regime_fsm = RegimeFSM(
        event_bus=object(),
        signal_id="alpha-signal",
    )

    policy_router = PolicyRouter()
    policy_router.register_tactic(
        PolicyTactic(
            tactic_id="alpha.live",
            base_weight=1.0,
            parameters={
                "symbol": "EURUSD",
                "side": "buy",
                "size": 25_000,
                "price": 1.2345,
            },
            guardrails={"requires_diary": True},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.25,
            description="AlphaTrade live staging tactic",
            objectives=("alpha",),
            tags=("alpha_trade",),
        )
    )
    understanding_router = UnderstandingRouter(policy_router)

    diary_store = DecisionDiaryStore(tmp_path / "diary.json", publish_on_record=False)

    ledger_store = PolicyLedgerStore(tmp_path / "ledger.json")
    ledger_store.upsert(
        policy_id="alpha.live",
        tactic_id="alpha.live",
        stage=PolicyLedgerStage.PAPER,
        approvals=("risk",),
        evidence_id="dd-alpha-live",
    )
    release_manager = LedgerReleaseManager(ledger_store)
    drift_gate = DriftSentryGate()

    orchestrator = AlphaTradeLoopOrchestrator(
        router=understanding_router,
        diary_store=diary_store,
        drift_gate=drift_gate,
        release_manager=release_manager,
    )

    runner_kwargs = runner_kwargs or {}

    runner = AlphaTradeLoopRunner(
        belief_emitter=belief_emitter,
        regime_fsm=regime_fsm,
        orchestrator=orchestrator,
        trading_manager=trading_manager,
        understanding_router=understanding_router,
        feature_toggles=feature_toggles,
        **runner_kwargs,
    )

    return runner, diary_store


def test_alpha_trade_runner_bounds_confidence_probability() -> None:
    assert AlphaTradeLoopRunner._bound_probability(1.5) == pytest.approx(1.0)
    assert AlphaTradeLoopRunner._bound_probability(-0.2) == pytest.approx(0.0)
    assert AlphaTradeLoopRunner._bound_probability(0.42) == pytest.approx(0.42)


def test_alpha_trade_runner_bounds_top_feature_norms() -> None:
    features = {
        "sigma_norm": 1_000_000.0,
        "alpha_norm": -500_000.0,
        "balanced": 0.75,
    }
    summary = AlphaTradeLoopRunner._select_top_features(features, limit=3)
    limit = AlphaTradeLoopRunner._TOP_FEATURE_NORM_LIMIT
    assert summary, "expected bounded features to be returned"
    for entry in summary:
        assert abs(entry["value"]) <= limit


def test_alpha_trade_runner_builds_brief_explanation() -> None:
    text = "AlphaTrade explanation " * 40
    trimmed = AlphaTradeLoopRunner._build_brief_explanation(text)
    assert len(trimmed) <= AlphaTradeLoopRunner._ATTRIBUTION_EXPLANATION_LIMIT
    assert trimmed.endswith("...")
    assert "  " not in trimmed


def test_alpha_trade_runner_build_trade_intent_threads_metadata() -> None:
    runner = AlphaTradeLoopRunner.__new__(AlphaTradeLoopRunner)

    belief_state = SimpleNamespace(
        symbol="EURUSD",
        generated_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
    )

    regime_state = SimpleNamespace(regime="balanced", confidence=0.8)
    regime_signal = SimpleNamespace(regime_state=regime_state)

    trade_metadata: dict[str, Any] = {
        "policy_id": "alpha.live",
        "symbol": "EURUSD",
        "side": "buy",
        "quantity": 25_000,
        "price": 1.2345,
        "fast_weight": {"enabled": True},
        "guardrails": {"requires_diary": True},
        "attribution": {
            "belief": {"belief_id": "belief-123"},
            "probes": [{"probe_id": "guardrail.requires_diary", "status": "ok"}],
            "brief_explanation": "unit test attribution context",
            "explanation": "unit test attribution context",
        },
        "diary_coverage": {"belief": True},
        "performance_health": {"healthy": True},
    }

    decision: dict[str, Any] = {
        "tactic_id": "alpha.live",
        "parameters": {
            "symbol": "EURUSD",
            "side": "buy",
            "quantity": 25_000,
            "price": 1.2345,
        },
    }

    intent = runner._build_trade_intent_from_decision(
        decision,
        belief_state,
        regime_signal,
        trade_metadata,
        overrides=None,
    )

    assert intent is not None
    metadata = intent["metadata"]
    assert metadata["fast_weight"] == trade_metadata["fast_weight"]
    assert metadata["guardrails"] == trade_metadata["guardrails"]
    assert metadata["attribution"] == trade_metadata["attribution"]
    assert metadata["diary_coverage"] == trade_metadata["diary_coverage"]
    assert metadata["performance_health"] == trade_metadata["performance_health"]
    assert metadata["confidence"] == pytest.approx(0.8)


def test_alpha_trade_runner_attribution_fallback_explanation() -> None:
    runner = AlphaTradeLoopRunner.__new__(AlphaTradeLoopRunner)

    belief_state = SimpleNamespace(
        belief_id="belief-fallback",
        metadata=None,
        symbol="EURUSD",
        generated_at=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
    )

    regime_state = SimpleNamespace(confidence=0.72, regime="balanced")
    belief_snapshot = SimpleNamespace(
        regime_state=regime_state,
        features={"sigma_norm": 1_000.0},
    )
    decision = SimpleNamespace(
        rationale="   \n\t  ",
        tactic_id="alpha.trade",
        guardrails={"requires_diary": True},
    )
    decision_bundle = SimpleNamespace(
        decision=decision,
        belief_snapshot=belief_snapshot,
    )

    diary_entry = SimpleNamespace(
        entry_id="dd-fallback",
        policy_id="alpha.trade",
        probes=(),
        metadata={"guardrails": {"requires_diary": True}},
    )

    attribution = runner._build_order_attribution(
        belief_state=belief_state,
        decision_bundle=decision_bundle,
        diary_entry=diary_entry,
    )

    assert attribution is not None
    explanation = attribution["explanation"]
    brief_explanation = attribution["brief_explanation"]
    assert explanation == "alpha.trade routed under balanced"
    assert brief_explanation == explanation
    assert len(brief_explanation) <= AlphaTradeLoopRunner._ATTRIBUTION_EXPLANATION_LIMIT

    belief_summary = attribution["belief"]
    assert belief_summary["belief_id"] == "belief-fallback"
    assert belief_summary["confidence"] == pytest.approx(0.72)
    assert belief_summary["regime"] == "balanced"

    probes = attribution["probes"]
    assert probes and {probe.get("probe_id") for probe in probes} == {
        "guardrail.requires_diary"
    }

@pytest.mark.asyncio()
async def test_alpha_trade_loop_runner_executes_trade(monkeypatch, tmp_path) -> None:
    trading_manager = _FakeTradingManager()
    runner, diary_store = _build_runner(monkeypatch, tmp_path, trading_manager)

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    result = await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    assert trading_manager.intents, "expected trade intent to be forwarded"
    intent = trading_manager.intents[0]
    assert intent["strategy_id"] == "alpha.live"
    assert intent["side"] == "BUY"
    assert intent["quantity"] == pytest.approx(25_000)
    assert intent["price"] == pytest.approx(1.2345)

    assert result.loop_result.policy_id == "alpha.live"
    assert result.trade_metadata["notional"] == pytest.approx(25_000 * 1.2345)
    assert result.trade_intent["metadata"]["regime"] == result.regime_signal.regime_state.regime
    fast_weight_metadata = result.trade_metadata.get("fast_weight")
    assert isinstance(fast_weight_metadata, dict)
    assert fast_weight_metadata.get("enabled") is True
    assert "metrics" in fast_weight_metadata
    guardrails = result.loop_result.decision.guardrails
    assert result.trade_metadata.get("guardrails") == guardrails
    intent_fast_weight = trading_manager.intents[0]["metadata"].get("fast_weight")
    assert intent_fast_weight == fast_weight_metadata
    assert trading_manager.intents[0]["metadata"].get("guardrails") == guardrails
    attribution = result.trade_metadata.get("attribution")
    assert isinstance(attribution, dict)
    belief_summary = attribution.get("belief")
    assert isinstance(belief_summary, dict)
    assert belief_summary.get("belief_id") == result.belief_state.belief_id
    assert attribution.get("brief_explanation")
    assert attribution.get("explanation") == attribution.get("brief_explanation")
    probes = attribution.get("probes")
    assert isinstance(probes, list) and probes, "expected attribution probes to be populated"
    probe_ids = {probe.get("probe_id") for probe in probes if isinstance(probe, Mapping)}
    assert "governance.guardrails" in probe_ids
    assert "governance.drift_sentry" in probe_ids
    intent_attribution = trading_manager.intents[0]["metadata"].get("attribution")
    assert intent_attribution == attribution
    assert diary_store.entries(), "expected decision diary entry to be recorded"
    assert result.trade_outcome is not None
    assert result.trade_outcome.status == "executed"
    assert result.trade_outcome.metadata.get("attribution") == attribution
    assert result.trade_outcome.metadata.get("guardrails") == guardrails
    entry = diary_store.entries()[0]
    trade_execution = entry.metadata.get("trade_execution")
    assert isinstance(trade_execution, dict)
    assert trade_execution["status"] == "executed"
    expected_coverage = dict(runner.describe_diary_coverage())
    outcome_coverage = result.trade_outcome.metadata.get("diary_coverage")
    assert outcome_coverage == expected_coverage
    intent_coverage = trading_manager.intents[0]["metadata"].get("diary_coverage")
    assert intent_coverage == expected_coverage
    assert result.trade_metadata.get("diary_coverage") == expected_coverage
    assert result.loop_result.metadata.get("diary_coverage") == expected_coverage
    assert entry.metadata.get("diary_coverage") == expected_coverage
    assert entry.metadata.get("attribution") == attribution
    performance_health = entry.metadata.get("performance_health")
    assert isinstance(performance_health, dict)
    assert performance_health.get("healthy") is True
    assert result.trade_metadata.get("performance_health") == performance_health


@pytest.mark.asyncio()
async def test_alpha_trade_runner_merges_counterfactual_guardrail(monkeypatch, tmp_path) -> None:
    trading_manager = _FakeTradingManager()

    buffer = BeliefBuffer(belief_id="alpha-live-belief")
    belief_emitter = BeliefEmitter(
        buffer=buffer,
        event_bus=object(),
    )
    regime_fsm = RegimeFSM(
        event_bus=object(),
        signal_id="alpha-live-signal",
    )

    policy_router = PolicyRouter()
    policy_router.register_tactic(
        PolicyTactic(
            tactic_id="alpha.live",
            base_weight=1.0,
            parameters={
                "symbol": "EURUSD",
                "side": "buy",
                "size": 25_000,
                "price": 1.2345,
            },
            guardrails={},
            regime_bias={"balanced": 1.0},
            confidence_sensitivity=0.0,
        )
    )
    understanding_router = UnderstandingRouter(policy_router)
    understanding_router.policy_router.register_experiment(
        FastWeightExperiment(
            experiment_id="live_counterfactual",
            tactic_id="alpha.live",
            delta=1.0,
            rationale="force counterfactual guardrail",
            min_confidence=0.0,
        )
    )

    diary_store = DecisionDiaryStore(tmp_path / "counterfactual_diary.json", publish_on_record=False)

    ledger_store = PolicyLedgerStore(tmp_path / "counterfactual_ledger.json")
    ledger_store.upsert(
        policy_id="alpha.live",
        tactic_id="alpha.live",
        stage=PolicyLedgerStage.LIMITED_LIVE,
        approvals=("risk", "ops"),
        evidence_id="dd-alpha-live",
        metadata=promotion_checklist_metadata(),
    )
    release_manager = LedgerReleaseManager(ledger_store)
    drift_gate = DriftSentryGate()

    orchestrator = AlphaTradeLoopOrchestrator(
        router=understanding_router,
        diary_store=diary_store,
        drift_gate=drift_gate,
        release_manager=release_manager,
    )

    runner = AlphaTradeLoopRunner(
        belief_emitter=belief_emitter,
        regime_fsm=regime_fsm,
        orchestrator=orchestrator,
        trading_manager=trading_manager,
        understanding_router=understanding_router,
    )

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
        "price": 1.2345,
        "quantity": 25_000,
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.1, "confidence": 0.9},
        },
        "integrated_signal": {"strength": 0.2, "confidence": 0.9},
    }

    existing_guardrails = {
        "risk_guardrail": {
            "breached": True,
            "force_paper": True,
            "reason": "risk_guardrail_active",
        }
    }

    result = await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={
            "policy_id": "alpha.live",
            "guardrails": existing_guardrails,
        },
    )

    assert trading_manager.intents, "expected trade intent to be forwarded"
    intent_metadata = trading_manager.intents[0]["metadata"]

    guardrails = result.trade_metadata.get("guardrails")
    assert isinstance(guardrails, dict)
    assert guardrails.get("force_paper") is True
    counterfactual = guardrails.get("counterfactual_guardrail")
    assert isinstance(counterfactual, dict)
    assert counterfactual.get("breached") is True
    assert counterfactual.get("reason") == "counterfactual_guardrail_delta_exceeded"
    assert counterfactual.get("action") == "force_paper"
    assert counterfactual.get("severity") == "aggro"
    assert counterfactual.get("delta_direction") == "aggro"
    assert counterfactual.get("relative_breach") is True
    assert counterfactual.get("absolute_breach") is False
    assert guardrails.get("risk_guardrail") == existing_guardrails["risk_guardrail"]

    assert intent_metadata.get("guardrails") == guardrails
    assert result.trade_outcome.metadata.get("guardrails") == guardrails


def test_alpha_trade_orchestrator_respects_passive_counterfactual_bound(tmp_path) -> None:
    store = PolicyLedgerStore(tmp_path / "passive_guardrail_ledger.json")
    release_manager = LedgerReleaseManager(store)

    policy_router = PolicyRouter()
    understanding_router = UnderstandingRouter(policy_router)
    diary_store = DecisionDiaryStore(
        tmp_path / "passive_guardrail_diary.json",
        publish_on_record=False,
    )
    drift_gate = DriftSentryGate()

    orchestrator = AlphaTradeLoopOrchestrator(
        router=understanding_router,
        diary_store=diary_store,
        drift_gate=drift_gate,
        release_manager=release_manager,
    )

    decision = PolicyDecision(
        tactic_id="alpha.live",
        parameters={},
        selected_weight=0.75,
        guardrails={},
        rationale="",
        experiments_applied=(),
        reflection_summary={},
        weight_breakdown={"base_score": 1.0, "final_score": 0.75},
    )

    guardrails: dict[str, Any] = {"force_paper": False}
    limits = {"relative": 0.20, "relative_passive": 0.4}

    updated = orchestrator._apply_counterfactual_guardrail(
        guardrails,
        decision,
        stage=PolicyLedgerStage.LIMITED_LIVE,
        limits=limits,
    )

    payload = updated.get("counterfactual_guardrail")
    assert isinstance(payload, dict)
    assert payload.get("delta_direction") == "passive"
    assert payload.get("score_delta") == pytest.approx(-0.25)
    assert payload.get("relative_breach") is False
    assert payload.get("breached") is False
    assert payload.get("max_relative_delta") == pytest.approx(0.4)
    assert payload.get("severity") is None
    assert updated.get("force_paper") is False

    guardrails_second: dict[str, Any] = {"force_paper": False}
    tighter_limits = {"relative": 0.20, "relative_passive": 0.1}

    updated_second = orchestrator._apply_counterfactual_guardrail(
        guardrails_second,
        decision,
        stage=PolicyLedgerStage.LIMITED_LIVE,
        limits=tighter_limits,
    )

    second_payload = updated_second.get("counterfactual_guardrail")
    assert isinstance(second_payload, dict)
    assert second_payload.get("breached") is True
    assert second_payload.get("relative_breach") is True
    assert second_payload.get("max_relative_delta") == pytest.approx(0.1)
    assert second_payload.get("severity") == "passive"
    assert updated_second.get("force_paper") is False


@pytest.mark.asyncio()
async def test_alpha_trade_loop_runner_records_throttle_metadata(
    monkeypatch, tmp_path
) -> None:
    trading_manager = _ThrottleTradingManager()
    runner, diary_store = _build_runner(monkeypatch, tmp_path, trading_manager)

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    result = await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    assert trading_manager.intents
    assert result.trade_outcome is not None
    assert result.trade_outcome.status == "throttled"
    entry = diary_store.entries()[0]
    trade_execution = entry.metadata.get("trade_execution")
    assert isinstance(trade_execution, dict)
    assert trade_execution["status"] == "throttled"
    throttle_snapshot = trade_execution.get("throttle")
    assert isinstance(throttle_snapshot, dict)
    assert throttle_snapshot.get("state") == "rate_limited"
    assert "too many trades" in throttle_snapshot.get("message", "")
    performance_health = entry.metadata.get("performance_health")
    assert isinstance(performance_health, dict)
    assert performance_health.get("healthy") is False
    throttle_summary = performance_health.get("throttle")
    assert isinstance(throttle_summary, dict)
    assert throttle_summary.get("state") == "rate_limited"
    assert result.trade_metadata.get("performance_health") == performance_health


@pytest.mark.asyncio()
async def test_alpha_trade_runner_honours_fast_weight_toggle(monkeypatch, tmp_path) -> None:
    trading_manager = _FakeTradingManager()
    toggles = AdaptationFeatureToggles(fast_weights=False)
    runner, _ = _build_runner(
        monkeypatch,
        tmp_path,
        trading_manager,
        feature_toggles=toggles,
    )

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    result = await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    fast_weight_metadata = result.trade_metadata["fast_weight"]
    assert fast_weight_metadata["enabled"] is False


@pytest.mark.asyncio()
async def test_alpha_trade_runner_freezes_on_risk_rejection(monkeypatch, tmp_path) -> None:
    trading_manager = _RiskRejectTradingManager()
    runner, _ = _build_runner(monkeypatch, tmp_path, trading_manager)
    policy_router = runner._understanding_router.policy_router
    release_manager = runner._orchestrator._release_manager
    release_manager._store.upsert(
        policy_id="alpha.live",
        tactic_id="alpha.live",
        stage=PolicyLedgerStage.EXPERIMENT,
        approvals=(),
        evidence_id="dd-alpha-live",
        allow_regression=True,
    )
    policy_router.register_tactic(
        PolicyTactic(
            tactic_id="alpha.explore",
            base_weight=1.5,
            regime_bias={"balanced": 1.0},
            exploration=True,
            parameters={
                "symbol": "EURUSD",
                "side": "buy",
                "size": 25_000,
                "price": 1.2345,
            },
        )
    )

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    assert policy_router.exploration_freeze_active() is True

    follow_up = await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    freeze_state = policy_router.exploration_freeze_state()
    assert freeze_state.get("active") is True
    assert freeze_state.get("reason") in {"policy_violation", "risk_rejected"}
    assert follow_up.loop_result.decision.tactic_id == "alpha.live"
    metadata = follow_up.loop_result.decision.exploration_metadata
    assert metadata.get("selected_is_exploration") is False
    blocked = metadata.get("blocked_candidates", [])
    assert any(item.get("reason") == "frozen" for item in blocked)


@pytest.mark.asyncio()
async def test_alpha_trade_runner_freezes_on_guardrail_invariant_violation(
    monkeypatch, tmp_path
) -> None:
    trading_manager = _GuardrailViolationTradingManager()
    runner, _ = _build_runner(monkeypatch, tmp_path, trading_manager)
    policy_router = runner._understanding_router.policy_router

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    assert policy_router.exploration_freeze_active() is True
    freeze_state = policy_router.exploration_freeze_state()
    assert freeze_state.get("reason") == "synthetic invariant breach"
    metadata = freeze_state.get("metadata")
    assert isinstance(metadata, Mapping)
    triggers = metadata.get("triggers")
    assert isinstance(triggers, list) and triggers
    guardrail_trigger = next(
        trigger
        for trigger in triggers
        if isinstance(trigger, Mapping) and trigger.get("triggered_by") == "risk_guardrail"
    )
    guardrail_metadata = guardrail_trigger.get("metadata")
    assert isinstance(guardrail_metadata, Mapping)
    violations = guardrail_metadata.get("violations")
    assert any(
        isinstance(entry, str) and "invariant" in entry for entry in (violations or [])
    )


@pytest.mark.asyncio()
async def test_alpha_trade_runner_releases_freeze_after_safe_iteration(
    monkeypatch, tmp_path
) -> None:
    trading_manager = _RiskRejectTradingManager()
    runner, _ = _build_runner(monkeypatch, tmp_path, trading_manager)
    policy_router = runner._understanding_router.policy_router
    policy_router.register_tactic(
        PolicyTactic(
            tactic_id="alpha.explore",
            base_weight=1.5,
            regime_bias={"balanced": 1.0},
            exploration=True,
            parameters={
                "symbol": "EURUSD",
                "side": "buy",
                "size": 25_000,
                "price": 1.2345,
            },
        )
    )

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    assert policy_router.exploration_freeze_active() is True

    runner._trading_manager = _FakeTradingManager()
    runner._exploration_release_threshold = 1

    recovery = await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    assert policy_router.exploration_freeze_active() is False
    state = policy_router.exploration_freeze_state()
    assert state.get("release_reason") == "stability_recovered"
    assert state.get("release_metadata", {}).get("safe_iterations") == 1


@pytest.mark.asyncio()
async def test_alpha_trade_runner_applies_drift_size_mitigation(monkeypatch, tmp_path) -> None:
    trading_manager = _FakeTradingManager()
    runner, _ = _build_runner(monkeypatch, tmp_path, trading_manager)

    drift_gate = runner._orchestrator._drift_gate
    config = DriftSentryConfig(
        baseline_window=10,
        evaluation_window=5,
        min_observations=5,
        page_hinkley_delta=0.001,
        page_hinkley_warn=0.25,
        page_hinkley_alert=0.6,
        cusum_warn=0.35,
        cusum_alert=0.9,
        variance_ratio_warn=5.0,
        variance_ratio_alert=12.0,
    )
    baseline = [0.12, 0.118, 0.121, 0.123, 0.119, 0.122, 0.117, 0.124, 0.12, 0.119]
    evaluation = [0.19, 0.198, 0.205, 0.21, 0.202]
    snapshot = evaluate_drift_sentry(
        {"belief_confidence": baseline + evaluation},
        config=config,
        generated_at=datetime(2025, 1, 5, tzinfo=UTC),
    )
    assert snapshot.status is DriftSeverity.warn
    drift_gate.update_snapshot(snapshot)

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    result = await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    trade_metadata = result.trade_metadata
    mitigation = trade_metadata.get("drift_mitigation")
    assert isinstance(mitigation, dict)
    assert pytest.approx(mitigation["size_multiplier"], rel=1e-9) == 0.5
    assert pytest.approx(mitigation["original_quantity"], rel=1e-9) == 25_000.0
    assert pytest.approx(mitigation["adjusted_quantity"], rel=1e-9) == 12_500.0

    assert pytest.approx(trade_metadata["quantity"], rel=1e-9) == 12_500.0
    expected_notional = 12_500.0 * 1.2345
    assert pytest.approx(trade_metadata["notional"], rel=1e-9) == expected_notional

    assert trading_manager.intents, "expected trade intent to be submitted"
    recorded_intent = trading_manager.intents[-1]
    assert pytest.approx(recorded_intent["quantity"], rel=1e-9) == 12_500.0

    theory_packet = trade_metadata.get("theory_packet")
    assert isinstance(theory_packet, dict)
    assert theory_packet.get("severity") == "warn"
    actions = theory_packet.get("actions")
    assert isinstance(actions, list)
    action_labels = {entry.get("action") for entry in actions if isinstance(entry, Mapping)}
    assert {"freeze_exploration", "size_multiplier"}.issubset(action_labels)

    diary_metadata = result.loop_result.diary_entry.metadata
    assert diary_metadata.get("theory_packet") == theory_packet
    assert diary_metadata.get("drift_mitigation") == mitigation

    loop_metadata = result.loop_result.metadata
    assert loop_metadata.get("theory_packet") == theory_packet
    assert loop_metadata.get("drift_mitigation") == mitigation

    policy_router = runner._understanding_router.policy_router
    assert policy_router.exploration_freeze_active() is True


@pytest.mark.asyncio()
async def test_alpha_trade_runner_freezes_on_compliance_risk_warning(monkeypatch, tmp_path) -> None:
    trading_manager = _FakeTradingManager()
    runner, _ = _build_runner(monkeypatch, tmp_path, trading_manager)

    orchestrator = runner._orchestrator
    original_run = orchestrator.run_iteration

    def patched_run_iteration(*args, **kwargs):
        result = original_run(*args, **kwargs)
        event = ComplianceEvent(
            event_type=ComplianceEventType.risk_warning,
            severity=ComplianceSeverity.warn,
            summary="Compliance risk warning triggered",
            occurred_at=datetime(2025, 2, 1, tzinfo=UTC),
            policy_id=result.policy_id,
            metadata={"source": "risk_monitor"},
        )
        return replace(result, compliance_events=result.compliance_events + (event,))

    monkeypatch.setattr(orchestrator, "run_iteration", patched_run_iteration)

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    policy_router = runner._understanding_router.policy_router
    assert policy_router.exploration_freeze_active() is False

    await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    assert policy_router.exploration_freeze_active() is True
    freeze_state = policy_router.exploration_freeze_state()
    assert freeze_state.get("reason") == "Compliance risk warning triggered"
    trigger_list = (
        freeze_state.get("metadata", {}).get("triggers")
        if isinstance(freeze_state.get("metadata"), Mapping)
        else None
    )
    assert isinstance(trigger_list, list) and trigger_list
    compliance_trigger = next(
        trigger
        for trigger in trigger_list
        if isinstance(trigger, Mapping)
        and trigger.get("triggered_by") == "compliance.risk_warning"
    )
    compliance_event = compliance_trigger.get("metadata", {}).get("compliance_event")
    assert isinstance(compliance_event, Mapping)
    assert compliance_event.get("event_type") == ComplianceEventType.risk_warning.value


@pytest.mark.asyncio()
async def test_alpha_trade_runner_records_alpha_decay_throttle(monkeypatch, tmp_path) -> None:
    multiplier = 0.7
    trading_manager = _AlphaDecayTradingManager(multiplier)
    runner, _ = _build_runner(monkeypatch, tmp_path, trading_manager)

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 2, 2, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.15, "confidence": 0.78},
            "momentum": {"signal": 0.42, "confidence": 0.84},
        },
        "integrated_signal": {"strength": 0.21, "confidence": 0.88},
        "price": 1.2345,
        "quantity": 25_000,
    }

    result = await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    trade_metadata = result.trade_metadata
    drift_throttle = trade_metadata.get("drift_throttle")
    assert isinstance(drift_throttle, dict)
    assert drift_throttle.get("reason") == "alpha_decay"
    assert drift_throttle.get("status") == "executed"
    assert drift_throttle.get("multiplier") == pytest.approx(multiplier)

    quantity_before = float(sensory_snapshot["quantity"])
    expected_quantity = quantity_before * multiplier
    assert drift_throttle.get("quantity_after") == pytest.approx(expected_quantity)
    assert trade_metadata.get("quantity") == pytest.approx(expected_quantity)

    recorded_price = trade_metadata.get("price") or sensory_snapshot["price"]
    expected_notional = expected_quantity * float(recorded_price)
    assert trade_metadata.get("notional") == pytest.approx(expected_notional)

    theory_packet = trade_metadata.get("theory_packet")
    assert isinstance(theory_packet, dict)
    assert theory_packet.get("alpha_decay_multiplier") == pytest.approx(multiplier)
    actions = theory_packet.get("actions") or []
    alpha_actions = [entry for entry in actions if entry.get("action") == "alpha_decay"]
    assert alpha_actions, "expected alpha_decay action recorded"
    assert alpha_actions[0].get("value") == pytest.approx(multiplier)

    loop_metadata = result.loop_result.metadata
    assert loop_metadata.get("drift_throttle") == drift_throttle
    assert loop_metadata.get("theory_packet") == theory_packet

    diary_metadata = result.loop_result.diary_entry.metadata
    assert diary_metadata.get("drift_throttle") == drift_throttle
    assert diary_metadata.get("theory_packet") == theory_packet


@pytest.mark.asyncio()
async def test_alpha_trade_runner_warns_when_diary_coverage_slips(
    monkeypatch, tmp_path, caplog
) -> None:
    trading_manager = _FakeTradingManager()
    base_time = datetime.now(tz=UTC)
    clock_values = iter(
        [
            base_time,
            base_time + timedelta(minutes=1),
            base_time + timedelta(minutes=2),
        ]
    )

    def fake_clock() -> datetime:
        try:
            return next(clock_values)
        except StopIteration:
            return base_time + timedelta(minutes=2)

    runner, _ = _build_runner(
        monkeypatch,
        tmp_path,
        trading_manager,
        runner_kwargs={
            "diary_minimum_samples": 2,
            "diary_coverage_target": 0.95,
            "clock": fake_clock,
        },
    )

    monkeypatch.setattr(runner._orchestrator._diary_store, "_now", fake_clock)

    original_run_iteration = runner._orchestrator.run_iteration
    invocation_count = {"value": 0}

    def stub_run_iteration(*args, **kwargs):
        invocation_count["value"] += 1
        if invocation_count["value"] == 2:
            raise RuntimeError("diary failure")
        return original_run_iteration(*args, **kwargs)

    monkeypatch.setattr(runner._orchestrator, "run_iteration", stub_run_iteration)

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    logger_name = "src.orchestration.alpha_trade_runner"
    caplog.set_level(logging.WARNING, logger=logger_name)

    await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    with pytest.raises(RuntimeError):
        await runner.process(
            sensory_snapshot,
            policy_id="alpha.live",
            trade_overrides={"policy_id": "alpha.live"},
        )

    stats = runner.describe_diary_coverage()
    assert stats["iterations"] == 2
    assert stats["recorded"] == 1
    assert stats["missing"] == 1
    assert stats["coverage"] == pytest.approx(0.5)

    warning_records = [
        record for record in caplog.records if "coverage" in record.message.lower()
    ]
    assert warning_records, "expected coverage warning to be emitted"


@pytest.mark.asyncio()
async def test_alpha_trade_runner_handles_missing_diary_entry(
    monkeypatch, tmp_path
) -> None:
    trading_manager = _FakeTradingManager()
    runner, diary_store = _build_runner(monkeypatch, tmp_path, trading_manager)

    orchestrator = runner._orchestrator
    original_run_iteration = orchestrator.run_iteration

    def stub_run_iteration(*args, **kwargs):
        result = original_run_iteration(*args, **kwargs)
        return replace(
            result,
            diary_entry=None,
            metadata=MappingProxyType(dict(result.metadata)),
        )

    monkeypatch.setattr(orchestrator, "run_iteration", stub_run_iteration)

    def fail_on_annotate(*_args, **_kwargs):
        pytest.fail("annotate_diary_entry should not be called when diary entry is missing")

    monkeypatch.setattr(orchestrator, "annotate_diary_entry", fail_on_annotate)

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    result = await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    assert result.loop_result.diary_entry is None
    attribution = result.trade_metadata.get("attribution")
    assert isinstance(attribution, Mapping)
    belief_snapshot = attribution.get("belief")
    assert isinstance(belief_snapshot, Mapping)
    assert belief_snapshot.get("belief_id") == result.belief_state.belief_id
    assert attribution.get("policy_id") == "alpha.live"
    assert "diary_entry_id" not in attribution
    probes = attribution.get("probes")
    assert isinstance(probes, list) and probes
    assert attribution.get("brief_explanation")
    assert attribution.get("explanation") == attribution.get("brief_explanation")
    assert result.trade_metadata.get("diary_coverage", {}).get("missing") == 1

    stats = runner.describe_diary_coverage()
    assert stats["iterations"] == 1
    assert stats["missing"] == 1
    assert stats["recorded"] == 0
    assert stats["coverage"] == 0.0

    assert trading_manager.intents, "expected trade intent to be submitted"
    intent_metadata = trading_manager.intents[0].get("metadata")
    if isinstance(intent_metadata, Mapping):
        intent_attribution = intent_metadata.get("attribution")
        assert intent_attribution == attribution
        coverage_snapshot = intent_metadata.get("diary_coverage")
        assert isinstance(coverage_snapshot, Mapping)
        assert coverage_snapshot.get("missing") == 1

    outcome = result.trade_outcome
    assert outcome is not None
    coverage_snapshot = outcome.metadata.get("diary_coverage")
    assert isinstance(coverage_snapshot, Mapping)
    assert coverage_snapshot.get("missing") == 1
    assert outcome.metadata.get("attribution") == attribution

    loop_metadata = result.loop_result.metadata
    assert loop_metadata.get("attribution") == attribution
    loop_coverage = loop_metadata.get("diary_coverage")
    assert isinstance(loop_coverage, Mapping)
    assert loop_coverage.get("missing") == 1

    assert diary_store.entries(), "expected diary store to keep previously recorded entries"


@pytest.mark.asyncio()
async def test_alpha_trade_runner_flags_diary_gap(monkeypatch, tmp_path, caplog) -> None:
    trading_manager = _FakeTradingManager()
    base_time = datetime.now(tz=UTC)
    clock_values = iter(
        [
            base_time,
            base_time + timedelta(minutes=3),
            base_time + timedelta(minutes=4),
            base_time + timedelta(minutes=5),
        ]
    )

    def fake_clock() -> datetime:
        try:
            return next(clock_values)
        except StopIteration:
            return base_time + timedelta(minutes=5)

    runner, _ = _build_runner(
        monkeypatch,
        tmp_path,
        trading_manager,
        runner_kwargs={
            "diary_gap_alert": timedelta(seconds=60),
            "diary_coverage_target": 0.0,
            "clock": fake_clock,
        },
    )

    monkeypatch.setattr(runner._orchestrator._diary_store, "_now", fake_clock)

    original_run_iteration = runner._orchestrator.run_iteration
    invocation_count = {"value": 0}

    def stub_run_iteration(*args, **kwargs):
        invocation_count["value"] += 1
        if invocation_count["value"] == 2:
            raise RuntimeError("diary failure")
        return original_run_iteration(*args, **kwargs)

    monkeypatch.setattr(runner._orchestrator, "run_iteration", stub_run_iteration)

    sensory_snapshot = {
        "symbol": "EURUSD",
        "generated_at": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "lineage": {"source": "unit-test"},
        "dimensions": {
            "liquidity": {"signal": -0.2, "confidence": 0.7},
            "momentum": {"signal": 0.55, "confidence": 0.65},
        },
        "integrated_signal": {"strength": 0.18, "confidence": 0.82},
        "price": 1.2345,
        "quantity": 25_000,
    }

    logger_name = "src.orchestration.alpha_trade_runner"

    caplog.set_level(logging.WARNING, logger=logger_name)

    await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    caplog.clear()

    with pytest.raises(RuntimeError):
        await runner.process(
            sensory_snapshot,
            policy_id="alpha.live",
            trade_overrides={"policy_id": "alpha.live"},
        )

    stale_records = [
        record for record in caplog.records if "stale" in record.message.lower()
    ]
    assert stale_records, "expected stale diary warning"

    stats = runner.describe_diary_coverage()
    assert stats.get("gap_breach") is True

    caplog.clear()
    caplog.set_level(logging.INFO, logger=logger_name)

    await runner.process(
        sensory_snapshot,
        policy_id="alpha.live",
        trade_overrides={"policy_id": "alpha.live"},
    )

    stats_after = runner.describe_diary_coverage()
    assert stats_after.get("gap_breach") is False

    refresh_records = [
        record for record in caplog.records if "refreshed" in record.message.lower()
    ]
    assert refresh_records, "expected gap recovery info log"
