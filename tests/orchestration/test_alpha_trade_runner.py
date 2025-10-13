from __future__ import annotations

from datetime import datetime, timezone

from typing import Mapping

import pytest

from src.operations.drift_sentry import DriftSentryConfig, evaluate_drift_sentry
from src.operations.sensory_drift import DriftSeverity
from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage, PolicyLedgerStore
from src.orchestration.alpha_trade_loop import AlphaTradeLoopOrchestrator
from src.orchestration.alpha_trade_runner import AlphaTradeLoopRunner
from src.trading.gating import DriftSentryGate
from src.trading.trading_manager import TradeIntentOutcome
from src.understanding.belief import BeliefBuffer, BeliefEmitter, RegimeFSM
from src.understanding.decision_diary import DecisionDiaryStore
from src.understanding.router import UnderstandingRouter
from src.thinking.adaptation.feature_toggles import AdaptationFeatureToggles
from src.thinking.adaptation.policy_router import (
    FastWeightExperiment,
    PolicyRouter,
    PolicyTactic,
)


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


def _build_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    trading_manager,
    feature_toggles: AdaptationFeatureToggles | None = None,
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

    runner = AlphaTradeLoopRunner(
        belief_emitter=belief_emitter,
        regime_fsm=regime_fsm,
        orchestrator=orchestrator,
        trading_manager=trading_manager,
        understanding_router=understanding_router,
        feature_toggles=feature_toggles,
    )

    return runner, diary_store


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
    assert attribution.get("explanation")
    assert isinstance(attribution.get("probes"), list)
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
    assert guardrails.get("risk_guardrail") == existing_guardrails["risk_guardrail"]

    assert intent_metadata.get("guardrails") == guardrails
    assert result.trade_outcome.metadata.get("guardrails") == guardrails


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
