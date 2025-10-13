from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage, PolicyLedgerStore
from src.orchestration.alpha_trade_loop import AlphaTradeLoopOrchestrator
from src.orchestration.alpha_trade_runner import AlphaTradeLoopRunner
from src.trading.gating import DriftSentryGate
from src.trading.trading_manager import TradeIntentOutcome
from src.understanding.belief import BeliefBuffer, BeliefEmitter, RegimeFSM
from src.understanding.decision_diary import DecisionDiaryStore
from src.understanding.router import UnderstandingRouter
from src.thinking.adaptation.policy_router import PolicyRouter, PolicyTactic


UTC = timezone.utc


class _FakeTradingManager:
    def __init__(self) -> None:
        self.intents: list[dict[str, object]] = []

    async def on_trade_intent(self, event: dict[str, object]) -> TradeIntentOutcome:
        self.intents.append(dict(event))
        return TradeIntentOutcome(status="executed", executed=True, metadata={})


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


def _build_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    trading_manager,
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
    intent_fast_weight = trading_manager.intents[0]["metadata"].get("fast_weight")
    assert intent_fast_weight == fast_weight_metadata
    assert diary_store.entries(), "expected decision diary entry to be recorded"
    assert result.trade_outcome is not None
    assert result.trade_outcome.status == "executed"
    entry = diary_store.entries()[0]
    trade_execution = entry.metadata.get("trade_execution")
    assert isinstance(trade_execution, dict)
    assert trade_execution["status"] == "executed"


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
