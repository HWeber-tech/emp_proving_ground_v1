from __future__ import annotations

"""Diagnostics helpers for the understanding loop sprint roadmap tasks."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import datetime as _datetime
import enum as _enum
try:  # Python 3.10 compatibility
    from enum import StrEnum
except ImportError:  # pragma: no cover - fallback for older runtimes
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of :class:`enum.StrEnum` for Python 3.10 runtimes."""

        pass
if not hasattr(_enum, "StrEnum"):  # pragma: no cover - expose fallback globally
    _enum.StrEnum = StrEnum  # type: ignore[attr-defined]
from typing import Callable, Mapping, MutableMapping, Sequence

try:  # Python 3.10 compatibility
    from datetime import UTC
except ImportError:  # pragma: no cover - fallback path for older runtimes
    UTC = timezone.utc
if not hasattr(_datetime, "UTC"):  # pragma: no cover - ensure downstream imports succeed
    _datetime.UTC = timezone.utc  # type: ignore[attr-defined]

try:
    from src.operations.observability_diary import (
        DecisionNarrationCapsule,
        PolicyLedgerDiff,
        build_decision_narration_capsule,
    )
except Exception:  # pragma: no cover - lightweight fallback for reduced test environments
    @dataclass(slots=True, frozen=True)
    class PolicyLedgerDiff:  # type: ignore[override]
        policy_id: str
        change_type: str
        before: Mapping[str, object] | None = None
        after: Mapping[str, object] | None = None
        approvals: tuple[str, ...] = ()
        notes: tuple[str, ...] = ()
        metadata: Mapping[str, object] = field(default_factory=dict)

        def as_dict(self) -> Mapping[str, object]:
            payload: MutableMapping[str, object] = {
                "policy_id": self.policy_id,
                "change_type": self.change_type,
                "approvals": list(self.approvals),
                "notes": list(self.notes),
                "metadata": dict(self.metadata),
            }
            if self.before is not None:
                payload["before"] = dict(self.before)
            if self.after is not None:
                payload["after"] = dict(self.after)
            return payload

    @dataclass(slots=True, frozen=True)
    class SigmaStabilitySnapshot:  # type: ignore[override]
        symbol: str
        sigma_before: float | None
        sigma_after: float | None
        sigma_target: float | None
        stability_index: float | None
        delta: float | None
        metadata: Mapping[str, object]

        def as_dict(self) -> Mapping[str, object]:
            return {
                "symbol": self.symbol,
                "sigma_before": self.sigma_before,
                "sigma_after": self.sigma_after,
                "sigma_target": self.sigma_target,
                "stability_index": self.stability_index,
                "delta": self.delta,
                "metadata": dict(self.metadata),
            }

    @dataclass(slots=True, frozen=True)
    class ThrottleStateSnapshot:  # type: ignore[override]
        name: str
        state: str
        active: bool
        multiplier: float | None
        reason: str | None
        metadata: Mapping[str, object] = field(default_factory=dict)

        def as_dict(self) -> Mapping[str, object]:
            return {
                "name": self.name,
                "state": self.state,
                "active": self.active,
                "multiplier": self.multiplier,
                "reason": self.reason,
                "metadata": dict(self.metadata),
            }

    @dataclass(slots=True, frozen=True)
    class DecisionNarrationCapsule:  # type: ignore[override]
        capsule_id: str
        generated_at: datetime
        window_start: datetime | None
        window_end: datetime | None
        policy_diffs: tuple[PolicyLedgerDiff, ...]
        sigma_stability: SigmaStabilitySnapshot
        throttle_states: tuple[ThrottleStateSnapshot, ...]
        notes: tuple[str, ...]
        metadata: Mapping[str, object]

        def as_dict(self) -> Mapping[str, object]:
            return {
                "capsule_id": self.capsule_id,
                "generated_at": self.generated_at.isoformat(),
                "window_start": self.window_start.isoformat() if self.window_start else None,
                "window_end": self.window_end.isoformat() if self.window_end else None,
                "policy_diffs": [diff.as_dict() for diff in self.policy_diffs],
                "sigma_stability": self.sigma_stability.as_dict(),
                "throttle_states": [state.as_dict() for state in self.throttle_states],
                "notes": list(self.notes),
                "metadata": dict(self.metadata),
            }

    def _coerce_datetime(value: datetime | str | None) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    def _normalise_notes(notes: Iterable[object] | None) -> tuple[str, ...]:
        if notes is None:
            return ()
        cleaned = []
        for note in notes:
            text = str(note).strip()
            if text:
                cleaned.append(text)
        return tuple(cleaned)

    def build_decision_narration_capsule(  # type: ignore[override]
        *,
        capsule_id: str,
        window_start: datetime | str | None,
        window_end: datetime | str | None,
        policy_diffs: Iterable[Mapping[str, object] | PolicyLedgerDiff],
        sigma_metrics: Mapping[str, object],
        throttle_states: Iterable[Mapping[str, object] | ThrottleStateSnapshot],
        notes: Iterable[object] | None = None,
        metadata: Mapping[str, object] | None = None,
        generated_at: datetime | None = None,
    ) -> DecisionNarrationCapsule:
        generated = generated_at or datetime.now(tz=UTC)
        window_start_dt = _coerce_datetime(window_start)
        window_end_dt = _coerce_datetime(window_end)

        diffs = []
        for diff in policy_diffs:
            if isinstance(diff, PolicyLedgerDiff):
                diffs.append(diff)
            else:
                diffs.append(
                    PolicyLedgerDiff(
                        policy_id=str(diff.get("policy_id", "unknown")),
                        change_type=str(diff.get("change_type", "updated")),
                        before=diff.get("before"),
                        after=diff.get("after"),
                        approvals=tuple(str(item) for item in diff.get("approvals", ()) if str(item)),
                        notes=_normalise_notes(diff.get("notes")),
                        metadata=dict(diff.get("metadata", {})),
                    )
                )

        sigma_snapshot = SigmaStabilitySnapshot(
            symbol=str(sigma_metrics.get("symbol", "UNKNOWN")),
            sigma_before=float(sigma_metrics.get("sigma_before", 0.0)) if sigma_metrics.get("sigma_before") is not None else None,
            sigma_after=float(sigma_metrics.get("sigma_after", 0.0)) if sigma_metrics.get("sigma_after") is not None else None,
            sigma_target=float(sigma_metrics.get("sigma_target", 0.0)) if sigma_metrics.get("sigma_target") is not None else None,
            stability_index=float(sigma_metrics.get("stability_index", 0.0)) if sigma_metrics.get("stability_index") is not None else None,
            delta=float(sigma_metrics.get("delta", 0.0)) if sigma_metrics.get("delta") is not None else None,
            metadata=dict(sigma_metrics),
        )

        throttle_snapshots = []
        for state in throttle_states:
            if isinstance(state, ThrottleStateSnapshot):
                throttle_snapshots.append(state)
            else:
                throttle_snapshots.append(
                    ThrottleStateSnapshot(
                        name=str(state.get("name", "throttle")),
                        state=str(state.get("state", "observing")),
                        active=bool(state.get("active", False)),
                        multiplier=float(state.get("multiplier")) if state.get("multiplier") is not None else None,
                        reason=str(state.get("reason")) if state.get("reason") is not None else None,
                        metadata=dict(state.get("metadata", {})),
                    )
                )

        return DecisionNarrationCapsule(
            capsule_id=str(capsule_id),
            generated_at=generated,
            window_start=window_start_dt,
            window_end=window_end_dt,
            policy_diffs=tuple(diffs),
            sigma_stability=sigma_snapshot,
            throttle_states=tuple(throttle_snapshots),
            notes=_normalise_notes(notes),
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True, frozen=True)
class SensorDriftBaseline:
    sensor: str
    mean: float
    std: float
    count: int


@dataclass(slots=True, frozen=True)
class SensorDriftParameters:
    baseline_window: int
    evaluation_window: int
    min_observations: int
    z_threshold: float


@dataclass(slots=True, frozen=True)
class SensorDriftResult:
    sensor: str
    baseline: SensorDriftBaseline
    evaluation_mean: float
    evaluation_std: float
    evaluation_count: int
    z_score: float | None
    drift_ratio: float
    exceeded: bool

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "sensor": self.sensor,
            "baseline_mean": self.baseline.mean,
            "baseline_std": self.baseline.std,
            "baseline_count": self.baseline.count,
            "evaluation_mean": self.evaluation_mean,
            "evaluation_std": self.evaluation_std,
            "evaluation_count": self.evaluation_count,
            "drift_ratio": self.drift_ratio,
            "exceeded": self.exceeded,
        }
        payload["z_score"] = self.z_score
        return payload


@dataclass(slots=True, frozen=True)
class SensorDriftSummary:
    parameters: SensorDriftParameters
    results: tuple[SensorDriftResult, ...]

    @property
    def exceeded(self) -> tuple[SensorDriftResult, ...]:
        return tuple(result for result in self.results if result.exceeded)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "parameters": {
                "baseline_window": self.parameters.baseline_window,
                "evaluation_window": self.parameters.evaluation_window,
                "min_observations": self.parameters.min_observations,
                "z_threshold": self.parameters.z_threshold,
            },
            "results": [result.as_dict() for result in self.results],
        }

from src.sensory.lineage import build_lineage_record
from src.thinking.adaptation.policy_router import (
    PolicyDecision,
    PolicyRouter,
    PolicyTactic,
    RegimeState,
)
from src.understanding.router import (
    BeliefSnapshot,
    FastWeightAdapter,
    FeatureGate,
    UnderstandingRouter,
)


class UnderstandingNodeKind(StrEnum):
    """Node categories that make up the understanding loop graph."""

    sensory = "sensory"
    belief = "belief"
    router = "router"
    policy = "policy"


class UnderstandingGraphStatus(StrEnum):
    """Status levels surfaced by understanding loop diagnostics."""

    ok = "ok"
    warn = "warn"
    fail = "fail"


@dataclass(slots=True, frozen=True)
class UnderstandingNode:
    """Single node within the understanding loop graph."""

    node_id: str
    name: str
    kind: UnderstandingNodeKind
    status: UnderstandingGraphStatus
    metadata: Mapping[str, object] = field(default_factory=dict)
    lineage: Mapping[str, object] | None = None

    def as_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "id": self.node_id,
            "name": self.name,
            "kind": self.kind.value,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }
        if self.lineage is not None:
            payload["lineage"] = dict(self.lineage)
        return payload


@dataclass(slots=True, frozen=True)
class UnderstandingEdge:
    """Directed edge describing information flow between graph nodes."""

    source: str
    target: str
    relationship: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class UnderstandingGraphDiagnostics:
    """Graph structure plus metadata describing the understanding loop."""

    status: UnderstandingGraphStatus
    nodes: Sequence[UnderstandingNode]
    edges: Sequence[UnderstandingEdge]
    generated_at: datetime
    metadata: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "metadata": dict(self.metadata),
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
        }

    def to_dot(self) -> str:
        """Render the graph into GraphViz ``dot`` format."""

        lines = ["digraph UnderstandingLoop {"]
        lines.append("  rankdir=LR;")
        for node in self.nodes:
            label = f"{node.name}\\nstatus={node.status.value}"
            lines.append(f'  "{node.node_id}" [label="{label}" shape=box];')
        for edge in self.edges:
            meta = edge.relationship
            lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{meta}"];')
        lines.append("}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render a Markdown table summarising the graph."""

        lines = ["| Node | Kind | Status | Headline |", "| --- | --- | --- | --- |"]
        for node in self.nodes:
            headline = node.metadata.get("headline") or node.metadata.get("summary") or "—"
            lines.append(
                f"| {node.name} | {node.kind.value} | {node.status.value.upper()} | {headline} |"
            )
        return "\n".join(lines)


@dataclass(slots=True, frozen=True)
class UnderstandingLoopSnapshot:
    """Aggregate snapshot consumed by the observability dashboard tile."""

    status: UnderstandingGraphStatus
    generated_at: datetime
    regime_state: RegimeState
    graph: UnderstandingGraphDiagnostics
    decision: PolicyDecision
    ledger_diff: PolicyLedgerDiff
    capsule: DecisionNarrationCapsule
    drift_summary: SensorDriftSummary

    def as_dict(self) -> Mapping[str, object]:
        metadata = {
            "regime": self.regime_state.regime,
            "regime_confidence": self.regime_state.confidence,
            "decision_id": self.decision.tactic_id,
            "capsule_id": self.capsule.capsule_id,
            "experiments": list(self.decision.experiments_applied),
        }
        return {
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "metadata": metadata,
            "graph": self.graph.as_dict(),
            "decision": {
                "tactic_id": self.decision.tactic_id,
                "parameters": dict(self.decision.parameters),
                "guardrails": dict(self.decision.guardrails),
                "selected_weight": self.decision.selected_weight,
                "experiments_applied": list(self.decision.experiments_applied),
                "reflection_summary": dict(self.decision.reflection_summary),
            },
            "ledger_diff": self.ledger_diff.as_dict(),
            "capsule": self.capsule.as_dict(),
            "drift_summary": self.drift_summary.as_dict(),
        }


@dataclass(slots=True, frozen=True)
class UnderstandingArtifacts:
    """Bundle of artefacts generated during an understanding loop cycle."""

    graph: UnderstandingGraphDiagnostics
    regime_state: RegimeState
    decision: PolicyDecision
    ledger_diff: PolicyLedgerDiff
    capsule: DecisionNarrationCapsule
    drift_summary: SensorDriftSummary

    def to_snapshot(self) -> UnderstandingLoopSnapshot:
        return UnderstandingLoopSnapshot(
            status=self.graph.status,
            generated_at=self.graph.generated_at,
            regime_state=self.regime_state,
            graph=self.graph,
            decision=self.decision,
            ledger_diff=self.ledger_diff,
            capsule=self.capsule,
            drift_summary=self.drift_summary,
        )


class UnderstandingDiagnosticsBuilder:
    """Construct synthetic-yet-deterministic diagnostics for the understanding loop."""

    def __init__(self, *, now: Callable[[], datetime] | None = None) -> None:
        self._now = now or (lambda: datetime.now(tz=UTC))

    def build(self) -> UnderstandingArtifacts:
        generated_at = self._now()

        sensory_dimensions = {
            "WHY": {"strength": 0.58, "confidence": 0.72},
            "WHAT": {"strength": 0.61, "confidence": 0.69},
            "WHEN": {"strength": 0.55, "confidence": 0.64},
            "HOW": {"strength": 0.59, "confidence": 0.66},
            "ANOMALY": {"strength": 0.47, "confidence": 0.71},
        }
        integrated_strength = 0.602
        integrated_confidence = 0.692
        sensory_lineage = build_lineage_record(
            "UNDERSTANDING_SENSORY",
            "sensory.understanding.synthetic",
            inputs={
                name: {"strength": payload["strength"], "confidence": payload["confidence"]}
                for name, payload in sensory_dimensions.items()
            },
            outputs={
                "strength": integrated_strength,
                "confidence": integrated_confidence,
            },
            telemetry={
                "dimensions": {
                    name: payload["strength"] for name, payload in sensory_dimensions.items()
                }
            },
            metadata={"symbol": "ALPHATRADE"},
        ).as_dict()

        regime_state = RegimeState(
            regime="balanced",
            confidence=0.78,
            features={
                "volatility_z": 0.43,
                "liquidity_z": -0.18,
                "sentiment_z": 0.22,
            },
            timestamp=generated_at,
        )

        belief_lineage = build_lineage_record(
            "UNDERSTANDING_BELIEF",
            "thinking.belief.synthetic",
            inputs={
                "integrated_strength": integrated_strength,
                "integrated_confidence": integrated_confidence,
            },
            outputs={
                "regime": regime_state.regime,
                "confidence": regime_state.confidence,
            },
            telemetry=regime_state.features,
            metadata={"window": "15m"},
        ).as_dict()

        policy_router = PolicyRouter(default_guardrails={"requires_diary": True, "max_latency_ms": 250})
        understanding_router = UnderstandingRouter(policy_router)
        understanding_router.register_tactic(
            PolicyTactic(
                tactic_id="momentum_breakout",
                base_weight=1.15,
                parameters={"style": "breakout", "timeframe": "15m"},
                guardrails={"risk_cap": "tier-balanced"},
                regime_bias={"balanced": 1.12, "bull": 1.18},
                confidence_sensitivity=0.6,
                description="Fast-follow breakout with liquidity throttle",
            )
        )
        understanding_router.register_tactic(
            PolicyTactic(
                tactic_id="mean_reversion",
                base_weight=0.92,
                parameters={"style": "reversion", "window": 5},
                guardrails={"requires_diary": True},
                regime_bias={"balanced": 0.95, "bear": 1.08},
                confidence_sensitivity=0.4,
                description="Liquidity-aware micro reversion",
            )
        )

        understanding_router.register_adapter(
            FastWeightAdapter(
                adapter_id="low_liquidity_boost",
                tactic_id="mean_reversion",
                multiplier=1.05,
                rationale="Boost mean reversion when liquidity is compressed and experiments are live",
                feature_gates=(FeatureGate(feature="liquidity_z", maximum=0.0),),
                required_flags={"experiments_live": True},
            )
        )
        understanding_router.register_adapter(
            FastWeightAdapter(
                adapter_id="momentum_sentiment_gate",
                tactic_id="momentum_breakout",
                multiplier=1.0,
                rationale="Documented gate for sentiment-positive momentum runs",
                feature_gates=(FeatureGate(feature="sentiment_z", minimum=0.2),),
                required_flags={"experiments_live": True},
            )
        )

        belief_snapshot = BeliefSnapshot(
            belief_id="understanding-balanced-shadow",
            regime_state=regime_state,
            features=regime_state.features,
            metadata={"window": "15m", "source": "synthetic-shadow"},
            fast_weights_enabled=True,
            feature_flags={"experiments_live": True},
        )

        understanding_decision = understanding_router.route(belief_snapshot)
        decision = understanding_decision.decision

        drift_parameters = SensorDriftParameters(
            baseline_window=4,
            evaluation_window=2,
            min_observations=3,
            z_threshold=3.0,
        )
        drift_baseline = SensorDriftBaseline(
            sensor="integrated_strength",
            mean=0.5925,
            std=0.015,
            count=4,
        )
        drift_result = SensorDriftResult(
            sensor="integrated_strength",
            baseline=drift_baseline,
            evaluation_mean=0.605,
            evaluation_std=0.0028,
            evaluation_count=2,
            z_score=None,
            drift_ratio=0.0211,
            exceeded=False,
        )
        drift_summary = SensorDriftSummary(
            parameters=drift_parameters,
            results=(drift_result,),
        )

        drift_exceeded = False

        ledger_diff = PolicyLedgerDiff(
            policy_id=decision.tactic_id,
            change_type="updated",
            before=None,
            after={
                "selected_weight": round(decision.selected_weight, 6),
                "parameters": dict(decision.parameters),
            },
            approvals=("risk-ops", "adaptation-lead"),
            notes=("correlated with balanced regime",),
            metadata={
                "experiments": list(decision.experiments_applied),
                "adapters": list(understanding_decision.applied_adapters),
                "decision_timestamp": decision.reflection_summary.get("timestamp"),
            },
        )

        capsule = build_decision_narration_capsule(
            capsule_id=f"understanding-{decision.tactic_id}",
            window_start=regime_state.timestamp - timedelta(minutes=15),
            window_end=regime_state.timestamp,
            policy_diffs=[ledger_diff],
            sigma_metrics={
                "symbol": "ALPHATRADE",
                "sigma_before": 1.24,
                "sigma_after": 1.18,
                "sigma_target": 1.10,
                "stability_index": 0.94,
                "notes": "sigma easing after throttle",
            },
            throttle_states=[
                {
                    "name": "drift_sentry",
                    "state": "observing",
                    "active": False,
                    "multiplier": 1.0,
                    "reason": "sigma within target band",
                }
            ],
            notes=[
                "Synced with decision diary",
                f"Experiments: {', '.join(decision.experiments_applied) or 'none'}",
            ],
            metadata={
                "decision_id": decision.tactic_id,
                "regime": regime_state.regime,
                "regime_confidence": regime_state.confidence,
            },
            generated_at=generated_at,
        )

        graph_status = UnderstandingGraphStatus.fail if drift_exceeded else UnderstandingGraphStatus.ok

        sensory_node = UnderstandingNode(
            node_id="sensory",
            name="Sensory cortex",
            kind=UnderstandingNodeKind.sensory,
            status=UnderstandingGraphStatus.warn if drift_exceeded else UnderstandingGraphStatus.ok,
            metadata={
                "symbol": "ALPHATRADE",
                "integrated_strength": integrated_strength,
                "integrated_confidence": integrated_confidence,
                "dimensions": sensory_dimensions,
                "headline": (
                    "strength={:.3f} confidence={:.3f}".format(
                        integrated_strength, integrated_confidence
                    )
                ),
                "drift_exceeded": drift_exceeded,
            },
            lineage=sensory_lineage,
        )

        belief_node = UnderstandingNode(
            node_id="belief",
            name="Belief state",
            kind=UnderstandingNodeKind.belief,
            status=UnderstandingGraphStatus.ok,
            metadata={
                "regime": regime_state.regime,
                "confidence": regime_state.confidence,
                "features": regime_state.features,
                "headline": f"regime={regime_state.regime} conf={regime_state.confidence:.2f}",
            },
            lineage=belief_lineage,
        )

        router_node = UnderstandingNode(
            node_id="router",
            name="Policy router",
            kind=UnderstandingNodeKind.router,
            status=UnderstandingGraphStatus.ok,
            metadata={
                "selected_tactic": decision.tactic_id,
                "headline": f"winner={decision.tactic_id} weight={decision.selected_weight:.3f}",
                "top_candidates": decision.reflection_summary.get("top_candidates", ()),
                "experiments": list(decision.experiments_applied),
                "adapters": list(understanding_decision.applied_adapters),
                "fast_weight_summary": understanding_decision.fast_weight_summary,
            },
            lineage=None,
        )

        policy_node = UnderstandingNode(
            node_id="policy",
            name="Policy decision",
            kind=UnderstandingNodeKind.policy,
            status=UnderstandingGraphStatus.ok,
            metadata={
                "headline": "ledger synced with capsule",
                "guardrails": decision.guardrails,
                "ledger_diff": ledger_diff.as_dict(),
                "capsule_id": capsule.capsule_id,
            },
        )

        edges = (
            UnderstandingEdge(
                source="sensory",
                target="belief",
                relationship="feeds",
                metadata={"symbol": "ALPHATRADE"},
            ),
            UnderstandingEdge(
                source="belief",
                target="router",
                relationship="conditions",
                metadata={"regime": regime_state.regime},
            ),
            UnderstandingEdge(
                source="router",
                target="policy",
                relationship="selects",
                metadata={"selected_weight": decision.selected_weight},
            ),
        )

        graph_metadata = {
            "regime": regime_state.regime,
            "regime_confidence": regime_state.confidence,
            "decision_id": decision.tactic_id,
            "capsule_id": capsule.capsule_id,
            "drift_exceeded": drift_exceeded,
            "experiments": list(decision.experiments_applied),
            "adapters": list(understanding_decision.applied_adapters),
        }

        graph = UnderstandingGraphDiagnostics(
            status=graph_status,
            nodes=(sensory_node, belief_node, router_node, policy_node),
            edges=edges,
            generated_at=generated_at,
            metadata=graph_metadata,
        )

        return UnderstandingArtifacts(
            graph=graph,
            regime_state=regime_state,
            decision=decision,
            ledger_diff=ledger_diff,
            capsule=capsule,
            drift_summary=drift_summary,
        )
