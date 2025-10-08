"""Governance helpers that track progress against the encyclopedia vision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "LayerAssessment",
    "VisionAlignmentReport",
]


def _status_from_coverage(coverage: float) -> str:
    if coverage >= 0.85:
        return "ready"
    if coverage >= 0.45:
        return "progressing"
    return "gap"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass(slots=True)
class LayerAssessment:
    """Snapshot describing how a single layer aligns with the roadmap."""

    layer: str
    coverage: float
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def status(self) -> str:
        return _status_from_coverage(self.coverage)

    def as_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "status": self.status(),
            "coverage": round(self.coverage, 3),
            "strengths": list(self.strengths),
            "gaps": list(self.gaps),
            "evidence": dict(self.evidence),
        }


class VisionAlignmentReport:
    """Evaluate runtime components against the encyclopedia's five layers."""

    def __init__(
        self,
        *,
        fabric: Any | None,
        pipeline: Any | None,
        trading_manager: Any | None,
        control_center: Any | None = None,
        evolution_orchestrator: Any | None = None,
        encyclopedia_version: str = "2.3",
    ) -> None:
        self.fabric = fabric
        self.pipeline = pipeline
        self.trading_manager = trading_manager
        self.control_center = control_center
        self.evolution_orchestrator = evolution_orchestrator
        self.encyclopedia_version = encyclopedia_version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self) -> Mapping[str, Any]:
        layers = [
            self._assess_data_foundation(),
            self._assess_sensory_cortex(),
            self._assess_intelligence_engine(),
            self._assess_evolutionary_layer(),
            self._assess_trading_layer(),
        ]

        readiness = 0.0
        if layers:
            readiness = sum(layer.coverage for layer in layers) / len(layers)

        summary_status = _status_from_coverage(readiness)

        gaps = self._collect_unique(it.gaps for it in layers)
        strengths = self._collect_unique(it.strengths for it in layers)

        summary: dict[str, Any] = {
            "readiness": round(readiness, 3),
            "status": summary_status,
            "layers_ready": sum(1 for layer in layers if layer.status() == "ready"),
            "layers_progressing": sum(
                1 for layer in layers if layer.status() == "progressing"
            ),
        }

        return {
            "encyclopedia_version": self.encyclopedia_version,
            "summary": summary,
            "layers": [layer.as_payload() for layer in layers],
            "strengths": strengths,
            "gaps": gaps,
        }

    def summary(self) -> Mapping[str, Any]:
        report = self.build()
        summary = report.get("summary")
        if isinstance(summary, Mapping):
            return dict(summary)
        empty: dict[str, Any] = {}
        return empty

    # ------------------------------------------------------------------
    # Layer assessors
    # ------------------------------------------------------------------
    def _assess_data_foundation(self) -> LayerAssessment:
        fabric = self.fabric
        connectors: list[str] = []
        diagnostics: Mapping[str, Any] = {}
        cache_ttl_seconds: float | None = None

        if fabric is not None:
            try:
                registered = getattr(fabric, "connectors", {})
                if isinstance(registered, Mapping):
                    connectors = sorted(str(name) for name in registered)
            except Exception:
                connectors = []

            try:
                cache_ttl = getattr(fabric, "cache_ttl", None)
                cache_ttl_seconds = getattr(cache_ttl, "total_seconds", lambda: None)()
            except Exception:
                cache_ttl_seconds = None

            if hasattr(fabric, "get_diagnostics"):
                try:
                    diagnostics = fabric.get_diagnostics()
                except Exception:
                    diagnostics = {}

        coverage = 0.0
        strengths: list[str] = []
        gaps: list[str] = []

        connector_count = len(connectors)
        if connector_count >= 2:
            coverage = 0.92
            strengths.append(f"{connector_count} market data connectors online")
        elif connector_count == 1:
            coverage = 0.62
            strengths.append("single market data connector active")
            gaps.append("Add redundant connectors for failover")
        else:
            coverage = 0.25
            gaps.append("Register market data connectors to activate the fabric")

        if diagnostics:
            successes = diagnostics.get("telemetry", {}).get("success", {})
            if isinstance(successes, Mapping) and any(successes.values()):
                strengths.append("connectors returning live data")
            failures = diagnostics.get("telemetry", {}).get("failures", {})
            if isinstance(failures, Mapping) and sum(_as_float(v) for v in failures.values()) > 0:
                gaps.append("Resolve failing data connectors")

        evidence: dict[str, Any] = {"connectors": connectors}
        if cache_ttl_seconds is not None:
            evidence["cache_ttl_seconds"] = cache_ttl_seconds
        if diagnostics:
            evidence["diagnostics"] = diagnostics

        return LayerAssessment(
            layer="Layer 1 – Data Foundation",
            coverage=coverage,
            strengths=strengths,
            gaps=gaps,
            evidence=evidence,
        )

    def _assess_sensory_cortex(self) -> LayerAssessment:
        pipeline = self.pipeline
        history_samples = 0
        fusion_readings = 0
        recent_symbols: list[str] = []

        if pipeline is not None:
            history = getattr(pipeline, "history", {})
            if isinstance(history, Mapping):
                for symbol, entries in history.items():
                    try:
                        count = len(entries)
                    except Exception:
                        count = 0
                    history_samples += count
                    if count:
                        recent_symbols.append(str(symbol))

            fusion_engine = getattr(pipeline, "fusion_engine", None)
            if fusion_engine is not None:
                readings = getattr(fusion_engine, "current_readings", {})
                if isinstance(readings, Mapping):
                    fusion_readings = len(readings)

        coverage = 0.2
        strengths: list[str] = []
        gaps: list[str] = []

        if history_samples >= 10 and fusion_readings >= 5:
            coverage = 0.9
            strengths.append("4D+1 sensory engines producing fused snapshots")
        elif history_samples > 0:
            coverage = 0.55
            strengths.append("sensory pipeline processing market ticks")
            if fusion_readings < 5:
                gaps.append("Complete dimensional coverage for all sensors")
        else:
            coverage = 0.25
            gaps.append("Stream market ticks through the sensory pipeline")

        evidence = {
            "history_samples": history_samples,
            "recent_symbols": recent_symbols,
            "dimensions_active": fusion_readings,
        }

        return LayerAssessment(
            layer="Layer 2 – 4D+1 Sensory Cortex",
            coverage=coverage,
            strengths=strengths,
            gaps=gaps,
            evidence=evidence,
        )

    def _assess_intelligence_engine(self) -> LayerAssessment:
        fusion_engine = getattr(self.pipeline, "fusion_engine", None)
        diagnostics: Mapping[str, Any] = {}
        adaptive_weights: Mapping[str, Any] = {}
        patterns: Iterable[Any] = ()

        if fusion_engine is not None and hasattr(fusion_engine, "get_diagnostic_information"):
            try:
                diagnostics = fusion_engine.get_diagnostic_information()
            except Exception:
                diagnostics = {}

        if isinstance(diagnostics, Mapping):
            adaptive_weights = diagnostics.get("adaptive_weights", {})
            patterns = diagnostics.get("patterns", ())

        weight_count = len(adaptive_weights) if isinstance(adaptive_weights, Mapping) else 0
        pattern_count = len(list(patterns)) if patterns else 0

        coverage = 0.3
        strengths: list[str] = []
        gaps: list[str] = []

        if weight_count >= 5 and pattern_count > 0:
            coverage = 0.88
            strengths.append("fusion engine adapting weights and spotting patterns")
        elif weight_count >= 3:
            coverage = 0.58
            strengths.append("contextual fusion recalibrating weights")
            gaps.append("Expand diagnostics to surface cross-dimensional patterns")
        else:
            coverage = 0.28
            gaps.append("Enable adaptive fusion diagnostics")

        evidence = {
            "adaptive_weights": dict(adaptive_weights)
            if isinstance(adaptive_weights, Mapping)
            else {},
            "pattern_count": pattern_count,
        }
        if isinstance(diagnostics, Mapping):
            evidence["diagnostics"] = diagnostics

        return LayerAssessment(
            layer="Layer 3 – Intelligence Engine",
            coverage=coverage,
            strengths=strengths,
            gaps=gaps,
            evidence=evidence,
        )

    def _assess_evolutionary_layer(self) -> LayerAssessment:
        orchestrator = self.evolution_orchestrator
        telemetry: Mapping[str, Any] | None = None
        champion_payload: Mapping[str, Any] | None = None
        readiness_payload: Mapping[str, Any] | None = None
        readiness_status: str | None = None

        if orchestrator is not None:
            telemetry_candidate = getattr(orchestrator, "telemetry", None)
            if isinstance(telemetry_candidate, Mapping):
                telemetry = telemetry_candidate

            champion = getattr(orchestrator, "champion", None)
            if champion is not None:
                if hasattr(champion, "as_payload"):
                    try:
                        champion_payload = champion.as_payload()
                    except Exception:
                        champion_payload = {"genome_id": getattr(champion, "genome_id", None)}
                elif isinstance(champion, Mapping):
                    champion_payload = champion

            readiness_builder = getattr(orchestrator, "build_readiness_snapshot", None)
            if callable(readiness_builder):
                try:
                    snapshot = readiness_builder()
                except Exception:
                    snapshot = None
                if snapshot is not None:
                    if hasattr(snapshot, "as_dict"):
                        try:
                            readiness_payload = snapshot.as_dict()  # type: ignore[call-arg]
                        except Exception:
                            readiness_payload = None
                    elif isinstance(snapshot, Mapping):
                        readiness_payload = dict(snapshot)

                    status_value = getattr(snapshot, "status", None)
                    if status_value is not None:
                        readiness_status = getattr(status_value, "value", None) or str(status_value)
                    elif readiness_payload is not None:
                        candidate_status = readiness_payload.get("status")
                        if isinstance(candidate_status, str):
                            readiness_status = candidate_status

        total_generations = 0
        if telemetry is not None:
            total_generations = int(telemetry.get("total_generations", 0))

        adaptive_runs_payload: Mapping[str, Any] | None = None
        if isinstance(telemetry, Mapping):
            adaptive_runs = telemetry.get("adaptive_runs")
            if isinstance(adaptive_runs, Mapping):
                adaptive_runs_payload = dict(adaptive_runs)

        coverage = 0.35
        strengths: list[str] = []
        gaps: list[str] = ["Wire evolutionary telemetry into governance reports"]

        if orchestrator is None:
            coverage = 0.3
            gaps.append("Integrate EvolutionCycleOrchestrator to activate Layer 4")
        else:
            status_key = readiness_status.lower() if readiness_status else None
            if status_key == "blocked":
                coverage = 0.32
                gaps.append("Resolve evolution readiness blockers")
            elif status_key == "review":
                coverage = 0.55
                strengths.append("evolution orchestrator ready for evaluation")
                gaps.append("Approve adaptive runs after governance review")
            elif status_key == "ready" and total_generations > 0:
                coverage = 0.9
                strengths.append(f"{total_generations} generations evolved under governance gate")
                gaps = [gap for gap in gaps if "Wire" not in gap]
            elif status_key == "ready":
                coverage = 0.72
                strengths.append("evolution orchestrator governance-ready")
                gaps.append("Run evolution cycles to produce live champions")
            elif total_generations > 0:
                coverage = 0.86
                strengths.append(f"{total_generations} generations evolved")
                gaps = [gap for gap in gaps if "Wire" not in gap]
            else:
                coverage = 0.55
                strengths.append("evolution orchestrator ready for evaluation")
                gaps.append("Run evolution cycles to produce live champions")

            if adaptive_runs_payload is not None and adaptive_runs_payload.get("enabled"):
                strengths.append("adaptive runs enabled via feature flag")
            elif adaptive_runs_payload is not None:
                gaps.append("Adaptive runs remain disabled pending flag approval")

        seen_gaps: set[str] = set()
        unique_gaps: list[str] = []
        for gap in gaps:
            if gap not in seen_gaps:
                unique_gaps.append(gap)
                seen_gaps.add(gap)
        gaps = unique_gaps

        evidence: dict[str, Any] = {
            "telemetry": telemetry or {},
            "champion": champion_payload,
        }
        if adaptive_runs_payload is not None:
            evidence["adaptive_runs"] = adaptive_runs_payload
        if readiness_payload is not None:
            evidence["readiness"] = readiness_payload
        if readiness_status is not None:
            metadata_payload = evidence.get("metadata")
            if isinstance(metadata_payload, Mapping):
                metadata_dict = dict(metadata_payload)
            else:
                metadata_dict = {}
            metadata_dict["readiness_status"] = readiness_status
            evidence["metadata"] = metadata_dict

        return LayerAssessment(
            layer="Layer 4 – Evolutionary Strategy Engine",
            coverage=coverage,
            strengths=strengths,
            gaps=gaps,
            evidence=evidence,
        )

    def _assess_trading_layer(self) -> LayerAssessment:
        trading_manager = self.trading_manager
        control_center = self.control_center
        coverage = 0.35
        strengths: list[str] = []
        gaps: list[str] = []
        evidence: dict[str, Any] = {}

        risk_gateway = getattr(trading_manager, "risk_gateway", None) if trading_manager else None
        portfolio_monitor = (
            getattr(trading_manager, "portfolio_monitor", None) if trading_manager else None
        )
        execution_engine = (
            getattr(trading_manager, "execution_engine", None) if trading_manager else None
        )

        if risk_gateway is not None:
            coverage += 0.25
            strengths.append("risk gateway enforcing encyclopedia limits")
            limits = (
                risk_gateway.get_risk_limits() if hasattr(risk_gateway, "get_risk_limits") else None
            )
            if isinstance(limits, Mapping):
                evidence["risk_limits"] = limits

        if portfolio_monitor is not None and hasattr(portfolio_monitor, "get_state"):
            coverage += 0.25
            strengths.append("portfolio monitor tracking exposure")
            try:
                state = portfolio_monitor.get_state()
            except Exception:
                state = {}
            if isinstance(state, Mapping):
                evidence["portfolio_state"] = dict(state)
        else:
            gaps.append("Activate portfolio monitor for exposure tracking")

        if execution_engine is not None:
            coverage += 0.2
            strengths.append("execution adapter processing fills")

        if control_center is not None and hasattr(control_center, "recent_decisions"):
            try:
                recent = control_center.recent_decisions(limit=5)
            except Exception:
                recent = []
            evidence["recent_decisions"] = recent
            if not recent:
                gaps.append("Route trade decisions into the control centre")

        coverage = min(coverage, 0.95)
        if coverage < 0.45 and not gaps:
            gaps.append("Complete risk, portfolio, and execution wiring")

        return LayerAssessment(
            layer="Layer 5 – Trading, Risk & Execution",
            coverage=coverage,
            strengths=strengths,
            gaps=gaps,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_unique(items: Iterable[Iterable[str]]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for sequence in items:
            for item in sequence:
                if not item:
                    continue
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
        return unique
