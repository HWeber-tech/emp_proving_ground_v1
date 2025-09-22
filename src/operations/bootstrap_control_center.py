"""Operational control centre consolidating bootstrap telemetry."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from itertools import islice
from typing import Any, Deque, Iterable, Mapping

from src.governance.vision_alignment import VisionAlignmentReport
from src.operations.roi import format_roi_markdown as format_roi_summary
from src.orchestration.bootstrap_stack import SensorySnapshot
from src.trading.risk.policy_telemetry import format_policy_markdown


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class BootstrapControlCenter:
    """Aggregate portfolio, risk, intelligence, liquidity, and execution telemetry."""

    def __init__(
        self,
        *,
        pipeline: Any,
        trading_manager: Any,
        execution_adapter: Any | None = None,
        liquidity_prober: Any | None = None,
        evolution_orchestrator: Any | None = None,
        history_limit: int = 256,
    ) -> None:
        self.pipeline = pipeline
        self.trading_manager = trading_manager
        self.portfolio_monitor = getattr(trading_manager, "portfolio_monitor", None)
        self.risk_gateway = getattr(trading_manager, "risk_gateway", None)
        self.execution_adapter = execution_adapter
        self.liquidity_prober = liquidity_prober
        self.evolution_orchestrator = evolution_orchestrator

        self._history: Deque[dict[str, Any]] = deque(maxlen=max(16, history_limit))
        self._last_snapshot: SensorySnapshot | None = None
        self._last_liquidity_summary: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def record_tick(self, *, snapshot: SensorySnapshot, result: Mapping[str, Any]) -> None:
        """Register a decision cycle for downstream reporting."""

        decision = result.get("decision") if isinstance(result, Mapping) else None
        liquidity_summary = result.get("liquidity_summary") if isinstance(result, Mapping) else None

        record: dict[str, Any] = {
            "symbol": snapshot.symbol,
            "generated_at": snapshot.generated_at.isoformat(),
            "status": result.get("status"),
            "unified_score": float(snapshot.synthesis.unified_score),
            "confidence": float(snapshot.synthesis.confidence),
            "narrative": snapshot.synthesis.dominant_narrative.name,
        }

        if isinstance(decision, Mapping):
            record["decision_status"] = decision.get("status")
            record["decision_reason"] = decision.get("reason")
            record["checks"] = list(decision.get("checks", ()))
        else:
            record["decision_status"] = None
            record["decision_reason"] = None
            record["checks"] = []

        if isinstance(liquidity_summary, Mapping):
            self._last_liquidity_summary = dict(liquidity_summary)

        self._history.appendleft(record)
        self._last_snapshot = snapshot

    # ------------------------------------------------------------------
    # Public reporting surfaces
    # ------------------------------------------------------------------
    def recent_decisions(self, limit: int = 5) -> list[Mapping[str, Any]]:
        return [dict(item) for item in islice(self._history, 0, max(1, limit))]

    def generate_report(self) -> Mapping[str, Any]:
        """Build a detailed operations snapshot."""

        portfolio = self._build_portfolio_section()
        risk = self._build_risk_section()
        intelligence = self._build_intelligence_section()
        liquidity = self._build_liquidity_section()
        performance = self._build_performance_section(portfolio)
        evolution = self._build_evolution_section()
        vision = self._build_vision_section()

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "portfolio": portfolio,
            "risk": risk,
            "intelligence": intelligence,
            "liquidity": liquidity,
            "performance": performance,
            "evolution": evolution,
            "vision_alignment": vision,
            "decisions": {
                "recent": self.recent_decisions(10),
                "total_logged": len(self._history),
            },
        }

    def overview(self) -> Mapping[str, Any]:
        """Return a lightweight summary for runtime.status() consumers."""

        portfolio_state = self._safe_portfolio_state()
        open_positions = portfolio_state.get("open_positions_count")
        if open_positions is None:
            open_positions = len(portfolio_state.get("open_positions", {}))

        latest = next(iter(self._history), None)

        overview = {
            "equity": _as_float(portfolio_state.get("equity"), default=0.0),
            "realized_pnl": _as_float(portfolio_state.get("realized_pnl"), default=0.0),
            "unrealized_pnl": _as_float(portfolio_state.get("unrealized_pnl"), default=0.0),
            "open_positions": int(open_positions or 0),
            "last_decision": dict(latest) if latest else None,
        }

        evolution_overview = self._build_evolution_overview()
        if evolution_overview:
            overview["evolution"] = evolution_overview

        vision_summary = self._build_vision_summary()
        if vision_summary:
            overview["vision_alignment"] = vision_summary

        if hasattr(self.trading_manager, "get_last_risk_snapshot"):
            try:
                snapshot_obj = self.trading_manager.get_last_risk_snapshot()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - diagnostics only
                snapshot_obj = None
            if snapshot_obj is not None:
                try:
                    overview["risk_posture"] = snapshot_obj.as_dict()  # type: ignore[attr-defined]
                except AttributeError:
                    if isinstance(snapshot_obj, Mapping):
                        overview["risk_posture"] = dict(snapshot_obj)

        if hasattr(self.trading_manager, "get_last_policy_snapshot"):
            try:
                policy_obj = self.trading_manager.get_last_policy_snapshot()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - diagnostics only
                policy_obj = None
            if policy_obj is not None:
                try:
                    overview["risk_policy"] = policy_obj.as_dict()  # type: ignore[attr-defined]
                except AttributeError:
                    if isinstance(policy_obj, Mapping):
                        overview["risk_policy"] = dict(policy_obj)

        if hasattr(self.trading_manager, "get_last_roi_snapshot"):
            try:
                roi_obj = self.trading_manager.get_last_roi_snapshot()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - diagnostics only
                roi_obj = None
            if roi_obj is not None:
                try:
                    overview["roi_posture"] = roi_obj.as_dict()  # type: ignore[attr-defined]
                except AttributeError:
                    if isinstance(roi_obj, Mapping):
                        overview["roi_posture"] = dict(roi_obj)

        return overview

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------
    def _build_portfolio_section(self) -> dict[str, Any]:
        state = self._safe_portfolio_state()

        exposures = {"long": 0.0, "short": 0.0}
        open_positions = state.get("open_positions", {})
        if isinstance(open_positions, Mapping):
            for payload in open_positions.values():
                if not isinstance(payload, Mapping):
                    continue
                quantity = _as_float(payload.get("quantity"), default=0.0)
                value = _as_float(payload.get("current_value"), default=0.0)
                if quantity >= 0:
                    exposures["long"] += value
                else:
                    exposures["short"] += abs(value)

        portfolio = dict(state)
        portfolio["exposures"] = exposures
        return portfolio

    def _build_risk_section(self) -> Mapping[str, Any]:
        gateway = self.risk_gateway
        if gateway is None:
            return {"limits": {}, "telemetry": {}, "last_decision": None}

        limits = gateway.get_risk_limits() if hasattr(gateway, "get_risk_limits") else {}
        last_decision = (
            gateway.get_last_decision() if hasattr(gateway, "get_last_decision") else None
        )

        limits_payload = (
            dict(limits) if isinstance(limits, Mapping) else {"limits": {}, "telemetry": {}}
        )
        snapshot: Mapping[str, Any] | None = None
        if hasattr(self.trading_manager, "get_last_risk_snapshot"):
            try:
                snapshot_obj = self.trading_manager.get_last_risk_snapshot()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive surface
                snapshot_obj = None
            if snapshot_obj is not None:
                try:
                    snapshot = snapshot_obj.as_dict()  # type: ignore[attr-defined]
                except AttributeError:
                    snapshot = dict(snapshot_obj) if isinstance(snapshot_obj, Mapping) else None

        policy_block: Mapping[str, Any] | None = None
        if hasattr(self.trading_manager, "get_last_policy_snapshot"):
            try:
                policy_obj = self.trading_manager.get_last_policy_snapshot()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - diagnostics only
                policy_obj = None
            if policy_obj is not None:
                try:
                    policy_snapshot = policy_obj.as_dict()  # type: ignore[attr-defined]
                    policy_markdown = format_policy_markdown(policy_obj)  # type: ignore[arg-type]
                except AttributeError:
                    policy_snapshot = dict(policy_obj) if isinstance(policy_obj, Mapping) else None
                    policy_markdown = None
                if policy_snapshot is not None:
                    block: dict[str, Any] = {"snapshot": policy_snapshot}
                    if policy_markdown:
                        block["markdown"] = policy_markdown
                    policy_block = block

        return {
            "limits": dict(limits_payload.get("limits", {})),
            "telemetry": dict(limits_payload.get("telemetry", {})),
            "last_decision": dict(last_decision)
            if isinstance(last_decision, Mapping)
            else last_decision,
            "snapshot": dict(snapshot) if isinstance(snapshot, Mapping) else None,
            "policy": policy_block,
        }

    def _build_intelligence_section(self) -> Mapping[str, Any]:
        snapshot = self._last_snapshot
        diagnostics = (
            self.pipeline.fusion_engine.get_diagnostic_information()
            if hasattr(self.pipeline, "fusion_engine")
            else {}
        )

        if snapshot is None:
            return {"diagnostics": diagnostics}

        synthesis = snapshot.synthesis
        return {
            "symbol": snapshot.symbol,
            "generated_at": snapshot.generated_at.isoformat(),
            "unified_score": float(synthesis.unified_score),
            "confidence": float(synthesis.confidence),
            "narrative": synthesis.dominant_narrative.name,
            "narrative_text": synthesis.narrative_text,
            "coherence": float(synthesis.narrative_coherence),
            "diagnostics": diagnostics,
        }

    def _build_liquidity_section(self) -> Mapping[str, Any]:
        summary = dict(self._last_liquidity_summary) if self._last_liquidity_summary else {}
        prober = self.liquidity_prober
        config = {}
        if prober is not None:
            config = {
                "decay": getattr(prober, "decay", None),
                "max_history": getattr(prober, "max_history", None),
                "min_quality": getattr(prober, "min_quality", None),
            }
        return {"summary": summary, "prober": config}

    def _build_performance_section(self, portfolio: Mapping[str, Any]) -> Mapping[str, Any]:
        fills: list[Any] = []
        if self.execution_adapter is not None:
            fills_attr = getattr(self.execution_adapter, "fills", None)
            if isinstance(fills_attr, Iterable):
                fills = list(fills_attr)

        return {
            "fills": len(fills),
            "last_fill": fills[-1] if fills else None,
            "realized_pnl": _as_float(portfolio.get("realized_pnl"), default=0.0),
            "unrealized_pnl": _as_float(portfolio.get("unrealized_pnl"), default=0.0),
            "total_pnl": _as_float(portfolio.get("total_pnl"), default=0.0),
            "equity": _as_float(portfolio.get("equity"), default=0.0),
            **self._build_roi_section(),
        }

    def _build_roi_section(self) -> Mapping[str, Any]:
        if not hasattr(self.trading_manager, "get_last_roi_snapshot"):
            return {}

        try:
            roi_obj = self.trading_manager.get_last_roi_snapshot()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - diagnostics only
            roi_obj = None
        if roi_obj is None:
            return {}

        try:
            roi_dict = roi_obj.as_dict()  # type: ignore[attr-defined]
            roi_markdown = format_roi_summary(roi_obj)  # type: ignore[arg-type]
        except AttributeError:
            roi_dict = dict(roi_obj) if isinstance(roi_obj, Mapping) else None
            roi_markdown = None
        if roi_dict is None:
            return {}

        payload: dict[str, Any] = {"roi": {"snapshot": roi_dict}}
        if roi_markdown:
            payload["roi"]["markdown"] = roi_markdown
        return payload

    def _build_evolution_section(self) -> Mapping[str, Any]:
        orchestrator = self.evolution_orchestrator
        if orchestrator is None:
            return {}

        telemetry = getattr(orchestrator, "telemetry", {})
        champion = getattr(orchestrator, "champion", None)
        stats = getattr(orchestrator, "population_statistics", {})

        if isinstance(telemetry, Mapping):
            telemetry_payload = dict(telemetry)
        else:
            telemetry_payload = {"raw": telemetry}

        if hasattr(champion, "as_payload"):
            champion_payload = champion.as_payload()
        elif isinstance(champion, Mapping):
            champion_payload = dict(champion)
        elif champion is None:
            champion_payload = None
        else:
            champion_payload = {"genome_id": getattr(champion, "genome_id", None)}

        population = dict(stats) if isinstance(stats, Mapping) else {}

        return {
            "telemetry": telemetry_payload,
            "champion": champion_payload,
            "population": population,
        }

    def _build_evolution_overview(self) -> Mapping[str, Any]:
        orchestrator = self.evolution_orchestrator
        if orchestrator is None:
            return {}

        champion = getattr(orchestrator, "champion", None)
        if hasattr(champion, "as_payload"):
            payload = champion.as_payload()
        elif isinstance(champion, Mapping):
            payload = dict(champion)
        elif champion is None:
            payload = {}
        else:
            payload = {"genome_id": getattr(champion, "genome_id", None)}

        telemetry = getattr(orchestrator, "telemetry", {})
        total_generations = None
        if isinstance(telemetry, Mapping):
            total_generations = telemetry.get("total_generations")

        overview: dict[str, Any] = {"champion": payload}
        if total_generations is not None:
            overview["generations"] = total_generations
        return overview

    def _build_vision_section(self) -> Mapping[str, Any]:
        reporter = VisionAlignmentReport(
            fabric=getattr(self.pipeline, "fabric", None),
            pipeline=self.pipeline,
            trading_manager=self.trading_manager,
            control_center=self,
            evolution_orchestrator=self.evolution_orchestrator,
        )
        return reporter.build()

    def _build_vision_summary(self) -> Mapping[str, Any]:
        reporter = VisionAlignmentReport(
            fabric=getattr(self.pipeline, "fabric", None),
            pipeline=self.pipeline,
            trading_manager=self.trading_manager,
            control_center=self,
            evolution_orchestrator=self.evolution_orchestrator,
        )
        return reporter.summary()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _safe_portfolio_state(self) -> Mapping[str, Any]:
        if self.portfolio_monitor is None:
            return {}
        state = self.portfolio_monitor.get_state()
        return state if isinstance(state, Mapping) else {}


__all__ = ["BootstrapControlCenter"]
