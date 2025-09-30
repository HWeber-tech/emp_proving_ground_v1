"""Operational control centre consolidating bootstrap telemetry."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from itertools import islice
import logging
from typing import Any, Callable, Deque, Iterable, Mapping

from src.governance.vision_alignment import VisionAlignmentReport
from src.operations.roi import format_roi_markdown as format_roi_summary
from src.orchestration.bootstrap_stack import SensorySnapshot
from src.trading.risk.policy_telemetry import format_policy_markdown


_log = logging.getLogger(__name__)


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _normalise_champion_payload(candidate: object | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    method = getattr(candidate, "as_payload", None)
    if callable(method):
        try:
            payload = method()
        except Exception as exc:  # pragma: no cover - defensive logging surface
            _log.warning(
                "Champion payload resolution failed for %s: %s",
                type(candidate).__name__,
                exc,
                exc_info=exc,
            )
            return {"genome_id": getattr(candidate, "genome_id", None)}
        if isinstance(payload, Mapping):
            return dict(payload)
        return {"raw": payload}
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return {"genome_id": getattr(candidate, "genome_id", None)}


def _call_trading_manager_method(trading_manager: Any, method_name: str) -> Any:
    """Invoke ``method_name`` on ``trading_manager`` if available."""

    target = getattr(trading_manager, method_name, None)
    if not callable(target):
        return None
    try:
        return target()
    except Exception as exc:  # pragma: no cover - diagnostics only
        _log.warning(
            "Trading manager method '%s' failed: %s",
            method_name,
            exc,
            exc_info=exc,
        )
        return None


def _coerce_snapshot_mapping(snapshot: Any) -> Mapping[str, Any] | None:
    """Convert snapshot-like objects to mappings for downstream formatting."""

    if snapshot is None:
        return None

    as_dict = getattr(snapshot, "as_dict", None)
    if callable(as_dict):
        try:
            payload = as_dict()
        except Exception as exc:  # pragma: no cover - diagnostics only
            _log.warning(
                "Snapshot serialization failed for %s: %s",
                type(snapshot).__name__,
                exc,
                exc_info=exc,
            )
            payload = None
        if isinstance(payload, Mapping):
            return dict(payload)

    if isinstance(snapshot, Mapping):
        return dict(snapshot)

    return None


def _format_optional_markdown(
    formatter: Callable[[Any], str | None], snapshot: Any
) -> str | None:
    """Best-effort markdown rendering for snapshot payloads."""

    try:
        return formatter(snapshot)
    except Exception as exc:  # pragma: no cover - diagnostics only
        formatter_name = getattr(formatter, "__name__", repr(formatter))
        _log.warning(
            "Formatter '%s' failed for %s: %s",
            formatter_name,
            type(snapshot).__name__,
            exc,
            exc_info=exc,
        )
        return None


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

        overview: dict[str, Any] = {
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

        snapshot_obj = _call_trading_manager_method(
            self.trading_manager, "get_last_risk_snapshot"
        )
        snapshot_mapping = _coerce_snapshot_mapping(snapshot_obj)
        if snapshot_mapping is not None:
            overview["risk_posture"] = snapshot_mapping

        policy_obj = _call_trading_manager_method(
            self.trading_manager, "get_last_policy_snapshot"
        )
        policy_mapping = _coerce_snapshot_mapping(policy_obj)
        if policy_mapping is not None:
            overview["risk_policy"] = policy_mapping

        roi_obj = _call_trading_manager_method(
            self.trading_manager, "get_last_roi_snapshot"
        )
        roi_mapping = _coerce_snapshot_mapping(roi_obj)
        if roi_mapping is not None:
            overview["roi_posture"] = roi_mapping

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
        snapshot_obj = _call_trading_manager_method(
            self.trading_manager, "get_last_risk_snapshot"
        )
        snapshot = _coerce_snapshot_mapping(snapshot_obj)

        policy_obj = _call_trading_manager_method(
            self.trading_manager, "get_last_policy_snapshot"
        )
        policy_snapshot = _coerce_snapshot_mapping(policy_obj)
        policy_markdown = (
            _format_optional_markdown(format_policy_markdown, policy_obj)
            if policy_obj is not None
            else None
        )
        policy_block: Mapping[str, Any] | None = None
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
        roi_obj = _call_trading_manager_method(
            self.trading_manager, "get_last_roi_snapshot"
        )
        if roi_obj is None:
            return {}

        roi_dict = _coerce_snapshot_mapping(roi_obj)
        roi_markdown = _format_optional_markdown(format_roi_summary, roi_obj)
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
        champion: object | None = getattr(orchestrator, "champion", None)
        stats = getattr(orchestrator, "population_statistics", {})

        if isinstance(telemetry, Mapping):
            telemetry_payload = dict(telemetry)
        else:
            telemetry_payload = {"raw": telemetry}

        champion_payload = _normalise_champion_payload(champion)

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

        champion: object | None = getattr(orchestrator, "champion", None)
        payload = _normalise_champion_payload(champion) or {}

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
