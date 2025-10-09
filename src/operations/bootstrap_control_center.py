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
from src.trading.risk.risk_api import (
    RISK_API_RUNBOOK,
    RiskApiError,
    build_runtime_risk_metadata,
    merge_risk_references,
)


_log = logging.getLogger(__name__)


_EXPECTED_OPERATION_ERRORS = (RuntimeError, ValueError, TypeError, LookupError)


# Keys from the deterministic risk summary we elevate into risk references.
_RISK_SUMMARY_KEYS = {
    "max_risk_per_trade_pct",
    "max_total_exposure_pct",
    "max_leverage",
    "max_drawdown_pct",
    "min_position_size",
    "max_position_size",
    "mandatory_stop_loss",
    "research_mode",
    "target_volatility_pct",
    "volatility_window",
    "max_volatility_leverage",
    "volatility_annualisation_factor",
    "sector_exposure_limits",
    "instrument_sector_map",
    "sector_budget_total_pct",
    "sector_headroom_pct",
    "sector_headroom_ratio",
    "max_sector_utilisation_ratio",
    "sector_instrument_counts",
}


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
        except _EXPECTED_OPERATION_ERRORS as exc:  # pragma: no cover - defensive logging surface
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
    except _EXPECTED_OPERATION_ERRORS as exc:  # pragma: no cover - diagnostics only
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
        except _EXPECTED_OPERATION_ERRORS as exc:  # pragma: no cover - diagnostics only
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


def _describe_risk_interface_payload(trading_manager: Any) -> Mapping[str, Any] | None:
    """Resolve the trading risk interface payload if the manager exposes it."""

    describe = getattr(trading_manager, "describe_risk_interface", None)
    if not callable(describe):
        return None
    try:
        payload = describe()
    except _EXPECTED_OPERATION_ERRORS as exc:  # pragma: no cover - diagnostics only
        _log.debug(
            "Trading manager describe_risk_interface failed: %s",
            exc,
            exc_info=exc,
        )
        return None
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"value": payload}


def _format_optional_markdown(
    formatter: Callable[[Any], str | None], snapshot: Any
) -> str | None:
    """Best-effort markdown rendering for snapshot payloads."""

    try:
        return formatter(snapshot)
    except _EXPECTED_OPERATION_ERRORS as exc:  # pragma: no cover - diagnostics only
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
    """Aggregate portfolio, risk, understanding, liquidity, and execution telemetry."""

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
        self._last_risk_metadata: dict[str, object] | None = None
        self._last_risk_error: dict[str, object] | None = None
        self._last_risk_reference: dict[str, object] | None = None
        self._last_risk_runbook: str = RISK_API_RUNBOOK

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

    def _update_risk_metadata(self) -> None:
        """Refresh the cached deterministic risk metadata for the trading manager."""

        self._last_risk_metadata = None
        self._last_risk_error = None

        try:
            metadata = build_runtime_risk_metadata(self.trading_manager)
        except RiskApiError as exc:
            self._last_risk_error = exc.to_metadata()
        except _EXPECTED_OPERATION_ERRORS as exc:  # pragma: no cover - diagnostic fallback
            _log.debug("Risk metadata resolution failed", exc_info=True)
            self._last_risk_error = {
                "message": "Bootstrap control centre risk metadata resolution failed",
                "error": str(exc),
                "runbook": RISK_API_RUNBOOK,
            }
        else:
            self._last_risk_metadata = dict(metadata)

    def _compose_risk_reference(
        self,
        limits_payload: Mapping[str, Any] | None,
        interface_payload: Mapping[str, Any] | None,
    ) -> tuple[dict[str, object] | None, str]:
        """Merge available risk references and determine the prevailing runbook."""

        runbook = RISK_API_RUNBOOK
        reference_candidates: list[Mapping[str, object]] = []

        if isinstance(limits_payload, Mapping):
            runbook_candidate = limits_payload.get("runbook")
            if isinstance(runbook_candidate, str) and runbook_candidate:
                runbook = runbook_candidate
            reference_candidate = limits_payload.get("risk_reference")
            if isinstance(reference_candidate, Mapping):
                reference_candidates.append(reference_candidate)

        if isinstance(interface_payload, Mapping):
            runbook_candidate = interface_payload.get("runbook")
            if isinstance(runbook_candidate, str) and runbook_candidate:
                runbook = runbook_candidate
            reference_candidate = interface_payload.get("risk_reference")
            if isinstance(reference_candidate, Mapping):
                reference_candidates.append(reference_candidate)

        metadata = self._last_risk_metadata
        if metadata is not None:
            metadata_runbook = metadata.get("runbook")
            if isinstance(metadata_runbook, str) and metadata_runbook:
                runbook = metadata_runbook
            summary: dict[str, object] = {}
            for key in _RISK_SUMMARY_KEYS:
                if key not in metadata:
                    continue
                value = metadata[key]
                summary[key] = dict(value) if isinstance(value, Mapping) else value
            if summary:
                reference_candidates.append({"risk_config_summary": summary})

        error = self._last_risk_error
        if error is not None:
            error_runbook = error.get("runbook")
            if isinstance(error_runbook, str) and error_runbook:
                runbook = error_runbook

        reference: dict[str, object] | None = None
        if reference_candidates:
            reference = merge_risk_references(*reference_candidates, runbook=runbook)

        self._last_risk_reference = reference
        self._last_risk_runbook = runbook
        return reference, runbook

    # ------------------------------------------------------------------
    # Public reporting surfaces
    # ------------------------------------------------------------------
    def recent_decisions(self, limit: int = 5) -> list[Mapping[str, Any]]:
        return [dict(item) for item in islice(self._history, 0, max(1, limit))]

    def generate_report(self) -> Mapping[str, Any]:
        """Build a detailed operations snapshot."""

        portfolio = self._build_portfolio_section()
        risk = self._build_risk_section()
        understanding = self._build_understanding_section()
        liquidity = self._build_liquidity_section()
        performance = self._build_performance_section(portfolio)
        evolution = self._build_evolution_section()
        vision = self._build_vision_section()

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "portfolio": portfolio,
            "risk": risk,
            "understanding": understanding,
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

        interface_payload = _describe_risk_interface_payload(self.trading_manager)
        if interface_payload is not None:
            overview["risk_interface"] = interface_payload

        limits_payload: Mapping[str, Any] | None = None
        gateway = self.risk_gateway
        if gateway is not None and hasattr(gateway, "get_risk_limits"):
            try:
                limits_candidate = gateway.get_risk_limits()
            except _EXPECTED_OPERATION_ERRORS:  # pragma: no cover - defensive diagnostics
                _log.debug("Risk gateway limits resolution failed for overview", exc_info=True)
            else:
                if isinstance(limits_candidate, Mapping):
                    limits_payload = dict(limits_candidate)

        self._update_risk_metadata()
        reference, runbook = self._compose_risk_reference(limits_payload, interface_payload)

        if self._last_risk_metadata is not None:
            overview["risk_metadata"] = dict(self._last_risk_metadata)
        if self._last_risk_error is not None:
            overview["risk_metadata_error"] = dict(self._last_risk_error)
        if reference is not None:
            overview["risk_reference"] = reference
        overview.setdefault("risk_runbook", runbook)

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
            return {
                "limits": {},
                "telemetry": {},
                "last_decision": None,
                "runbook": RISK_API_RUNBOOK,
            }

        limits = gateway.get_risk_limits() if hasattr(gateway, "get_risk_limits") else {}
        last_decision = (
            gateway.get_last_decision() if hasattr(gateway, "get_last_decision") else None
        )

        limits_payload = (
            dict(limits) if isinstance(limits, Mapping) else {"limits": {}, "telemetry": {}}
        )
        interface_payload = _describe_risk_interface_payload(self.trading_manager)

        self._update_risk_metadata()
        metadata = dict(self._last_risk_metadata) if self._last_risk_metadata is not None else None
        metadata_error = (
            dict(self._last_risk_error) if self._last_risk_error is not None else None
        )
        reference, runbook = self._compose_risk_reference(limits_payload, interface_payload)

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

        release_execution = _call_trading_manager_method(
            self.trading_manager, "describe_release_execution"
        )
        release_block: Mapping[str, Any] | None
        if isinstance(release_execution, Mapping):
            release_block = dict(release_execution)
        else:
            release_block = None

        payload: dict[str, Any] = {
            "limits": dict(limits_payload.get("limits", {})),
            "telemetry": dict(limits_payload.get("telemetry", {})),
            "last_decision": dict(last_decision)
            if isinstance(last_decision, Mapping)
            else last_decision,
            "snapshot": dict(snapshot) if isinstance(snapshot, Mapping) else None,
            "policy": policy_block,
            "interface": interface_payload,
            "release_execution": release_block,
            "runbook": runbook,
        }

        if metadata is not None:
            payload["metadata"] = metadata
        if metadata_error is not None:
            payload["metadata_error"] = metadata_error
        if reference is not None:
            payload["risk_reference"] = reference

        return payload

    def _build_understanding_section(self) -> Mapping[str, Any]:
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

    def _resolve_evolution_readiness(self, orchestrator: Any) -> Mapping[str, Any] | None:
        """Best-effort resolution of the evolution readiness snapshot."""

        builder = getattr(orchestrator, "build_readiness_snapshot", None)
        if not callable(builder):
            return None

        try:
            snapshot = builder()
        except _EXPECTED_OPERATION_ERRORS as exc:  # pragma: no cover - diagnostics only
            _log.debug(
                "Evolution readiness snapshot resolution failed: %s",
                exc,
                exc_info=exc,
            )
            return None

        payload = _coerce_snapshot_mapping(snapshot)
        if payload is None and hasattr(snapshot, "as_dict"):
            try:
                payload_candidate = snapshot.as_dict()  # type: ignore[call-arg]
            except _EXPECTED_OPERATION_ERRORS:
                payload_candidate = None
            if isinstance(payload_candidate, Mapping):
                payload = dict(payload_candidate)

        return payload if isinstance(payload, Mapping) else None

    def _build_evolution_section(self) -> Mapping[str, Any]:
        orchestrator = self.evolution_orchestrator
        if orchestrator is None:
            return {}

        telemetry = getattr(orchestrator, "telemetry", {})
        champion: object | None = getattr(orchestrator, "champion", None)
        stats = getattr(orchestrator, "population_statistics", {})
        readiness = self._resolve_evolution_readiness(orchestrator)

        if isinstance(telemetry, Mapping):
            telemetry_payload = dict(telemetry)
        else:
            telemetry_payload = {"raw": telemetry}

        champion_payload = _normalise_champion_payload(champion)

        population = dict(stats) if isinstance(stats, Mapping) else {}

        payload: dict[str, Any] = {
            "telemetry": telemetry_payload,
            "champion": champion_payload,
            "population": population,
        }
        if readiness:
            payload["readiness"] = readiness
        return payload

    def _build_evolution_overview(self) -> Mapping[str, Any]:
        orchestrator = self.evolution_orchestrator
        if orchestrator is None:
            return {}

        champion: object | None = getattr(orchestrator, "champion", None)
        payload = _normalise_champion_payload(champion) or {}

        telemetry = getattr(orchestrator, "telemetry", {})
        overview: dict[str, Any] = {"champion": payload}

        if isinstance(telemetry, Mapping):
            total_generations = telemetry.get("total_generations")
            if total_generations is not None:
                try:
                    overview["generations"] = int(total_generations)
                except (TypeError, ValueError):
                    overview["generations"] = total_generations

            adaptive_runs = telemetry.get("adaptive_runs")
            if isinstance(adaptive_runs, Mapping):
                overview["adaptive_runs"] = dict(adaptive_runs)
            elif adaptive_runs is not None:
                overview["adaptive_runs"] = {"raw": adaptive_runs}

            lineage = telemetry.get("lineage")
            if isinstance(lineage, Mapping):
                overview["lineage"] = dict(lineage)

            catalogue = telemetry.get("catalogue")
            if isinstance(catalogue, Mapping):
                overview["catalogue"] = dict(catalogue)

        lineage_snapshot = getattr(orchestrator, "lineage_snapshot", None)
        lineage_payload = _coerce_snapshot_mapping(lineage_snapshot)
        if lineage_payload and "lineage" not in overview:
            overview["lineage"] = lineage_payload

        readiness = self._resolve_evolution_readiness(orchestrator)
        if readiness:
            overview["readiness"] = readiness

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
