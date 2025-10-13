"""Release-aware execution router bridging policy ledger stages to engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Mapping, MutableMapping, Sequence

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage
from src.operations.sensory_drift import DriftSeverity
from ._risk_context import (
    RiskContextProvider,
    capture_risk_context,
    describe_risk_context,
)

__all__ = ["ReleaseAwareExecutionRouter"]

logger = logging.getLogger(__name__)


_ENGINE_UNSET = object()


@dataclass
class ReleaseAwareExecutionRouter:
    """Route validated intents to the appropriate execution engine per release stage."""

    release_manager: LedgerReleaseManager
    paper_engine: Any
    pilot_engine: Any | None = None
    live_engine: Any | None = None
    default_stage: PolicyLedgerStage = PolicyLedgerStage.EXPERIMENT
    risk_context_provider: RiskContextProvider | None = field(default=None, repr=False)
    _last_route: MutableMapping[str, Any] = field(default_factory=dict, init=False, repr=False)
    _last_risk_metadata: dict[str, object] | None = field(default=None, init=False, repr=False)
    _last_risk_error: dict[str, object] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.release_manager is None:
            raise ValueError("release_manager is required")
        if self.paper_engine is None:
            raise ValueError("paper_engine is required")
        if not isinstance(self.default_stage, PolicyLedgerStage):
            self.default_stage = PolicyLedgerStage.from_value(self.default_stage)
        self._last_risk_metadata = None
        self._last_risk_error = None
        self._propagate_risk_context_provider()

    def set_risk_context_provider(
        self, provider: RiskContextProvider | None
    ) -> None:
        """Install or replace the callable that resolves trading risk metadata."""

        self.risk_context_provider = provider
        self._propagate_risk_context_provider()

    def _propagate_risk_context_provider(self) -> None:
        for engine in (self.paper_engine, self.pilot_engine, self.live_engine):
            if engine is None:
                continue
            setter = getattr(engine, "set_risk_context_provider", None)
            if callable(setter):
                try:
                    setter(self.risk_context_provider)
                except Exception:  # pragma: no cover - defensive guard
                    logger.debug("Failed to propagate risk context provider", exc_info=True)

    def _capture_risk_context(self) -> None:
        metadata, error = capture_risk_context(self.risk_context_provider)
        self._last_risk_metadata = metadata
        self._last_risk_error = error

    def describe_risk_context(self) -> dict[str, object]:
        """Expose the most recent deterministic risk metadata snapshot."""

        return describe_risk_context(self._last_risk_metadata, self._last_risk_error)

    def should_block_orders(self, intent: Any) -> Mapping[str, Any] | None:
        """Determine whether the routed engine is currently blocking execution."""

        self._capture_risk_context()
        metadata = self._extract_metadata(intent)
        strategy_id = self._extract_policy_id(intent)
        stage, posture = self._resolve_stage(strategy_id)
        stage_force_reason = self._stage_force_reason(stage)
        audit_force, audit_reason, audit_details = self._audit_enforcement(stage, posture)
        gate_force, gate_reason, forced_severity = self._should_force_paper(metadata)
        combined_force = audit_force or gate_force or bool(stage_force_reason)
        engine, route_label = self._select_engine(stage, force_paper=combined_force)

        sentinel = getattr(engine, "should_block_orders", None)
        if not callable(sentinel):
            return None

        try:
            result = sentinel(intent)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("release_router_block_check_failed", exc_info=True)
            return None

        if isinstance(result, Mapping):
            payload: MutableMapping[str, Any] = {
                str(key): value for key, value in result.items()
            }
        elif result:
            payload = {"reason": str(result)}
        else:
            return None

        payload.setdefault("stage", stage.value)
        payload.setdefault("route", route_label)
        if combined_force:
            payload.setdefault("forced_route", "paper")

        forced_reasons: list[str] = []
        if gate_force and gate_reason:
            payload.setdefault("drift_gate_reason", gate_reason)
            forced_reasons.append(gate_reason)
        if audit_force:
            payload.setdefault("audit_forced", True)
            if audit_details:
                payload.setdefault("audit", audit_details)
            if audit_reason:
                forced_reasons.append(audit_reason)
        if stage_force_reason:
            payload.setdefault("stage_force_reason", stage_force_reason)
            forced_reasons.append(stage_force_reason)
        if forced_severity:
            payload.setdefault("drift_severity", forced_severity.value)
        if forced_reasons and "forced_reasons" not in payload:
            payload["forced_reasons"] = forced_reasons

        if self._last_risk_metadata is not None:
            payload.setdefault("risk", dict(self._last_risk_metadata))
        if self._last_risk_error is not None:
            payload.setdefault("risk_error", dict(self._last_risk_error))

        return dict(payload)

    async def process_order(self, intent: Any) -> Any:
        """Process an order using the engine that matches the ledger stage."""

        self._capture_risk_context()

        metadata = self._extract_metadata(intent)
        strategy_id = self._extract_policy_id(intent)
        stage, posture = self._resolve_stage(strategy_id)
        stage_force_reason = self._stage_force_reason(stage)

        audit_force_paper, audit_reason, audit_details = self._audit_enforcement(stage, posture)

        gate_force_paper, gate_reason, forced_severity = self._should_force_paper(metadata)

        combined_force_paper = audit_force_paper or gate_force_paper or bool(stage_force_reason)
        forced_reasons: list[str] = []
        if gate_force_paper:
            forced_reasons.append(gate_reason or "drift_gate_force_paper")
        if audit_force_paper:
            forced_reasons.append(audit_reason or "release_audit_enforced")
        if stage_force_reason and stage_force_reason not in forced_reasons:
            forced_reasons.append(stage_force_reason)
        primary_forced_reason = forced_reasons[0] if forced_reasons else None

        engine, route_label = self._select_engine(stage, force_paper=combined_force_paper)
        self._attach_metadata(
            intent,
            stage,
            route_label,
            forced_reason=primary_forced_reason,
            forced_reasons=forced_reasons or None,
            audit_details=audit_details,
        )

        result = await engine.process_order(intent)
        self._last_route.clear()
        self._last_route.update(
            {
                "strategy_id": strategy_id,
                "stage": stage.value,
                "route": route_label,
                "engine": engine.__class__.__name__,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
        )
        if self._last_risk_metadata is not None:
            self._last_route["risk"] = dict(self._last_risk_metadata)
        if self._last_risk_error is not None:
            self._last_route["risk_error"] = dict(self._last_risk_error)
        if combined_force_paper:
            self._last_route["forced_route"] = "paper"
            if primary_forced_reason:
                self._last_route["forced_reason"] = primary_forced_reason
            if forced_reasons:
                self._last_route["forced_reasons"] = list(dict.fromkeys(forced_reasons))
            if forced_severity:
                self._last_route["drift_severity"] = forced_severity.value
            if audit_details:
                self._last_route["audit_forced"] = audit_force_paper
        if audit_details:
            self._last_route["audit"] = audit_details
        return result

    def describe(self) -> Mapping[str, Any]:
        """Return a serialisable snapshot of the router configuration."""

        engines = {
            "paper": self._engine_name(self.paper_engine),
        }
        if self.pilot_engine is not None:
            engines["pilot"] = self._engine_name(self.pilot_engine)
        if self.live_engine is not None:
            engines["live"] = self._engine_name(self.live_engine)
        payload: MutableMapping[str, Any] = {
            "default_stage": self.default_stage.value,
            "engines": engines,
        }
        if self._last_route:
            payload["last_route"] = dict(self._last_route)
        payload["risk_context"] = self.describe_risk_context()
        return payload

    def last_route(self) -> Mapping[str, Any] | None:
        """Expose the most recent routing decision."""

        if not self._last_route:
            return None
        return dict(self._last_route)

    def configure_engines(
        self,
        *,
        paper_engine: Any | object = _ENGINE_UNSET,
        pilot_engine: Any | object = _ENGINE_UNSET,
        live_engine: Any | object = _ENGINE_UNSET,
    ) -> None:
        """Update the execution engines used for each release stage.

        Any argument left unspecified retains the previously configured engine.
        Passing ``None`` explicitly clears the engine for that stage.
        """

        updated = False

        if paper_engine is not _ENGINE_UNSET and paper_engine is not self.paper_engine:
            self.paper_engine = paper_engine
            updated = True

        if pilot_engine is not _ENGINE_UNSET and pilot_engine is not self.pilot_engine:
            self.pilot_engine = pilot_engine
            updated = True

        if live_engine is not _ENGINE_UNSET and live_engine is not self.live_engine:
            self.live_engine = live_engine
            updated = True

        if updated:
            self._propagate_risk_context_provider()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_stage(
        self, policy_id: str | None
    ) -> tuple[PolicyLedgerStage, Mapping[str, Any] | None]:
        if not policy_id:
            return self.default_stage, None

        posture: Mapping[str, Any] | None = None
        describe = getattr(self.release_manager, "describe", None)
        if callable(describe):
            try:
                candidate = describe(policy_id)
            except Exception:  # pragma: no cover - diagnostics only
                logger.debug(
                    "Failed to describe release posture for %s", policy_id, exc_info=True
                )
            else:
                if isinstance(candidate, Mapping) and candidate.get("stage"):
                    posture = candidate
                    try:
                        stage_value = PolicyLedgerStage.from_value(candidate["stage"])
                    except Exception:
                        posture = None
                    else:
                        return stage_value, posture

        try:
            stage = self.release_manager.resolve_stage(policy_id)
        except Exception as exc:  # pragma: no cover - defensive logging guard
            logger.warning("Failed to resolve release stage for %s: %s", policy_id, exc)
            return self.default_stage, posture
        return (stage or self.default_stage), posture

    @staticmethod
    def _audit_enforcement(
        stage: PolicyLedgerStage,
        posture: Mapping[str, Any] | None,
    ) -> tuple[bool, str | None, Mapping[str, Any] | None]:
        if not posture:
            return False, None, None

        audit_details: dict[str, Any] = {}
        declared_raw = posture.get("declared_stage")
        audit_stage_raw = posture.get("audit_stage")
        audit_gaps_raw = posture.get("audit_gaps")
        audit_enforced_flag = bool(posture.get("audit_enforced"))

        declared_stage: PolicyLedgerStage | None = None
        if declared_raw:
            try:
                declared_stage = PolicyLedgerStage.from_value(declared_raw)
            except Exception:
                declared_stage = None
            else:
                audit_details["declared_stage"] = declared_stage.value

        audit_stage: PolicyLedgerStage | None = None
        if audit_stage_raw:
            try:
                audit_stage = PolicyLedgerStage.from_value(audit_stage_raw)
            except Exception:
                audit_stage = None
            else:
                audit_details["audit_stage"] = audit_stage.value

        audit_gaps: list[str] = []
        if isinstance(audit_gaps_raw, (list, tuple)):
            audit_gaps = [str(gap) for gap in audit_gaps_raw if gap]
            if audit_gaps:
                audit_details["gaps"] = audit_gaps

        if declared_stage is not None and declared_stage is not stage:
            audit_enforced_flag = True

        if audit_stage is not None and audit_stage is not stage:
            audit_enforced_flag = True

        if audit_enforced_flag:
            audit_details["enforced"] = True

        if posture.get("updated_at") and "updated_at" not in audit_details:
            audit_details["updated_at"] = posture["updated_at"]

        audit_force_paper = audit_enforced_flag and stage in (
            PolicyLedgerStage.EXPERIMENT,
            PolicyLedgerStage.PAPER,
        )

        audit_reason: str | None = None
        if audit_force_paper:
            if audit_gaps:
                audit_reason = f"release_audit_gap_{audit_gaps[0]}"
            else:
                audit_reason = "release_audit_enforced"

        return audit_force_paper, audit_reason, audit_details or None

    def _select_engine(
        self,
        stage: PolicyLedgerStage,
        *,
        force_paper: bool = False,
    ) -> tuple[Any, str]:
        if force_paper:
            return self.paper_engine, "paper"
        if stage is PolicyLedgerStage.LIMITED_LIVE:
            if self.live_engine is not None:
                return self.live_engine, "live"
            if self.pilot_engine is not None:
                return self.pilot_engine, "pilot"
            return self.paper_engine, "paper"
        if stage is PolicyLedgerStage.PILOT:
            if self.pilot_engine is not None:
                return self.pilot_engine, "pilot"
            if self.live_engine is not None:
                return self.live_engine, "live"
            return self.paper_engine, "paper"
        if stage is PolicyLedgerStage.PAPER:
            return self.paper_engine, "paper"
        # Default: experiment or unknown stages fall back to paper execution.
        return self.paper_engine, "paper"

    @staticmethod
    def _extract_policy_id(intent: Any) -> str | None:
        if isinstance(intent, Mapping):
            if "policy_id" in intent:
                return str(intent["policy_id"])
            if "strategy_id" in intent:
                return str(intent["strategy_id"])
        if hasattr(intent, "policy_id"):
            value = getattr(intent, "policy_id")
            if value:
                return str(value)
        if hasattr(intent, "strategy_id"):
            value = getattr(intent, "strategy_id")
            if value:
                return str(value)
        metadata = None
        if isinstance(intent, Mapping):
            metadata = intent.get("metadata")
        elif hasattr(intent, "metadata"):
            metadata = getattr(intent, "metadata")
        if isinstance(metadata, Mapping):
            policy = metadata.get("policy_id") or metadata.get("strategy_id")
            if policy:
                return str(policy)
        return None

    @staticmethod
    def _engine_name(engine: Any) -> str:
        return getattr(engine, "name", engine.__class__.__name__)

    @staticmethod
    def _extract_metadata(intent: Any) -> Mapping[str, Any] | None:
        if isinstance(intent, Mapping):
            candidate = intent.get("metadata")
        else:
            candidate = getattr(intent, "metadata", None)
        if isinstance(candidate, Mapping):
            return candidate
        return None

    def _should_force_paper(
        self,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[bool, str | None, DriftSeverity | None]:
        if not metadata:
            return False, None, None
        gate_payload = metadata.get("drift_gate")
        if not isinstance(gate_payload, Mapping):
            return False, None, None

        allowed = gate_payload.get("allowed")
        severity_value = gate_payload.get("severity")
        severity: DriftSeverity | None = None
        if severity_value is not None:
            try:
                severity = DriftSeverity(str(severity_value).lower())
            except Exception:
                severity = None

        if allowed is False:
            forced_reason = gate_payload.get("reason") or "drift_gate_blocked"
            return True, forced_reason, severity

        force_flag = gate_payload.get("force_paper")
        if isinstance(force_flag, bool) and force_flag:
            forced_reason = gate_payload.get("reason")
            if not forced_reason:
                requirements = gate_payload.get("requirements")
                if isinstance(requirements, Mapping):
                    candidate = requirements.get("release_stage_gate")
                    if isinstance(candidate, str) and candidate:
                        forced_reason = candidate
            if not forced_reason and severity is not None and severity is not DriftSeverity.normal:
                forced_reason = f"drift_gate_severity_{severity.value}"
            if not forced_reason:
                forced_reason = "drift_gate_force_paper"
            return True, forced_reason, severity

        if severity is None or severity is DriftSeverity.normal:
            return False, None, severity

        reason = f"drift_gate_severity_{severity.value}"
        return True, reason, severity

    @staticmethod
    def _stage_force_reason(stage: PolicyLedgerStage) -> str | None:
        if stage is PolicyLedgerStage.EXPERIMENT:
            return "release_stage_experiment_requires_paper_or_better"
        if stage is PolicyLedgerStage.PAPER:
            return "release_stage_paper_requires_paper_execution"
        return None

    @staticmethod
    def _attach_metadata(
        intent: Any,
        stage: PolicyLedgerStage,
        route_label: str,
        *,
        forced_reason: str | None = None,
        forced_reasons: Sequence[str] | None = None,
        audit_details: Mapping[str, Any] | None = None,
    ) -> None:
        metadata: MutableMapping[str, Any] | None = None
        created_metadata = False
        if isinstance(intent, MutableMapping):
            meta_value = intent.get("metadata")
            if isinstance(meta_value, MutableMapping):
                metadata = meta_value
            else:
                metadata = {}
                intent["metadata"] = metadata
                created_metadata = True
        else:
            meta_value = getattr(intent, "metadata", None)
            if isinstance(meta_value, MutableMapping):
                metadata = meta_value
            else:
                if meta_value is None:
                    metadata = {}
                elif isinstance(meta_value, Mapping):
                    metadata = dict(meta_value)
                else:
                    metadata = {"original_metadata": meta_value}
                setattr(intent, "metadata", metadata)
                created_metadata = True
        if metadata is None:
            return
        metadata["release_stage"] = stage.value
        metadata["release_execution_route"] = route_label
        override_flag = False
        reasons_list: list[str] = []
        if forced_reasons:
            reasons_list.extend(str(reason) for reason in forced_reasons if reason)

        if forced_reason:
            metadata["release_execution_forced"] = forced_reason
            override_flag = True
        elif reasons_list:
            metadata["release_execution_forced"] = reasons_list[0]
            override_flag = True
        else:
            metadata.pop("release_execution_forced", None)

        if reasons_list:
            metadata["release_execution_forced_reasons"] = list(
                dict.fromkeys(reasons_list)
            )
        else:
            metadata.pop("release_execution_forced_reasons", None)

        if override_flag:
            metadata.setdefault("release_execution_route_overridden", True)
        else:
            metadata.pop("release_execution_route_overridden", None)

        if audit_details:
            metadata["release_execution_audit"] = dict(audit_details)

        # When we materialise metadata for previously missing payloads, ensure
        # we still surface that a forced route was applied.
        if created_metadata and (forced_reason or reasons_list):
            metadata.setdefault("release_execution_route_overridden", True)
