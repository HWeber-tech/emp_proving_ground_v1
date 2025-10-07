"""Release-aware execution router bridging policy ledger stages to engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Mapping, MutableMapping

from src.governance.policy_ledger import LedgerReleaseManager, PolicyLedgerStage
from src.operations.sensory_drift import DriftSeverity

__all__ = ["ReleaseAwareExecutionRouter"]

logger = logging.getLogger(__name__)


@dataclass
class ReleaseAwareExecutionRouter:
    """Route validated intents to the appropriate execution engine per release stage."""

    release_manager: LedgerReleaseManager
    paper_engine: Any
    pilot_engine: Any | None = None
    live_engine: Any | None = None
    default_stage: PolicyLedgerStage = PolicyLedgerStage.EXPERIMENT
    _last_route: MutableMapping[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.release_manager is None:
            raise ValueError("release_manager is required")
        if self.paper_engine is None:
            raise ValueError("paper_engine is required")
        if not isinstance(self.default_stage, PolicyLedgerStage):
            self.default_stage = PolicyLedgerStage.from_value(self.default_stage)

    async def process_order(self, intent: Any) -> Any:
        """Process an order using the engine that matches the ledger stage."""

        metadata = self._extract_metadata(intent)
        strategy_id = self._extract_policy_id(intent)
        stage = self._resolve_stage(strategy_id)
        force_paper, forced_reason, forced_severity = self._should_force_paper(metadata)
        engine, route_label = self._select_engine(stage, force_paper=force_paper)
        self._attach_metadata(
            intent,
            stage,
            route_label,
            forced_reason=forced_reason,
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
        if force_paper:
            self._last_route["forced_route"] = "paper"
            if forced_reason:
                self._last_route["forced_reason"] = forced_reason
            if forced_severity:
                self._last_route["drift_severity"] = forced_severity.value
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
        return payload

    def last_route(self) -> Mapping[str, Any] | None:
        """Expose the most recent routing decision."""

        if not self._last_route:
            return None
        return dict(self._last_route)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_stage(self, policy_id: str | None) -> PolicyLedgerStage:
        if not policy_id:
            return self.default_stage
        try:
            stage = self.release_manager.resolve_stage(policy_id)
        except Exception as exc:  # pragma: no cover - defensive logging guard
            logger.warning("Failed to resolve release stage for %s: %s", policy_id, exc)
            return self.default_stage
        return stage or self.default_stage

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
            return True, "drift_gate_blocked", severity

        if severity is None or severity is DriftSeverity.normal:
            return False, None, severity

        reason = f"drift_gate_severity_{severity.value}"
        return True, reason, severity

    @staticmethod
    def _attach_metadata(
        intent: Any,
        stage: PolicyLedgerStage,
        route_label: str,
        *,
        forced_reason: str | None = None,
    ) -> None:
        metadata: MutableMapping[str, Any] | None = None
        if isinstance(intent, MutableMapping):
            meta_value = intent.get("metadata")
            if isinstance(meta_value, MutableMapping):
                metadata = meta_value
        else:
            meta_value = getattr(intent, "metadata", None)
            if isinstance(meta_value, MutableMapping):
                metadata = meta_value
        if metadata is None:
            return
        metadata["release_stage"] = stage.value
        metadata["release_execution_route"] = route_label
        if forced_reason:
            metadata["release_execution_forced"] = forced_reason
            metadata.setdefault("release_execution_route_overridden", True)
