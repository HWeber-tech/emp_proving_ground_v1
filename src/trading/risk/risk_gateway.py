"""Risk gateway coordinating multi-layer trade validation.

This module gives the trading runtime a concrete implementation of the
"risk gateway" promised in the encyclopaedia.  It fuses portfolio context,
strategy governance, capital-at-risk guards, and optional liquidity probing
before an order is allowed to reach execution.

The implementation favours defensive programming so that it can work with
the heterogeneous TradeIntent representations that already exist across the
codebase (dict-based legacy payloads, dataclasses, and Pydantic-style
objects).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    cast,
)

from src.config.risk.risk_config import RiskConfig
from src.data_foundation.config.execution_config import (
    ExecutionConfig,
    ExecutionRiskLimits,
    load_execution_config,
)
from src.core.event_bus import EventBus, Event
from src.trading.execution.execution_model import (
    ExecContext,
    calculate_edge_ticks,
    estimate_commission_bps,
    estimate_slippage_bps,
    estimate_total_cost_ticks,
)
from .policy_telemetry import (
    RiskPolicyEvaluationSnapshot,
    build_policy_snapshot,
    build_policy_violation_alert,
    format_policy_violation_markdown,
    publish_policy_violation,
    RISK_POLICY_VIOLATION_RUNBOOK,
)
from .risk_api import RISK_API_RUNBOOK, merge_risk_references, summarise_risk_config
from .risk_policy import RiskPolicy, RiskPolicyDecision
from .guardrail_incidents import GuardrailIncident, extract_guardrail_incident

try:  # pragma: no cover - metrics optional in certain runtimes
    from src.operational import metrics as operational_metrics
except Exception:  # pragma: no cover
    operational_metrics = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


_EPSILON = 1e-9
_SECTOR_KEYS = ("sector", "instrument_sector", "asset_sector", "industry")


class SupportsLiquidityProbing(Protocol):
    """Protocol for optional liquidity probing capabilities."""

    async def probe_liquidity(
        self, symbol: str, price_levels: list[float], side: str
    ) -> Mapping[float, float]: ...

    def calculate_liquidity_confidence_score(
        self, probe_results: Mapping[float, float], intended_volume: float
    ) -> float: ...

    def get_probe_summary(self, probe_results: Mapping[float, float]) -> Mapping[str, Any]: ...


class SupportsStrategyRegistry(Protocol):
    """Protocol capturing the single method RiskGateway relies on."""

    def get_strategy(self, strategy_id: str) -> Mapping[str, Any] | None: ...


if TYPE_CHECKING:
    from src.risk.real_risk_manager import RealRiskManager


class SupportsPortfolioRisk(Protocol):
    """Protocol describing the :class:`RealRiskManager` surface we rely on."""

    def update_equity(self, equity: float | Decimal) -> None: ...

    def assess_risk(self, positions: Mapping[str, float]) -> float: ...

    @property
    def last_snapshot(self) -> Mapping[str, float]: ...

PositionSizer = Callable[[Decimal, Decimal, Decimal], Decimal]


def _to_decimal(value: Any, *, default: Decimal = Decimal("0")) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _ensure_metadata(intent: Any) -> MutableMapping[str, Any]:
    """Return a mutable metadata mapping for ``intent`` and attach one if needed."""

    if isinstance(intent, MutableMapping):
        existing_meta = intent.get("metadata")
        if isinstance(existing_meta, MutableMapping):
            return existing_meta
        meta_map: dict[str, Any] = {}
        intent["metadata"] = meta_map
        return meta_map

    meta_attr = getattr(intent, "metadata", None)
    if isinstance(meta_attr, MutableMapping):
        return meta_attr

    # Attach a new metadata mapping for objects lacking one
    new_meta: dict[str, Any] = {}
    try:
        setattr(intent, "metadata", new_meta)
    except Exception as exc:  # pragma: no cover - best effort when attribute blocked
        logger.debug("Unable to attach metadata attribute to intent %r", intent, exc_info=exc)
    return new_meta


def _extract_attr(intent: Any, *names: str, default: Any = None) -> Any:
    """Return the first attribute/key found on ``intent`` from ``names``."""

    if isinstance(intent, Mapping):
        for name in names:
            if name in intent:
                return intent[name]
        return default

    for name in names:
        if hasattr(intent, name):
            return getattr(intent, name)
    return default


def _set_attr(intent: Any, value: Any, *names: str) -> None:
    for name in names:
        if isinstance(intent, MutableMapping) and name in intent:
            intent[name] = value
            return
        if hasattr(intent, name):
            try:
                setattr(intent, name, value)
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(
                    "Failed to set attribute %s on intent %r", name, intent, exc_info=exc
                )
            return

    # Attribute was missing entirely; fall back to first candidate for dicts
    if isinstance(intent, MutableMapping) and names:
        intent[names[0]] = value


class RiskGateway:
    """Central risk gate that validates intents before execution."""

    def __init__(
        self,
        strategy_registry: SupportsStrategyRegistry | None,
        position_sizer: PositionSizer | None,
        portfolio_monitor: Any,
        *,
        risk_per_trade: Decimal | float = Decimal("0.01"),
        max_open_positions: int = 5,
        max_daily_drawdown: float = 0.05,
        min_intent_confidence: float = 0.2,
        stop_loss_floor: float = 0.005,
        liquidity_prober: SupportsLiquidityProbing | None = None,
        liquidity_probe_threshold: float = 1.0,
        min_liquidity_confidence: float = 0.3,
        risk_policy: RiskPolicy | None = None,
        execution_config: ExecutionConfig | None = None,
        portfolio_risk_manager: SupportsPortfolioRisk | None = None,
        risk_config: RiskConfig | None = None,
        event_bus: EventBus | None = None,
        policy_violation_runbook: str | None = None,
        liquidity_capacity_ratio: float = 0.02,
        liquidity_capacity_percentile: float = 50.0,
    ) -> None:
        self.strategy_registry = strategy_registry
        self.position_sizer = position_sizer
        self.portfolio_monitor = portfolio_monitor
        self.risk_per_trade = _to_decimal(risk_per_trade, default=Decimal("0.01"))
        self.max_open_positions = int(max_open_positions)
        self.max_daily_drawdown = float(max_daily_drawdown)
        self.min_intent_confidence = float(min_intent_confidence)
        self.stop_loss_floor = float(stop_loss_floor)
        self.liquidity_prober = liquidity_prober
        self.liquidity_probe_threshold = float(liquidity_probe_threshold)
        self.min_liquidity_confidence = float(min_liquidity_confidence)
        self.risk_policy = risk_policy
        self._execution_config = execution_config or load_execution_config()
        self.portfolio_risk_manager = portfolio_risk_manager
        self._risk_per_trade_float = float(self.risk_per_trade)

        ratio_value = max(0.0, float(liquidity_capacity_ratio))
        self._liquidity_capacity_ratio = ratio_value
        self._liquidity_capacity_ratio_decimal = (
            Decimal(str(ratio_value)) if ratio_value > 0.0 else Decimal("0")
        )
        self._liquidity_capacity_percentile = max(
            0.0, min(100.0, float(liquidity_capacity_percentile))
        )
        self._liquidity_capacity_epsilon = Decimal("1e-9")

        self.telemetry: dict[str, Any] = {
            "total_checks": 0,
            "approved": 0,
            "rejected": 0,
            "last_decision": None,
        }
        self.telemetry.update(
            {
                "guardrail_near_misses": 0,
                "guardrail_violations": 0,
                "last_guardrail_incident": None,
            }
        )
        self._last_policy_decision: RiskPolicyDecision | None = None
        self._last_policy_snapshot: RiskPolicyEvaluationSnapshot | None = None
        self._risk_config: RiskConfig | None = risk_config
        self._risk_reference_cache: dict[str, object] | None = None
        self.event_bus = event_bus
        self._policy_violation_runbook = (
            policy_violation_runbook or RISK_POLICY_VIOLATION_RUNBOOK
        )
        self._last_guardrail_incident: GuardrailIncident | None = None

        self._confidence_limit_pct: float = 1.0
        self._max_leverage_limit: float = 0.0
        self._instrument_sector_map: dict[str, str] = {}
        self._sector_limits: dict[str, float] = {}
        self._initialise_policy_limits(self._risk_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def validate_trade_intent(
        self, intent: Any, portfolio_state: Mapping[str, Any] | None
    ) -> Any | None:
        """Return an enriched intent if all risk checks pass, otherwise ``None``."""

        self.telemetry["total_checks"] += 1
        self._last_policy_decision = None
        self._last_policy_snapshot = None
        decision: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": _extract_attr(intent, "symbol", default="UNKNOWN"),
            "strategy_id": _extract_attr(intent, "strategy_id", "strategy", default=""),
            "checks": [],
        }

        limits_snapshot = self._build_limits_snapshot()
        risk_reference = self._resolve_risk_reference()
        if risk_reference is None:
            reference_payload: dict[str, Any] = {"risk_api_runbook": RISK_API_RUNBOOK}
        else:
            reference_payload = merge_risk_references(risk_reference)
        reference_payload.setdefault("limits", dict(limits_snapshot))
        decision["risk_reference"] = reference_payload

        try:
            if not self._check_strategy(decision["strategy_id"]):
                decision.update(status="rejected", reason="strategy_disabled")
                await self._reject_and_maybe_publish(decision)
                return None

            quantity = self._extract_quantity(intent)
            confidence = _as_float(
                _extract_attr(intent, "confidence", "score", default=1.0), default=1.0
            )
            if confidence < self.min_intent_confidence:
                decision.update(status="rejected", reason="low_confidence")
                decision["checks"].append(
                    {
                        "name": "confidence_floor",
                        "value": confidence,
                        "threshold": self.min_intent_confidence,
                    }
                )
                await self._reject_and_maybe_publish(decision)
                return None

            portfolio_state = portfolio_state or {}
            if not self._check_synthetic_invariant(portfolio_state, decision):
                decision.update(status="rejected", reason="synthetic_invariant_breach")
                await self._reject_and_maybe_publish(decision)
                return None
            if not self._check_drawdown(portfolio_state, decision):
                decision.update(status="rejected", reason="max_drawdown_exceeded")
                await self._reject_and_maybe_publish(decision)
                return None

            if not self._check_open_positions(portfolio_state, decision):
                decision.update(status="rejected", reason="too_many_open_positions")
                await self._reject_and_maybe_publish(decision)
                return None

            adjusted_quantity = self._apply_position_sizing(intent, portfolio_state, decision)

            market_price = self._extract_price(intent, portfolio_state)
            stop_loss_pct = self._extract_stop_loss_pct(intent, portfolio_state, market_price)

            policy_metadata: Mapping[str, Any] | None = None
            if self.risk_policy is not None:
                policy_decision: RiskPolicyDecision = self.risk_policy.evaluate(
                    symbol=decision["symbol"],
                    quantity=float(adjusted_quantity),
                    price=market_price,
                    stop_loss_pct=stop_loss_pct,
                    portfolio_state=portfolio_state,
                )
                self._last_policy_decision = policy_decision
                try:
                    self._last_policy_snapshot = build_policy_snapshot(
                        policy_decision, self.risk_policy
                    )
                except Exception:  # pragma: no cover - defensive telemetry path
                    logger.debug("Failed to build policy snapshot", exc_info=True)
                    self._last_policy_snapshot = None
                decision["checks"].extend(policy_decision.checks)
                policy_metadata = policy_decision.metadata
                decision["policy"] = {
                    "approved": policy_decision.approved,
                    "reason": policy_decision.reason,
                    "metadata": dict(policy_decision.metadata),
                    "violations": list(policy_decision.violations),
                }
                if not policy_decision.approved:
                    decision.update(
                        status="rejected",
                        reason=policy_decision.reason or "policy_violation",
                    )
                    await self._reject_and_maybe_publish(decision)
                    return None

            symbol_value = str(decision.get("symbol") or "")
            equity = self._resolve_equity_for_risk(portfolio_state)
            trade_price = self._resolve_trade_price(
                market_price,
                portfolio_state,
                symbol_value,
            )
            side_value = str(_extract_attr(intent, "side", "direction", default="BUY"))

            if not self._enforce_confidence_limit(
                decision,
                confidence=confidence,
                quantity=float(adjusted_quantity),
                trade_price=trade_price,
                equity=equity,
            ):
                decision.update(status="rejected", reason="confidence_notional_limit")
                await self._reject_and_maybe_publish(decision)
                return None

            if not self._enforce_sector_limits(
                decision,
                portfolio_state,
                symbol=symbol_value,
                quantity=float(adjusted_quantity),
                trade_price=trade_price,
                equity=equity,
                side=side_value,
                intent=intent,
            ):
                decision.update(status="rejected", reason="sector_exposure_limit")
                await self._reject_and_maybe_publish(decision)
                return None

            if not self._enforce_leverage_limit(
                decision,
                portfolio_state,
                quantity=float(adjusted_quantity),
                trade_price=trade_price,
                equity=equity,
            ):
                decision.update(status="rejected", reason="leverage_limit")
                await self._reject_and_maybe_publish(decision)
                return None

            if not self._enforce_portfolio_risk(
                decision,
                portfolio_state,
                symbol=str(decision["symbol"]),
                quantity=float(adjusted_quantity),
                market_price=market_price,
                stop_loss_pct=stop_loss_pct,
            ):
                await self._reject_and_maybe_publish(decision)
                return None

            execution_assessment = self._evaluate_execution_cost(
                intent,
                float(adjusted_quantity),
                market_price,
                portfolio_state,
            )
            decision["checks"].extend(execution_assessment["checks"])
            if execution_assessment["metadata"]:
                decision.setdefault("execution", {}).update(
                    execution_assessment["metadata"]
                )
            if execution_assessment["breaches"]:
                decision.update(
                    status="rejected",
                    reason=execution_assessment["reason"],
                )
                await self._reject_and_maybe_publish(decision)
                return None

            liquidity_summary: Mapping[str, Any] | None = None
            liquidity_score: float | None = None
            if self.liquidity_prober and adjusted_quantity >= self.liquidity_probe_threshold:
                liquidity_score, liquidity_summary = await self._run_liquidity_probe(
                    intent,
                    adjusted_quantity,
                    decision,
                    market_price=market_price,
                    portfolio_state=portfolio_state,
                )
                if liquidity_score is not None and liquidity_score < self.min_liquidity_confidence:
                    decision.update(status="rejected", reason="insufficient_liquidity")
                    await self._reject_and_maybe_publish(decision)
                    return None

            adjusted_quantity = self._apply_liquidity_capacity_guard(
                intent,
                adjusted_quantity,
                decision,
                portfolio_state=portfolio_state,
            )

            self._enrich_intent(
                intent,
                adjusted_quantity,
                liquidity_score,
                decision,
                policy_metadata=policy_metadata,
            )

            if liquidity_summary:
                decision["checks"].append(
                    {
                        "name": "liquidity_probe",
                        "summary": dict(liquidity_summary),
                    }
                )

            decision.update(status="approved", quantity=float(adjusted_quantity))
            decision_payload = self._augment_with_risk_reference(decision)
            self.telemetry["approved"] += 1
            self.telemetry["last_decision"] = decision_payload
            incident = self._record_guardrail_incident(decision_payload)
            await self._maybe_publish_guardrail_incident(incident)
            return intent

        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("RiskGateway encountered unexpected error: %s", exc)
            decision.update(status="rejected", reason="exception", error=str(exc))
            await self._reject_and_maybe_publish(decision)
            return None

    def get_risk_limits(self) -> dict[str, Any]:
        """Return a snapshot of configured limits and recent telemetry."""

        limits = self._build_limits_snapshot()
        payload: dict[str, Any] = {"limits": dict(limits), "telemetry": dict(self.telemetry)}
        risk_reference = self._resolve_risk_reference()
        if risk_reference is not None:
            payload["risk_reference"] = risk_reference
            summary = risk_reference.get("risk_config_summary")
            if isinstance(summary, Mapping):
                payload["risk_config_summary"] = dict(summary)
            runbook = risk_reference.get("risk_api_runbook")
            if isinstance(runbook, str) and runbook:
                payload["runbook"] = runbook
            config_payload = risk_reference.get("risk_config")
            if isinstance(config_payload, Mapping):
                payload["risk_config"] = dict(config_payload)
        if "runbook" not in payload:
            payload["runbook"] = RISK_API_RUNBOOK
        if "risk_reference" not in payload:
            payload["risk_reference"] = {"risk_api_runbook": RISK_API_RUNBOOK}
        return payload

    def get_last_decision(self) -> Mapping[str, Any] | None:
        """Expose the most recent risk decision record."""

        return cast(Optional[Mapping[str, Any]], self.telemetry.get("last_decision"))

    def get_last_policy_decision(self) -> RiskPolicyDecision | None:
        """Return the last evaluated :class:`RiskPolicyDecision`, if available."""

        return self._last_policy_decision

    def get_last_policy_snapshot(self) -> RiskPolicyEvaluationSnapshot | None:
        """Expose the last :class:`RiskPolicyEvaluationSnapshot`, if available."""

        return self._last_policy_snapshot

    def get_last_guardrail_incident(self) -> GuardrailIncident | None:
        """Return the most recent guardrail incident, if one was recorded."""

        return self._last_guardrail_incident

    def apply_risk_config(
        self,
        config: RiskConfig,
        *,
        risk_policy: RiskPolicy | None = None,
    ) -> None:
        """Refresh the gateway with ``config`` and optional ``risk_policy``."""

        if not isinstance(config, RiskConfig):
            raise TypeError("config must be an instance of RiskConfig")

        self._risk_config = config
        self._risk_reference_cache = None

        self.risk_per_trade = _to_decimal(
            config.max_risk_per_trade_pct, default=self.risk_per_trade
        )
        self._risk_per_trade_float = float(self.risk_per_trade)
        self.max_daily_drawdown = float(config.max_drawdown_pct)

        if risk_policy is not None:
            self.risk_policy = risk_policy
        elif self.risk_policy is not None:
            self.risk_policy = RiskPolicy.from_config(config)

        self._initialise_policy_limits(config)

    def _build_limits_snapshot(self) -> dict[str, Any]:
        """Construct the current limits payload exposed by the gateway."""

        limits = {
            "max_open_positions": self.max_open_positions,
            "max_daily_drawdown": self.max_daily_drawdown,
            "min_intent_confidence": self.min_intent_confidence,
            "liquidity_probe_threshold": self.liquidity_probe_threshold,
            "min_liquidity_confidence": self.min_liquidity_confidence,
            "risk_per_trade": float(self.risk_per_trade),
        }
        if self.risk_policy is not None:
            limits.update(self.risk_policy.limit_snapshot())
            if self.risk_policy.research_mode:
                limits["research_mode"] = True
        return limits

    def _initialise_policy_limits(self, risk_config: RiskConfig | None) -> None:
        """Initialise deterministic risk limit caches from ``risk_config``."""

        confidence_limit = 1.0
        leverage_limit = 0.0
        instrument_map: dict[str, str] = {}
        sector_limits: dict[str, float] = {}

        if risk_config is not None:
            try:
                confidence_limit = max(0.0, float(risk_config.max_total_exposure_pct))
            except Exception:
                confidence_limit = 1.0
            try:
                leverage_limit = max(0.0, float(risk_config.max_leverage))
            except Exception:
                leverage_limit = 0.0

            raw_map = getattr(risk_config, "instrument_sector_map", {}) or {}
            for symbol, sector in raw_map.items():
                if not symbol or not sector:
                    continue
                instrument_map[str(symbol).upper()] = str(sector).upper()

            raw_limits = getattr(risk_config, "sector_exposure_limits", {}) or {}
            for sector, limit in raw_limits.items():
                try:
                    resolved = float(limit)
                except Exception:
                    continue
                if resolved > 0:
                    sector_limits[str(sector).upper()] = resolved
        elif self.risk_policy is not None:
            confidence_limit = max(0.0, float(self.risk_policy.max_total_exposure_pct))
            leverage_limit = max(0.0, float(self.risk_policy.max_leverage))

        self._confidence_limit_pct = confidence_limit or 1.0
        self._max_leverage_limit = leverage_limit
        self._instrument_sector_map = instrument_map
        self._sector_limits = sector_limits

    @staticmethod
    def _normalise_sector(value: Any) -> str | None:
        """Return an upper-cased sector string when ``value`` is meaningful."""

        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return text.upper()

    def _resolve_sector_hint(
        self,
        symbol: str | None = None,
        *,
        payload: Mapping[str, Any] | None = None,
        intent: Any | None = None,
    ) -> str | None:
        """Resolve the sector for ``symbol`` using config, payload, or intent hints."""

        if symbol:
            mapped = self._instrument_sector_map.get(str(symbol).upper())
            if mapped:
                return mapped

        if intent is not None:
            direct = _extract_attr(
                intent,
                "sector",
                "instrument_sector",
                "asset_sector",
                "industry",
                default=None,
            )
            resolved = self._normalise_sector(direct)
            if resolved:
                return resolved

            intent_meta: Mapping[str, Any] | None = None
            if isinstance(intent, Mapping):
                raw_meta = intent.get("metadata")
                if isinstance(raw_meta, Mapping):
                    intent_meta = raw_meta
            else:
                raw_meta = getattr(intent, "metadata", None)
                if isinstance(raw_meta, Mapping):
                    intent_meta = raw_meta

            for meta in self._iter_sector_metadata(intent_meta):
                candidate = self._extract_sector_from_mapping(meta)
                if candidate:
                    return candidate

        for meta in self._iter_sector_metadata(payload):
            candidate = self._extract_sector_from_mapping(meta)
            if candidate:
                return candidate

        return None

    def _extract_sector_from_mapping(self, mapping: Mapping[str, Any]) -> str | None:
        for key in _SECTOR_KEYS:
            value = mapping.get(key)
            resolved = self._normalise_sector(value)
            if resolved:
                return resolved

        instrument_meta = mapping.get("instrument")
        if isinstance(instrument_meta, Mapping):
            for key in _SECTOR_KEYS:
                value = instrument_meta.get(key)
                resolved = self._normalise_sector(value)
                if resolved:
                    return resolved
        return None

    @staticmethod
    def _iter_sector_metadata(
        mapping: Mapping[str, Any] | None,
    ) -> Iterable[Mapping[str, Any]]:
        if not isinstance(mapping, Mapping):
            return ()

        stack: list[Mapping[str, Any]] = [mapping]
        visited: set[int] = set()

        while stack:
            current = stack.pop()
            ident = id(current)
            if ident in visited:
                continue
            visited.add(ident)
            yield current

            nested = current.get("metadata")
            if isinstance(nested, Mapping):
                stack.append(nested)

            risk_section = current.get("risk_assessment")
            if isinstance(risk_section, Mapping):
                stack.append(risk_section)

            instrument_meta = current.get("instrument")
            if isinstance(instrument_meta, Mapping):
                stack.append(instrument_meta)

    def _resolve_risk_reference(self) -> dict[str, Any] | None:
        """Build a deterministic risk reference payload for telemetry surfaces."""

        if self._risk_config is None and self.risk_policy is None:
            return None
        return self._risk_reference_base()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reject(
        self, decision: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], GuardrailIncident | None]:
        decision_payload = self._augment_with_risk_reference(decision)
        self.telemetry["rejected"] += 1
        self.telemetry["last_decision"] = decision_payload
        reason = str(decision_payload.get("reason") or "")
        symbol = str(decision_payload.get("symbol") or "")
        if operational_metrics is not None:
            safe_reason = reason or "unknown"
            try:
                operational_metrics.inc_pretrade_denial(symbol, safe_reason)
            except Exception:  # pragma: no cover - metrics layer optional
                logger.debug("Failed to emit pre-trade denial metric", exc_info=True)
        logger.info("RiskGateway rejected trade: %s", decision_payload)
        incident = self._record_guardrail_incident(decision_payload)
        return decision_payload, incident

    async def _reject_and_maybe_publish(self, decision: Mapping[str, Any]) -> None:
        """Record a rejection and emit policy telemetry when applicable."""

        _, incident = self._reject(decision)
        await self._maybe_publish_policy_violation()
        await self._maybe_publish_guardrail_incident(incident)

    async def _maybe_publish_policy_violation(self) -> None:
        """Publish a policy violation alert when the last snapshot failed."""

        event_bus = self.event_bus
        if event_bus is None:
            return

        snapshot = self._last_policy_snapshot
        if snapshot is None:
            return

        if snapshot.approved and not snapshot.violations:
            return

        try:
            severity = "critical" if not snapshot.approved else "warning"
            alert = build_policy_violation_alert(
                snapshot,
                severity=severity,
                runbook=self._policy_violation_runbook,
            )
        except Exception:  # pragma: no cover - defensive telemetry path
            logger.debug("Failed to build policy violation alert", exc_info=True)
            return

        try:
            logger.warning(
                "🚨 Policy violation alert\n%s",
                format_policy_violation_markdown(alert),
            )
        except Exception:  # pragma: no cover - formatting guardrail
            logger.debug("Failed to format policy violation alert", exc_info=True)

        try:
            await publish_policy_violation(
                event_bus,
                alert,
                source="risk_gateway",
            )
        except Exception:  # pragma: no cover - event bus optional/diagnostic
            logger.debug("Failed to publish policy violation telemetry", exc_info=True)

    def _record_guardrail_incident(
        self, decision_payload: Mapping[str, Any]
    ) -> GuardrailIncident | None:
        incident = extract_guardrail_incident(decision_payload)
        if incident is None:
            return None

        key = "guardrail_violations" if incident.severity == "violation" else "guardrail_near_misses"
        try:
            current = int(self.telemetry.get(key, 0) or 0)
        except (TypeError, ValueError):
            current = 0
        self.telemetry[key] = current + 1

        incident_payload = incident.as_dict()
        self.telemetry["last_guardrail_incident"] = incident_payload
        self._last_guardrail_incident = incident

        log_extra = {"incident": incident_payload}
        if incident.severity == "violation":
            logger.error("Risk guardrail violation captured", extra=log_extra)
        else:
            logger.warning("Risk guardrail near miss captured", extra=log_extra)
        return incident

    async def _maybe_publish_guardrail_incident(
        self, incident: GuardrailIncident | None
    ) -> None:
        if incident is None or self.event_bus is None:
            return
        try:
            payload = incident.as_dict()
            event = Event(
                type="telemetry.risk.guardrail",
                payload=payload,
                source="risk_gateway",
            )
            await self.event_bus.publish(event)
        except Exception:  # pragma: no cover - telemetry guards only
            logger.debug("Failed to publish guardrail incident telemetry", exc_info=True)

    def _risk_reference_base(self) -> dict[str, object]:
        if self._risk_reference_cache is None:
            reference: dict[str, object] = {}
            if self._risk_config is not None:
                try:
                    reference["risk_config_summary"] = summarise_risk_config(self._risk_config)
                except Exception:  # pragma: no cover - defensive metadata guard
                    logger.debug(
                        "Failed to summarise risk config for risk reference cache",
                        exc_info=True,
                    )
                else:
                    try:
                        config_payload = self._risk_config.dict()
                    except Exception:  # pragma: no cover - serialisation guard
                        logger.debug(
                            "Failed to serialise risk config for risk reference cache",
                            exc_info=True,
                        )
                    else:
                        if isinstance(config_payload, Mapping):
                            reference["risk_config"] = dict(config_payload)
            if not reference and self.risk_policy is not None:
                reference["risk_config_summary"] = self._policy_summary()
            reference.setdefault("risk_api_runbook", RISK_API_RUNBOOK)
            self._risk_reference_cache = reference
        return merge_risk_references(self._risk_reference_cache)

    def _augment_with_risk_reference(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(decision)
        existing = payload.get("risk_reference")
        base_reference = self._risk_reference_base()
        payload["risk_reference"] = merge_risk_references(existing, base_reference)
        return payload

    def _policy_summary(self) -> dict[str, object]:
        policy = self.risk_policy
        if policy is None:
            return {}
        summary = {
            "max_risk_per_trade_pct": float(policy.max_risk_per_trade_pct),
            "max_total_exposure_pct": float(policy.max_total_exposure_pct),
            "max_leverage": float(policy.max_leverage),
            "max_drawdown_pct": float(policy.max_drawdown_pct),
            "min_position_size": float(policy.min_position_size),
            "max_position_size": float(policy.max_position_size),
            "mandatory_stop_loss": bool(policy.mandatory_stop_loss),
            "research_mode": bool(policy.research_mode),
        }
        return summary

    def _check_strategy(self, strategy_id: str | None) -> bool:
        if not strategy_id or self.strategy_registry is None:
            return True
        try:
            record = self.strategy_registry.get_strategy(strategy_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Strategy registry lookup failed for %s: %s", strategy_id, exc)
            return True

        if not record:
            return True

        status = str(record.get("status", "")).lower()
        return status in {"active", "approved"}

    def _check_drawdown(self, portfolio_state: Mapping[str, Any], decision: dict[str, Any]) -> bool:
        current = _as_float(portfolio_state.get("current_daily_drawdown"), default=0.0)
        decision["checks"].append(
            {
                "name": "daily_drawdown",
                "value": current,
                "threshold": self.max_daily_drawdown,
            }
        )
        return current <= self.max_daily_drawdown

    def _check_synthetic_invariant(
        self,
        portfolio_state: Mapping[str, Any],
        decision: dict[str, Any],
    ) -> bool:
        """Detect synthetic invariant breaches supplied via ``portfolio_state``."""

        metadata = self._resolve_synthetic_invariant_breach(portfolio_state)
        entry: dict[str, Any] = {
            "name": "risk.synthetic_invariant_posture",
            "status": "ok" if metadata is None else "violation",
        }
        if metadata is not None:
            entry["metadata"] = metadata
        decision.setdefault("checks", []).append(entry)
        return metadata is None

    @staticmethod
    def _resolve_synthetic_invariant_breach(
        portfolio_state: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        for key, value in portfolio_state.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if "synthetic" in lowered and "invariant" in lowered:
                metadata = RiskGateway._format_invariant_metadata(key, value)
                if metadata is not None:
                    return metadata

        targeted_indicators = (
            "invariant_breaches",
            "invariant_assessment",
            "metadata",
            "portfolio_metadata",
            "risk_metadata",
            "insights",
            "indicator_payloads",
            "indicator_payload",
            "indicatorPayloads",
            "indicatorPayload",
        )

        for indicator in targeted_indicators:
            payload = portfolio_state.get(indicator)
            metadata = RiskGateway._search_invariant_payload(payload, base=indicator)
            if metadata is not None:
                return metadata

        for key, value in portfolio_state.items():
            if isinstance(key, str) and key in targeted_indicators:
                continue
            base = str(key)
            metadata = RiskGateway._search_invariant_payload(value, base=base)
            if metadata is not None:
                return metadata

        return None

    @staticmethod
    def _search_invariant_payload(payload: object, *, base: str) -> dict[str, Any] | None:
        if payload is None:
            return None

        if isinstance(payload, Mapping):
            if RiskGateway._mapping_describes_synthetic(payload):
                metadata = RiskGateway._format_invariant_metadata(base, payload)
                if metadata is not None:
                    return metadata
            for key, value in payload.items():
                key_str = str(key)
                indicator = f"{base}.{key_str}" if key_str else base
                metadata = RiskGateway._search_invariant_payload(value, base=indicator)
                if metadata is not None:
                    return metadata
            return None

        if isinstance(payload, (list, tuple)):
            for index, entry in enumerate(payload):
                indicator = f"{base}[{index}]"
                metadata = RiskGateway._search_invariant_payload(entry, base=indicator)
                if metadata is not None:
                    return metadata
            return None

        if isinstance(payload, str):
            if "synthetic" in payload.lower():
                return RiskGateway._format_invariant_metadata(base, payload)
            return None

        return None

    @staticmethod
    def _format_invariant_metadata(indicator: str, value: object) -> dict[str, Any] | None:
        if not RiskGateway._value_indicates_breach(value):
            return None

        metadata: dict[str, Any] = {"indicator": indicator}
        details = RiskGateway._coerce_invariant_details(value)
        if details is not None:
            metadata["details"] = details
        return metadata

    @staticmethod
    def _mapping_describes_synthetic(mapping: Mapping[str, Any]) -> bool:
        for key, value in mapping.items():
            if isinstance(key, str) and "synthetic" in key.lower():
                return True
            if isinstance(value, str) and "synthetic" in value.lower():
                return True
            if isinstance(value, Mapping) and RiskGateway._mapping_describes_synthetic(value):
                return True
            if isinstance(value, (list, tuple)) and RiskGateway._sequence_contains_synthetic(value):
                return True
        return False

    @staticmethod
    def _sequence_contains_synthetic(sequence: Sequence[Any]) -> bool:
        for entry in sequence:
            if isinstance(entry, str) and "synthetic" in entry.lower():
                return True
            if isinstance(entry, Mapping) and RiskGateway._mapping_describes_synthetic(entry):
                return True
            if isinstance(entry, (list, tuple)) and RiskGateway._sequence_contains_synthetic(entry):
                return True
        return False

    @staticmethod
    def _value_indicates_breach(value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if not lowered:
                return False
            if lowered in {"false", "0", "none", "ok", "clear", "passed", "nominal"}:
                return False
            tokens = (
                "breach",
                "breached",
                "violation",
                "violations",
                "violat",
                "fail",
                "failed",
                "trigger",
                "triggered",
                "engaged",
            )
            return any(token in lowered for token in tokens)
        if isinstance(value, Mapping):
            status = value.get("status")
            if isinstance(status, str) and RiskGateway._value_indicates_breach(status):
                return True
            count = value.get("count")
            if isinstance(count, (int, float)) and count > 0:
                return True
            breaches = value.get("breaches")
            if isinstance(breaches, (list, tuple)) and breaches:
                return True
            for candidate_key in ("breach", "breached", "violations", "violation", "active", "engaged", "triggered"):
                if candidate_key in value and RiskGateway._value_indicates_breach(value[candidate_key]):
                    return True
            for nested in value.values():
                if isinstance(nested, (Mapping, list, tuple, str)) and RiskGateway._value_indicates_breach(nested):
                    return True
            return False
        if isinstance(value, (list, tuple, set)):
            return any(RiskGateway._value_indicates_breach(item) for item in value)
        return bool(value)

    @staticmethod
    def _coerce_invariant_details(value: object) -> object | None:
        if isinstance(value, Mapping):
            return {
                str(key): RiskGateway._coerce_invariant_details(inner)
                for key, inner in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [RiskGateway._coerce_invariant_details(entry) for entry in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _check_open_positions(
        self, portfolio_state: Mapping[str, Any], decision: dict[str, Any]
    ) -> bool:
        # Portfolio monitor exposes helper, otherwise rely on supplied state
        try:
            count = _as_float(self.portfolio_monitor.get_open_position_count())
        except Exception:
            count = _as_float(portfolio_state.get("open_positions_count"), default=0.0)

        decision["checks"].append(
            {
                "name": "open_positions",
                "value": count,
                "threshold": self.max_open_positions,
            }
        )
        return count < self.max_open_positions

    def _extract_quantity(self, intent: Any) -> Decimal:
        quantity = _extract_attr(intent, "quantity", "size", "volume", default=0)
        return _to_decimal(quantity)

    def _extract_price(self, intent: Any, portfolio_state: Mapping[str, Any]) -> float:
        price = _extract_attr(intent, "price", "limit_price", "entry_price")
        if price is None:
            price = portfolio_state.get("current_price")
        return _as_float(price, default=0.0)

    def _extract_stop_loss_pct(
        self, intent: Any, portfolio_state: Mapping[str, Any], market_price: float
    ) -> float:
        metadata = _ensure_metadata(intent)
        if "stop_loss_pct" in metadata:
            return float(metadata["stop_loss_pct"])

        if "stop_loss_pips" in metadata:
            pip_value = _as_float(portfolio_state.get("pip_value"), default=0.0001)
            pips = _as_float(metadata.get("stop_loss_pips"), default=0.0)
            if pip_value and market_price:
                return max(self.stop_loss_floor, abs(pips * pip_value) / market_price)

        return self.stop_loss_floor

    def _apply_position_sizing(
        self, intent: Any, portfolio_state: Mapping[str, Any], decision: dict[str, Any]
    ) -> Decimal:
        quantity = self._extract_quantity(intent)
        adjusted = quantity

        if self.position_sizer:
            balance = _to_decimal(portfolio_state.get("equity"), default=Decimal("0"))
            if balance > 0:
                market_price = self._extract_price(intent, portfolio_state)
                stop_loss_pct = self._extract_stop_loss_pct(
                    intent, portfolio_state, market_price
                )
                recommended = self.position_sizer(
                    balance, self.risk_per_trade, Decimal(str(stop_loss_pct))
                )

                decision["checks"].append(
                    {
                        "name": "position_sizer",
                        "recommended": float(recommended),
                        "requested": float(quantity),
                    }
                )

                if quantity > recommended:
                    logger.info(
                        "RiskGateway clipping position size from %s to %s", quantity, recommended
                    )
                    adjusted = recommended
            else:
                adjusted = quantity

        return adjusted

    async def _run_liquidity_probe(
        self,
        intent: Any,
        quantity: Decimal,
        decision: dict[str, Any],
        *,
        market_price: float,
        portfolio_state: Mapping[str, Any],
    ) -> tuple[float | None, Mapping[str, Any] | None]:
        if not self.liquidity_prober:
            return None, None

        symbol = _extract_attr(intent, "symbol", default="UNKNOWN")
        side = str(_extract_attr(intent, "side", "direction", default="buy")).lower()
        resolved_price = self._resolve_trade_price(
            market_price, portfolio_state, str(symbol)
        )
        market_price = resolved_price if resolved_price > 0 else 0.0
        spread = max(market_price * 0.001, 0.0005)
        price_levels = (
            [
                market_price + spread * offset
                if side.startswith("buy")
                else market_price - spread * offset
                for offset in range(-2, 3)
            ]
            if market_price
            else []
        )

        if not price_levels:
            price_levels = [0.0]

        probe_results = await self.liquidity_prober.probe_liquidity(symbol, price_levels, side)
        score = self.liquidity_prober.calculate_liquidity_confidence_score(
            probe_results, float(quantity)
        )
        summary = self.liquidity_prober.get_probe_summary(probe_results)

        decision["checks"].append(
            {
                "name": "liquidity_confidence",
                "value": score,
                "threshold": self.min_liquidity_confidence,
            }
        )
        return score, summary

    def _enrich_intent(
        self,
        intent: Any,
        quantity: Decimal,
        liquidity_score: float | None,
        decision: Mapping[str, Any],
        *,
        policy_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        _set_attr(intent, quantity, "quantity", "size", "volume")

        metadata = _ensure_metadata(intent)
        metadata.setdefault("risk_assessment", {})
        assessment = metadata["risk_assessment"]
        if isinstance(assessment, Mapping):
            assessment = dict(assessment)
            metadata["risk_assessment"] = assessment

        assessment.update(
            {
                "checks": list(decision.get("checks", [])),
                "final_quantity": float(quantity),
            }
        )
        if policy_metadata:
            policy_section = assessment.get("policy")
            if isinstance(policy_section, Mapping):
                merged = dict(policy_section)
                merged.update(policy_metadata)
                assessment["policy"] = merged
            else:
                assessment["policy"] = dict(policy_metadata)
        if liquidity_score is not None:
            assessment["liquidity_confidence"] = float(liquidity_score)
        risk_summary = decision.get("risk_manager")
        if isinstance(risk_summary, Mapping):
            assessment["portfolio_risk"] = dict(risk_summary)

        risk_reference = decision.get("risk_reference")
        if isinstance(risk_reference, Mapping):
            assessment["risk_reference"] = dict(risk_reference)

        if hasattr(intent, "liquidity_confidence_score"):
            try:
                setattr(intent, "liquidity_confidence_score", liquidity_score)
            except Exception as exc:  # pragma: no cover - attribute blocked
                logger.debug(
                    "Failed to persist liquidity confidence score on intent %r",
                    intent,
                    exc_info=exc,
                )

        if is_dataclass(intent):
            try:
                params = getattr(intent, "__dataclass_params__", None)
                is_frozen = bool(getattr(params, "frozen", False))
                if is_frozen:
                    data: dict[str, Any] = asdict(cast(Any, intent))
                    data.update(
                        {
                            "metadata": metadata,
                            "liquidity_confidence_score": liquidity_score,
                            "quantity": quantity,
                            "size": quantity,
                        }
                    )
                    for field, value in data.items():
                        try:
                            setattr(intent, field, value)
                        except Exception as exc:
                            logger.debug(
                                "Unable to update frozen dataclass field %s on %r",
                                field,
                                intent,
                                exc_info=exc,
                            )
            except Exception as exc:  # pragma: no cover
                logger.debug("Failed to update dataclass intent %r", intent, exc_info=exc)

    def _evaluate_execution_cost(
        self,
        intent: Any,
        quantity: float,
        price: float,
        portfolio_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        metadata = _ensure_metadata(intent)
        feature_source = self._extract_microstructure_features(metadata)
        context = self._build_execution_context(metadata, portfolio_state, quantity, price)

        execution_cfg = self._execution_config
        limits: ExecutionRiskLimits | None = execution_cfg.limits

        slippage_bps = estimate_slippage_bps(context, execution_cfg)
        commission_bps = estimate_commission_bps(execution_cfg)
        total_cost_bps = slippage_bps + commission_bps

        price_float = max(_as_float(price, default=0.0), 0.0)
        tick_size = self._extract_tick_size(feature_source, metadata, portfolio_state)
        spread_ticks = self._derive_spread_ticks(context, feature_source, tick_size)
        edge_floor_ticks = self._resolve_edge_floor_ticks(feature_source, tick_size)
        effective_spread_ticks = max(spread_ticks, edge_floor_ticks) if edge_floor_ticks > 0 else spread_ticks

        delta_hat = self._try_get_float(feature_source, ("delta_hat", "deltaHat"))
        if delta_hat is None and isinstance(feature_source, Mapping):
            for key in ("edge", "signals"):
                nested = feature_source.get(key)
                if isinstance(nested, Mapping):
                    delta_hat = self._try_get_float(nested, ("delta_hat", "deltaHat"))
                    if delta_hat is not None:
                        break
        if delta_hat is None:
            delta_hat = self._try_get_float(metadata, ("delta_hat",))
        delta_value = delta_hat if delta_hat is not None else 0.0

        edge_ticks = calculate_edge_ticks(
            delta_value,
            spread_ticks,
            spread_floor=edge_floor_ticks if edge_floor_ticks > 0 else None,
        )

        slippage_ticks = self._price_to_ticks(price_float * slippage_bps / 10000.0, tick_size)
        fees_ticks = self._price_to_ticks(price_float * commission_bps / 10000.0, tick_size)
        as_penalty_ticks = self._extract_adverse_selection_penalty_ticks(
            feature_source,
            metadata,
            portfolio_state,
            tick_size=tick_size,
            price=price_float,
        )
        total_cost_ticks = estimate_total_cost_ticks(
            spread_ticks,
            slippage=slippage_ticks,
            fees=fees_ticks,
            adverse_selection_penalty=as_penalty_ticks,
        )
        spread_cost_ticks = 0.5 * max(spread_ticks, 0.0)

        notional = abs(quantity) * max(price, 0.0)
        equity = self._resolve_equity_for_execution(portfolio_state)
        notional_pct = (notional / equity) if equity > 0 else None
        estimated_cost = notional * total_cost_bps / 10000 if notional else 0.0

        checks: list[dict[str, Any]] = []
        breaches: list[str] = []

        def _record(name: str, value: float | None, threshold: float | None) -> None:
            entry: dict[str, Any] = {
                "name": name,
                "value": value,
                "threshold": threshold,
            }
            status = "info"
            if threshold is not None and value is not None:
                if value > threshold:
                    status = "violation"
                elif threshold > 0 and value / threshold >= 0.8:
                    status = "warn"
                else:
                    status = "ok"
            elif value is not None and threshold is None:
                status = "info"
            entry["status"] = status
            if status == "violation":
                breaches.append(name)
            checks.append(entry)

        limit_slippage = limits.max_slippage_bps if limits else None
        limit_total = limits.max_total_cost_bps if limits else None
        limit_notional = limits.max_notional_pct_of_equity if limits else None

        _record("execution.slippage_bps", slippage_bps, limit_slippage)
        _record("execution.total_cost_bps", total_cost_bps, limit_total)
        _record("execution.notional_pct_of_equity", notional_pct, limit_notional)

        execution_metadata: dict[str, Any] = {
            "slippage_bps": slippage_bps,
            "commission_bps": commission_bps,
            "total_cost_bps": total_cost_bps,
            "notional": notional,
            "estimated_cost": estimated_cost,
            "notional_pct_of_equity": notional_pct,
            "size_ratio": context.size_ratio,
            "spread": context.spread,
            "top_imbalance": context.top_imbalance,
            "sigma_ann": context.sigma_ann,
        }

        execution_metadata["spread_ticks"] = spread_ticks
        execution_metadata["effective_spread_ticks"] = effective_spread_ticks
        if edge_floor_ticks > 0:
            execution_metadata["edge_spread_floor_ticks"] = edge_floor_ticks
        if tick_size > 0:
            execution_metadata["tick_size"] = tick_size
        if delta_hat is not None:
            execution_metadata["delta_hat"] = delta_hat
        execution_metadata["edge_ticks"] = edge_ticks
        execution_metadata["spread_cost_ticks"] = spread_cost_ticks
        execution_metadata["slippage_ticks"] = slippage_ticks
        execution_metadata["fees_ticks"] = fees_ticks
        execution_metadata["adverse_selection_penalty_ticks"] = as_penalty_ticks
        execution_metadata["total_cost_ticks"] = total_cost_ticks

        metadata.setdefault("execution_risk", {}).update(execution_metadata)

        return {
            "checks": tuple(checks),
            "metadata": execution_metadata,
            "breaches": tuple(breaches),
            "reason": "execution_risk",
        }

    def _enforce_portfolio_risk(
        self,
        decision: dict[str, Any],
        portfolio_state: Mapping[str, Any],
        *,
        symbol: str,
        quantity: float,
        market_price: float,
        stop_loss_pct: float,
    ) -> bool:
        manager = self.portfolio_risk_manager
        if manager is None:
            return True

        equity = 0.0
        try:
            equity = self._resolve_equity_for_risk(portfolio_state)
            if equity > 0:
                manager.update_equity(equity)
        except Exception:  # pragma: no cover - diagnostics only
            logger.debug("Failed to update portfolio equity for risk manager", exc_info=True)

        risk_map = self._build_portfolio_risk_map(portfolio_state)

        price = market_price
        if price <= 0:
            existing_position = self._locate_position(portfolio_state, symbol)
            if existing_position is not None:
                price = self._resolve_position_price(existing_position)

        trade_exposure = self._compute_risk_exposure(quantity, price, stop_loss_pct)
        if trade_exposure > 0:
            risk_map[symbol] = risk_map.get(symbol, 0.0) + trade_exposure

        try:
            risk_score = manager.assess_risk(risk_map)
            snapshot = dict(manager.last_snapshot)
        except Exception as exc:  # pragma: no cover - fail safe
            logger.exception("Portfolio risk assessment failed: %s", exc)
            decision.setdefault("checks", []).append(
                {
                    "name": "portfolio_risk_score",
                    "value": None,
                    "threshold": 1.0,
                    "status": "error",
                    "error": str(exc),
                }
            )
            decision.update(status="rejected", reason="portfolio_risk_error")
            return False

        check_entry = {
            "name": "portfolio_risk_score",
            "value": risk_score,
            "threshold": 1.0,
            "status": "ok" if risk_score <= 1.0 else "violation",
            "metadata": snapshot,
        }
        decision.setdefault("checks", []).append(check_entry)
        risk_section = decision.setdefault("risk_manager", {})
        risk_section.update(
            {
                "risk_score": risk_score,
                "snapshot": snapshot,
                "equity": equity,
                "projected_risk_map": dict(risk_map),
            }
        )

        if risk_score > 1.0:
            decision.update(status="rejected", reason="portfolio_risk_breach")
            return False

        return True

    def _enforce_confidence_limit(
        self,
        decision: dict[str, Any],
        *,
        confidence: float,
        quantity: float,
        trade_price: float,
        equity: float,
    ) -> bool:
        limit_pct = self._confidence_limit_pct
        if limit_pct <= 0:
            return True

        checks = decision.setdefault("checks", [])
        clamped_confidence = max(0.0, min(float(confidence), 1.0))

        if equity <= 0 or trade_price <= 0:
            checks.append(
                {
                    "name": "risk.confidence_notional_limit",
                    "value": None,
                    "threshold": None,
                    "status": "info",
                    "metadata": {
                        "confidence": clamped_confidence,
                        "equity": equity,
                        "price": trade_price,
                    },
                }
            )
            return True

        threshold_pct = limit_pct * clamped_confidence
        notional = abs(quantity) * trade_price
        allowed_notional = equity * threshold_pct

        status = "ok"
        if threshold_pct <= 0 and notional > _EPSILON:
            status = "violation"
        elif threshold_pct > 0:
            utilisation = notional / allowed_notional if allowed_notional > 0 else float("inf")
            if utilisation > 1.0 + _EPSILON:
                status = "violation"
            elif utilisation >= 0.8:
                status = "warn"

        checks.append(
            {
                "name": "risk.confidence_notional_limit",
                "value": notional,
                "threshold": allowed_notional,
                "status": status,
                "metadata": {
                    "confidence": clamped_confidence,
                    "limit_pct": threshold_pct,
                },
            }
        )

        return status != "violation"

    def _enforce_leverage_limit(
        self,
        decision: dict[str, Any],
        portfolio_state: Mapping[str, Any],
        *,
        quantity: float,
        trade_price: float,
        equity: float,
    ) -> bool:
        limit = self._max_leverage_limit
        if limit <= 0:
            return True

        checks = decision.setdefault("checks", [])

        if equity <= 0 or trade_price <= 0:
            checks.append(
                {
                    "name": "risk.leverage_ratio",
                    "value": None,
                    "threshold": limit,
                    "status": "info",
                }
            )
            return True

        total_exposure = self._compute_total_exposure(portfolio_state)
        symbol = str(decision.get("symbol") or "")
        existing_position = self._locate_position(portfolio_state, symbol)
        existing_notional = self._compute_position_notional(existing_position)
        proposed_notional = abs(quantity) * trade_price
        projected_total = total_exposure - existing_notional + proposed_notional

        projected_leverage = projected_total / equity if equity > 0 else float("inf")

        status = "ok"
        if projected_leverage > limit + _EPSILON:
            status = "violation"
        elif projected_leverage >= 0.8 * limit:
            status = "warn"

        checks.append(
            {
                "name": "risk.leverage_ratio",
                "value": projected_leverage,
                "threshold": limit,
                "status": status,
                "metadata": {
                    "projected_total_exposure": projected_total,
                    "equity": equity,
                },
            }
        )

        return status != "violation"

    def _enforce_sector_limits(
        self,
        decision: dict[str, Any],
        portfolio_state: Mapping[str, Any],
        *,
        symbol: str,
        quantity: float,
        trade_price: float,
        equity: float,
        side: str | None,
        intent: Any | None,
    ) -> bool:
        checks = decision.setdefault("checks", [])

        if not self._sector_limits or not symbol:
            return True

        sector = self._resolve_sector_hint(symbol, intent=intent)
        if sector is None:
            checks.append(
                {
                    "name": "risk.sector_limit.unmapped",
                    "value": None,
                    "threshold": None,
                    "status": "info",
                    "metadata": {
                        "symbol": symbol,
                        "reason": "no_sector_hint",
                    },
                }
            )
            return True

        limit_pct = self._sector_limits.get(sector)
        if limit_pct is None or limit_pct <= 0:
            return True

        if equity <= 0 or trade_price <= 0:
            checks.append(
                {
                    "name": f"risk.sector_limit.{sector}",
                    "value": None,
                    "threshold": None,
                    "status": "info",
                    "metadata": {"sector": sector, "equity": equity, "price": trade_price},
                }
            )
            return True

        exposures = self._compute_sector_exposures(portfolio_state)
        existing_position = self._locate_position(portfolio_state, symbol)
        existing_qty = 0.0
        existing_price = trade_price
        if existing_position is not None:
            existing_qty = _as_float(existing_position.get("quantity"), default=0.0)
            price_candidate = self._resolve_position_price(existing_position)
            if price_candidate > 0:
                existing_price = price_candidate

        # Determine signed order quantity using side hint when provided.
        side_upper = (side or "").upper()
        signed_quantity = quantity
        if side_upper.startswith("SELL"):
            signed_quantity = -abs(quantity)
        elif side_upper.startswith("BUY"):
            signed_quantity = abs(quantity)

        order_quantity = signed_quantity
        if not side_upper and quantity == 0:
            order_quantity = 0.0

        projected_qty = existing_qty + order_quantity

        current_sector_exposure = exposures.get(sector, 0.0)
        existing_notional = abs(existing_qty) * existing_price
        exposures_other = max(0.0, current_sector_exposure - existing_notional)
        projected_notional = abs(projected_qty) * trade_price
        projected_exposure = exposures_other + projected_notional

        allowed_exposure = equity * limit_pct

        status = "ok"
        if projected_exposure - allowed_exposure > _EPSILON:
            status = "violation"
        elif allowed_exposure > 0 and projected_exposure / allowed_exposure >= 0.8:
            status = "warn"

        checks.append(
            {
                "name": f"risk.sector_limit.{sector}",
                "value": projected_exposure,
                "threshold": allowed_exposure,
                "status": status,
                "metadata": {
                    "current_exposure": current_sector_exposure,
                    "existing_notional": existing_notional,
                    "projected_quantity": projected_qty,
                    "order_quantity": order_quantity,
                },
            }
        )

        return status != "violation"

    def _resolve_trade_price(
        self, market_price: float, portfolio_state: Mapping[str, Any], symbol: str
    ) -> float:
        if market_price > 0:
            return market_price
        position = self._locate_position(portfolio_state, symbol)
        if position is not None:
            resolved = self._resolve_position_price(position)
            if resolved > 0:
                return resolved
        fallback = _as_float(portfolio_state.get("current_price"), default=0.0)
        return fallback

    def _compute_position_notional(self, position: Mapping[str, Any] | None) -> float:
        if position is None:
            return 0.0
        quantity = _as_float(position.get("quantity"), default=0.0)
        price = self._resolve_position_price(position)
        return abs(quantity) * price

    def _compute_total_exposure(self, portfolio_state: Mapping[str, Any]) -> float:
        total = 0.0
        positions = portfolio_state.get("open_positions")
        if not isinstance(positions, Mapping):
            return total
        for payload in positions.values():
            if not isinstance(payload, Mapping):
                continue
            total += self._compute_position_notional(payload)
        return total

    def _compute_sector_exposures(
        self, portfolio_state: Mapping[str, Any]
    ) -> dict[str, float]:
        exposures: dict[str, float] = {}
        positions = portfolio_state.get("open_positions")
        if not isinstance(positions, Mapping):
            return exposures

        for symbol, payload in positions.items():
            if not isinstance(payload, Mapping):
                continue
            sector = self._resolve_sector_hint(str(symbol), payload=payload)
            if sector is None:
                continue
            exposures[sector] = exposures.get(sector, 0.0) + self._compute_position_notional(payload)

        return exposures

    def _resolve_equity_for_risk(self, portfolio_state: Mapping[str, Any]) -> float:
        equity = _as_float(portfolio_state.get("equity"), default=0.0)
        if equity > 0:
            return equity
        cash = _as_float(portfolio_state.get("cash"), default=0.0)
        total_exposure = 0.0
        positions = portfolio_state.get("open_positions")
        if isinstance(positions, Mapping):
            for payload in positions.values():
                if not isinstance(payload, Mapping):
                    continue
                quantity = abs(_as_float(payload.get("quantity"), default=0.0))
                price = self._resolve_position_price(payload)
                total_exposure += quantity * price
        return max(0.0, cash + total_exposure)

    def _build_portfolio_risk_map(
        self, portfolio_state: Mapping[str, Any]
    ) -> dict[str, float]:
        exposures: dict[str, float] = {}
        positions = portfolio_state.get("open_positions")
        if not isinstance(positions, Mapping):
            return exposures

        for symbol, payload in positions.items():
            if not isinstance(payload, Mapping):
                continue
            quantity = _as_float(payload.get("quantity"), default=0.0)
            price = self._resolve_position_price(payload)
            stop_loss = self._extract_position_stop_loss(payload)
            exposure = self._compute_risk_exposure(quantity, price, stop_loss)
            if exposure > 0:
                exposures[str(symbol)] = exposure

        return exposures

    def _compute_risk_exposure(
        self, quantity: float, price: float, stop_loss_pct: float
    ) -> float:
        abs_quantity = abs(quantity)
        if abs_quantity <= 0 or price <= 0:
            return 0.0
        resolved_stop = max(
            abs(float(stop_loss_pct)),
            self._risk_per_trade_float,
            self.stop_loss_floor,
        )
        return abs_quantity * price * resolved_stop

    def _extract_position_stop_loss(self, payload: Mapping[str, Any]) -> float:
        stop_loss = _as_float(payload.get("stop_loss_pct"), default=0.0)
        if stop_loss > 0:
            return stop_loss
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            candidate = _as_float(metadata.get("stop_loss_pct"), default=0.0)
            if candidate > 0:
                return candidate
        risk_section = payload.get("risk_assessment")
        if isinstance(risk_section, Mapping):
            candidate = _as_float(risk_section.get("stop_loss_pct"), default=0.0)
            if candidate > 0:
                return candidate
        return 0.0

    def _resolve_position_price(self, payload: Mapping[str, Any]) -> float:
        for key in ("current_price", "last_price", "avg_price", "price"):
            price = _as_float(payload.get(key), default=0.0)
            if price > 0:
                return price
        quantity = _as_float(payload.get("quantity"), default=0.0)
        current_value = _as_float(payload.get("current_value"), default=0.0)
        if quantity and current_value:
            return abs(current_value) / abs(quantity)
        return _as_float(payload.get("entry_price"), default=0.0)

    def _locate_position(
        self, portfolio_state: Mapping[str, Any], symbol: str
    ) -> Mapping[str, Any] | None:
        positions = portfolio_state.get("open_positions")
        if isinstance(positions, Mapping):
            payload = positions.get(symbol)
            if isinstance(payload, Mapping):
                return payload
        return None

    def _build_execution_context(
        self,
        metadata: Mapping[str, Any],
        portfolio_state: Mapping[str, Any],
        quantity: float,
        price: float,
    ) -> ExecContext:
        feature_source = self._extract_microstructure_features(metadata)
        if not feature_source:
            feature_source = {}

        spread = _as_float(feature_source.get("spread"), default=0.0)
        if spread <= 0 and price > 0:
            spread_bps = _as_float(feature_source.get("spread_bps"), default=0.0)
            if spread_bps:
                spread = price * spread_bps / 10000.0
        if spread <= 0:
            spread = _as_float(portfolio_state.get("current_spread"), default=0.0)

        imbalance = _as_float(feature_source.get("liquidity_imbalance"), default=0.0)
        if imbalance == 0.0:
            imbalance = _as_float(feature_source.get("order_flow_imbalance"), default=0.0)

        sigma_ann = _as_float(feature_source.get("sigma_ann"), default=0.0)
        if sigma_ann == 0.0:
            sigma_ann = _as_float(feature_source.get("price_volatility"), default=0.0)
        if sigma_ann == 0.0:
            sigma_ann = _as_float(feature_source.get("volatility"), default=0.0)

        notional = abs(quantity) * max(price, 0.0)
        equity = self._resolve_equity_for_execution(portfolio_state)
        size_ratio = notional / equity if equity > 0 else abs(quantity)

        return ExecContext(
            spread=max(0.0, spread),
            top_imbalance=imbalance,
            sigma_ann=max(0.0, sigma_ann),
            size_ratio=max(0.0, size_ratio),
        )

    @staticmethod
    def _extract_microstructure_features(
        metadata: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if not isinstance(metadata, Mapping):
            return {}

        for key in ("execution_context", "microstructure", "market_microstructure"):
            candidate = metadata.get(key)
            if isinstance(candidate, Mapping):
                return candidate

        exec_context = metadata.get("execution_context")
        if isinstance(exec_context, Mapping):
            for key in ("microstructure", "market_microstructure"):
                nested = exec_context.get(key)
                if isinstance(nested, Mapping):
                    return nested
        return {}

    @staticmethod
    def _try_get_float(source: Mapping[str, Any] | None, keys: Sequence[str]) -> float | None:
        if not isinstance(source, Mapping):
            return None
        for key in keys:
            if key not in source:
                continue
            value = source.get(key)
            if value is None:
                continue
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate):
                return candidate
        return None

    def _extract_tick_size(
        self,
        feature_source: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
        portfolio_state: Mapping[str, Any] | None,
    ) -> float:
        containers: list[Mapping[str, Any]] = []
        for candidate in (feature_source, metadata, portfolio_state):
            if isinstance(candidate, Mapping):
                containers.append(candidate)
                instrument = candidate.get("instrument")
                if isinstance(instrument, Mapping):
                    containers.append(instrument)

        for container in containers:
            tick = self._try_get_float(
                container,
                ("tick_size", "tick", "price_tick", "minimum_price_increment"),
            )
            if tick is not None and tick > 0.0:
                return float(tick)
        return 0.0

    def _derive_spread_ticks(
        self,
        context: ExecContext,
        feature_source: Mapping[str, Any] | None,
        tick_size: float,
    ) -> float:
        explicit_ticks = self._try_get_float(feature_source, ("spread_ticks",))
        spread_ticks = self._price_to_ticks(context.spread, tick_size)
        if explicit_ticks is not None:
            spread_ticks = float(explicit_ticks)
        if spread_ticks < 0.0:
            spread_ticks = 0.0
        return spread_ticks

    def _resolve_edge_floor_ticks(
        self,
        feature_source: Mapping[str, Any] | None,
        tick_size: float,
    ) -> float:
        explicit = self._try_get_float(
            feature_source,
            (
                "edge_spread_floor_ticks",
                "edge_floor_ticks",
                "spread_floor_ticks",
                "k_sigma_ticks",
                "min_spread_ticks",
            ),
        )
        if explicit is not None and explicit > 0.0:
            return float(explicit)

        price_floor = self._try_get_float(
            feature_source,
            ("edge_spread_floor", "edge_floor", "spread_floor", "k_sigma", "min_spread"),
        )
        if price_floor is not None and price_floor > 0.0:
            return self._price_to_ticks(price_floor, tick_size)
        return 0.0

    @staticmethod
    def _price_to_ticks(value: float, tick_size: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(numeric):
            return 0.0
        if tick_size > 0.0:
            return numeric / tick_size
        return numeric

    def _extract_adverse_selection_penalty_ticks(
        self,
        feature_source: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None,
        portfolio_state: Mapping[str, Any] | None,
        *,
        tick_size: float,
        price: float,
    ) -> float:
        candidates: list[Mapping[str, Any]] = []

        def _extend(source: Mapping[str, Any] | None) -> None:
            if not isinstance(source, Mapping):
                return
            candidates.append(source)
            nested = source.get("adverse_selection")
            if isinstance(nested, Mapping):
                candidates.append(nested)
            penalties = source.get("penalties")
            if isinstance(penalties, Mapping):
                candidates.append(penalties)

        for item in (feature_source, metadata, portfolio_state):
            _extend(item)

        penalty_ticks: float | None = None
        for container in candidates:
            ticks = self._try_get_float(
                container,
                ("adverse_selection_penalty_ticks", "as_penalty_ticks", "penalty_ticks"),
            )
            if ticks is not None:
                penalty_ticks = float(ticks)
                break

        if penalty_ticks is None:
            for container in candidates:
                penalty_price = self._try_get_float(
                    container,
                    ("adverse_selection_penalty", "as_penalty", "penalty"),
                )
                if penalty_price is not None:
                    penalty_ticks = self._price_to_ticks(penalty_price, tick_size)
                    break

        if penalty_ticks is None and price > 0.0:
            for container in candidates:
                penalty_bps = self._try_get_float(
                    container,
                    ("adverse_selection_penalty_bps", "as_penalty_bps", "penalty_bps"),
                )
                if penalty_bps is not None:
                    penalty_ticks = self._price_to_ticks(price * penalty_bps / 10000.0, tick_size)
                    break

        if penalty_ticks is None:
            return 0.0
        return max(0.0, float(penalty_ticks))

    def _resolve_equity_for_execution(self, portfolio_state: Mapping[str, Any]) -> float:
        for key in ("equity", "cash", "account_equity", "total_equity"):
            equity = _as_float(portfolio_state.get(key), default=0.0)
            if equity > 0:
                return equity
        return 0.0

    def _apply_liquidity_capacity_guard(
        self,
        intent: Any,
        quantity: Decimal,
        decision: dict[str, Any],
        *,
        portfolio_state: Mapping[str, Any],
    ) -> Decimal:
        if self._liquidity_capacity_ratio_decimal <= 0:
            return quantity

        metadata = _ensure_metadata(intent)
        depth_hint = self._extract_depth_percentile_hint(metadata)
        if depth_hint <= 0.0:
            depth_hint = self._extract_depth_percentile_hint(portfolio_state)
        if depth_hint <= 0.0:
            symbol = str(_extract_attr(intent, "symbol", default=""))
            depth_hint = self._resolve_depth_percentile(symbol)

        if depth_hint <= 0.0:
            return quantity

        capacity_units = Decimal(str(depth_hint)) * self._liquidity_capacity_ratio_decimal
        if capacity_units <= 0:
            return quantity

        abs_quantity = abs(quantity)
        if abs_quantity <= capacity_units + self._liquidity_capacity_epsilon:
            return quantity

        clipped_abs = min(abs_quantity, capacity_units)
        clipped_quantity = clipped_abs if quantity >= 0 else -clipped_abs

        decision["checks"].append(
            {
                "name": "liquidity_capacity",
                "value": float(abs_quantity),
                "threshold": float(capacity_units),
                "status": "clipped",
                "adjusted": float(clipped_abs),
                "depth_percentile": float(depth_hint),
                "percentile": self._liquidity_capacity_percentile,
                "ratio": self._liquidity_capacity_ratio,
            }
        )

        logger.info(
            "RiskGateway clipping position due to liquidity capacity: %s -> %s (symbol=%s, depth=%.2f, percentile=%.1f, ratio=%.4f)",
            quantity,
            clipped_quantity,
            _extract_attr(intent, "symbol", default=""),
            depth_hint,
            self._liquidity_capacity_percentile,
            self._liquidity_capacity_ratio,
        )

        return clipped_quantity

    @staticmethod
    def _extract_depth_percentile_hint(source: Mapping[str, Any] | None) -> float:
        if not isinstance(source, Mapping):
            return 0.0

        candidate_keys = (
            "l1_depth_percentile",
            "l1_depth_p50",
            "depth_percentile",
            "depth_p50",
            "l1_depth",
        )

        for key in candidate_keys:
            value = source.get(key)
            depth = _as_float(value, default=0.0)
            if depth > 0.0:
                return depth

        nested_keys = (
            "liquidity",
            "microstructure",
            "market_microstructure",
            "execution_context",
        )
        for nested_key in nested_keys:
            nested_value = source.get(nested_key)
            if isinstance(nested_value, Mapping):
                depth = RiskGateway._extract_depth_percentile_hint(nested_value)
                if depth > 0.0:
                    return depth
        return 0.0

    def _resolve_depth_percentile(self, symbol: str) -> float:
        if not symbol:
            return 0.0
        prober = self.liquidity_prober
        if prober is None:
            return 0.0
        depth_resolver = getattr(prober, "depth_percentile", None)
        if not callable(depth_resolver):
            return 0.0
        try:
            depth_value = float(depth_resolver(symbol, self._liquidity_capacity_percentile))
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("liquidity_depth_percentile_failed", exc_info=True)
            return 0.0
        return depth_value if depth_value > 0.0 else 0.0
