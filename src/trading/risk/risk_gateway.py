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
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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
from src.trading.execution.execution_model import (
    ExecContext,
    estimate_commission_bps,
    estimate_slippage_bps,
)
from .policy_telemetry import (
    RiskPolicyEvaluationSnapshot,
    build_policy_snapshot,
)
from .risk_api import RISK_API_RUNBOOK, summarise_risk_config
from .risk_policy import RiskPolicy, RiskPolicyDecision

try:  # pragma: no cover - metrics optional in certain runtimes
    from src.operational import metrics as operational_metrics
except Exception:  # pragma: no cover
    operational_metrics = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


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
    except Exception:  # pragma: no cover - best effort when attribute blocked
        pass
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
            except Exception:  # pragma: no cover - best effort
                pass
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

        self.telemetry: dict[str, Any] = {
            "total_checks": 0,
            "approved": 0,
            "rejected": 0,
            "last_decision": None,
        }
        self._last_policy_decision: RiskPolicyDecision | None = None
        self._last_policy_snapshot: RiskPolicyEvaluationSnapshot | None = None
        self._risk_config: RiskConfig | None = risk_config
        self._risk_reference_cache: dict[str, object] | None = None

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
            reference_payload = dict(risk_reference)
        reference_payload.setdefault("limits", dict(limits_snapshot))
        decision["risk_reference"] = reference_payload

        try:
            if not self._check_strategy(decision["strategy_id"]):
                decision.update(status="rejected", reason="strategy_disabled")
                self._reject(decision)
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
                self._reject(decision)
                return None

            portfolio_state = portfolio_state or {}
            if not self._check_drawdown(portfolio_state, decision):
                decision.update(status="rejected", reason="max_drawdown_exceeded")
                self._reject(decision)
                return None

            if not self._check_open_positions(portfolio_state, decision):
                decision.update(status="rejected", reason="too_many_open_positions")
                self._reject(decision)
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
                    self._reject(decision)
                    return None

            if not self._enforce_portfolio_risk(
                decision,
                portfolio_state,
                symbol=str(decision["symbol"]),
                quantity=float(adjusted_quantity),
                market_price=market_price,
                stop_loss_pct=stop_loss_pct,
            ):
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
                self._reject(decision)
                return None

            liquidity_summary: Mapping[str, Any] | None = None
            liquidity_score: float | None = None
            if self.liquidity_prober and adjusted_quantity >= self.liquidity_probe_threshold:
                liquidity_score, liquidity_summary = await self._run_liquidity_probe(
                    intent, adjusted_quantity, decision
                )
                if liquidity_score is not None and liquidity_score < self.min_liquidity_confidence:
                    decision.update(status="rejected", reason="insufficient_liquidity")
                    self._reject(decision)
                    return None

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
            return intent

        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("RiskGateway encountered unexpected error: %s", exc)
            decision.update(status="rejected", reason="exception", error=str(exc))
            self._reject(decision)
            return None

    def get_risk_limits(self) -> dict[str, Any]:
        """Return a snapshot of configured limits and recent telemetry."""

        limits = self._build_limits_snapshot()
        payload: dict[str, Any] = {"limits": dict(limits), "telemetry": dict(self.telemetry)}
        risk_reference = self._resolve_risk_reference()
        if risk_reference is not None:
            summary = risk_reference.get("risk_config_summary")
            if isinstance(summary, Mapping):
                payload["risk_config_summary"] = dict(summary)
            payload["runbook"] = str(risk_reference.get("risk_api_runbook", RISK_API_RUNBOOK))
            config_payload = risk_reference.get("risk_config")
            if isinstance(config_payload, Mapping):
                payload["risk_config"] = dict(config_payload)
        elif self._risk_config is not None:
            payload["risk_config_summary"] = summarise_risk_config(self._risk_config)
            payload["runbook"] = RISK_API_RUNBOOK
        else:
            payload["runbook"] = RISK_API_RUNBOOK
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

    def _resolve_risk_reference(self) -> dict[str, Any] | None:
        """Build a deterministic risk reference payload for telemetry surfaces."""

        config = self._risk_config
        if config is None:
            return None
        try:
            summary = summarise_risk_config(config)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to summarise risk config for reference", exc_info=True)
            return None

        reference: dict[str, Any] = {
            "risk_config_summary": summary,
            "risk_api_runbook": RISK_API_RUNBOOK,
        }
        try:
            config_payload = config.dict()
        except Exception:  # pragma: no cover - diagnostics only
            config_payload = None
        if isinstance(config_payload, Mapping):
            reference["risk_config"] = dict(config_payload)
        return reference

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reject(self, decision: Mapping[str, Any]) -> None:
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
        return None

    def _risk_reference_base(self) -> dict[str, object]:
        if self._risk_reference_cache is None:
            reference: dict[str, object] = {"risk_api_runbook": RISK_API_RUNBOOK}
            if self._risk_config is not None:
                try:
                    reference["risk_config_summary"] = summarise_risk_config(self._risk_config)
                except Exception:  # pragma: no cover - defensive metadata guard
                    logger.debug(
                        "Failed to summarise risk config for risk reference cache",
                        exc_info=True,
                    )
            elif self.risk_policy is not None:
                reference["risk_config_summary"] = self._policy_summary()
            self._risk_reference_cache = reference
        return dict(self._risk_reference_cache)

    def _augment_with_risk_reference(self, decision: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(decision)
        existing = payload.get("risk_reference")
        base_reference = self._risk_reference_base()
        if isinstance(existing, Mapping):
            merged = dict(base_reference)
            merged.update(existing)
            payload["risk_reference"] = merged
        else:
            payload["risk_reference"] = base_reference
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
        if not self.position_sizer:
            return quantity

        balance = _to_decimal(portfolio_state.get("equity"), default=Decimal("0"))
        if balance <= 0:
            return quantity

        market_price = self._extract_price(intent, portfolio_state)
        stop_loss_pct = self._extract_stop_loss_pct(intent, portfolio_state, market_price)
        recommended = self.position_sizer(balance, self.risk_per_trade, Decimal(str(stop_loss_pct)))

        decision["checks"].append(
            {
                "name": "position_sizer",
                "recommended": float(recommended),
                "requested": float(quantity),
            }
        )

        if quantity <= recommended:
            return quantity

        logger.info("RiskGateway clipping position size from %s to %s", quantity, recommended)
        return recommended

    async def _run_liquidity_probe(
        self, intent: Any, quantity: Decimal, decision: dict[str, Any]
    ) -> tuple[float | None, Mapping[str, Any] | None]:
        if not self.liquidity_prober:
            return None, None

        symbol = _extract_attr(intent, "symbol", default="UNKNOWN")
        side = str(_extract_attr(intent, "side", "direction", default="buy")).lower()
        market_price = self._extract_price(intent, {}) or 0.0
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
            except Exception:  # pragma: no cover - attribute blocked
                pass

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
                        except Exception:
                            pass
            except Exception:  # pragma: no cover
                pass

    def _evaluate_execution_cost(
        self,
        intent: Any,
        quantity: float,
        price: float,
        portfolio_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        metadata = _ensure_metadata(intent)
        context = self._build_execution_context(metadata, portfolio_state, quantity, price)

        execution_cfg = self._execution_config
        limits: ExecutionRiskLimits | None = execution_cfg.limits

        slippage_bps = estimate_slippage_bps(context, execution_cfg)
        commission_bps = estimate_commission_bps(execution_cfg)
        total_cost_bps = slippage_bps + commission_bps

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

        execution_metadata = {
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
            self._reject(decision)
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
            self._reject(decision)
            return False

        return True

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
        feature_source: Mapping[str, Any] | None = None
        for key in ("execution_context", "microstructure", "market_microstructure"):
            candidate = metadata.get(key)
            if isinstance(candidate, Mapping):
                feature_source = candidate
                break

        feature_source = feature_source or {}

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

    def _resolve_equity_for_execution(self, portfolio_state: Mapping[str, Any]) -> float:
        for key in ("equity", "cash", "account_equity", "total_equity"):
            equity = _as_float(portfolio_state.get(key), default=0.0)
            if equity > 0:
                return equity
        return 0.0
