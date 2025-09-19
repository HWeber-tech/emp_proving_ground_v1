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
from typing import Any, Callable, Mapping, MutableMapping, Optional, Protocol, cast


logger = logging.getLogger(__name__)


class SupportsLiquidityProbing(Protocol):
    """Protocol for optional liquidity probing capabilities."""

    async def probe_liquidity(
        self, symbol: str, price_levels: list[float], side: str
    ) -> Mapping[float, float]:
        ...

    def calculate_liquidity_confidence_score(
        self, probe_results: Mapping[float, float], intended_volume: float
    ) -> float:
        ...

    def get_probe_summary(self, probe_results: Mapping[float, float]) -> Mapping[str, Any]:
        ...


class SupportsStrategyRegistry(Protocol):
    """Protocol capturing the single method RiskGateway relies on."""

    def get_strategy(self, strategy_id: str) -> Mapping[str, Any] | None: ...


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
        meta = intent.get("metadata")
        if isinstance(meta, MutableMapping):
            return meta
        meta = {}
        intent["metadata"] = meta
        return meta

    meta = getattr(intent, "metadata", None)
    if isinstance(meta, MutableMapping):
        return meta

    # Attach a new metadata mapping for objects lacking one
    meta_map: dict[str, Any] = {}
    try:
        setattr(intent, "metadata", meta_map)
    except Exception:  # pragma: no cover - best effort when attribute blocked
        pass
    return meta_map


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

        self.telemetry: dict[str, Any] = {
            "total_checks": 0,
            "approved": 0,
            "rejected": 0,
            "last_decision": None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def validate_trade_intent(
        self, intent: Any, portfolio_state: Mapping[str, Any] | None
    ) -> Any | None:
        """Return an enriched intent if all risk checks pass, otherwise ``None``."""

        self.telemetry["total_checks"] += 1
        decision: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": _extract_attr(intent, "symbol", default="UNKNOWN"),
            "strategy_id": _extract_attr(intent, "strategy_id", "strategy", default=""),
            "checks": [],
        }

        try:
            if not self._check_strategy(decision["strategy_id"]):
                decision.update(status="rejected", reason="strategy_disabled")
                return self._reject(decision)

            quantity = self._extract_quantity(intent)
            confidence = _as_float(
                _extract_attr(intent, "confidence", "score", default=1.0), default=1.0
            )
            if confidence < self.min_intent_confidence:
                decision.update(status="rejected", reason="low_confidence")
                decision["checks"].append({
                    "name": "confidence_floor",
                    "value": confidence,
                    "threshold": self.min_intent_confidence,
                })
                return self._reject(decision)

            portfolio_state = portfolio_state or {}
            if not self._check_drawdown(portfolio_state, decision):
                decision.update(status="rejected", reason="max_drawdown_exceeded")
                return self._reject(decision)

            if not self._check_open_positions(portfolio_state, decision):
                decision.update(status="rejected", reason="too_many_open_positions")
                return self._reject(decision)

            adjusted_quantity = self._apply_position_sizing(intent, portfolio_state, decision)

            liquidity_summary: Mapping[str, Any] | None = None
            liquidity_score: float | None = None
            if self.liquidity_prober and adjusted_quantity >= self.liquidity_probe_threshold:
                liquidity_score, liquidity_summary = await self._run_liquidity_probe(
                    intent, adjusted_quantity, decision
                )
                if liquidity_score is not None and liquidity_score < self.min_liquidity_confidence:
                    decision.update(status="rejected", reason="insufficient_liquidity")
                    return self._reject(decision)

            self._enrich_intent(intent, adjusted_quantity, liquidity_score, decision)

            if liquidity_summary:
                decision["checks"].append({
                    "name": "liquidity_probe",
                    "summary": dict(liquidity_summary),
                })

            decision.update(status="approved", quantity=float(adjusted_quantity))
            self.telemetry["approved"] += 1
            self.telemetry["last_decision"] = decision
            return intent

        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("RiskGateway encountered unexpected error: %s", exc)
            decision.update(status="rejected", reason="exception", error=str(exc))
            return self._reject(decision)

    def get_risk_limits(self) -> dict[str, Any]:
        """Return a snapshot of configured limits and recent telemetry."""

        limits = {
            "max_open_positions": self.max_open_positions,
            "max_daily_drawdown": self.max_daily_drawdown,
            "min_intent_confidence": self.min_intent_confidence,
            "liquidity_probe_threshold": self.liquidity_probe_threshold,
            "min_liquidity_confidence": self.min_liquidity_confidence,
            "risk_per_trade": float(self.risk_per_trade),
        }
        return {"limits": limits, "telemetry": dict(self.telemetry)}

    def get_last_decision(self) -> Mapping[str, Any] | None:
        """Expose the most recent risk decision record."""

        return cast(Optional[Mapping[str, Any]], self.telemetry.get("last_decision"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reject(self, decision: Mapping[str, Any]) -> None:
        self.telemetry["rejected"] += 1
        self.telemetry["last_decision"] = dict(decision)
        logger.info("RiskGateway rejected trade: %s", decision)
        return None

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

        logger.info(
            "RiskGateway clipping position size from %s to %s", quantity, recommended
        )
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
        price_levels = [
            market_price + spread * offset if side.startswith("buy") else market_price - spread * offset
            for offset in range(-2, 3)
        ] if market_price else []

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
    ) -> None:
        _set_attr(intent, quantity, "quantity", "size", "volume")

        metadata = _ensure_metadata(intent)
        metadata.setdefault("risk_assessment", {})
        assessment = metadata["risk_assessment"]
        if isinstance(assessment, Mapping):
            assessment = dict(assessment)
            metadata["risk_assessment"] = assessment

        assessment.update({
            "checks": list(decision.get("checks", [])),
            "final_quantity": float(quantity),
        })
        if liquidity_score is not None:
            assessment["liquidity_confidence"] = float(liquidity_score)

        if hasattr(intent, "liquidity_confidence_score"):
            try:
                setattr(intent, "liquidity_confidence_score", liquidity_score)
            except Exception:  # pragma: no cover - attribute blocked
                pass

        if is_dataclass(intent):
            try:
                # Persist enrichment for frozen dataclasses by rebuilding
                if getattr(intent, "__dataclass_params__", None).frozen:  # type: ignore[attr-defined]
                    data = asdict(intent)
                    data.update({
                        "metadata": metadata,
                        "liquidity_confidence_score": liquidity_score,
                        "quantity": quantity,
                        "size": quantity,
                    })
                    # dataclasses.replace would be cleaner but may not support arbitrary attrs
                    for field, value in data.items():
                        try:
                            setattr(intent, field, value)
                        except Exception:
                            pass
            except Exception:  # pragma: no cover
                pass

