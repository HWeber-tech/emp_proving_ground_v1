from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.config.risk.risk_config import RiskConfig


def _as_float(value: object | None, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_equity(state: Mapping[str, object]) -> float:
    equity = _as_float(state.get("equity"), default=0.0)
    if equity > 0.0:
        return equity
    cash = _as_float(state.get("cash"), default=0.0)
    return max(0.0, cash + _compute_total_exposure(state.get("open_positions")))


def _compute_total_exposure(positions: object) -> float:
    total = 0.0
    if isinstance(positions, Mapping):
        for payload in positions.values():
            if not isinstance(payload, Mapping):
                continue
            qty = abs(_as_float(payload.get("quantity"), default=0.0))
            price = _resolve_position_price(payload)
            if qty and price:
                total += qty * price
    return total


def _resolve_position_price(payload: Mapping[str, object] | None, fallback: float = 0.0) -> float:
    if not isinstance(payload, Mapping):
        return max(0.0, fallback)
    for key in ("last_price", "avg_price", "price"):
        price = _as_float(payload.get(key))
        if price > 0:
            return price
    current_value = _as_float(payload.get("current_value"))
    quantity = abs(_as_float(payload.get("quantity")))
    if quantity > 0 and current_value:
        return abs(current_value) / quantity
    return max(0.0, fallback)


def _locate_position(state: Mapping[str, object], symbol: str) -> Mapping[str, object] | None:
    positions = state.get("open_positions")
    if isinstance(positions, Mapping):
        payload = positions.get(symbol)
        if isinstance(payload, Mapping):
            return payload
    return None


@dataclass(frozen=True)
class RiskPolicyDecision:
    """Result of evaluating a potential trade against policy limits."""

    approved: bool
    reason: str | None
    checks: tuple[dict[str, object], ...]
    metadata: Mapping[str, object]
    violations: tuple[str, ...]


@dataclass(frozen=True)
class RiskPolicy:
    """Evaluate intent-level risk limits sourced from :class:`RiskConfig`."""

    max_risk_per_trade_pct: float
    max_total_exposure_pct: float
    max_leverage: float
    max_drawdown_pct: float
    min_position_size: float
    max_position_size: float
    mandatory_stop_loss: bool
    research_mode: bool

    @classmethod
    def from_config(cls, config: RiskConfig) -> "RiskPolicy":
        return cls(
            max_risk_per_trade_pct=float(config.max_risk_per_trade_pct),
            max_total_exposure_pct=float(config.max_total_exposure_pct),
            max_leverage=float(config.max_leverage),
            max_drawdown_pct=float(config.max_drawdown_pct),
            min_position_size=float(config.min_position_size),
            max_position_size=float(config.max_position_size),
            mandatory_stop_loss=bool(config.mandatory_stop_loss),
            research_mode=bool(config.research_mode),
        )

    def limit_snapshot(self) -> Mapping[str, float]:
        """Return a serialisable snapshot of configured policy limits."""

        return {
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "max_leverage": self.max_leverage,
            "max_risk_per_trade_pct": self.max_risk_per_trade_pct,
            "min_position_size": self.min_position_size,
            "max_position_size": self.max_position_size,
            "max_drawdown_pct": self.max_drawdown_pct,
        }

    def evaluate(
        self,
        *,
        symbol: str,
        quantity: float,
        price: float,
        stop_loss_pct: float,
        portfolio_state: Mapping[str, object],
    ) -> RiskPolicyDecision:
        """Assess whether the proposed trade respects configured limits."""

        checks: list[dict[str, object]] = []
        violations: list[str] = []
        metadata: dict[str, object] = {
            "symbol": symbol,
            "research_mode": self.research_mode,
        }

        abs_quantity = abs(quantity)

        def _record(
            name: str,
            value: float,
            threshold: float,
            *,
            status: str,
            extra: Mapping[str, object] | None = None,
        ) -> None:
            entry: dict[str, object] = {
                "name": name,
                "value": value,
                "threshold": threshold,
                "status": status,
            }
            if extra:
                entry.update(extra)
            checks.append(entry)
            if status == "violation":
                violations.append(name)

        min_status = "ok" if abs_quantity >= self.min_position_size else "violation"
        _record("policy.min_position_size", abs_quantity, self.min_position_size, status=min_status)

        max_status = "ok" if abs_quantity <= self.max_position_size else "violation"
        _record("policy.max_position_size", abs_quantity, self.max_position_size, status=max_status)

        resolved_price = price if price > 0 else self._resolve_price(symbol, price, portfolio_state)
        metadata["resolved_price"] = resolved_price
        if resolved_price <= 0:
            _record("policy.market_price", resolved_price, 0.0, status="violation")

        equity = _resolve_equity(portfolio_state)
        metadata["equity"] = equity
        if equity <= 0:
            _record("policy.equity", equity, 0.0, status="violation")

        existing_position = _locate_position(portfolio_state, symbol)
        existing_quantity = (
            _as_float(existing_position.get("quantity")) if existing_position else 0.0
        )
        existing_price = _resolve_position_price(existing_position, fallback=resolved_price)
        existing_notional = abs(existing_quantity) * existing_price

        total_exposure = _compute_total_exposure(portfolio_state.get("open_positions"))
        metadata["current_total_exposure"] = total_exposure
        metadata["existing_position_notional"] = existing_notional

        projected_quantity = existing_quantity + quantity
        projected_notional = abs(projected_quantity) * max(resolved_price, 0.0)
        metadata["projected_notional"] = projected_notional

        projected_total_exposure = max(0.0, total_exposure - existing_notional) + projected_notional
        metadata["projected_total_exposure"] = projected_total_exposure

        exposure_increase = max(0.0, projected_notional - existing_notional)
        metadata["exposure_increase"] = exposure_increase

        stop_loss = max(0.0, float(stop_loss_pct))
        metadata["stop_loss_pct"] = stop_loss
        if self.mandatory_stop_loss and stop_loss <= 0 and not self.research_mode:
            _record("policy.stop_loss", stop_loss, 0.0, status="violation")

        risk_budget = equity * self.max_risk_per_trade_pct
        estimated_risk = exposure_increase * stop_loss
        metadata["risk_budget"] = risk_budget
        metadata["estimated_risk"] = estimated_risk

        risk_status = "ok"
        if risk_budget > 0:
            risk_status = "ok" if estimated_risk <= risk_budget else "violation"
        elif exposure_increase > 0:
            risk_status = "violation"
        _record(
            "policy.max_risk_per_trade_pct",
            estimated_risk,
            risk_budget,
            status=risk_status,
            extra={"ratio": (estimated_risk / risk_budget) if risk_budget else None},
        )

        max_total = equity * self.max_total_exposure_pct
        metadata["max_total_exposure"] = max_total
        exposure_status = "ok"
        exposure_ratio = None
        if max_total > 0:
            exposure_ratio = projected_total_exposure / max_total
            if projected_total_exposure > max_total:
                exposure_status = "violation"
            elif exposure_ratio >= 0.8 and exposure_increase > 0:
                exposure_status = "warn"
        _record(
            "policy.max_total_exposure_pct",
            projected_total_exposure,
            max_total,
            status=exposure_status,
            extra={"ratio": exposure_ratio},
        )

        leverage = projected_total_exposure / equity if equity > 0 else float("inf")
        metadata["projected_leverage"] = leverage
        leverage_status = "ok"
        leverage_ratio = None
        if self.max_leverage > 0:
            leverage_ratio = leverage / self.max_leverage
            if leverage > self.max_leverage:
                leverage_status = "violation"
            elif leverage_ratio >= 0.8 and exposure_status != "violation":
                leverage_status = "warn"
        _record(
            "policy.max_leverage",
            leverage,
            self.max_leverage,
            status=leverage_status,
            extra={"ratio": leverage_ratio},
        )

        current_drawdown = _as_float(portfolio_state.get("current_daily_drawdown"), default=0.0)
        drawdown_status = "ok" if current_drawdown <= self.max_drawdown_pct else "violation"
        _record(
            "policy.max_drawdown_pct",
            current_drawdown,
            self.max_drawdown_pct,
            status=drawdown_status,
        )

        approved = not violations and exposure_status != "violation" and risk_status != "violation"
        reason = violations[0] if violations else None

        if not approved and self.research_mode:
            approved = True
        elif not approved and reason is None:
            reason = "policy_violation"

        metadata["violations"] = tuple(violations)

        return RiskPolicyDecision(
            approved=approved,
            reason=reason,
            checks=tuple(checks),
            metadata=metadata,
            violations=tuple(violations),
        )

    def _resolve_price(
        self, symbol: str, price: float, portfolio_state: Mapping[str, object]
    ) -> float:
        position = _locate_position(portfolio_state, symbol)
        resolved = _resolve_position_price(position, fallback=price)
        if resolved > 0:
            return resolved
        fallback_price = _as_float(portfolio_state.get("current_price"))
        if fallback_price > 0:
            return fallback_price
        return price


__all__ = ["RiskPolicy", "RiskPolicyDecision"]
