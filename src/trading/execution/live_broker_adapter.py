"""Live broker execution adapter with risk policy enforcement.

This adapter wraps a broker interface (typically the production FIX stack) and
re-validates intents against the trading risk gateway immediately before
submission.  It preserves the telemetry surface provided by the paper trading
adapter while adding policy snapshots and deterministic rejection metadata so
operators can audit why an order was blocked at the final hop.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from .paper_broker_adapter import PaperBrokerExecutionAdapter, PaperBrokerError
from src.trading.risk.risk_policy import RiskPolicy, RiskPolicyDecision

__all__ = ["LiveBrokerError", "LiveBrokerExecutionAdapter"]


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _extract(intent: Any, *names: str, default: Any = None) -> Any:
    if isinstance(intent, Mapping):
        for name in names:
            if name in intent:
                return intent[name]
    for name in names:
        if hasattr(intent, name):
            return getattr(intent, name)
    return default


def _extract_metadata(intent: Any) -> MutableMapping[str, Any] | None:
    candidate = None
    if isinstance(intent, Mapping):
        candidate = intent.get("metadata")
    elif hasattr(intent, "metadata"):
        candidate = getattr(intent, "metadata")
    if isinstance(candidate, MutableMapping):
        return candidate
    return None


def _normalise_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if is_dataclass(value):
        return _normalise_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalise_value(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_value(item) for item in value]
    return str(value)


def _serialise_mapping(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if is_dataclass(payload):
        payload = asdict(payload)
    if isinstance(payload, Mapping):
        return {str(key): _normalise_value(value) for key, value in payload.items()}
    return None


def _serialise_policy_decision(decision: Any) -> dict[str, Any] | None:
    if decision is None:
        return None
    if isinstance(decision, RiskPolicyDecision):
        serialised = {
            "approved": decision.approved,
            "reason": decision.reason,
            "metadata": _serialise_mapping(decision.metadata) or {},
            "violations": list(decision.violations),
            "checks": [
                _serialise_mapping(check) or {}
                for check in decision.checks
            ],
        }
        return serialised
    if is_dataclass(decision):
        return _serialise_policy_decision(asdict(decision))
    if isinstance(decision, Mapping):
        return _serialise_mapping(decision)
    return {"value": _normalise_value(decision)}


def _resolve_stop_loss_pct(intent: Any) -> float:
    candidate = _extract(intent, "stop_loss_pct", "stop_loss", default=None)
    if candidate is not None:
        return _as_float(candidate, default=0.0)

    metadata = _extract_metadata(intent)
    if metadata:
        for key in ("stop_loss_pct", "stop_loss"):
            if key in metadata:
                return _as_float(metadata[key], default=0.0)
        risk_section = metadata.get("risk_assessment")
        if isinstance(risk_section, Mapping):
            policy_section = risk_section.get("policy")
            if isinstance(policy_section, Mapping):
                value = policy_section.get("stop_loss_pct")
                if value is not None:
                    return _as_float(value, default=0.0)
    return 0.0


def _symbol_from_portfolio(portfolio_state: Mapping[str, Any], symbol: str) -> Mapping[str, Any] | None:
    positions = portfolio_state.get("open_positions")
    if isinstance(positions, Mapping):
        payload = positions.get(symbol)
        if isinstance(payload, Mapping):
            return payload
    return None


class LiveBrokerError(PaperBrokerError):
    """Raised when the live broker adapter blocks or fails an order."""


class LiveBrokerExecutionAdapter(PaperBrokerExecutionAdapter):
    """Execution adapter that re-validates intents before hitting a live broker."""

    def __init__(
        self,
        *,
        broker_interface: Any,
        portfolio_monitor: Any,
        risk_gateway: Any | None,
        risk_policy: RiskPolicy | None,
        order_timeout: float | None = 5.0,
        failover_threshold: int = 3,
        failover_cooldown_seconds: float = 30.0,
        risk_block_cooldown_seconds: float | None = 5.0,
    ) -> None:
        super().__init__(
            broker_interface=broker_interface,
            portfolio_monitor=portfolio_monitor,
            order_timeout=order_timeout,
            failover_threshold=failover_threshold,
            failover_cooldown_seconds=failover_cooldown_seconds,
        )
        self._risk_gateway = risk_gateway
        self._risk_policy = risk_policy
        self._risk_block_cooldown = max(0.0, float(risk_block_cooldown_seconds or 0.0))
        self._risk_block_until: datetime | None = None
        self._risk_blocks = 0
        self._last_risk_violation: dict[str, Any] | None = None
        self._last_policy_snapshot: dict[str, Any] | None = None
        self._policy_history: deque[dict[str, Any]] = deque(maxlen=64)
        self._last_policy_warning: dict[str, Any] | None = None

    async def process_order(self, intent: Any) -> Any:
        if self._risk_gateway is None:
            return await self._delegate_to_paper_adapter(intent)

        portfolio_state = self.portfolio_monitor.get_state()
        revalidation_reason: str | None = None

        validated_intent: Any | None
        try:
            validated_intent = await self._risk_gateway.validate_trade_intent(
                intent,
                portfolio_state,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            self._record_risk_violation(
                reason="risk_gateway_error",
                details={"error": str(exc)},
                portfolio_state=portfolio_state,
            )
            raise LiveBrokerError("Risk gateway validation failed") from exc

        if not validated_intent:
            decision = self._safe_call(self._risk_gateway, "get_last_decision")
            reason = None
            if isinstance(decision, Mapping):
                reason = str(decision.get("reason") or "") or None
                if not reason:
                    policy_section = decision.get("policy")
                    if isinstance(policy_section, Mapping):
                        reason = str(policy_section.get("reason") or "") or None
            if reason == "policy.market_price":
                validated_intent = intent
                revalidation_reason = reason
            else:
                details: dict[str, Any] = {"decision": _serialise_mapping(decision) or {}}
                if reason == "policy.min_position_size":
                    details.setdefault("delegated_to", "paper")
                    self._record_risk_near_miss(
                        intent=intent,
                        reason=reason,
                        details=details,
                        portfolio_state=portfolio_state,
                    )
                    return await self._delegate_to_paper_adapter(intent)
                self._record_risk_violation(
                    reason="risk_gateway_rejected",
                    details=details,
                    portfolio_state=portfolio_state,
                )
                raise LiveBrokerError(reason or "Risk gateway rejected trade")

        policy_decision = self._enforce_policy(validated_intent, portfolio_state)
        if policy_decision is not None and not policy_decision.approved:
            summary = _serialise_policy_decision(policy_decision) or {}
            summary.setdefault("violations", list(policy_decision.violations))
            self._record_risk_violation(
                reason=policy_decision.reason or "risk_policy_violation",
                details={"policy_decision": summary},
                portfolio_state=portfolio_state,
            )
            raise LiveBrokerError(policy_decision.reason or "Risk policy violation")

        symbol = str(_extract(validated_intent, "symbol", "instrument", "asset", default="UNKNOWN")).upper()
        side = str(_extract(validated_intent, "side", "direction", default="BUY")).upper()
        quantity = _as_float(
            _extract(validated_intent, "quantity", "size", "volume", default=0.0),
            default=0.0,
        )

        self._record_policy_snapshot(
            policy_decision=policy_decision,
            portfolio_state=portfolio_state,
            symbol=symbol,
            side=side,
            quantity=quantity,
            revalidation_reason=revalidation_reason,
        )
        self._clear_risk_violation()

        try:
            order_id = await super().process_order(validated_intent)
        except PaperBrokerError as exc:
            raise LiveBrokerError(str(exc)) from exc

        if isinstance(order_id, Mapping):  # pragma: no cover - defensive guard
            self._annotate_policy_snapshot({"order_reference": _serialise_mapping(order_id)})
        else:
            self._annotate_policy_snapshot({"order_id": order_id})
        return order_id

    async def _delegate_to_paper_adapter(self, intent: Any) -> Any:
        try:
            return await super().process_order(intent)
        except PaperBrokerError as exc:
            raise LiveBrokerError(str(exc)) from exc

    def _clear_risk_violation(self) -> None:
        self._last_risk_violation = None
        self._risk_block_until = None

    def _annotate_policy_snapshot(self, payload: Mapping[str, Any]) -> None:
        if not payload:
            return
        base = dict(self._last_policy_snapshot or {})
        base.update({str(key): _normalise_value(value) for key, value in payload.items()})
        base.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._last_policy_snapshot = base
        if self._policy_history:
            self._policy_history.pop()
        self._policy_history.append(dict(base))

    def _safe_call(self, target: Any, name: str) -> Any:
        attr = getattr(target, name, None)
        if not callable(attr):
            return None
        try:
            return attr()
        except Exception:  # pragma: no cover - defensive guard
            return None

    def _enforce_policy(
        self,
        intent: Any,
        portfolio_state: Mapping[str, Any],
    ) -> RiskPolicyDecision | None:
        if self._risk_policy is None:
            decision = self._safe_call(self._risk_gateway, "get_last_policy_decision")
            if isinstance(decision, RiskPolicyDecision):
                return decision
            return None

        symbol = str(_extract(intent, "symbol", "instrument", "asset", default="")).upper()
        quantity = _as_float(
            _extract(intent, "quantity", "size", "volume", default=0.0),
            default=0.0,
        )
        price = _as_float(_extract(intent, "price", "reference_price"), default=0.0)
        metadata = _extract_metadata(intent)
        self._last_policy_warning = None
        if price <= 0 and metadata:
            for candidate_key in ("price", "reference_price"):
                if candidate_key in metadata:
                    price_candidate = _as_float(metadata[candidate_key], default=0.0)
                    if price_candidate > 0:
                        price = price_candidate
                        break
            if price <= 0:
                risk_section = metadata.get("risk_assessment")
                if isinstance(risk_section, Mapping):
                    policy_section = risk_section.get("policy")
                    if isinstance(policy_section, Mapping):
                        resolved = _as_float(policy_section.get("resolved_price"), default=0.0)
                        if resolved > 0:
                            price = resolved
                    reference = risk_section.get("reference_price")
                    resolved_reference = _as_float(reference, default=0.0)
                    if price <= 0 and resolved_reference > 0:
                        price = resolved_reference
        stop_loss_pct = _resolve_stop_loss_pct(intent)

        decision = self._risk_policy.evaluate(
            symbol=symbol,
            quantity=quantity,
            price=price,
            stop_loss_pct=stop_loss_pct,
            portfolio_state=portfolio_state,
        )
        if (
            not decision.approved
            and decision.reason == "policy.market_price"
            and price <= 0.0
        ):
            resolved_price = _as_float(decision.metadata.get("resolved_price"), default=0.0)
            if resolved_price <= 0.0:
                self._last_policy_warning = {
                    "reason": decision.reason,
                    "metadata": _serialise_mapping(decision.metadata) or {},
                }
                return None
        return decision

    def _record_policy_snapshot(
        self,
        *,
        policy_decision: RiskPolicyDecision | None,
        portfolio_state: Mapping[str, Any],
        symbol: str,
        side: str,
        quantity: float,
        revalidation_reason: str | None,
    ) -> None:
        snapshot: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": _as_float(portfolio_state.get("equity"), default=0.0),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "risk_blocks": self._risk_blocks,
        }

        open_positions = portfolio_state.get("open_positions")
        if isinstance(open_positions, Mapping):
            snapshot["open_positions"] = len(open_positions)
        else:
            snapshot["open_positions"] = 0

        decision = self._safe_call(self._risk_gateway, "get_last_decision")
        if decision:
            snapshot["risk_decision"] = _serialise_mapping(decision)

        policy_snapshot = self._safe_call(self._risk_gateway, "get_last_policy_snapshot")
        if policy_snapshot:
            snapshot["risk_gateway_policy"] = _serialise_mapping(policy_snapshot)

        serialised_policy = _serialise_policy_decision(policy_decision)
        if serialised_policy:
            snapshot["policy_decision"] = serialised_policy
        elif policy_snapshot and isinstance(policy_snapshot, Mapping):
            snapshot.setdefault("policy_decision", _serialise_mapping(policy_snapshot))

        position_state = _symbol_from_portfolio(portfolio_state, symbol)
        if position_state:
            snapshot["position"] = _serialise_mapping(position_state)

        if revalidation_reason:
            snapshot["revalidation_reason"] = revalidation_reason
        if self._last_policy_warning:
            snapshot["policy_warning"] = dict(self._last_policy_warning)
            self._last_policy_warning = None

        self._last_policy_snapshot = snapshot
        self._policy_history.append(snapshot)

    def _record_risk_near_miss(
        self,
        *,
        intent: Any,
        reason: str | None,
        details: Mapping[str, Any] | None,
        portfolio_state: Mapping[str, Any],
    ) -> None:
        timestamp = datetime.now(timezone.utc)
        symbol = str(
            _extract(intent, "symbol", "instrument", "asset", default="UNKNOWN")
        ).upper()
        side = str(_extract(intent, "side", "direction", default="BUY")).upper()
        quantity = _as_float(
            _extract(intent, "quantity", "size", "volume", default=0.0),
            default=0.0,
        )
        snapshot_reason = reason or "risk_gateway_near_miss"
        snapshot: dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "reason": snapshot_reason,
            "severity": "near_miss",
            "delegated_route": "paper",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "equity": _as_float(portfolio_state.get("equity"), default=0.0),
            "risk_blocks": self._risk_blocks,
        }
        if details:
            snapshot["details"] = {
                str(key): _normalise_value(value) for key, value in details.items()
            }
        open_positions = portfolio_state.get("open_positions")
        if isinstance(open_positions, Mapping):
            snapshot["open_positions"] = len(open_positions)

        self._last_policy_snapshot = snapshot
        self._policy_history.append(snapshot)
        self._last_policy_warning = {
            "reason": snapshot_reason,
            "metadata": snapshot.get("details", {}),
        }

    def _record_risk_violation(
        self,
        *,
        reason: str,
        details: Mapping[str, Any] | None,
        portfolio_state: Mapping[str, Any],
    ) -> None:
        self._risk_blocks += 1
        timestamp = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "reason": reason,
            "timestamp": timestamp.isoformat(),
            "equity": _as_float(portfolio_state.get("equity"), default=0.0),
            "risk_blocks": self._risk_blocks,
        }
        if details:
            payload["details"] = {
                str(key): _normalise_value(value) for key, value in details.items()
            }
        open_positions = portfolio_state.get("open_positions")
        if isinstance(open_positions, Mapping):
            payload["open_positions"] = len(open_positions)
        self._last_risk_violation = payload
        self._last_error = {
            "stage": "risk_policy",
            "message": reason,
            "timestamp": payload["timestamp"],
            "details": payload.get("details", {}),
        }
        self._last_error_time = timestamp
        self._error_history.appendleft(dict(self._last_error))

        if self._risk_block_cooldown > 0.0:
            self._risk_block_until = timestamp + timedelta(seconds=self._risk_block_cooldown)

        snapshot = {
            "timestamp": payload["timestamp"],
            "risk_blocks": self._risk_blocks,
            "reason": reason,
        }
        if "details" in payload:
            snapshot["details"] = dict(payload["details"])
        if "equity" in payload:
            snapshot["equity"] = payload["equity"]
        if "open_positions" in payload:
            snapshot["open_positions"] = payload["open_positions"]
        self._last_policy_snapshot = snapshot
        self._policy_history.append(snapshot)

    def describe_metrics(self) -> Mapping[str, Any]:
        metrics = dict(super().describe_metrics())
        metrics["risk_blocks"] = self._risk_blocks
        metrics["risk_block_cooldown_seconds"] = self._risk_block_cooldown
        metrics["risk_block_active"] = False
        if self._risk_block_until is not None:
            now = datetime.now(timezone.utc)
            if now < self._risk_block_until:
                metrics["risk_block_active"] = True
                metrics["risk_block_retry_in_seconds"] = (
                    self._risk_block_until - now
                ).total_seconds()
            else:
                self._risk_block_until = None
        if self._last_policy_snapshot:
            metrics["last_policy_snapshot"] = dict(self._last_policy_snapshot)
        return metrics

    def describe_policy_snapshot(self) -> Mapping[str, Any] | None:
        if self._last_policy_snapshot is None:
            return None
        return dict(self._last_policy_snapshot)

    def policy_history(self) -> Sequence[Mapping[str, Any]]:
        return list(self._policy_history)

    def should_block_orders(self, intent: Any | None = None) -> Mapping[str, Any] | None:
        failover_block = super().should_block_orders(intent)
        risk_block = self._build_risk_block_payload()
        if failover_block and risk_block:
            combined = dict(failover_block)
            combined.setdefault("reason", combined.get("reason", "broker_blocked"))
            combined["risk_policy"] = risk_block
            return combined
        if risk_block:
            return risk_block
        return failover_block

    def _build_risk_block_payload(self) -> Mapping[str, Any] | None:
        payload = self._last_risk_violation
        if not payload:
            return None
        snapshot = dict(payload)
        if self._risk_block_until is not None:
            now = datetime.now(timezone.utc)
            if now >= self._risk_block_until:
                self._risk_block_until = None
            else:
                snapshot["retry_in_seconds"] = (
                    self._risk_block_until - now
                ).total_seconds()
                snapshot["risk_block_resumes_at"] = self._risk_block_until.isoformat()
        snapshot.setdefault("risk_blocks", self._risk_blocks)
        snapshot.setdefault("reason", "risk_policy_violation")
        return snapshot
