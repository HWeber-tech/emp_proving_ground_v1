"""Canonical helpers for interacting with trading risk interfaces.

The roadmap calls for a deterministic Risk API that other runtime components can
depend on.  Historically, the runtime builder reached into ``TradingManager``
internals to coerce a :class:`~src.config.risk.risk_config.RiskConfig` payload
and construct ad-hoc metadata.  This module centralises that contract so every
consumer resolves risk configuration through the same hardened path and receives
consistent metadata snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from collections.abc import Mapping

from pydantic import ValidationError

from src.config.risk.risk_config import RiskConfig


class RiskApiError(RuntimeError):
    """Raised when a trading manager violates the risk API contract."""


def _extract_risk_status(trading_manager: Any) -> Mapping[str, object] | None:
    """Best-effort retrieval of a risk status payload from a trading manager."""

    getter = getattr(trading_manager, "get_risk_status", None)
    if callable(getter):
        try:
            payload = getter()
        except Exception as exc:  # pragma: no cover - diagnostic guardrail
            raise RiskApiError("Trading manager risk status retrieval failed") from exc
        if isinstance(payload, Mapping):
            return payload
    return None


def resolve_trading_risk_config(trading_manager: Any) -> RiskConfig:
    """Resolve the canonical :class:`RiskConfig` for a trading manager.

    The function defends against partially-initialised managers by checking the
    private ``_risk_config`` attribute first (which canonical implementations
    expose) and falling back to :meth:`TradingManager.get_risk_status`.  Any
    malformed payload raises :class:`RiskApiError` so upstream supervisors fail
    deterministically.
    """

    candidate = getattr(trading_manager, "_risk_config", None)
    if isinstance(candidate, RiskConfig):
        return candidate

    status_payload = _extract_risk_status(trading_manager)
    if status_payload is not None:
        config_payload = status_payload.get("risk_config")
        if isinstance(config_payload, Mapping):
            try:
                return RiskConfig.parse_obj(config_payload)
            except ValidationError as exc:
                raise RiskApiError("Trading manager risk configuration is invalid") from exc

    raise RiskApiError("Trading manager does not expose a canonical RiskConfig")


def summarise_risk_config(config: RiskConfig) -> dict[str, object]:
    """Render a serialisable summary of the supplied risk configuration."""

    return {
        "max_risk_per_trade_pct": float(config.max_risk_per_trade_pct),
        "max_total_exposure_pct": float(config.max_total_exposure_pct),
        "max_leverage": float(config.max_leverage),
        "max_drawdown_pct": float(config.max_drawdown_pct),
        "min_position_size": int(config.min_position_size),
        "max_position_size": int(config.max_position_size),
        "mandatory_stop_loss": bool(config.mandatory_stop_loss),
        "research_mode": bool(config.research_mode),
    }


@dataclass(frozen=True)
class TradingRiskInterface:
    """Deterministic view over a trading manager's risk configuration surface."""

    config: RiskConfig
    status: Mapping[str, object] | None = None

    def summary(self) -> dict[str, object]:
        """Build a metadata summary incorporating policy metadata when available."""

        payload = summarise_risk_config(self.config)
        status = self.status
        if status is None:
            return payload

        policy_limits = status.get("policy_limits") if isinstance(status, Mapping) else None
        if isinstance(policy_limits, Mapping):
            payload["policy_limits"] = dict(policy_limits)

        research_mode = status.get("policy_research_mode") if isinstance(status, Mapping) else None
        if research_mode is not None:
            payload["policy_research_mode"] = bool(research_mode)

        last_snapshot = status.get("snapshot") if isinstance(status, Mapping) else None
        if isinstance(last_snapshot, Mapping):
            payload["latest_snapshot"] = dict(last_snapshot)

        return payload


def resolve_trading_risk_interface(trading_manager: Any) -> TradingRiskInterface:
    """Resolve a :class:`TradingRiskInterface` for the supplied manager."""

    config = resolve_trading_risk_config(trading_manager)
    status = _extract_risk_status(trading_manager)
    return TradingRiskInterface(config=config, status=status)


def build_runtime_risk_metadata(trading_manager: Any) -> dict[str, object]:
    """Produce runtime metadata used by supervisors and telemetry surfaces."""

    interface = resolve_trading_risk_interface(trading_manager)
    return interface.summary()


__all__ = [
    "RiskApiError",
    "TradingRiskInterface",
    "build_runtime_risk_metadata",
    "resolve_trading_risk_config",
    "resolve_trading_risk_interface",
    "summarise_risk_config",
]

