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
from decimal import Decimal
import logging
from typing import Any

from collections.abc import Mapping

from pydantic import ValidationError

from src.config.risk.risk_config import RiskConfig


logger = logging.getLogger(__name__)

_DEFAULT_RUNBOOK = "docs/operations/runbooks/risk_api_contract.md"


# Public alias so runtime supervisors can surface consistent remediation links.
RISK_API_RUNBOOK = _DEFAULT_RUNBOOK


class RiskApiError(RuntimeError):
    """Raised when a trading manager violates the risk API contract."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
        runbook: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})
        self.runbook = runbook or _DEFAULT_RUNBOOK

    def to_metadata(self) -> dict[str, object]:
        """Serialise the error details for telemetry surfaces."""

        payload: dict[str, object] = {"message": self.message, "runbook": self.runbook}
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def _extract_risk_status(trading_manager: Any) -> Mapping[str, object] | None:
    """Best-effort retrieval of a risk status payload from a trading manager."""

    getter = getattr(trading_manager, "get_risk_status", None)
    if callable(getter):
        try:
            payload = getter()
        except Exception as exc:  # pragma: no cover - diagnostic guardrail
            raise RiskApiError(
                "Trading manager risk status retrieval failed",
                details={"manager": type(trading_manager).__name__},
            ) from exc
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
                logger.warning(
                    "Risk API rejected invalid trading manager risk configuration",
                    extra={
                        "risk.manager": type(trading_manager).__name__,
                        "risk.invalid_fields": sorted(config_payload.keys()),
                    },
                )
                raise RiskApiError(
                    "Trading manager risk configuration is invalid",
                    details={"manager": type(trading_manager).__name__},
                ) from exc

    raise RiskApiError(
        "Trading manager does not expose a canonical RiskConfig",
        details={"manager": type(trading_manager).__name__},
    )


def summarise_risk_config(config: RiskConfig) -> dict[str, object]:
    """Render a serialisable summary of the supplied risk configuration."""

    summary: dict[str, object] = {
        "max_risk_per_trade_pct": float(config.max_risk_per_trade_pct),
        "max_total_exposure_pct": float(config.max_total_exposure_pct),
        "max_leverage": float(config.max_leverage),
        "max_drawdown_pct": float(config.max_drawdown_pct),
        "min_position_size": int(config.min_position_size),
        "max_position_size": int(config.max_position_size),
        "mandatory_stop_loss": bool(config.mandatory_stop_loss),
        "research_mode": bool(config.research_mode),
        "target_volatility_pct": float(config.target_volatility_pct),
        "volatility_window": int(config.volatility_window),
        "max_volatility_leverage": float(config.max_volatility_leverage),
        "volatility_annualisation_factor": float(
            config.volatility_annualisation_factor
        ),
        "runbook": RISK_API_RUNBOOK,
    }

    if config.sector_exposure_limits:
        summary["sector_exposure_limits"] = {
            sector: float(limit) for sector, limit in config.sector_exposure_limits.items()
        }
        sector_total = sum(config.sector_exposure_limits.values(), Decimal("0"))
        summary["sector_budget_total_pct"] = float(sector_total)
        headroom = config.max_total_exposure_pct - sector_total
        if headroom < Decimal("0"):
            headroom = Decimal("0")
        summary["sector_headroom_pct"] = float(headroom)
        summary["sector_headroom_ratio"] = float(
            headroom / config.max_total_exposure_pct
        )
        max_sector_limit = max(
            config.sector_exposure_limits.values(), default=Decimal("0")
        )
        summary["max_sector_utilisation_ratio"] = float(
            max_sector_limit / config.max_total_exposure_pct
        )

    if config.instrument_sector_map:
        instrument_sector_map = dict(config.instrument_sector_map)
        summary["instrument_sector_map"] = instrument_sector_map
        sector_instrument_counts: dict[str, int] = {}
        for sector in instrument_sector_map.values():
            sector_instrument_counts[sector] = sector_instrument_counts.get(sector, 0) + 1
        summary["sector_instrument_counts"] = sector_instrument_counts

    return summary


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


def _merge_mapping(target: dict[str, object], source: Mapping[str, object]) -> None:
    """Recursively merge ``source`` into ``target`` while copying nested mappings."""

    for key, value in source.items():
        normalised_key = "risk_api_runbook" if key == "runbook" else key
        if isinstance(value, Mapping):
            existing = target.get(normalised_key)
            nested: dict[str, object]
            if isinstance(existing, Mapping):
                nested = dict(existing)
            else:
                nested = {}
            _merge_mapping(nested, value)
            target[normalised_key] = nested
        elif value is not None:
            target[normalised_key] = value


def merge_risk_references(
    *references: Mapping[str, object] | None,
    runbook: str | None = None,
) -> dict[str, object]:
    """Merge multiple risk reference payloads into a canonical mapping.

    Each ``references`` entry may be ``None`` or a mapping containing nested
    structures such as ``risk_config_summary`` or ``risk_interface_status``.  The
    function copies nested mappings so callers can safely mutate the returned
    payload without affecting the inputs and ensures the canonical
    ``risk_api_runbook`` key is always present (falling back to
    :data:`RISK_API_RUNBOOK` when the sources omit a runbook).
    """

    merged: dict[str, object] = {}
    for reference in references:
        if isinstance(reference, Mapping):
            _merge_mapping(merged, reference)
    merged.setdefault("risk_api_runbook", runbook or RISK_API_RUNBOOK)
    return merged


__all__ = [
    "RISK_API_RUNBOOK",
    "RiskApiError",
    "TradingRiskInterface",
    "build_runtime_risk_metadata",
    "merge_risk_references",
    "resolve_trading_risk_config",
    "resolve_trading_risk_interface",
    "summarise_risk_config",
]
