"""Typed, Enum-based SystemConfig with safe environment coercion and no import-time side effects."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
from typing import Mapping, TypeVar

logger = logging.getLogger(__name__)

# Prefer StrEnum on Python 3.11; fallback to a compatible shim otherwise
try:  # Python 3.11+
    from enum import StrEnum  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback path for older Python
    class StrEnum(str, Enum):  # minimal shim
        pass


class RunMode(StrEnum):
    mock = "mock"
    paper = "paper"
    live = "live"


class EmpTier(StrEnum):
    tier_0 = "tier_0"
    tier_1 = "tier_1"
    tier_2 = "tier_2"


class EmpEnvironment(StrEnum):
    demo = "demo"
    staging = "staging"
    production = "production"


class ConnectionProtocol(StrEnum):
    fix = "fix"


E = TypeVar("E", bound=Enum)


def _coerce_enum(
    value: str | E | None,
    enum_type: type[E],
    default: E,
    aliases: Mapping[str, E] | None = None,
) -> E:
    """
    Normalize and coerce a string to an enum value without raising.
    - Normalization: strip, lower, replace '-' with '_'
    - If aliases provided and match, return alias
    - Else try enum_type(normalized) by value; on failure return default
    """
    if value is None:
        return default
    if isinstance(value, enum_type):
        return value
    raw_lower = str(value).strip().lower()
    normalized = raw_lower.replace("-", "_")
    if aliases:
        alt = aliases.get(normalized) or aliases.get(raw_lower)
        if alt is not None:
            return alt
    try:
        # enumType("value") resolves by value for Enum/StrEnum with string values
        return enum_type(normalized)  # type: ignore[call-arg]
    except Exception:
        return default


def _coerce_bool(value: str | bool | None, default: bool) -> bool:
    """
    Normalize and coerce to bool without raising.
    - For bool: return as-is
    - For str: normalize (strip, lower, '-'->'_') and map common variants
    - On unknown: return default
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass
class SystemConfig:
    # Typed fields (Enums and primitives)
    run_mode: RunMode = RunMode.paper
    environment: EmpEnvironment = EmpEnvironment.demo
    tier: EmpTier = EmpTier.tier_0
    confirm_live: bool = False
    connection_protocol: ConnectionProtocol = ConnectionProtocol.fix
    extras: dict[str, str] = field(default_factory=lambda: {})

    # Backward-compatible string views for legacy comparisons
    @property
    def run_mode_str(self) -> str:
        return self.run_mode.value

    @property
    def environment_str(self) -> str:
        return self.environment.value

    @property
    def tier_str(self) -> str:
        return self.tier.value

    @property
    def connection_protocol_str(self) -> str:
        return self.connection_protocol.value

    # Additional backward-compat alias for existing code that refers to "emp_tier"
    @property
    def emp_tier(self) -> str:  # legacy alias, string view
        return self.tier.value

    def to_dict(self) -> dict[str, object]:
        """Export a dict representation with enum values as strings."""
        return {
            "run_mode": self.run_mode.value,
            "environment": self.environment.value,
            "tier": self.tier.value,
            "confirm_live": self.confirm_live,
            "connection_protocol": self.connection_protocol.value,
            "extras": dict(self.extras),
        }

    def to_env(self) -> dict[str, str]:
        """Export environment variables as a string dictionary."""
        return {
            "RUN_MODE": self.run_mode.value,
            "EMP_ENVIRONMENT": self.environment.value,
            "EMP_TIER": self.tier.value,
            "CONFIRM_LIVE": "true" if self.confirm_live else "false",
            "CONNECTION_PROTOCOL": self.connection_protocol.value,
        }

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        defaults: SystemConfig | None = None,
    ) -> SystemConfig:
        """
        Construct from environment mapping without raising.
        - Reads from os.environ if env is None
        - Uses coercion helpers
        - On invalid recognized values, fall back to defaults and add extras entries:
          <KEY>_invalid: "<raw>"
        - Include all other non-recognized env entries into extras
        """
        env = env if env is not None else os.environ  # read only here

        base = defaults if defaults is not None else cls()
        extras: dict[str, str] = dict(base.extras) if base.extras else {}

        recognized_keys = {"RUN_MODE", "EMP_ENVIRONMENT", "EMP_TIER", "CONFIRM_LIVE", "CONNECTION_PROTOCOL"}

        # Prepare helpers for validity checks
        def _is_valid_enum_value(raw: str | None, enum_type: type[E], aliases: Mapping[str, E] | None = None) -> bool:
            if raw is None:
                return False  # absence is not "invalid", it's "unset"
            normalized = raw.strip().lower().replace("-", "_")
            values = {m.value for m in enum_type}  # type: ignore[attr-defined]
            if aliases and (normalized in aliases):
                return True
            return normalized in values

        def _mark_invalid(key: str, raw_value: str, fallback_desc: str) -> None:
            extras[f"{key}_invalid"] = raw_value
            logger.debug("Invalid %s value %r; falling back to default %s", key, raw_value, fallback_desc)

        # RUN_MODE
        raw_run_mode = env.get("RUN_MODE")
        if raw_run_mode is not None and not _is_valid_enum_value(raw_run_mode, RunMode):
            _mark_invalid("RUN_MODE", raw_run_mode, base.run_mode.value)
        run_mode = _coerce_enum(raw_run_mode, RunMode, base.run_mode)

        # EMP_ENVIRONMENT
        raw_env = env.get("EMP_ENVIRONMENT")
        if raw_env is not None and not _is_valid_enum_value(raw_env, EmpEnvironment):
            _mark_invalid("EMP_ENVIRONMENT", raw_env, base.environment.value)
        environment = _coerce_enum(raw_env, EmpEnvironment, base.environment)

        # EMP_TIER (add aliases for 'tier-1' and normalized 'tier_1')
        raw_tier = env.get("EMP_TIER")
        tier_aliases: Mapping[str, EmpTier] = {
            "tier-1": EmpTier.tier_1,
            "tier_1": EmpTier.tier_1,
        }
        if raw_tier is not None and not _is_valid_enum_value(raw_tier, EmpTier, tier_aliases):
            _mark_invalid("EMP_TIER", raw_tier, base.tier.value)
        tier = _coerce_enum(raw_tier, EmpTier, base.tier, tier_aliases)

        # CONFIRM_LIVE
        raw_confirm_live = env.get("CONFIRM_LIVE")
        if raw_confirm_live is not None:
            normalized_bool = str(raw_confirm_live).strip().lower().replace("-", "_")
            valid_bool = normalized_bool in {"1", "true", "yes", "y", "on", "0", "false", "no", "n", "off"}
            if not valid_bool:
                _mark_invalid("CONFIRM_LIVE", raw_confirm_live, str(base.confirm_live).lower())
        confirm_live = _coerce_bool(raw_confirm_live, base.confirm_live)

        # CONNECTION_PROTOCOL
        raw_cp = env.get("CONNECTION_PROTOCOL")
        if raw_cp is not None and not _is_valid_enum_value(raw_cp, ConnectionProtocol):
            _mark_invalid("CONNECTION_PROTOCOL", raw_cp, base.connection_protocol.value)
        connection_protocol = _coerce_enum(raw_cp, ConnectionProtocol, base.connection_protocol)

        # Include all other non-recognized env entries in extras (stringify values)
        for k, v in env.items():
            if k not in recognized_keys:
                extras[k] = str(v)

        return cls(
            run_mode=run_mode,
            environment=environment,
            tier=tier,
            confirm_live=confirm_live,
            connection_protocol=connection_protocol,
            extras=extras,
        )

    def with_updated(self, **kwargs) -> SystemConfig:
        """
        Return a new instance with updated fields; deep-copy extras to avoid aliasing.
        Accepts either Enum instances or strings for Enum fields, and str/bool for confirm_live.
        """
        # Start with existing fields
        run_mode = self.run_mode
        environment = self.environment
        tier = self.tier
        confirm_live = self.confirm_live
        connection_protocol = self.connection_protocol
        extras: dict[str, str] = dict(self.extras)  # deep copy for simple dict[str, str]

        if "run_mode" in kwargs:
            val = kwargs["run_mode"]
            run_mode = _coerce_enum(val, RunMode, run_mode) if not isinstance(val, RunMode) else val  # type: ignore[arg-type]
        if "environment" in kwargs:
            val = kwargs["environment"]
            environment = _coerce_enum(val, EmpEnvironment, environment) if not isinstance(val, EmpEnvironment) else val  # type: ignore[arg-type]
        if "tier" in kwargs:
            val = kwargs["tier"]
            tier = _coerce_enum(val, EmpTier, tier) if not isinstance(val, EmpTier) else val  # type: ignore[arg-type]
        if "confirm_live" in kwargs:
            val = kwargs["confirm_live"]
            confirm_live = _coerce_bool(val, confirm_live)  # type: ignore[arg-type]
        if "connection_protocol" in kwargs:
            val = kwargs["connection_protocol"]
            connection_protocol = (
                _coerce_enum(val, ConnectionProtocol, connection_protocol) if not isinstance(val, ConnectionProtocol) else val  # type: ignore[arg-type]
            )
        if "extras" in kwargs:
            new_extras = kwargs["extras"]
            extras = dict(new_extras) if isinstance(new_extras, dict) else extras

        return SystemConfig(
            run_mode=run_mode,
            environment=environment,
            tier=tier,
            confirm_live=confirm_live,
            connection_protocol=connection_protocol,
            extras=extras,
        )


__all__ = [
    "SystemConfig",
    "RunMode",
    "EmpTier",
    "EmpEnvironment",
    "ConnectionProtocol",
]
