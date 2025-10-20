"""Typed, Enum-based SystemConfig with safe environment coercion and no import-time side effects."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
try:
    from typing import Mapping, TypedDict, TypeVar, Unpack
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    from typing import Mapping, TypedDict, TypeVar
    from typing_extensions import Unpack

import yaml

logger = logging.getLogger(__name__)


class SystemConfigLoadError(RuntimeError):
    """Raised when a SystemConfig YAML payload cannot be decoded."""


class _ValueStrEnum(StrEnum):
    def __str__(self) -> str:  # pragma: no cover - trivial delegation
        return str(self.value)


class RunMode(_ValueStrEnum):
    mock = "mock"
    paper = "paper"
    live = "live"


class EmpTier(_ValueStrEnum):
    tier_0 = "tier_0"
    tier_1 = "tier_1"
    tier_2 = "tier_2"


class EmpEnvironment(_ValueStrEnum):
    demo = "demo"
    staging = "staging"
    production = "production"


class ConnectionProtocol(_ValueStrEnum):
    bootstrap = "bootstrap"
    paper = "paper"
    fix = "fix"


E = TypeVar("E", bound=StrEnum)


class DataBackboneMode(_ValueStrEnum):
    """Selectable data backbone implementations for ingest and caching."""

    bootstrap = "bootstrap"
    institutional = "institutional"


class SystemConfigUpdate(TypedDict, total=False):
    run_mode: RunMode | str
    environment: EmpEnvironment | str
    tier: EmpTier | str
    confirm_live: bool | str | int
    connection_protocol: ConnectionProtocol | str
    data_backbone_mode: DataBackboneMode | str
    kill_switch_path: str | os.PathLike[str] | Path | None
    extras: dict[str, str]


_SYSTEM_CONFIG_FIELDS = (
    "run_mode",
    "environment",
    "tier",
    "confirm_live",
    "connection_protocol",
    "data_backbone_mode",
    "kill_switch_path",
    "extras",
)


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
    - Else match by member.value; on failure return default
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
    for member in enum_type:
        if member.value == normalized:
            return member
    return default


def _coerce_bool(value: str | bool | int | None, default: bool) -> bool:
    """
    Normalize and coerce to bool without raising.
    - For bool: return as-is
    - For int: 1/0 treated via string normalization
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


def _default_kill_switch_path() -> Path:
    return Path(tempfile.gettempdir()) / "emp_pg.KILL"


def _normalize_kill_switch_path(
    raw: str | os.PathLike[str] | Path | None,
) -> Path | None:
    if raw is None:
        return None
    raw_str = str(raw).strip()
    if not raw_str:
        return None
    lowered = raw_str.lower()
    if lowered in {"none", "disabled", "off"}:
        return None
    candidate = Path(raw_str).expanduser()
    if not candidate.is_absolute():
        candidate = Path(tempfile.gettempdir()) / candidate
    return candidate


def _normalise_extras(payload: Mapping[str, object]) -> dict[str, str]:
    """Convert arbitrary mapping values to ``dict[str, str]``."""

    extras: dict[str, str] = {}
    for key, value in payload.items():
        extras[str(key)] = str(value)
    return extras


def _normalise_model_section(section: Mapping[str, object]) -> dict[str, str]:
    """Convert a nested ``model`` mapping into ``extras`` style keys."""

    extras: dict[str, str] = {}
    for raw_key, value in section.items():
        if value is None:
            continue
        key_str = str(raw_key).strip()
        if not key_str:
            continue
        extras[f"MODEL_{key_str.upper().replace('-', '_')}"] = str(value)

    if "MODEL_SSM_IMPL" in extras:
        extras.setdefault("MODEL_PRIMARY_IMPL", extras["MODEL_SSM_IMPL"])
    if "MODEL_FALLBACK_IMPL" in extras:
        extras.setdefault("MODEL_SECONDARY_IMPL", extras["MODEL_FALLBACK_IMPL"])
    return extras


def _merge_config_overrides(target: dict[str, object], source: Mapping[str, object]) -> None:
    """Merge supported SystemConfig fields from ``source`` into ``target``."""

    for field in _SYSTEM_CONFIG_FIELDS:
        if field not in source:
            continue
        if field == "extras":
            raw_extras = source[field]
            if not isinstance(raw_extras, Mapping):
                continue
            existing = target.get("extras")
            merged: dict[str, str]
            if isinstance(existing, Mapping):
                merged = dict(existing)
            else:
                merged = {}
            merged.update(_normalise_extras(raw_extras))
            target["extras"] = merged
            continue
        target[field] = source[field]


@dataclass
class SystemConfig:
    # Typed fields (Enums and primitives)
    run_mode: RunMode = RunMode.paper
    environment: EmpEnvironment = EmpEnvironment.demo
    tier: EmpTier = EmpTier.tier_0
    confirm_live: bool = False
    connection_protocol: ConnectionProtocol = ConnectionProtocol.bootstrap
    data_backbone_mode: DataBackboneMode = DataBackboneMode.bootstrap
    kill_switch_path: Path | None = field(default_factory=_default_kill_switch_path)
    extras: dict[str, str] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:  # pragma: no cover - trivial normalization
        self.kill_switch_path = _normalize_kill_switch_path(self.kill_switch_path) or None

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

    @property
    def data_backbone_mode_str(self) -> str:
        return self.data_backbone_mode.value

    def to_dict(self) -> dict[str, object]:
        """Export a dict representation with enum values as strings."""
        return {
            "run_mode": self.run_mode.value,
            "environment": self.environment.value,
            "tier": self.tier.value,
            "confirm_live": self.confirm_live,
            "connection_protocol": self.connection_protocol.value,
            "data_backbone_mode": self.data_backbone_mode.value,
            "kill_switch_path": str(self.kill_switch_path) if self.kill_switch_path else None,
            "extras": dict(self.extras),
        }

    def to_env(self) -> dict[str, str]:
        """Export environment variables as a string dictionary."""
        env = {
            "RUN_MODE": self.run_mode.value,
            "EMP_ENVIRONMENT": self.environment.value,
            "EMP_TIER": self.tier.value,
            "CONFIRM_LIVE": "true" if self.confirm_live else "false",
            "CONNECTION_PROTOCOL": self.connection_protocol.value,
            "DATA_BACKBONE_MODE": self.data_backbone_mode.value,
        }
        if self.kill_switch_path:
            env["EMP_KILL_SWITCH"] = str(self.kill_switch_path)
        return env

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
        source_env: Mapping[str, str] = os.environ if env is None else env

        base = defaults if defaults is not None else cls()
        extras: dict[str, str] = dict(base.extras) if base.extras else {}

        recognized_keys = {
            "RUN_MODE",
            "EMP_ENVIRONMENT",
            "EMP_TIER",
            "CONFIRM_LIVE",
            "CONNECTION_PROTOCOL",
            "DATA_BACKBONE_MODE",
            "EMP_KILL_SWITCH",
        }

        def _infer_data_backbone_mode(current: DataBackboneMode) -> DataBackboneMode:
            """Infer backbone mode if credentials for institutional services are present."""

            indicator_keys = (
                "REDIS_URL",
                "REDIS_HOST",
                "KAFKA_BROKERS",
                "KAFKA_BOOTSTRAP_SERVERS",
                "KAFKA_URL",
            )
            for key in indicator_keys:
                raw_indicator = source_env.get(key)
                if raw_indicator and str(raw_indicator).strip():
                    return DataBackboneMode.institutional
            return current

        # Prepare helpers for validity checks
        def _is_valid_enum_value(
            raw: str | None, enum_type: type[E], aliases: Mapping[str, E] | None = None
        ) -> bool:
            if raw is None:
                return False  # absence is not "invalid", it's "unset"
            normalized = raw.strip().lower().replace("-", "_")
            values: set[str] = {m.value for m in enum_type}
            if aliases and (normalized in aliases):
                return True
            return normalized in values

        def _mark_invalid(key: str, raw_value: str, fallback_desc: str) -> None:
            extras[f"{key}_invalid"] = raw_value
            logger.debug(
                "Invalid %s value %r; falling back to default %s", key, raw_value, fallback_desc
            )

        # RUN_MODE
        raw_run_mode = source_env.get("RUN_MODE")
        if raw_run_mode is not None and not _is_valid_enum_value(raw_run_mode, RunMode):
            _mark_invalid("RUN_MODE", raw_run_mode, base.run_mode.value)
        run_mode = _coerce_enum(raw_run_mode, RunMode, base.run_mode)

        # EMP_ENVIRONMENT
        raw_env = source_env.get("EMP_ENVIRONMENT")
        if raw_env is not None and not _is_valid_enum_value(raw_env, EmpEnvironment):
            _mark_invalid("EMP_ENVIRONMENT", raw_env, base.environment.value)
        environment = _coerce_enum(raw_env, EmpEnvironment, base.environment)

        # EMP_TIER (add aliases for 'tier-1' and normalized 'tier_1')
        raw_tier = source_env.get("EMP_TIER")
        tier_aliases: Mapping[str, EmpTier] = {
            "tier-1": EmpTier.tier_1,
            "tier_1": EmpTier.tier_1,
        }
        if raw_tier is not None and not _is_valid_enum_value(raw_tier, EmpTier, tier_aliases):
            _mark_invalid("EMP_TIER", raw_tier, base.tier.value)
        tier = _coerce_enum(raw_tier, EmpTier, base.tier, tier_aliases)

        # CONFIRM_LIVE
        raw_confirm_live = source_env.get("CONFIRM_LIVE")
        if raw_confirm_live is not None:
            normalized_bool = str(raw_confirm_live).strip().lower().replace("-", "_")
            valid_bool = normalized_bool in {
                "1",
                "true",
                "yes",
                "y",
                "on",
                "0",
                "false",
                "no",
                "n",
                "off",
            }
            if not valid_bool:
                _mark_invalid("CONFIRM_LIVE", raw_confirm_live, str(base.confirm_live).lower())
        confirm_live = _coerce_bool(raw_confirm_live, base.confirm_live)

        # CONNECTION_PROTOCOL
        raw_cp = source_env.get("CONNECTION_PROTOCOL")
        cp_aliases: Mapping[str, ConnectionProtocol] = {
            "mock": ConnectionProtocol.bootstrap,
            "bootstrap": ConnectionProtocol.bootstrap,
            "paper": ConnectionProtocol.paper,
        }
        if raw_cp is not None and not _is_valid_enum_value(raw_cp, ConnectionProtocol, cp_aliases):
            _mark_invalid("CONNECTION_PROTOCOL", raw_cp, base.connection_protocol.value)
        connection_protocol = _coerce_enum(
            raw_cp, ConnectionProtocol, base.connection_protocol, cp_aliases
        )

        raw_backbone = source_env.get("DATA_BACKBONE_MODE")
        if raw_backbone is not None and not _is_valid_enum_value(raw_backbone, DataBackboneMode):
            _mark_invalid("DATA_BACKBONE_MODE", raw_backbone, base.data_backbone_mode.value)
        data_backbone_mode = _coerce_enum(raw_backbone, DataBackboneMode, base.data_backbone_mode)
        if raw_backbone is None:
            data_backbone_mode = _infer_data_backbone_mode(data_backbone_mode)

        kill_switch_path = base.kill_switch_path
        raw_kill_switch = source_env.get("EMP_KILL_SWITCH")
        if raw_kill_switch is not None:
            normalized_kill_switch = _normalize_kill_switch_path(raw_kill_switch)
            if normalized_kill_switch is None:
                raw_str = str(raw_kill_switch).strip()
                if raw_str and raw_str.lower() not in {"none", "disabled", "off"}:
                    _mark_invalid("EMP_KILL_SWITCH", raw_str, str(kill_switch_path or "disabled"))
            kill_switch_path = normalized_kill_switch

        # Include all other non-recognized env entries in extras (stringify values)
        for k, v in source_env.items():
            if k not in recognized_keys:
                extras[k] = str(v)

        # Final dry run orchestration injects evidence paths via dedicated
        # environment variables. Mirror them onto the canonical extras keys so
        # runtime components that look for DECISION_DIARY_PATH /
        # PERFORMANCE_METRICS_PATH pick them up automatically when operators
        # rely on the new harness.
        final_dry_run_diary = extras.get("FINAL_DRY_RUN_DIARY_PATH")
        if final_dry_run_diary and "DECISION_DIARY_PATH" not in extras:
            extras["DECISION_DIARY_PATH"] = final_dry_run_diary

        final_dry_run_performance = extras.get("FINAL_DRY_RUN_PERFORMANCE_PATH")
        if final_dry_run_performance and "PERFORMANCE_METRICS_PATH" not in extras:
            extras["PERFORMANCE_METRICS_PATH"] = final_dry_run_performance

        return cls(
            run_mode=run_mode,
            environment=environment,
            tier=tier,
            confirm_live=confirm_live,
            connection_protocol=connection_protocol,
            data_backbone_mode=data_backbone_mode,
            kill_switch_path=kill_switch_path,
            extras=extras,
        )

    def with_updated(self, **overrides: Unpack[SystemConfigUpdate]) -> SystemConfig:
        """
        Return a new instance with updated fields; deep-copy extras to avoid aliasing.
        Accepts either Enum instances or strings for Enum fields, and bool/str/int for confirm_live.
        """
        # Start with existing fields
        run_mode = self.run_mode
        environment = self.environment
        tier = self.tier
        confirm_live = self.confirm_live
        connection_protocol = self.connection_protocol
        data_backbone_mode = self.data_backbone_mode
        kill_switch_path = self.kill_switch_path
        extras: dict[str, str] = dict(self.extras)  # deep copy for simple dict[str, str]

        if "run_mode" in overrides:
            v_rm = overrides["run_mode"]
            run_mode = v_rm if isinstance(v_rm, RunMode) else _coerce_enum(v_rm, RunMode, run_mode)
        if "environment" in overrides:
            v_env = overrides["environment"]
            environment = (
                v_env
                if isinstance(v_env, EmpEnvironment)
                else _coerce_enum(v_env, EmpEnvironment, environment)
            )
        if "tier" in overrides:
            v_tier = overrides["tier"]
            tier = v_tier if isinstance(v_tier, EmpTier) else _coerce_enum(v_tier, EmpTier, tier)
        if "confirm_live" in overrides:
            v_cl = overrides["confirm_live"]
            confirm_live = _coerce_bool(v_cl, confirm_live)
        if "connection_protocol" in overrides:
            v_cp = overrides["connection_protocol"]
            cp_aliases = {
                "mock": ConnectionProtocol.bootstrap,
                "bootstrap": ConnectionProtocol.bootstrap,
                "paper": ConnectionProtocol.paper,
            }
            connection_protocol = (
                v_cp
                if isinstance(v_cp, ConnectionProtocol)
                else _coerce_enum(v_cp, ConnectionProtocol, connection_protocol, cp_aliases)
            )
        if "data_backbone_mode" in overrides:
            v_bb = overrides["data_backbone_mode"]
            data_backbone_mode = (
                v_bb
                if isinstance(v_bb, DataBackboneMode)
                else _coerce_enum(v_bb, DataBackboneMode, data_backbone_mode)
            )
        if "kill_switch_path" in overrides:
            kill_switch_path = _normalize_kill_switch_path(overrides["kill_switch_path"])
        if "extras" in overrides:
            new_extras = overrides["extras"]
            extras = dict(new_extras) if isinstance(new_extras, dict) else extras

        return SystemConfig(
            run_mode=run_mode,
            environment=environment,
            tier=tier,
            confirm_live=confirm_live,
            connection_protocol=connection_protocol,
            data_backbone_mode=data_backbone_mode,
            kill_switch_path=kill_switch_path,
            extras=extras,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        env: Mapping[str, str] | None = None,
    ) -> SystemConfig:
        """Load a ``SystemConfig`` from a YAML payload.

        The loader keeps backward compatibility with legacy configuration files by
        accepting either top-level configuration keys or a nested ``system_config``
        mapping. Extras are normalised to ``dict[str, str]`` and environment
        variables remain authoritative unless the YAML file explicitly overrides
        them.
        """

        cfg_path = Path(path)
        if not cfg_path.exists():
            raise SystemConfigLoadError(f"Configuration file not found: {cfg_path}")

        try:
            payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - defensive
            raise SystemConfigLoadError(f"Error loading configuration: {exc}") from exc

        if not isinstance(payload, Mapping):
            raise SystemConfigLoadError("Configuration file must contain a mapping")

        overrides: dict[str, object] = {}
        _merge_config_overrides(overrides, payload)

        nested = payload.get("system_config")
        if isinstance(nested, Mapping):
            _merge_config_overrides(overrides, nested)

        model_extras: dict[str, str] = {}

        def _accumulate_model(candidate: Mapping[str, object]) -> None:
            section = candidate.get("model") if isinstance(candidate, Mapping) else None
            if isinstance(section, Mapping):
                model_extras.update(_normalise_model_section(section))

        _accumulate_model(payload)
        if isinstance(nested, Mapping):
            _accumulate_model(nested)

        if model_extras:
            existing_extras = overrides.get("extras")
            merged_extras: dict[str, str]
            if isinstance(existing_extras, Mapping):
                merged_extras = dict(existing_extras)
            else:
                merged_extras = {}
            merged_extras.update(model_extras)
            overrides["extras"] = merged_extras

        base = cls.from_env(env=env)
        if overrides:
            base = base.with_updated(**overrides)
        return base


__all__ = [
    "SystemConfig",
    "RunMode",
    "EmpTier",
    "EmpEnvironment",
    "ConnectionProtocol",
    "DataBackboneMode",
    "SystemConfigLoadError",
]
