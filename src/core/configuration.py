"""Compatibility configuration shim backed by the canonical ``SystemConfig``."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml

from src.governance.system_config import SystemConfig

from .exceptions import ConfigurationException

logger = logging.getLogger(__name__)


def _coerce_debug(value: str | None, *, default: bool = False) -> bool:
    """Normalize debug flags sourced from environment variables."""

    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def _normalize_token(raw: str) -> str:
    """Lower/strip/underscore tokens for comparisons."""

    return raw.strip().lower().replace("-", "_")


def _apply_environment_override(
    config: SystemConfig, requested: object | None
) -> tuple[SystemConfig, str]:
    """Apply legacy environment overrides while preserving canonical enums."""

    extras = dict(config.extras)
    if requested is None:
        if extras.pop("legacy_environment", None) is not None:
            config = config.with_updated(extras=extras)
        return config, config.environment.value

    requested_str = str(requested)
    normalized = _normalize_token(requested_str)
    candidate = config.with_updated(environment=requested_str)

    if candidate.environment.value == normalized:
        if extras.pop("legacy_environment", None) is not None:
            candidate = candidate.with_updated(extras=extras)
        return candidate, candidate.environment.value

    extras["legacy_environment"] = requested_str
    updated = config.with_updated(extras=extras)
    return updated, requested_str


@dataclass(slots=True)
class Configuration:
    """
    Legacy configuration surface that proxies to ``SystemConfig`` for core settings.

    The compatibility layer keeps historical attributes such as ``trading`` and
    ``sensory`` dictionaries while ensuring environment/run-mode decisions are
    sourced from :class:`SystemConfig`. Consumers can migrate gradually by
    reading the ``system_config`` attribute instead of re-implementing parsing
    logic.
    """

    _SECTION_FIELDS: ClassVar[tuple[str, ...]] = (
        "sensory",
        "thinking",
        "trading",
        "evolution",
        "governance",
        "operational",
        "redis",
        "postgresql",
        "nats",
        "ctrader",
    )

    system_name: str = "EMP"
    system_version: str = "1.1.0"
    environment: str | None = None
    debug: bool | None = None
    sensory: dict[str, Any] = field(default_factory=dict)
    thinking: dict[str, Any] = field(default_factory=dict)
    trading: dict[str, Any] = field(default_factory=dict)
    evolution: dict[str, Any] = field(default_factory=dict)
    governance: dict[str, Any] = field(default_factory=dict)
    operational: dict[str, Any] = field(default_factory=dict)
    redis: dict[str, Any] = field(default_factory=dict)
    postgresql: dict[str, Any] = field(default_factory=dict)
    nats: dict[str, Any] = field(default_factory=dict)
    ctrader: dict[str, Any] = field(default_factory=dict)
    system_config: SystemConfig = field(default_factory=SystemConfig, repr=False)

    def __post_init__(self) -> None:
        self._sync_system_config()

    def _sync_system_config(self) -> None:
        """Refresh ``system_config`` using environment variables and overrides."""

        resolved = SystemConfig.from_env(defaults=self.system_config)
        resolved, environment_value = _apply_environment_override(resolved, self.environment)
        self.system_config = resolved
        self.environment = environment_value
        self.debug = _coerce_debug(os.getenv("EMP_DEBUG")) if self.debug is None else bool(self.debug)

    def apply_system_config_overrides(self, overrides: Mapping[str, object]) -> None:
        """Apply mapping overrides to the backing ``SystemConfig`` instance."""

        override_dict = dict(overrides)
        extras_override = override_dict.get("extras")
        if isinstance(extras_override, Mapping):
            override_dict["extras"] = dict(extras_override)
            legacy_env = override_dict["extras"].get("legacy_environment")
        else:
            legacy_env = None

        requested_env = override_dict.pop("environment", None)
        if legacy_env is not None:
            requested_env = legacy_env

        updated = self.system_config.with_updated(**override_dict)
        updated, environment_value = _apply_environment_override(updated, requested_env)
        self.system_config = updated
        self.environment = environment_value

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> Configuration:
        """Load configuration from YAML file, propagating settings to ``SystemConfig``."""

        path = Path(config_path)
        if not path.exists():
            raise ConfigurationException(f"Configuration file not found: {path}")

        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - defensive
            raise ConfigurationException(f"Error loading configuration: {exc}") from exc

        if not isinstance(raw, Mapping):
            raise ConfigurationException("Configuration file must contain a mapping")

        base_kwargs: dict[str, Any] = {}
        for key in ("system_name", "system_version", "environment", "debug"):
            if key in raw:
                base_kwargs[key] = raw[key]

        section_kwargs: dict[str, Any] = {}
        for section in cls._SECTION_FIELDS:
            value = raw.get(section) or {}
            section_kwargs[section] = dict(value) if isinstance(value, Mapping) else {}

        config = cls(**base_kwargs, **section_kwargs)

        system_config_section = raw.get("system_config")
        if isinstance(system_config_section, Mapping):
            config.apply_system_config_overrides(system_config_section)

        return config

    def to_yaml(self, config_path: str | Path) -> None:
        """Persist configuration to YAML, including the ``system_config`` view."""

        path = Path(config_path)
        payload: dict[str, Any] = {
            "system_name": self.system_name,
            "system_version": self.system_version,
            "environment": self.environment,
            "debug": bool(self.debug),
        }

        for section in self._SECTION_FIELDS:
            payload[section] = dict(getattr(self, section))

        payload["system_config"] = self.system_config.to_dict()

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)
        except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - defensive
            raise ConfigurationException(f"Error saving configuration: {exc}") from exc

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """Get configuration value using dot notation."""

        value: Any = self
        for part in key.split("."):
            if isinstance(value, Mapping):
                if part not in value:
                    return default
                value = value[part]
                continue

            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""

        parts = key.split(".")
        if parts[0] == "system_config":
            overrides = {parts[1]: value} if len(parts) == 2 else value
            if isinstance(overrides, Mapping):
                self.apply_system_config_overrides(overrides)
            else:
                raise ConfigurationException("System config overrides must be mappings")
            return

        target: Any = self
        for part in parts[:-1]:
            if isinstance(target, MutableMapping):
                target = target.setdefault(part, {})
                continue

            current = getattr(target, part, None)
            if not isinstance(current, MutableMapping):
                current = {}
                setattr(target, part, current)
            target = current

        final_key = parts[-1]
        if isinstance(target, MutableMapping):
            target[final_key] = value
        else:
            setattr(target, final_key, value)

    def validate(self) -> bool:
        """Validate configuration."""

        if not self.system_name:
            raise ConfigurationException("Required field missing: system_name")
        if not self.system_version:
            raise ConfigurationException("Required field missing: system_version")
        if not self.environment:
            raise ConfigurationException("Required field missing: environment")
        return True


def config_factory() -> Configuration:
    """Factory indirection to ensure environments are applied during import."""

    return Configuration()


config = config_factory()


def load_config(config_path: str | Path | None = None) -> Configuration:
    """Load configuration from file or use defaults."""

    global config

    if config_path is not None:
        config = Configuration.from_yaml(config_path)
    else:
        for candidate in (Path("config/emp.yaml"), Path("config.yaml"), Path("emp.yaml")):
            if candidate.exists():
                config = Configuration.from_yaml(candidate)
                break

    config.validate()
    logger.info("Configuration loaded: %s v%s", config.system_name, config.system_version)
    return config


def get_config() -> Configuration:
    """Get the global configuration instance."""

    return config
