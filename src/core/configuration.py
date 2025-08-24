"""
EMP Core Configuration v1.1

Provides centralized configuration management for the EMP system.
Supports loading from YAML files, environment variables, and secrets.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .exceptions import ConfigurationException

logger = logging.getLogger(__name__)


@dataclass
class Configuration:
    """Central configuration class for EMP system."""

    # System configuration
    system_name: str = "EMP"
    system_version: str = "1.1.0"
    environment: str = "development"
    debug: bool = False

    # Layer configurations
    sensory: dict[str, object] = field(default_factory=dict)
    thinking: dict[str, object] = field(default_factory=dict)
    trading: dict[str, object] = field(default_factory=dict)
    evolution: dict[str, object] = field(default_factory=dict)
    governance: dict[str, object] = field(default_factory=dict)
    operational: dict[str, object] = field(default_factory=dict)

    # External service configurations
    redis: dict[str, object] = field(default_factory=dict)
    postgresql: dict[str, object] = field(default_factory=dict)
    nats: dict[str, object] = field(default_factory=dict)
    ctrader: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization setup."""
        self._load_environment_variables()

    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        self.environment = os.getenv("EMP_ENVIRONMENT", self.environment)
        self.debug = os.getenv("EMP_DEBUG", "false").lower() == "true"

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Configuration":
        """Load configuration from YAML file."""
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                raise ConfigurationException(f"Configuration file not found: {config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            return cls(**config_data)

        except Exception as e:
            raise ConfigurationException(f"Error loading configuration: {e}")

    def to_yaml(self, config_path: Union[str, Path]):
        """Save configuration to YAML file."""
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            config_data = {
                "system_name": self.system_name,
                "system_version": self.system_version,
                "environment": self.environment,
                "debug": self.debug,
                "sensory": self.sensory,
                "thinking": self.thinking,
                "trading": self.trading,
                "evolution": self.evolution,
                "governance": self.governance,
                "operational": self.operational,
                "redis": self.redis,
                "postgresql": self.postgresql,
                "nats": self.nats,
                "ctrader": self.ctrader,
            }

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

        except Exception as e:
            raise ConfigurationException(f"Error saving configuration: {e}")

    def get(self, key: str, default: object = None) -> object:
        """Get configuration value using dot notation."""
        keys = key.split(".")
        value = self

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: object):
        """Set configuration value using dot notation."""
        keys = key.split(".")
        config = self

        for k in keys[:-1]:
            if k not in config.__dict__:
                config.__dict__[k] = {}
            config = config.__dict__[k]

        config[keys[-1]] = value

    def validate(self) -> bool:
        """Validate configuration."""
        required_fields = ["system_name", "system_version", "environment"]

        for field in required_fields:
            if not getattr(self, field):
                raise ConfigurationException(f"Required field missing: {field}")

        return True


# Global configuration instance
config = Configuration()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Configuration:
    """Load configuration from file or use defaults."""
    global config

    if config_path:
        config = Configuration.from_yaml(config_path)
    else:
        # Try to load from default locations
        default_paths = [Path("config/emp.yaml"), Path("config.yaml"), Path("emp.yaml")]

        for path in default_paths:
            if path.exists():
                config = Configuration.from_yaml(path)
                break

    config.validate()
    logger.info(f"Configuration loaded: {config.system_name} v{config.system_version}")
    return config


def get_config() -> Configuration:
    """Get the global configuration instance."""
    return config
