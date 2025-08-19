"""
Core Configuration Access Port (Protocol)
========================================

Provides a minimal, domain-agnostic interface for retrieving configuration values
without importing governance or higher-layer packages.

Concrete adapters should live in higher layers (e.g., orchestration) and be injected
where needed. A NoOpConfigurationProvider is provided for safe defaults.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ConfigurationProvider(Protocol):
    """
    Abstract configuration provider interface.

    Implementations should never raise and must return safe defaults on error.
    """

    def get_value(self, key: str, default: Any = None) -> Any:
        """Return config value for a flat key (or default if not present)."""
        ...

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """Return a dict for a configuration namespace (or empty dict)."""
        ...


class NoOpConfigurationProvider:
    """Safe default provider that returns defaults or empty data structures."""

    def get_value(self, key: str, default: Any = None) -> Any:
        return default

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        return {}


def is_configuration_provider(obj: object) -> bool:
    """Runtime duck-typing helper."""
    try:
        return isinstance(obj, ConfigurationProvider)
    except Exception:
        return False


__all__ = ["ConfigurationProvider", "NoOpConfigurationProvider", "is_configuration_provider"]