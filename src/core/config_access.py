"""
Core configuration access port.

This module defines a minimal ConfigurationProvider protocol to decouple domain
modules (e.g., thinking) from concrete governance/system implementations.
It also provides a NoOpConfigurationProvider that safely returns defaults.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ConfigurationProvider(Protocol):
    """
    Minimal configuration access protocol used by thinking modules.

    - get_value: retrieve a single key with an optional default
    - get_namespace: retrieve a namespaced dictionary (or empty dict if missing)
    """

    def get_value(self, key: str, default: Any = None) -> Any:
        ...

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        ...


class NoOpConfigurationProvider:
    """
    No-op implementation that never raises and always returns safe defaults.
    """

    def get_value(self, key: str, default: Any = None) -> Any:
        return default

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        return {}


__all__ = ["ConfigurationProvider", "NoOpConfigurationProvider"]