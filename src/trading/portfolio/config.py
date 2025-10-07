"""Canonical portfolio monitoring configuration primitives."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping

from src.governance.system_config import SystemConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PortfolioMonitorConfig:
    """Typed configuration for `RealPortfolioMonitor`.

    The legacy `src.config.portfolio_config` module duplicated these settings
    without integrating with `SystemConfig`.  This canonical dataclass keeps
    the public fields stable while allowing callers to resolve configuration
    from the governance-backed `SystemConfig` extras.
    """

    database_path: Path = Path("portfolio.db")
    initial_balance: float = 10_000.0
    save_snapshots: bool = True
    snapshot_interval_minutes: int = 15
    max_positions: int = 10
    max_position_size_pct: float = 0.10
    max_total_exposure: float = 0.80
    performance_report_days: int = 30
    detailed_logging: bool = True

    def validate(self) -> None:
        """Raise ``ValueError`` when configuration values are invalid."""

        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if not 0 < self.max_position_size_pct <= 1:
            raise ValueError("max_position_size_pct must be between 0 and 1")
        if not 0 < self.max_total_exposure <= 1:
            raise ValueError("max_total_exposure must be between 0 and 1")
        if self.snapshot_interval_minutes <= 0:
            raise ValueError("snapshot_interval_minutes must be positive")
        if self.performance_report_days <= 0:
            raise ValueError("performance_report_days must be positive")


_DEFAULT_CONFIG = PortfolioMonitorConfig()


def resolve_portfolio_monitor_config(
    system_config: SystemConfig,
    overrides: Mapping[str, object] | None = None,
) -> PortfolioMonitorConfig:
    """Build configuration from ``SystemConfig`` extras and optional overrides.

    Resolution order (highest precedence first):
    1. ``overrides`` mapping (keys ``{field}`` or ``portfolio.{field}``)
    2. ``system_config.extras`` (same keys as above)
    3. Dataclass defaults
    """

    resolved: dict[str, object] = {}
    raw_overrides = overrides or {}
    extras = system_config.extras or {}

    for field in fields(PortfolioMonitorConfig):
        default_value = getattr(_DEFAULT_CONFIG, field.name)
        raw_value = _select_raw_value(field.name, raw_overrides, extras)
        caster = _CASTERS[field.name]
        resolved[field.name] = caster(raw_value, default_value)

    config = PortfolioMonitorConfig(**resolved)
    try:
        config.validate()
    except ValueError as exc:
        logger.warning("Invalid portfolio config derived from SystemConfig: %s", exc)
        raise
    return config


def _select_raw_value(
    field_name: str,
    overrides: Mapping[str, object],
    extras: Mapping[str, str],
) -> object | None:
    for key in (field_name, f"portfolio.{field_name}"):
        if key in overrides:
            return overrides[key]
    for key in (f"portfolio.{field_name}", field_name):
        if key in extras:
            return extras[key]
    return None


def _coerce_path(value: object | None, default: Path) -> Path:
    if value in (None, ""):
        return default
    try:
        return Path(str(value)).expanduser()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to parse portfolio database path %r: %s", value, exc)
        return default


def _coerce_bool(value: object | None, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    logger.warning("Invalid boolean value for portfolio config: %r", value)
    return default


def _coerce_positive_float(value: object | None, default: float) -> float:
    if value is None:
        return default
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid float value for portfolio config: %r", value)
        return default
    if candidate <= 0:
        logger.warning("Non-positive float for portfolio config: %r", value)
        return default
    return candidate


def _coerce_ratio(value: object | None, default: float) -> float:
    candidate = _coerce_positive_float(value, default)
    if not 0 < candidate <= 1:
        logger.warning("Ratio must be between 0 and 1, received %r", value)
        return default
    return candidate


def _coerce_positive_int(value: object | None, default: int) -> int:
    if value is None:
        return default
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid integer value for portfolio config: %r", value)
        return default
    if candidate <= 0:
        logger.warning("Non-positive integer for portfolio config: %r", value)
        return default
    return candidate


_CASTERS: dict[str, Callable[[object | None, Any], Any]] = {
    "database_path": _coerce_path,
    "initial_balance": _coerce_positive_float,
    "save_snapshots": _coerce_bool,
    "snapshot_interval_minutes": _coerce_positive_int,
    "max_positions": _coerce_positive_int,
    "max_position_size_pct": _coerce_ratio,
    "max_total_exposure": _coerce_ratio,
    "performance_report_days": _coerce_positive_int,
    "detailed_logging": _coerce_bool,
}


__all__ = ["PortfolioMonitorConfig", "resolve_portfolio_monitor_config"]
