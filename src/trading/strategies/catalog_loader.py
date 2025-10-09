"""Helpers for loading the roadmap strategy catalog configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml

from src.core.strategy.engine import BaseStrategy

from . import (
    MeanReversionStrategy,
    MeanReversionStrategyConfig,
    MomentumStrategy,
    MomentumStrategyConfig,
    MultiTimeframeMomentumConfig,
    MultiTimeframeMomentumStrategy,
    TimeframeMomentumLegConfig,
    VolatilityBreakoutConfig,
    VolatilityBreakoutStrategy,
)

__all__ = [
    "BaselineDefinition",
    "StrategyCatalog",
    "StrategyDefinition",
    "load_strategy_catalog",
    "instantiate_strategy",
]


@dataclass(slots=True)
class StrategyDefinition:
    """Canonical description of a strategy from the catalog."""

    key: str
    identifier: str
    name: str
    class_name: str
    enabled: bool
    capital: float
    parameters: Mapping[str, Any]
    symbols: tuple[str, ...]
    tags: tuple[str, ...]
    description: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "identifier": self.identifier,
            "name": self.name,
            "class_name": self.class_name,
            "enabled": self.enabled,
            "capital": self.capital,
            "parameters": dict(self.parameters),
            "symbols": list(self.symbols),
            "tags": list(self.tags),
            "description": self.description,
        }


@dataclass(slots=True)
class BaselineDefinition:
    """Definition of the baseline moving-average strategy."""

    identifier: str
    short_window: int
    long_window: int
    risk_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "short_window": self.short_window,
            "long_window": self.long_window,
            "risk_fraction": self.risk_fraction,
        }


@dataclass(slots=True)
class StrategyCatalog:
    """Container for the full strategy catalog configuration."""

    version: str
    default_capital: float
    definitions: tuple[StrategyDefinition, ...]
    baseline: BaselineDefinition
    description: str | None = None
    _definition_index: dict[str, StrategyDefinition] = field(
        init=False, repr=False, default_factory=dict
    )
    _key_index: dict[str, StrategyDefinition] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "definitions", tuple(self.definitions))
        identifier_index = {
            definition.identifier: definition for definition in self.definitions
        }
        key_index = {definition.key: definition for definition in self.definitions}
        object.__setattr__(self, "_definition_index", identifier_index)
        object.__setattr__(self, "_key_index", key_index)

    def enabled_strategies(self) -> tuple[StrategyDefinition, ...]:
        return tuple(defn for defn in self.definitions if defn.enabled)

    def get_definition(self, identifier: str) -> StrategyDefinition | None:
        """Return a strategy definition by identifier."""

        if not identifier:
            return None
        return self._definition_index.get(str(identifier))

    def get_definition_by_key(self, key: str) -> StrategyDefinition | None:
        """Return a strategy definition by catalogue key."""

        if not key:
            return None
        return self._key_index.get(str(key))

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "default_capital": self.default_capital,
            "description": self.description,
            "baseline": self.baseline.as_dict(),
            "definitions": [definition.as_dict() for definition in self.definitions],
        }


_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "trading" / "strategy_catalog.yaml"

_STRATEGY_CLASS_MAP: Mapping[str, tuple[type[BaseStrategy], type[Any] | None]] = {
    "MomentumStrategy": (MomentumStrategy, MomentumStrategyConfig),
    "MeanReversionStrategy": (MeanReversionStrategy, MeanReversionStrategyConfig),
    "VolatilityBreakoutStrategy": (VolatilityBreakoutStrategy, VolatilityBreakoutConfig),
    "MultiTimeframeMomentumStrategy": (
        MultiTimeframeMomentumStrategy,
        MultiTimeframeMomentumConfig,
    ),
}


def _normalise_symbols(payload: Any, fallback: Iterable[str]) -> tuple[str, ...]:
    if isinstance(payload, str):
        return (payload,)
    if isinstance(payload, Iterable):
        result = []
        for item in payload:
            if isinstance(item, str) and item:
                result.append(item)
        if result:
            return tuple(result)
    return tuple(fallback)


def _normalise_tags(payload: Any) -> tuple[str, ...]:
    if isinstance(payload, str):
        return (payload,)
    if isinstance(payload, Iterable):
        tags = [str(tag) for tag in payload if isinstance(tag, str) and tag]
        if tags:
            return tuple(tags)
    return ()


def load_strategy_catalog(path: str | Path | None = None) -> StrategyCatalog:
    """Load the catalog configuration from YAML."""

    resolved_path = Path(path) if path is not None else _DEFAULT_PATH
    data = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    catalog_section = data.get("catalog", data)

    version = str(catalog_section.get("version", "0.0"))
    default_capital = float(catalog_section.get("default_capital", 0.0))
    description = catalog_section.get("description")

    symbols_cfg = catalog_section.get("symbols", {})
    default_symbols: tuple[str, ...]
    if isinstance(symbols_cfg, Mapping):
        default_symbols = _normalise_symbols(symbols_cfg.get("default"), ("EURUSD",))
    else:
        default_symbols = ("EURUSD",)

    baseline_cfg = catalog_section.get("baseline", {})
    baseline = BaselineDefinition(
        identifier=str(baseline_cfg.get("id", "baseline")),
        short_window=int(baseline_cfg.get("short_window", 5)),
        long_window=int(baseline_cfg.get("long_window", 20)),
        risk_fraction=float(baseline_cfg.get("risk_fraction", 0.2)),
    )

    definitions: list[StrategyDefinition] = []
    strategies_cfg = catalog_section.get("strategies", {})
    if isinstance(strategies_cfg, Mapping):
        for key, payload in strategies_cfg.items():
            if not isinstance(payload, Mapping):
                continue
            identifier = str(payload.get("id", key))
            name = str(payload.get("name", identifier))
            class_name = str(payload.get("class", ""))
            enabled = bool(payload.get("enabled", True))
            capital = float(payload.get("capital", default_capital))

            parameters_obj = payload.get("parameters")
            parameters: MutableMapping[str, Any]
            if isinstance(parameters_obj, Mapping):
                parameters = {str(k): v for k, v in parameters_obj.items()}
            else:
                parameters = {}

            symbols_payload = payload.get("symbols")
            symbols = _normalise_symbols(symbols_payload, default_symbols)

            description_text = payload.get("description")
            tags = _normalise_tags(payload.get("tags"))

            definition = StrategyDefinition(
                key=str(key),
                identifier=identifier,
                name=name,
                class_name=class_name,
                enabled=enabled,
                capital=capital,
                parameters=parameters,
                symbols=symbols,
                tags=tags,
                description=str(description_text) if description_text else None,
            )
            definitions.append(definition)

    return StrategyCatalog(
        version=version,
        default_capital=default_capital,
        definitions=tuple(definitions),
        baseline=baseline,
        description=str(description) if description else None,
    )


def instantiate_strategy(definition: StrategyDefinition) -> BaseStrategy:
    """Instantiate a strategy using the catalog definition."""

    try:
        cls, config_cls = _STRATEGY_CLASS_MAP[definition.class_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported strategy class: {definition.class_name!r}"
        ) from exc

    symbols = list(definition.symbols)
    capital = float(definition.capital)
    parameters = dict(definition.parameters)

    config = None
    if config_cls is not None:
        if config_cls is MultiTimeframeMomentumConfig and "timeframes" in parameters:
            timeframes_payload = parameters.get("timeframes")
            if isinstance(timeframes_payload, Iterable):
                legs: list[TimeframeMomentumLegConfig] = []
                for entry in timeframes_payload:
                    if isinstance(entry, TimeframeMomentumLegConfig):
                        legs.append(entry)
                    elif isinstance(entry, Mapping):
                        legs.append(TimeframeMomentumLegConfig(**entry))
                parameters = dict(parameters)
                parameters["timeframes"] = tuple(legs)
        config = config_cls(**parameters)

    kwargs: dict[str, Any] = {"capital": capital}
    if config is not None:
        kwargs["config"] = config

    return cls(definition.identifier, symbols, **kwargs)
