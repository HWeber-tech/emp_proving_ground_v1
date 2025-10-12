"""Adapters for constructing strategy instances used in the experimentation loop."""
from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Optional

_STRATEGY_FACTORY: Optional[Callable[[dict], Any]] = None


def register_strategy_factory(factory: Callable[[dict], Any]) -> None:
    """Register an explicit factory function."""

    global _STRATEGY_FACTORY
    _STRATEGY_FACTORY = factory


def _load_factory() -> Callable[[dict], Any]:
    global _STRATEGY_FACTORY
    if _STRATEGY_FACTORY is not None:
        return _STRATEGY_FACTORY

    env_path = os.getenv("EMP_STRATEGY_FACTORY")
    candidate_paths = []
    if env_path:
        candidate_paths.append(env_path)

    # Potential defaults for future native integrations.
    candidate_paths.extend(
        [
            "strategies.factory:make_strategy",
            "src.trading.strategy_engine.factory:make_strategy",
        ]
    )

    for dotted in candidate_paths:
        module_name, _, attr = dotted.partition(":")
        if not module_name or not attr:
            continue
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attr)
        except (ImportError, AttributeError):
            continue
        if callable(factory):
            _STRATEGY_FACTORY = factory
            return factory

    raise RuntimeError(
        "No strategy factory available. Set EMP_STRATEGY_FACTORY to a 'module:function' path."
    )


def make_strategy(params: dict) -> Any:
    """Instantiate a strategy using the configured factory."""

    factory = _load_factory()
    return factory(params)


__all__ = ["make_strategy", "register_strategy_factory"]
