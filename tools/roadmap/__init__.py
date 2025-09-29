"""Roadmap utilities for assessing concept-to-code parity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time conveniences for type checkers
    from .high_impact import StreamStatus, evaluate_streams, main as high_impact_main
    from .snapshot import (
        InitiativeStatus,
        evaluate_portfolio_snapshot,
        main as snapshot_main,
    )
    from . import doc_tasks as doc_tasks

__all__ = [
    "InitiativeStatus",
    "StreamStatus",
    "evaluate_portfolio_snapshot",
    "evaluate_streams",
    "snapshot_main",
    "high_impact_main",
    "doc_tasks",
    "main",
]


def __getattr__(name: str) -> Any:
    """Lazily import roadmap helpers to avoid module initialisation loops."""

    from importlib import import_module

    module_map = {
        "InitiativeStatus": ("tools.roadmap.snapshot", "InitiativeStatus"),
        "evaluate_portfolio_snapshot": (
            "tools.roadmap.snapshot",
            "evaluate_portfolio_snapshot",
        ),
        "snapshot_main": ("tools.roadmap.snapshot", "main"),
        "StreamStatus": ("tools.roadmap.high_impact", "StreamStatus"),
        "evaluate_streams": ("tools.roadmap.high_impact", "evaluate_streams"),
        "high_impact_main": ("tools.roadmap.high_impact", "main"),
        "doc_tasks": ("tools.roadmap.doc_tasks", None),
        "main": ("tools.roadmap.snapshot", "main"),
    }

    target = module_map.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = import_module(module_name)
    attribute = module if attribute_name is None else getattr(module, attribute_name)

    # Cache the resolved attribute to avoid repeated imports.
    globals()[name] = attribute

    # Keep the alias between ``main`` and ``snapshot_main`` in sync.
    if attribute_name is None:
        return attribute

    if name == "main" and "snapshot_main" not in globals():
        globals()["snapshot_main"] = attribute
    elif name == "snapshot_main" and "main" not in globals():
        globals()["main"] = attribute

    return attribute


def __dir__() -> list[str]:
    """Expose lazily-imported symbols through dir()."""

    return sorted(__all__)
