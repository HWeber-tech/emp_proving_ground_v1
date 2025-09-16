"""Competitive Intelligence façade (lazy, side-effect free).

This module exposes the canonical Competitive Intelligence types while
avoiding heavy imports at module import-time. Public symbols are re-exported
lazily via PEP 562 (__getattr__), preserving legacy import paths and
keeping import-time costs low.

Design principles:
- No logging configuration, threads, or I/O at import time.
- Heavy third-party imports (numpy, sklearn, torch) localized at use time.
- TYPE_CHECKING and lazy lookups used to avoid runtime imports for types.
- Backward compatibility: public names remain available from this module.

Examples:
    from src.intelligence.competitive_intelligence import CompetitiveIntelligenceSystem
    system = CompetitiveIntelligenceSystem(...)

    # Legacy shim example (heavy deps only imported on use)
    from src.intelligence.competitive_intelligence import StrategyInsightLegacy
    labels = StrategyInsightLegacy().cluster_signatures([[0.1, 0.2], [0.3, 0.4]])
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Public canonical symbols from the canonical implementation.
# __all__ is defined dynamically after _LAZY_EXPORTS

# Map public symbols to canonical implementations (lazy import).
_LAZY_EXPORTS: Dict[str, str] = {
    "AlgorithmFingerprinter": (
        "src.thinking.competitive.competitive_intelligence_system:AlgorithmFingerprinter"
    ),
    "BehaviorAnalyzer": (
        "src.thinking.competitive.competitive_intelligence_system:BehaviorAnalyzer"
    ),
    "CompetitiveIntelligenceSystem": (
        "src.thinking.competitive.competitive_intelligence_system:CompetitiveIntelligenceSystem"
    ),
    "CounterStrategyDeveloper": (
        "src.thinking.competitive.competitive_intelligence_system:CounterStrategyDeveloper"
    ),
    "MarketShareTracker": (
        "src.thinking.competitive.competitive_intelligence_system:MarketShareTracker"
    ),
}
# Derive public API dynamically from the lazy export map and legacy shims.
__all__ = list(_LAZY_EXPORTS.keys()) + ["StrategyInsightLegacy", "AlgorithmFingerprinterLegacy"]


def __getattr__(name: str) -> object:
    """PEP 562 lazy re-export for canonical symbols.

    Imports the canonical implementation on first attribute access, caches
    it in module globals, and returns it.

    Raises:
        AttributeError: if `name` is not a known public symbol.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)

    module_path, _, attr = target.partition(":")
    mod = importlib.import_module(module_path)
    obj = getattr(mod, attr)
    # Cache to avoid repeated importlib lookups.
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    """Include lazy-exported names in dir()."""
    return sorted(set(list(globals().keys()) + list(__all__)))


# ---- Legacy shims (minimal; heavy imports localized at use-time) ----------------


class _NumpyProxy:
    """Lightweight proxy that imports numpy lazily on first attribute access.

    This mitigates import-time cost for environments without numpy wheels by
    deferring the import until a legacy method actually needs it.
    """

    def __getattr__(self, item: str) -> object:  # pragma: no cover - exercised indirectly
        try:
            import numpy as _np
        except Exception as exc:
            raise AttributeError(f"numpy not available: {exc}") from exc
        return getattr(_np, item)


_np = _NumpyProxy()


class StrategyInsightLegacy:
    """Minimal legacy shim for competitive intelligence analytics.

    Heavy third-party imports (e.g., scikit-learn, torch) are localized in methods.
    This class is kept to preserve compatibility in environments where callers
    still rely on legacy algorithmic entry points. It should not introduce import-time
    side effects and should fail with a clear error if a required heavy dependency
    is missing at use time.
    """

    def cluster_signatures(
        self, X: "list[list[float]]", eps: float = 0.5, min_samples: int = 5
    ) -> "list[int]":
        """Cluster signature vectors using DBSCAN (scikit-learn).

        Imports scikit-learn locally to avoid import-time heavy dependencies.

        Args:
            X: 2D list of floats (n_samples x n_features).
            eps: DBSCAN eps parameter.
            min_samples: DBSCAN min_samples parameter.

        Returns:
            List of cluster labels per sample.

        Raises:
            RuntimeError: when scikit-learn is not available.
        """
        try:
            from sklearn.cluster import DBSCAN  # Localized heavy import
        except Exception as exc:  # pragma: no cover - exercised under import failure path
            raise RuntimeError(
                "scikit-learn is required for StrategyInsightLegacy.cluster_signatures"
            ) from exc

        # Import numpy locally for this operation; require real numpy at use-time.
        try:
            import numpy as _np_mod
        except Exception as exc:
            raise RuntimeError(
                "numpy is required for StrategyInsightLegacy.cluster_signatures"
            ) from exc

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels_arr = model.fit(_np_mod.asarray(X)).labels_
        labels_list: list[int] = [int(x) for x in getattr(labels_arr, "tolist")()]
        return labels_list

    def torch_sanity(self) -> bool:
        """Check that torch is importable at use-time (not at module import).

        Returns:
            True if torch imports successfully, otherwise raises.

        Raises:
            RuntimeError: if torch is not importable.
        """
        try:
            importlib.import_module("torch")
        except Exception as exc:  # pragma: no cover - import-failure path
            raise RuntimeError("torch is required for StrategyInsightLegacy.torch_sanity") from exc
        return True


class AlgorithmFingerprinterLegacy:
    def __init__(self) -> None:
        # Localized heavy import to ensure construction fails when sklearn is blocked
        try:
            from sklearn.preprocessing import StandardScaler  # Localized heavy import
        except Exception as exc:
            raise ImportError("scikit-learn is required for AlgorithmFingerprinterLegacy") from exc
        self._scaler = StandardScaler()

    def fingerprint(self, X: "list[list[float]]") -> "tuple[int,int]":
        # Compute shape without relying on numpy typing
        rows = len(X)
        cols = len(X[0]) if (rows > 0 and isinstance(X[0], list)) else 0
        return (rows, cols)


# End of module
