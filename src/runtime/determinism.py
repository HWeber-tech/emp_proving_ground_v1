"""Deterministic seeding helpers for runtime replays and simulations.

This module centralises lightweight seeding so replay-driven acceptance tests
can reset Python's pseudo-random sources prior to executing the trading loop.
It deliberately stays dependency-free beyond ``numpy`` so it can run inside the
minimal test harness used by the roadmap regression suite.
"""

from __future__ import annotations

import os
import random
from typing import Callable


def _seed_numpy(seed: int) -> None:  # pragma: no cover - optional dependency
    try:
        import numpy as np
    except Exception:  # pragma: no cover - best-effort seeding
        return
    np.random.seed(seed)


def _seed_torch(seed: int) -> None:  # pragma: no cover - optional dependency
    try:
        import torch
    except Exception:
        return

    # ``torch.manual_seed`` returns a torch.Generator instance; ignore it.
    try:
        torch.manual_seed(seed)
    except Exception:
        return


def seed_runtime(seed: int | None) -> None:
    """Seed common pseudo-random sources when ``seed`` is provided.

    The helper is intentionally idempotent: passing ``None`` leaves the global
    RNG state untouched, while providing an integer resets the Python ``random``
    module and any available optional libraries such as ``numpy`` or ``torch``.

    Parameters
    ----------
    seed:
        Integer seed applied to supported RNG providers.  ``None`` disables
        seeding so callers can opt out without additional conditionals.
    """

    if seed is None:
        return

    random.seed(seed)

    # ``PYTHONHASHSEED`` only takes effect on interpreter start, but storing the
    # value documents the replay seed in child processes and diagnostics.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    for hook in (_seed_numpy, _seed_torch):
        hook(seed)


__all__ = ["seed_runtime"]
