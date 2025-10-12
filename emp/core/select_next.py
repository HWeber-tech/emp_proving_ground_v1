"""Selection helpers for choosing the next candidate to fully test."""
from __future__ import annotations

import math
from typing import Optional

from . import findings_memory


def ucb_lite(quick_score: float, novelty: float, c: float = 0.2) -> float:
    """Compute a light-weight UCB style score."""

    quick = float(quick_score)
    explore = float(novelty)
    return quick + c * explore


def pick_next(conn, c: float = 0.2) -> Optional[int]:
    """Pick the candidate id with the best exploration/exploitation balance."""

    candidates = findings_memory.fetch_candidates(conn)
    best_id: Optional[int] = None
    best_value: float | None = None
    best_quick: float = 0.0
    best_novel: float = 0.0
    for fid, _params_json, quick_score, novelty in candidates:
        quick = float(quick_score or 0.0)
        novel = float(novelty or 0.0)
        value = ucb_lite(quick, novel, c=c)
        if best_value is None:
            best_id = fid
            best_value = value
            best_quick = quick
            best_novel = novel
            continue

        if value > best_value + 1e-9:
            best_id = fid
            best_value = value
            best_quick = quick
            best_novel = novel
            continue

        if math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-9):
            if quick > best_quick + 1e-9:
                best_id = fid
                best_value = value
                best_quick = quick
                best_novel = novel
                continue
            if math.isclose(quick, best_quick, rel_tol=1e-9, abs_tol=1e-9):
                if novel > best_novel + 1e-9:
                    best_id = fid
                    best_value = value
                    best_quick = quick
                    best_novel = novel
                    continue
                if math.isclose(novel, best_novel, rel_tol=1e-9, abs_tol=1e-9) and fid < (best_id or fid):
                    best_id = fid
                    best_value = value
                    best_quick = quick
                    best_novel = novel
    return best_id


__all__ = ["ucb_lite", "pick_next"]
