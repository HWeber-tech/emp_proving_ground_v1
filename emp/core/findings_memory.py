"""SQLite-backed findings memory for experimentation loop."""
from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

_DB_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS findings (
      id INTEGER PRIMARY KEY,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      stage TEXT CHECK(stage IN ('idea','screened','tested','progress')) NOT NULL,
      params_json TEXT NOT NULL,
      novelty REAL DEFAULT 0.0,
      quick_metrics_json TEXT,
      quick_score REAL,
      full_metrics_json TEXT,
      notes TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_stage ON findings(stage);",
)

DEFAULT_DB_PATH = Path("data/experiments.sqlite")


def connect(db_path: str | os.PathLike[str] = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create (if necessary) and return a SQLite connection.

    The database schema is created idempotently on first connection.
    """

    path = Path(db_path)
    if path != Path(":memory:"):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    with conn:
        for stmt in _DB_SCHEMA:
            conn.execute(stmt)
    return conn


def _encode_params(params: Dict) -> List[str]:
    tokens: List[str] = []

    def _walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                _walk(f"{prefix}{key}.", value[key])
        elif isinstance(value, (list, tuple, set)):
            for idx, item in enumerate(value):
                _walk(f"{prefix}{idx}.", item)
        else:
            token = f"{prefix[:-1]}={value}"
            tokens.append(token)

    for key in sorted(params):
        _walk(f"{key}.", params[key])
    return tokens


def _vectorize(tokens: Sequence[str], buckets: int = 8) -> List[int]:
    vector = [0] * buckets
    for token in sorted(tokens):
        # Rolling hash with deterministic bucket assignment.
        h = 0
        for char in token:
            h = (h * 33 + ord(char)) & 0xFFFFFFFF
        bucket = h % buckets
        weight = (h // buckets) % 7 + 1
        vector[bucket] += weight
    return vector


def _cosine_distance(vec_a: Sequence[int], vec_b: Sequence[int]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def add_idea(conn: sqlite3.Connection, params: Dict, novelty: float) -> int:
    """Insert a new idea row and return its identifier."""

    params_json = json.dumps(params, sort_keys=True)
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO findings(stage, params_json, novelty)
            VALUES(?, ?, ?)
            """,
            ("idea", params_json, float(novelty)),
        )
    return int(cursor.lastrowid)


def update_quick(
    conn: sqlite3.Connection, fid: int, quick_metrics: Dict, quick_score: float
) -> None:
    """Persist quick-evaluation metrics and promote to the screened stage."""

    metrics_json = json.dumps(quick_metrics, sort_keys=True)
    with conn:
        conn.execute(
            """
            UPDATE findings
               SET quick_metrics_json = ?, quick_score = ?, stage = 'screened'
             WHERE id = ?
            """,
            (metrics_json, float(quick_score), int(fid)),
        )


def promote_tested(
    conn: sqlite3.Connection, fid: int, full_metrics: Dict, is_progress: bool
) -> None:
    """Update an idea with full evaluation metrics and final stage."""

    metrics_json = json.dumps(full_metrics, sort_keys=True)
    stage = "progress" if is_progress else "tested"
    with conn:
        conn.execute(
            """
            UPDATE findings
               SET full_metrics_json = ?, stage = ?
             WHERE id = ?
            """,
            (metrics_json, stage, int(fid)),
        )


def fetch_candidates(conn: sqlite3.Connection, k: int = 200) -> List[Tuple[int, str, float | None, float]]:
    """Return screened candidates ordered by quick score and novelty."""

    cursor = conn.execute(
        """
        SELECT id, params_json, quick_score, novelty
          FROM findings
         WHERE stage = 'screened'
         ORDER BY COALESCE(quick_score, 0.0) DESC, novelty DESC, id ASC
         LIMIT ?
        """,
        (int(k),),
    )
    return [(int(row["id"]), row["params_json"], row["quick_score"], row["novelty"]) for row in cursor]


def nearest_novelty(conn: sqlite3.Connection, params: Dict) -> float:
    """Compute novelty score relative to prior ideas in the database."""

    tokens = _encode_params(params)
    candidate_vec = _vectorize(tokens)
    cursor = conn.execute("SELECT params_json FROM findings")
    min_distance: float | None = None
    for (params_json,) in cursor.fetchall():
        prior_params = json.loads(params_json)
        prior_tokens = _encode_params(prior_params)
        prior_vec = _vectorize(prior_tokens)
        distance = _cosine_distance(candidate_vec, prior_vec)
        if min_distance is None or distance < min_distance:
            min_distance = distance
    if min_distance is None:
        return 1.0
    return float(max(0.0, min(1.0, min_distance)))


__all__ = [
    "connect",
    "DEFAULT_DB_PATH",
    "add_idea",
    "update_quick",
    "promote_tested",
    "fetch_candidates",
    "nearest_novelty",
]
