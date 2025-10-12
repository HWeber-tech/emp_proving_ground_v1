"""SQLite-backed findings memory for experimentation loop."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

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
      notes TEXT,
      params_hash TEXT,
      params_vec TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_stage ON findings(stage);",
    "CREATE INDEX IF NOT EXISTS idx_params_hash ON findings(params_hash);",
    "CREATE INDEX IF NOT EXISTS idx_created_at ON findings(created_at);",
)

DEFAULT_DB_PATH = Path("data/experiments.sqlite")


class ParamsArtifacts(NamedTuple):
    """Pre-computed artefacts for params JSON hashing and vectorisation."""

    params_json: str
    params_hash: str
    params_vec: Tuple[int, ...]


class IdeaInsertResult(NamedTuple):
    """Return metadata from :func:`add_idea`."""

    id: int
    inserted: bool
    stage: str
    duplicate_of: Optional[int] = None


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
        _ensure_columns(conn)
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> None:
    info = conn.execute("PRAGMA table_info(findings)").fetchall()
    columns = {row[1] for row in info}
    with conn:
        if "params_hash" not in columns:
            conn.execute("ALTER TABLE findings ADD COLUMN params_hash TEXT")
        if "params_vec" not in columns:
            conn.execute("ALTER TABLE findings ADD COLUMN params_vec TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_params_hash ON findings(params_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON findings(created_at)")


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


def _vectorize(tokens: Sequence[str], buckets: int = 8) -> Tuple[int, ...]:
    vector = [0] * buckets
    for token in sorted(tokens):
        # Rolling hash with deterministic bucket assignment.
        h = 0
        for char in token:
            h = (h * 33 + ord(char)) & 0xFFFFFFFF
        bucket = h % buckets
        weight = (h // buckets) % 7 + 1
        vector[bucket] += weight
    return tuple(vector)


def _cosine_distance(vec_a: Sequence[int], vec_b: Sequence[int]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def compute_params_artifacts(params: Dict) -> ParamsArtifacts:
    """Compute JSON/hash/vector representations for *params*."""

    params_json = json.dumps(params, sort_keys=True)
    params_hash = hashlib.sha1(params_json.encode("utf-8")).hexdigest()
    tokens = _encode_params(params)
    params_vec = _vectorize(tokens)
    return ParamsArtifacts(params_json=params_json, params_hash=params_hash, params_vec=params_vec)


def _json_vector(vec: Sequence[int]) -> str:
    return json.dumps(list(vec), separators=(",", ":"))


def _backfill_missing_artifacts(conn: sqlite3.Connection, batch_size: int = 512) -> None:
    cursor = conn.execute(
        "SELECT id, params_json FROM findings WHERE params_hash IS NULL OR params_vec IS NULL LIMIT ?",
        (int(batch_size),),
    )
    rows = cursor.fetchall()
    if not rows:
        return
    with conn:
        for row in rows:
            params = json.loads(row["params_json"])
            artefacts = compute_params_artifacts(params)
            conn.execute(
                "UPDATE findings SET params_hash = ?, params_vec = ? WHERE id = ?",
                (artefacts.params_hash, _json_vector(artefacts.params_vec), int(row["id"])),
            )


def add_idea(
    conn: sqlite3.Connection,
    params: Dict,
    novelty: float,
    *,
    artefacts: ParamsArtifacts | None = None,
    note: str | None = None,
) -> IdeaInsertResult:
    """Insert a new idea row and return metadata.

    When a duplicate idea exists in ``idea`` or ``screened`` stage, the existing
    identifier is returned and no new row is inserted.
    """

    artefacts = artefacts or compute_params_artifacts(params)
    _backfill_missing_artifacts(conn)
    existing = conn.execute(
        """
        SELECT id, stage
          FROM findings
         WHERE params_hash = ?
         ORDER BY id ASC
         LIMIT 1
        """,
        (artefacts.params_hash,),
    ).fetchone()
    if existing and existing["stage"] in {"idea", "screened"}:
        fid = int(existing["id"])
        if note:
            append_note(conn, fid, note)
        return IdeaInsertResult(id=fid, inserted=False, stage=str(existing["stage"]))

    duplicate_of: Optional[int] = None
    if existing:
        duplicate_of = int(existing["id"])

    params_vec_json = _json_vector(artefacts.params_vec)
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO findings(stage, params_json, novelty, params_hash, params_vec, notes)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                "idea",
                artefacts.params_json,
                float(novelty),
                artefacts.params_hash,
                params_vec_json,
                note if note else None,
            ),
        )
        fid = int(cursor.lastrowid)
    if duplicate_of is not None:
        append_note(conn, fid, f"duplicate_of:{duplicate_of}")
    return IdeaInsertResult(id=fid, inserted=True, stage="idea", duplicate_of=duplicate_of)


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


def nearest_novelty(
    conn: sqlite3.Connection,
    params: Dict,
    *,
    artefacts: ParamsArtifacts | None = None,
    window: int = 5000,
) -> float:
    """Compute novelty score relative to prior ideas in the database."""

    artefacts = artefacts or compute_params_artifacts(params)
    _backfill_missing_artifacts(conn)
    cursor = conn.execute(
        """
        SELECT id, params_vec, params_hash
          FROM findings
         WHERE params_vec IS NOT NULL AND id IS NOT NULL
         ORDER BY datetime(created_at) DESC, id DESC
         LIMIT ?
        """,
        (int(window),),
    )
    candidate_vec = artefacts.params_vec
    min_distance: float | None = None
    for row in cursor.fetchall():
        params_vec_json = row["params_vec"]
        if not params_vec_json:
            continue
        prior_vec = tuple(int(v) for v in json.loads(params_vec_json))
        distance = _cosine_distance(candidate_vec, prior_vec)
        if min_distance is None or distance < min_distance:
            min_distance = distance
            if min_distance == 0.0:
                break
    if min_distance is None:
        return 1.0
    return float(max(0.0, min(1.0, min_distance)))


def append_note(conn: sqlite3.Connection, fid: int, note: str) -> None:
    """Append a semi-colon separated note to the given finding."""

    if not note:
        return
    with conn:
        conn.execute(
            """
            UPDATE findings
               SET notes = CASE
                               WHEN notes IS NULL OR notes = '' THEN ?
                               ELSE notes || ';' || ?
                           END
             WHERE id = ?
            """,
            (note, note, int(fid)),
        )


__all__ = [
    "connect",
    "DEFAULT_DB_PATH",
    "add_idea",
    "update_quick",
    "promote_tested",
    "fetch_candidates",
    "nearest_novelty",
    "append_note",
    "IdeaInsertResult",
    "ParamsArtifacts",
    "compute_params_artifacts",
]
