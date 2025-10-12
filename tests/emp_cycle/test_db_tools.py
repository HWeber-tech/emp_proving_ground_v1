from __future__ import annotations

from datetime import datetime, timedelta, timezone

from emp.cli import emp_db_tools
from emp.core import findings_memory


def test_prune_and_vacuum(tmp_path):
    db_path = tmp_path / "experiments.sqlite"
    conn = findings_memory.connect(db_path)

    idea = {"alpha": 1}
    artefacts = findings_memory.compute_params_artifacts(idea)
    novelty = findings_memory.nearest_novelty(conn, idea, artefacts=artefacts)
    res = findings_memory.add_idea(conn, idea, novelty, artefacts=artefacts)
    assert res.inserted

    conn.execute(
        "UPDATE findings SET created_at = ?",
        ((datetime.now(timezone.utc) - timedelta(days=200)).isoformat(timespec="seconds"),),
    )
    conn.commit()

    code = emp_db_tools.main(
        [
            "--db-path",
            str(db_path),
            "prune",
            "--keep-days",
            "30",
            "--stages",
            "idea,screened",
        ]
    )
    assert code == 0

    remaining = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
    assert remaining == 0

    code = emp_db_tools.main(["--db-path", str(db_path), "vacuum"])
    assert code == 0
