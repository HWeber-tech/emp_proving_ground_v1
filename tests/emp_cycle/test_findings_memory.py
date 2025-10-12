from __future__ import annotations
from emp.core import findings_memory


def test_findings_memory_flow(tmp_path):
    db_path = tmp_path / "experiments.sqlite"
    conn = findings_memory.connect(db_path)

    idea1 = {"alpha": 1, "beta": 2}
    idea2 = {"alpha": 2, "beta": 3}
    idea3 = {"alpha": 5, "beta": 8}

    novelty1 = findings_memory.nearest_novelty(conn, idea1)
    fid1 = findings_memory.add_idea(conn, idea1, novelty1)

    novelty2 = findings_memory.nearest_novelty(conn, idea2)
    fid2 = findings_memory.add_idea(conn, idea2, novelty2)

    novelty3 = findings_memory.nearest_novelty(conn, idea3)
    fid3 = findings_memory.add_idea(conn, idea3, novelty3)

    findings_memory.update_quick(conn, fid1, {"score": 0.8}, 0.8)
    findings_memory.update_quick(conn, fid2, {"score": 0.6}, 0.6)

    candidates = findings_memory.fetch_candidates(conn)
    assert [cid for cid, *_ in candidates] == [fid1, fid2]

    findings_memory.promote_tested(conn, fid1, {"sharpe": 1.2}, True)

    # Reconnect to ensure schema creation is idempotent.
    conn2 = findings_memory.connect(db_path)
    cursor = conn2.execute("SELECT stage FROM findings WHERE id = ?", (fid1,))
    assert cursor.fetchone()[0] == "progress"

    # Novelty should be bounded between 0 and 1
    assert 0.0 <= novelty1 <= 1.0
    assert 0.0 <= novelty2 <= 1.0
    assert 0.0 <= novelty3 <= 1.0

    cursor = conn2.execute("SELECT COUNT(*) FROM findings")
    assert cursor.fetchone()[0] == 3
