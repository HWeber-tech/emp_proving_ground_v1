from __future__ import annotations
import json

from emp.core import findings_memory


def test_findings_memory_flow(tmp_path):
    db_path = tmp_path / "experiments.sqlite"
    conn = findings_memory.connect(db_path)

    idea1 = {"alpha": 1, "beta": 2}
    idea2 = {"alpha": 2, "beta": 3}
    idea3 = {"alpha": 5, "beta": 8}

    artefacts1 = findings_memory.compute_params_artifacts(idea1)
    novelty1 = findings_memory.nearest_novelty(conn, idea1, artefacts=artefacts1)
    res1 = findings_memory.add_idea(conn, idea1, novelty1, artefacts=artefacts1)
    fid1 = res1.id

    artefacts2 = findings_memory.compute_params_artifacts(idea2)
    novelty2 = findings_memory.nearest_novelty(conn, idea2, artefacts=artefacts2)
    res2 = findings_memory.add_idea(conn, idea2, novelty2, artefacts=artefacts2)
    fid2 = res2.id

    artefacts3 = findings_memory.compute_params_artifacts(idea3)
    novelty3 = findings_memory.nearest_novelty(conn, idea3, artefacts=artefacts3)
    res3 = findings_memory.add_idea(conn, idea3, novelty3, artefacts=artefacts3)
    fid3 = res3.id

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

    cursor = conn2.execute("SELECT params_hash, params_vec FROM findings WHERE id = ?", (fid1,))
    row = cursor.fetchone()
    assert row["params_hash"]
    assert json.loads(row["params_vec"])  # vector cached
