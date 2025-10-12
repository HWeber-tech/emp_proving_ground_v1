from __future__ import annotations

from emp.core import findings_memory, select_next


def test_pick_next_prefers_ucb_score(tmp_path):
    conn = findings_memory.connect(tmp_path / "experiments.sqlite")
    params = {"alpha": 1}
    fid1 = findings_memory.add_idea(conn, params, novelty=0.9)
    fid2 = findings_memory.add_idea(conn, params, novelty=0.3)
    fid3 = findings_memory.add_idea(conn, params, novelty=0.5)

    findings_memory.update_quick(conn, fid1, {"score": 0.4}, 0.4)
    findings_memory.update_quick(conn, fid2, {"score": 0.7}, 0.7)
    findings_memory.update_quick(conn, fid3, {"score": 0.6}, 0.6)

    best = select_next.pick_next(conn, c=0.5)
    assert best == fid2

    # Increase novelty to make fid1 more appealing despite lower quick score
    conn.execute("UPDATE findings SET novelty = ? WHERE id = ?", (1.0, fid1))
    best_after = select_next.pick_next(conn, c=1.0)
    assert best_after == fid1
