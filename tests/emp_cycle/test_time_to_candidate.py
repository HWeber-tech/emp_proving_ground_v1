from __future__ import annotations

import pytest

from emp.core import findings_memory


def test_time_to_candidate_stats(tmp_path):
    db_path = tmp_path / "experiments.sqlite"
    conn = findings_memory.connect(db_path)

    idea_fast = {"alpha": 1}
    art_fast = findings_memory.compute_params_artifacts(idea_fast)
    novelty_fast = findings_memory.nearest_novelty(conn, idea_fast, artefacts=art_fast)
    fid_fast = findings_memory.add_idea(conn, idea_fast, novelty_fast, artefacts=art_fast).id

    with conn:
        conn.execute(
            "UPDATE findings SET created_at = ? WHERE id = ?",
            ("2024-01-01 00:00:00", fid_fast),
        )

    findings_memory.update_quick(conn, fid_fast, {"score": 0.9}, 0.9)
    findings_memory.promote_tested(conn, fid_fast, {"sharpe": 1.0}, True)

    with conn:
        conn.execute(
            "UPDATE findings SET tested_at = ?, progress_at = ? WHERE id = ?",
            ("2024-01-01 12:00:00", "2024-01-01 12:00:00", fid_fast),
        )

    idea_slow = {"alpha": 2}
    art_slow = findings_memory.compute_params_artifacts(idea_slow)
    novelty_slow = findings_memory.nearest_novelty(conn, idea_slow, artefacts=art_slow)
    fid_slow = findings_memory.add_idea(conn, idea_slow, novelty_slow, artefacts=art_slow).id

    with conn:
        conn.execute(
            "UPDATE findings SET created_at = ? WHERE id = ?",
            ("2024-01-01 00:00:00", fid_slow),
        )

    findings_memory.update_quick(conn, fid_slow, {"score": 0.7}, 0.7)
    findings_memory.promote_tested(conn, fid_slow, {"sharpe": 0.5}, False)

    with conn:
        conn.execute(
            "UPDATE findings SET tested_at = ? WHERE id = ?",
            ("2024-01-02 18:00:00", fid_slow),
        )

    stats = findings_memory.time_to_candidate_stats(conn)

    assert stats.count == 2
    assert stats.sla_met is False
    assert pytest.approx(stats.average_hours, rel=1e-4) == 27.0
    assert pytest.approx(stats.median_hours, rel=1e-4) == 27.0
    assert pytest.approx(stats.p90_hours, rel=1e-4) == 39.0
    assert pytest.approx(stats.max_hours, rel=1e-4) == 42.0
    assert len(stats.breaches) == 1
    assert stats.breaches[0].id == fid_slow
    assert pytest.approx(stats.breaches[0].hours, rel=1e-4) == 42.0
