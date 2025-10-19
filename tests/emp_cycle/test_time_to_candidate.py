from __future__ import annotations

import math

import pytest

from emp.cli import emp_cycle_metrics

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


def test_nearest_novelty_excludes_current_row(tmp_path):
    db_path = tmp_path / "experiments.sqlite"
    conn = findings_memory.connect(db_path)

    params_a = {"alpha": 1}
    art_a = findings_memory.compute_params_artifacts(params_a)
    novelty_a = findings_memory.nearest_novelty(conn, params_a, artefacts=art_a)
    fid_a = findings_memory.add_idea(conn, params_a, novelty_a, artefacts=art_a).id

    params_b = {"alpha": 2}
    art_b = findings_memory.compute_params_artifacts(params_b)
    novelty_b = findings_memory.nearest_novelty(conn, params_b, artefacts=art_b)
    result_b = findings_memory.add_idea(conn, params_b, novelty_b, artefacts=art_b)

    row = conn.execute("SELECT novelty FROM findings WHERE id = ?", (result_b.id,)).fetchone()
    stored_novelty = float(row["novelty"])

    recomputed = findings_memory.nearest_novelty(
        conn,
        params_b,
        artefacts=art_b,
        exclude_id=result_b.id,
    )

    assert math.isclose(recomputed, stored_novelty, rel_tol=1e-6)


def test_metrics_cli_exit_code_on_sla_breach(tmp_path):
    db_path = tmp_path / "experiments.sqlite"
    conn = findings_memory.connect(db_path)

    params = {"alpha": 3}
    artefacts = findings_memory.compute_params_artifacts(params)
    novelty = findings_memory.nearest_novelty(conn, params, artefacts=artefacts)
    fid = findings_memory.add_idea(conn, params, novelty, artefacts=artefacts).id

    with conn:
        conn.execute(
            "UPDATE findings SET created_at = ? WHERE id = ?",
            ("2024-01-01 00:00:00", fid),
        )

    findings_memory.update_quick(conn, fid, {"score": 0.8}, 0.8)
    findings_memory.promote_tested(conn, fid, {"sharpe": 0.6}, False)

    with conn:
        conn.execute(
            "UPDATE findings SET tested_at = ? WHERE id = ?",
            ("2024-01-03 01:00:00", fid),
        )

    exit_code = emp_cycle_metrics.main(["--db-path", str(db_path), "--threshold-hours", "24"])
    assert exit_code == 1
