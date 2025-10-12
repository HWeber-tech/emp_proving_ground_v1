from __future__ import annotations

from emp.core import findings_memory


def test_dedupe_identical_ideas(tmp_path):
    conn = findings_memory.connect(tmp_path / "experiments.sqlite")
    idea = {"alpha": 1, "beta": 2}

    artefacts = findings_memory.compute_params_artifacts(idea)
    novelty = findings_memory.nearest_novelty(conn, idea, artefacts=artefacts)
    first = findings_memory.add_idea(conn, idea, novelty, artefacts=artefacts, note="seed:1 git:abc")
    assert first.inserted

    duplicate = findings_memory.add_idea(conn, idea, novelty, artefacts=artefacts, note="seed:1 git:abc")
    assert not duplicate.inserted
    assert duplicate.stage == "idea"

    findings_memory.update_quick(conn, first.id, {"score": 0.9}, 0.9)
    second_duplicate = findings_memory.add_idea(conn, idea, novelty, artefacts=artefacts, note="seed:1 git:abc")
    assert not second_duplicate.inserted
    assert second_duplicate.stage == "screened"

    findings_memory.promote_tested(conn, first.id, {"sharpe": 1.1, "max_dd": 10}, False)
    new_attempt = findings_memory.add_idea(conn, idea, novelty, artefacts=artefacts)
    assert new_attempt.inserted
    assert new_attempt.duplicate_of == first.id

    notes = conn.execute("SELECT notes FROM findings WHERE id = ?", (new_attempt.id,)).fetchone()[0]
    assert "duplicate_of:" in (notes or "")
