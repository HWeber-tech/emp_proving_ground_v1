from __future__ import annotations

import json

from emp.core import findings_memory


def test_novelty_uses_cached_vectors(tmp_path):
    conn = findings_memory.connect(tmp_path / "experiments.sqlite")

    last_novelty = None
    for idx in range(500):
        params = {"alpha": idx, "beta": idx % 5}
        artefacts = findings_memory.compute_params_artifacts(params)
        novelty = findings_memory.nearest_novelty(conn, params, artefacts=artefacts)
        res = findings_memory.add_idea(conn, params, novelty, artefacts=artefacts)
        assert res.inserted
        last_novelty = novelty

    assert last_novelty is not None
    assert 0.0 <= last_novelty <= 1.0

    row = conn.execute("SELECT params_vec FROM findings ORDER BY id DESC LIMIT 1").fetchone()
    vec = json.loads(row["params_vec"])
    assert len(vec) == 8
