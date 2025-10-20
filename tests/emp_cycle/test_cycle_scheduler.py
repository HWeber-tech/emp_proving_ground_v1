from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from emp.cli import emp_cycle_scheduler
from emp.core import findings_memory


@pytest.fixture(autouse=True)
def _strategy_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMP_STRATEGY_FACTORY", "tests.emp_cycle.strategy_stub:make_strategy")


def _setup_idea(conn, params):
    artefacts = findings_memory.compute_params_artifacts(params)
    novelty = findings_memory.nearest_novelty(conn, params, artefacts=artefacts)
    return findings_memory.add_idea(conn, params, novelty, artefacts=artefacts).id


def test_scheduler_promotes_candidate(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.sqlite"
    baseline_path = tmp_path / "baseline.json"
    conn = findings_memory.connect(db_path)

    candidate_params = {"weight": 1.0}
    fid = _setup_idea(conn, candidate_params)

    args = [
        "--db-path",
        str(db_path),
        "--baseline-json",
        str(baseline_path),
        "--quick-threshold",
        "0.3",
        "--ucb-c",
        "0.2",
        "--full-timeout-secs",
        "5",
        "--max-full",
        "1",
        "--note",
        "scheduler:test",
    ]

    exit_code = emp_cycle_scheduler.main(args)
    assert exit_code == 0

    row = conn.execute("SELECT stage, tested_at, notes FROM findings WHERE id = ?", (fid,)).fetchone()
    assert row is not None
    assert row["stage"] in {"tested", "progress"}
    assert row["tested_at"] is not None
    notes = row["notes"] or ""
    assert "scheduler:test" in notes
    assert "evidence:quick-screen" in notes
    assert "evidence:full-eval" in notes
    assert "coverage:regression" in notes

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert "sharpe" in baseline


def test_scheduler_handles_failed_quick(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.sqlite"
    baseline_path = tmp_path / "baseline.json"
    conn = findings_memory.connect(db_path)

    failing_params = {"weight": 0.1}
    fid = _setup_idea(conn, failing_params)

    args = [
        "--db-path",
        str(db_path),
        "--baseline-json",
        str(baseline_path),
        "--quick-threshold",
        "3.0",
        "--max-full",
        "0",
    ]

    exit_code = emp_cycle_scheduler.main(args)
    assert exit_code == 0

    row = conn.execute("SELECT stage, notes FROM findings WHERE id = ?", (fid,)).fetchone()
    assert row is not None
    assert row["stage"] == "idea"
    assert "failed_screen" in (row["notes"] or "")


def test_scheduler_respects_instrument_fair_share(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.sqlite"
    baseline_path = tmp_path / "baseline.json"
    conn = findings_memory.connect(db_path)

    fids_by_instrument: dict[str, list[int]] = defaultdict(list)
    payloads = [
        ("ES", {"weight": 1.15, "instrument": "ES"}),
        ("ES", {"weight": 1.1, "meta": {"symbol": "es"}}),
        ("NQ", {"weight": 1.05, "instrument": "NQ"}),
        ("CL", {"weight": 1.05, "asset": "CL"}),
    ]
    for key, params in payloads:
        fid = _setup_idea(conn, params)
        fids_by_instrument[key].append(fid)

    args = [
        "--db-path",
        str(db_path),
        "--baseline-json",
        str(baseline_path),
        "--quick-threshold",
        "0.2",
        "--max-full",
        "3",
    ]

    exit_code = emp_cycle_scheduler.main(args)
    assert exit_code == 0

    rows = conn.execute("SELECT id, stage FROM findings").fetchall()
    stage_map = {int(row["id"]): str(row["stage"]) for row in rows}

    def _tested(fid: int) -> bool:
        return stage_map.get(fid) in {"tested", "progress"}

    assert sum(1 for fid in stage_map if stage_map[fid] in {"tested", "progress"}) == 3
    assert sum(1 for fid in fids_by_instrument["ES"] if _tested(fid)) == 1
    assert _tested(fids_by_instrument["NQ"][0])
    assert _tested(fids_by_instrument["CL"][0])
    assert any(stage_map[fid] == "screened" for fid in fids_by_instrument["ES"])
