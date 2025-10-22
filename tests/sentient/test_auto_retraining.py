from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.sentient.memory.auto_retraining import MemoryAutoRetrainer
from src.sentient.memory.faiss_pattern_memory import FAISSPatternMemory


class _ManualClock:
    def __init__(self, start: datetime) -> None:
        self._current = start

    def advance(self, seconds: float) -> None:
        self._current += timedelta(seconds=seconds)

    def now(self) -> datetime:
        return self._current


def _memory_config(tmp_path: Path, overrides: Dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "vector_dimension": 6,
        "index_path": str(tmp_path / "index.faiss"),
        "metadata_path": str(tmp_path / "metadata.json"),
        "stale_after_seconds": 3600,
        "decay_interval": 0,
        "max_memories": 3,
        "reinforcement_threshold": 2,
    }
    if overrides:
        base.update(overrides)
    return base


def _sample_metadata(tag: str) -> dict[str, Any]:
    return {
        "learning_signal": {
            "pnl": 1.0,
            "signal_type": "profitable",
            "context": {"market_condition": "bull", "strategy": tag},
        },
        "context": {"market_condition": "bull", "strategy": tag},
        "outcome": {"pnl": 1.0},
    }


def test_auto_retrainer_runs_compaction_and_prunes(tmp_path: Path) -> None:
    memory = FAISSPatternMemory(_memory_config(tmp_path))
    clock = _ManualClock(datetime(2024, 1, 1, 12, 0, 0))
    state_path = tmp_path / "state.json"
    retrainer = MemoryAutoRetrainer(
        memory,
        {"retrain_interval_seconds": 60, "state_path": str(state_path)},
        clock=clock.now,
    )

    base_vector = np.linspace(0.1, 0.6, memory.dimension, dtype=np.float32)
    duplicate = base_vector + 0.01
    vectors = [base_vector, base_vector, duplicate, duplicate, base_vector + 0.02]
    for idx, vector in enumerate(vectors):
        memory.add_experience(vector, _sample_metadata(f"alpha-{idx}"))

    summary = retrainer.run_cycle(force=True)

    assert summary["retrained"] is True
    rebuild = summary["rebuild"]
    assert isinstance(rebuild, dict)
    assert rebuild["compacted"] <= memory.max_memories
    assert rebuild["pruned"] >= 1

    stats = memory.get_memory_stats()
    assert stats["total_memories"] <= memory.max_memories
    assert retrainer.last_retrain_at == clock.now()
    assert state_path.exists()


def test_auto_retrainer_respects_interval_and_flushes_queue(tmp_path: Path) -> None:
    memory = FAISSPatternMemory(_memory_config(tmp_path))
    clock = _ManualClock(datetime(2024, 1, 1, 8, 30, 0))
    retrainer = MemoryAutoRetrainer(
        memory,
        {"retrain_interval_seconds": 120, "state_path": str(tmp_path / "state.json")},
        clock=clock.now,
    )

    base_vector = np.linspace(0.05, 0.35, memory.dimension, dtype=np.float32)
    memory.add_experience(base_vector, _sample_metadata("initial"))

    retrainer.run_cycle(force=True)
    first_run = retrainer.last_retrain_at
    stats_after_first = memory.get_memory_stats()["total_memories"]

    clock.advance(30)
    retrainer.register_experience(base_vector + 0.5, _sample_metadata("queued"))
    assert retrainer.pending_updates == 1

    summary = retrainer.run_cycle()

    assert summary["retrained"] is False
    assert summary["flushed"] == 1
    assert retrainer.pending_updates == 0
    assert retrainer.last_retrain_at == first_run
    assert summary["next_due_at"] == retrainer.next_due_at.isoformat()

    stats_after_second = memory.get_memory_stats()["total_memories"]
    assert stats_after_second == stats_after_first + 1
