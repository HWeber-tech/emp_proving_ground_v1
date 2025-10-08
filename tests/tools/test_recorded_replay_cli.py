from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.evolution import recorded_replay_cli


try:
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - python 3.10 fallback
    UTC = timezone.utc  # type: ignore[assignment]


def _snapshot(at: datetime, price: float, strength: float, confidence: float) -> dict[str, object]:
    return {
        "generated_at": at.isoformat(),
        "integrated_signal": {
            "strength": strength,
            "confidence": confidence,
        },
        "dimensions": {
            "WHAT": {
                "metadata": {
                    "last_close": price,
                }
            }
        },
    }


def _write_dataset(path: Path) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    snapshots = [
        _snapshot(start + timedelta(minutes=idx), 100.0 + idx * 0.5, 0.6 - 0.1 * (idx % 4), 0.7 + 0.05 * (idx % 3))
        for idx in range(12)
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in snapshots) + "\n", encoding="utf-8")


def test_cli_evaluates_genome_json(tmp_path, capsys) -> None:
    dataset = tmp_path / "replay.jsonl"
    _write_dataset(dataset)

    genome = {
        "id": "aggressive",
        "parameters": {
            "entry_threshold": 0.45,
            "exit_threshold": 0.25,
            "risk_fraction": 0.3,
            "min_confidence": 0.55,
        },
    }

    exit_code = recorded_replay_cli.main(
        [
            "--dataset",
            str(dataset),
            "--genome",
            json.dumps(genome),
            "--format",
            "json",
            "--indent",
            "0",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["dataset"] == str(dataset)
    assert payload["snapshots_loaded"] == 12
    assert len(payload["evaluations"]) == 1
    evaluation = payload["evaluations"][0]
    assert evaluation["genome_id"] == "aggressive"
    assert "max_drawdown" in evaluation["metrics"]
    assert evaluation["lineage"]["dimension"] == "EVOLUTION"


def test_cli_renders_markdown_from_file(tmp_path, capsys) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)

    genome_definitions = [
        {
            "id": "balanced",
            "parameters": {"entry_threshold": 0.4, "exit_threshold": 0.2, "risk_fraction": 0.25},
        },
        {
            "id": "defensive",
            "parameters": {"entry_threshold": 0.65, "exit_threshold": 0.35, "risk_fraction": 0.15},
            "metadata": {"notes": "stress test"},
        },
    ]
    genome_file = tmp_path / "genomes.json"
    genome_file.write_text(json.dumps(genome_definitions), encoding="utf-8")

    exit_code = recorded_replay_cli.main(
        [
            "--dataset",
            str(dataset),
            "--genome-file",
            str(genome_file),
            "--format",
            "markdown",
            "--dataset-id",
            "demo-dataset",
            "--evaluation-id",
            "run-001",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    output = captured.out
    assert "Recorded sensory replay evaluation" in output
    assert "Genome `balanced`" in output
    assert "Genome `defensive`" in output
    assert str(dataset) in output
    assert "Warn drawdown threshold" in output


def test_cli_requires_dataset(tmp_path) -> None:
    genome = {"id": "sample", "parameters": {"entry_threshold": 0.4}}
    missing_dataset = tmp_path / "missing.jsonl"

    exit_code = recorded_replay_cli.main(
        [
            "--dataset",
            str(missing_dataset),
            "--genome",
            json.dumps(genome),
        ]
    )

    assert exit_code == 1


def test_cli_requires_genomes(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)

    exit_code = recorded_replay_cli.main(["--dataset", str(dataset)])

    assert exit_code == 1
