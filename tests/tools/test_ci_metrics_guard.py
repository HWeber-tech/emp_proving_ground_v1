import json
from pathlib import Path

import pytest

from tools.telemetry.ci_metrics_guard import main as guard_main


@pytest.fixture()
def metrics_path(tmp_path: Path) -> Path:
    path = tmp_path / "metrics.json"
    return path


def _write_metrics(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_guard_flags_stale_trend(metrics_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_metrics(
        metrics_path,
        {
            "coverage_trend": [{"label": "2024-01-01T00:00:00+00:00"}],
            "formatter_trend": [],
            "coverage_domain_trend": [],
            "remediation_trend": [],
        },
    )

    exit_code = guard_main([
        "--metrics",
        str(metrics_path),
        "--max-age-hours",
        "1.0",
    ])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "STALE" in captured.out
    assert "coverage_trend" in captured.out
    assert "Stale telemetry trends" in captured.err


def test_guard_passes_when_within_threshold(
    metrics_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_metrics(
        metrics_path,
        {
            "coverage_trend": [{"label": "2024-01-01T00:00:00+00:00"}],
            "formatter_trend": [],
            "coverage_domain_trend": [],
            "remediation_trend": [],
        },
    )

    exit_code = guard_main([
        "--metrics",
        str(metrics_path),
        "--max-age-hours",
        "999999",
        "--require-trend",
        "coverage_trend",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "coverage_trend" in captured.out
    assert "STALE" not in captured.out
    assert captured.err == ""


def test_guard_outputs_json(metrics_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_metrics(
        metrics_path,
        {
            "coverage_trend": [{"label": "2024-06-05T00:00:00+00:00"}],
            "formatter_trend": [],
            "coverage_domain_trend": [],
            "remediation_trend": [],
        },
    )

    exit_code = guard_main([
        "--metrics",
        str(metrics_path),
        "--max-age-hours",
        "999999",
        "--format",
        "json",
        "--require-trend",
        "coverage_trend",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["trends"]["coverage_trend"]["entry_count"] == 1
    assert captured.err == ""
